"""
Based on PureJaxRL Implementation of PPO.

Note, this file will only work for MPE environments with homogenous agents (e.g. Simple Spread).

Functions:
    - main(config): Train an IPPO-RNN agent and save to wandb
    - plot(saved_path, max_steps, num_episodes, seed): Load a trained model from offline wandb 
      and visualize it running in the environment

Example usage for visualization:
    from baselines.Custom import ippo_rnn_mpe
    anim = ippo_rnn_mpe.plot(saved_path="wandb/offline-run-20251027_174612-g3qd2e3o")

"""

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf
import os

import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper
from baselines.Custom.simple_spread_visualizer import MPEVisualizer

import wandb
import functools
import json
from pathlib import Path


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)        

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env_kwargs = config.get("ENV_KWARGS", {})
    env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
    
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    env = MPELogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jnp.reshape(
                    init_hstate, (1, config["NUM_ACTORS"], -1)
                )
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            ratio_0 = loss_info[1][3].at[0,0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    {
                        # the metrics have an agent dimension, but this is identical
                        # for all agents so index into the 0th item of that dimension.
                        "returns": metric["returned_episode_returns"][:, :, 0][
                            metric["returned_episode"][:, :, 0]
                        ].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                        **metric["loss"],
                    }
                )

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def main(config):
    config = OmegaConf.to_container(config)
    
    # Add environment configuration details to wandb config for easy retrieval
    env_kwargs = config.get("ENV_KWARGS", {})
    if env_kwargs:
        config["num_agents"] = env_kwargs.get("num_agents", 3)  # default 3
        config["num_landmarks"] = env_kwargs.get("num_landmarks", 3)  # default 3
    
    # Set wandb run directory name with custom format
    run_folder = config['WANDB_DIR']
    wandb_dir = Path(run_folder)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    config["WANDB_DIR"] = str(wandb_dir)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        name=config["RUN_NAME"],
        tags=["IPPO", "RNN"],
        config=config,
        mode=config["WANDB_MODE"],
        dir=str(wandb_dir),  # Set the directory where wandb files are saved
        reinit=True  # Allow multiple wandb.init() calls in same process
    )
    
    try:
        rng = jax.random.PRNGKey(config["SEED"])
        train_jit = jax.jit(make_train(config), device=jax.devices()[0])
        out = train_jit(rng)

        
        # === SAVE FINAL MODEL PARAMETERS TO WANDB ===
        final_train_state = out["runner_state"][0][0]  # (train_state, env_state, ...)
        params_bytes = flax.serialization.to_bytes(final_train_state.params)
        
        # Save directly to wandb.run.dir to avoid symlink issues
        params_path = os.path.join(wandb.run.dir, "final_model_params.msgpack")
        with open(params_path, "wb") as f:
            f.write(params_bytes)
        
        # Save config with environment details to JSON for easy loading
        config_to_save = {
            "num_agents": config.get("num_agents", 3),
            "num_landmarks": config.get("num_landmarks", 3),
            "ENV_NAME": config["ENV_NAME"],
            "GRU_HIDDEN_DIM": config["GRU_HIDDEN_DIM"],
            "FC_DIM_SIZE": config["FC_DIM_SIZE"],
            "algo_type": "ippo_rnn_mpe"
        }
        config_path = os.path.join(wandb.run.dir, "env_config.json")
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)
        
        # Log final metrics summary
        wandb.summary["num_agents"] = config.get("num_agents", 3)
        wandb.summary["num_landmarks"] = config.get("num_landmarks", 3)
        
        return out
    finally:
        # Always finish the wandb run to clean up properly
        wandb.finish()


def load_params_from_path(saved_path):
    """Load model parameters from a given path."""
    saved_path = Path(saved_path)
    if saved_path.name != "files":
        saved_path = saved_path / "files"
    saved_path = saved_path.resolve()
    params_file = saved_path / "final_model_params.msgpack"
    if not params_file.exists():
        raise ValueError(f"Model parameters file not found: {params_file}")
    
    print(f"Loading model parameters from: {params_file}")
    with open(params_file, "rb") as f:
        params = flax.serialization.from_bytes(None, f.read())

    env_config_file = saved_path / "env_config.json"
    if not env_config_file.exists():
        raise ValueError(f"Environment config file not found: {env_config_file}")
    print(f"Loading environment config from: {env_config_file}")
    with open(env_config_file, "r") as f:
        env_config = json.load(f)
    return params, env_config

def evaluate_policy(params, env_config, num_eval_episodes=100, eval_seed=42, max_steps=100):
    """
    Evaluate a trained IPPO policy on fresh episodes using greedy (deterministic) actions.
    
    Args:
        params: Trained model parameters
        env_config: Environment configuration dict with keys like:
                   'num_agents', 'num_landmarks', 'ENV_NAME', 'GRU_HIDDEN_DIM', 'FC_DIM_SIZE'
        num_eval_episodes: Number of episodes to evaluate
        eval_seed: Random seed for evaluation (different from training)
        max_steps: Maximum steps per episode (default: 100)
    
    Returns:
        tuple: (metrics dict, episode_returns array, episode_lengths array)
    """
    
    # Build config dict for the network
    config = {
        "ENV_NAME": env_config.get("ENV_NAME", "MPE_simple_spread_v3"),
        "GRU_HIDDEN_DIM": env_config.get("GRU_HIDDEN_DIM", 128),
        "FC_DIM_SIZE": env_config.get("FC_DIM_SIZE", 128),
        "ENV_KWARGS": {
            "num_agents": env_config.get("num_agents", 3),
            "num_landmarks": env_config.get("num_landmarks", 3)
        }
    }
    
    # Create environment with appropriate max_steps
    env = MPELogWrapper(jaxmarl.make(config["ENV_NAME"], max_steps=max_steps, **config["ENV_KWARGS"]))
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config)
    
    # Use single environment for evaluation (could also use multiple for speed)
    num_envs = 1
    num_actors = env.num_agents * num_envs
    
    # Storage for results
    episode_returns = []
    episode_lengths = []
    
    rng = jax.random.PRNGKey(eval_seed)
    
    for episode_idx in range(num_eval_episodes):
        # Reset environment
        print(f"Starting evaluation episode {episode_idx + 1}/{num_eval_episodes}...")
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng)
        
        # Initialize hidden state
        hstate = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            # Prepare observation batch
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            obs_batch = obs_batch.reshape(-1, obs_batch.shape[-1])
            dones = jnp.zeros((num_actors,), dtype=jnp.float32)
            ac_in = (obs_batch[None, :, :], dones[None, :])
            
            # Get action from policy
            hstate, pi, value = network.apply(params, hstate, ac_in)
            
            # Deterministic action (greedy)
            actions = jnp.argmax(pi.logits, axis=-1)
            
            # Actions shape is (1, num_agents), squeeze to (num_agents,)
            actions = actions.squeeze()
            
            # Step environment
            act_dict = {a: int(actions[i]) for i, a in enumerate(env.agents)}
            rng, step_rng = jax.random.split(rng)
            obs, env_state, reward, done_dict, info = env.step(step_rng, env_state, act_dict)
            
            # Accumulate rewards (sum over all agents)
            episode_reward += sum(reward.values())
            episode_length += 1
            
            # Check if episode is done
            done = done_dict["__all__"]
        
        episode_returns.append(float(episode_reward))
        episode_lengths.append(episode_length)
        
        if (episode_idx + 1) % 20 == 0:
            print(f"Evaluated {episode_idx + 1}/{num_eval_episodes} episodes...")
    
    # Compute statistics
    episode_returns = np.array(episode_returns)
    episode_lengths = np.array(episode_lengths)
    
    metrics = {
        "mean_return": episode_returns.mean(),
        "std_return": episode_returns.std(),
        "min_return": episode_returns.min(),
        "max_return": episode_returns.max(),
        "median_return": np.median(episode_returns),
        "mean_episode_length": episode_lengths.mean(),
        "std_episode_length": episode_lengths.std(),
        "num_episodes": num_eval_episodes,
        "max_steps": max_steps
    }
    
    return metrics, episode_returns, episode_lengths


def plot_simulation(saved_path="", max_steps=100, num_episodes=1, seed=42):
    """
    Load a trained IPPO-RNN model from an offline wandb path and visualize it.
    
    Args:
        saved_path: Path to the offline wandb run directory, e.g., 
                   "wandb/offline-run-20251008_121315-shellov1" or
                   "wandb/offline-run-20251008_121315-shellov1/files"
        max_steps: Maximum number of steps to simulate (default: 100)
        num_episodes: Number of episodes to visualize (default: 1)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Animation object from MPEVisualizer
    """
    
    # Normalize path - handle both "files" subdirectory and parent directory
    saved_path = Path(saved_path)
    if saved_path.name != "files":
        saved_path = saved_path / "files"
    
    if not saved_path.exists():
        raise ValueError(f"Path does not exist: {saved_path}")
    
    saved_path = saved_path.resolve()
    # Load model parameters
    params_file = saved_path / "final_model_params.msgpack"
    if not params_file.exists():
        raise ValueError(f"Model parameters file not found: {params_file}")
    
    print(f"Loading model parameters from: {params_file}")
    with open(params_file, "rb") as f:
        params = flax.serialization.from_bytes(None, f.read())
    
    # Try to load config from env_config.json first
    config_file = saved_path / "env_config.json"
    print(f"Loading environment config from: {config_file}")
    with open(config_file, "r") as f:
        env_config = json.load(f)
    num_agents = env_config.get("num_agents", 3)
    num_landmarks = env_config.get("num_landmarks", 3)
    env_name = env_config.get("ENV_NAME", "MPE_simple_spread_v3")
    gru_hidden_dim = env_config.get("GRU_HIDDEN_DIM", 128)
    fc_dim_size = env_config.get("FC_DIM_SIZE", 128)

    
    print(f"Environment config: num_agents={num_agents}, num_landmarks={num_landmarks}")
    
    # Create configuration dict for the network
    config = {
        "ENV_NAME": env_name,
        "GRU_HIDDEN_DIM": gru_hidden_dim,
        "FC_DIM_SIZE": fc_dim_size,
        "NUM_ENVS": 1,
    }
    
    # Create environment
    env_kwargs = {"num_agents": num_agents, "num_landmarks": num_landmarks}
    env = MPELogWrapper(jaxmarl.make(env_name, max_steps=max_steps, **env_kwargs))
    
    # Create network
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config)
    
    # Initialize RNG
    rng = jax.random.PRNGKey(seed)
    
    # Run simulation for each episode
    all_state_sequences = []
    
    for episode_idx in range(num_episodes):
        print(f"Running episode {episode_idx + 1}/{num_episodes}...")
        
        # Reset environment
        rng, reset_rng = jax.random.split(rng)
        rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obs, env_state = jax.vmap(env.reset, in_axes=(0,))(rngs)
        
        # Initialize hidden state
        num_actors = env.num_agents * config["NUM_ENVS"]
        hstate = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])
        
        # Collect states
        state_seq = [env_state.env_state]
        
        for step in range(max_steps):
            # Batchify observations for RNN
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            obs_batch = obs_batch.transpose(1, 0, 2).reshape(-1, obs_batch.shape[-1])
            dones = jnp.zeros((num_actors,), dtype=jnp.float32)
            ac_in = (obs_batch[None, :, :], dones[None, :])  # add time dim for scan
            
            # Get action from policy
            hstate, pi, _ = network.apply(params, hstate, ac_in)
            
            # Sample actions
            rng, action_rng = jax.random.split(rng)
            actions = pi.sample(seed=action_rng)
            
            # Unbatch actions for environment
            actions = actions.reshape(config["NUM_ENVS"], env.num_agents)
            act_dict = {a: np.array(actions[:, i]) for i, a in enumerate(env.agents)}
            
            # Step environment
            rng_step = jax.random.split(action_rng, config["NUM_ENVS"])
            obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                rng_step, env_state, act_dict
            )
            
            state_seq.append(env_state.env_state)
            
            # Check if episode is done
            if done["__all__"][0]:
                print(f"Episode {episode_idx + 1} finished at step {step + 1}")
                break
        
        all_state_sequences.append(state_seq)
    
    # Visualize the first episode (or all if needed)
    print("Creating visualization...")
    # Create base environment for visualization (without wrapper)
    env_base = jaxmarl.make(env_name, **env_kwargs)
    viz = MPEVisualizer(env_base, all_state_sequences[0])
    anim = viz.animate()
    
    print("Visualization complete!")
    return anim


if __name__ == "__main__":
    main()
