""" 
Based on the PureJaxRL Implementation of PPO

Functions:
    - main(config): Train an IPPO-FF agent and save to wandb
    - load_params_from_path(saved_path): Load trained model parameters and config
    - evaluate_policy(params, env_config, ...): Evaluate a trained policy
    - plot_simulation(saved_path, ...): Load and visualize a trained model

Example usage for visualization:
    from baselines.Custom import ippo_ff_mpe
    anim = ippo_ff_mpe.plot_simulation(saved_path="wandb/offline-run-XXXXX")
"""

import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper 
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import wandb
import os
import json
from pathlib import Path

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
    
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    max_dim = max([x[a].shape[-1] for a in agent_list])
    def pad(z, length):
        return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + [length - z.shape[-1]])], -1)

    x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    env = LogWrapper(env)
    
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        network = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act,
                )

                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
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
                    unroll=8,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
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

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    loss_info = {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "critic_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                        "ratio": total_loss[1][3],
                    }
                    
                    return train_state, loss_info

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            def callback(metric):
                wandb.log(
                    metric
                )

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            r0 = {"ratio0": loss_info["ratio"][0,0].mean()}
            # jax.debug.print('ratio0 {x}', x=r0["ratio0"])
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric = {**metric, **loss_info, **r0}
            jax.experimental.io_callback(callback, None, metric)
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def main(config):
    config = OmegaConf.to_container(config) 
    
    # Add environment configuration details to wandb config for easy retrieval
    env_kwargs = config.get("ENV_KWARGS", {})
    if env_kwargs:
        config["num_agents"] = env_kwargs.get("num_agents", 3)
        config["num_landmarks"] = env_kwargs.get("num_landmarks", 3)
    run_name = f"ippo_ff_mpe-offline/agents{config.get('num_agents', 3)}-landmarks{config.get('num_landmarks', 3)}"
    wandb_dir = Path(run_name)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    config["WANDB_DIR"] = str(wandb_dir)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        name=config["RUN_NAME"],
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
        dir=config["WANDB_DIR"],
        reinit=True,  # Allow multiple runs in same session
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])    
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    updates_x = jnp.arange(out["metrics"]["total_loss"][0].shape[0])
    loss_table = jnp.stack([updates_x, out["metrics"]["total_loss"].mean(axis=0), out["metrics"]["actor_loss"].mean(axis=0), out["metrics"]["critic_loss"].mean(axis=0), out["metrics"]["entropy"].mean(axis=0), out["metrics"]["ratio"].mean(axis=0)], axis=1)    
    loss_table = wandb.Table(data=loss_table.tolist(), columns=["updates", "total_loss", "actor_loss", "critic_loss", "entropy", "ratio"])
    updates_x = jnp.arange(out["metrics"]["returned_episode_returns"][0].shape[0])
    returns_table = jnp.stack([updates_x, out["metrics"]["returned_episode_returns"].mean(axis=0)], axis=1)
    returns_table = wandb.Table(data=returns_table.tolist(), columns=["updates", "returns"])
    wandb.log({
        "returns_plot": wandb.plot.line(returns_table, "updates", "returns", title="returns_vs_updates"),
        "returns": out["metrics"]["returned_episode_returns"][:,-1].mean(),
        "total_loss_plot": wandb.plot.line(loss_table, "updates", "total_loss", title="total_loss_vs_updates"),
        "actor_loss_plot": wandb.plot.line(loss_table, "updates", "actor_loss", title="actor_loss_vs_updates"),
        "critic_loss_plot": wandb.plot.line(loss_table, "updates", "critic_loss", title="critic_loss_vs_updates"),
        "entropy_plot": wandb.plot.line(loss_table, "updates", "entropy", title="entropy_vs_updates"),
        "ratio_plot": wandb.plot.line(loss_table, "updates", "ratio", title="ratio_vs_updates"),
    })

    # === SAVE FINAL MODEL PARAMETERS TO WANDB ===
    # In feedforward version, runner_state is (train_state, env_state, last_obs, rng)
    # After vmap, it becomes a tuple where each element has shape (NUM_SEEDS, ...)
    # So out["runner_state"][0] gives train_state array, [0] gets first seed
    final_train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])
    params_bytes = flax.serialization.to_bytes(final_train_state.params)
    
    # Save as artifact
    with open("final_model_params.msgpack", "wb") as f:
        f.write(params_bytes)
    wandb.save("final_model_params.msgpack")  # Upload to W&B run folder
    
    # Save config with environment details to JSON for easy loading
    config_to_save = {
        "num_agents": config.get("num_agents", 3),
        "num_landmarks": config.get("num_landmarks", 3),
        "ENV_NAME": config["ENV_NAME"],
        "ACTIVATION": config.get("ACTIVATION", "tanh"),
        "algo_type": "ippo_ff_mpe"
    }
    with open("env_config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)
    wandb.save("env_config.json")
    
    # Log final metrics summary
    wandb.summary["num_agents"] = config.get("num_agents", 3)
    wandb.summary["num_landmarks"] = config.get("num_landmarks", 3)
    wandb.finish()
    return out
    


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
    Evaluate a trained IPPO-FF policy on fresh episodes using greedy (deterministic) actions.
    
    Args:
        params: Trained model parameters
        env_config: Environment configuration dict with keys like:
                   'num_agents', 'num_landmarks', 'ENV_NAME', 'ACTIVATION'
        num_eval_episodes: Number of episodes to evaluate
        eval_seed: Random seed for evaluation (different from training)
        max_steps: Maximum steps per episode (default: 100)
    
    Returns:
        tuple: (metrics dict, episode_returns array, episode_lengths array)
    """
    
    # Build config dict for the network
    config = {
        "ENV_NAME": env_config.get("ENV_NAME", "MPE_simple_spread_v3"),
        "ACTIVATION": env_config.get("ACTIVATION", "tanh"),
        "ENV_KWARGS": {
            "num_agents": env_config.get("num_agents", 3),
            "num_landmarks": env_config.get("num_landmarks", 3)
        }
    }
    
    # Create environment with appropriate max_steps
    env = LogWrapper(jaxmarl.make(config["ENV_NAME"], max_steps=max_steps, **config["ENV_KWARGS"]))
    network = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
    
    # Use single environment for evaluation
    num_envs = 1
    num_actors = env.num_agents * num_envs
    
    # Storage for results
    episode_returns = []
    episode_lengths = []
    
    rng = jax.random.PRNGKey(eval_seed)
    
    for episode_idx in range(num_eval_episodes):
        # Reset environment
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng)
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            # Prepare observation batch
            obs_batch = batchify(obs, env.agents, num_actors)
            
            # Get action from policy (feedforward - no hidden state)
            pi, value = network.apply(params, obs_batch)
            
            # Deterministic action (greedy)
            actions = jnp.argmax(pi.logits, axis=-1)
            
            # Unbatch actions for environment
            act_dict = unbatchify(actions, env.agents, num_envs, env.num_agents)
            act_dict = {k: int(v.squeeze()) for k, v in act_dict.items()}
            
            # Step environment
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


def plot_simulation(saved_path, max_steps=100, num_episodes=1, seed=42):
    """
    Load a trained IPPO-FF model from an offline wandb path and visualize it.
    
    Args:
        saved_path: Path to the offline wandb run directory (required)
        max_steps: Maximum number of steps to simulate (default: 100)
        num_episodes: Number of episodes to visualize (default: 1)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Animation object from MPEVisualizer
    """
    # Load parameters and config
    params, env_config = load_params_from_path(saved_path)
    
    num_agents = env_config.get("num_agents", 3)
    num_landmarks = env_config.get("num_landmarks", 3)
    env_name = env_config.get("ENV_NAME", "MPE_simple_spread_v3")
    activation = env_config.get("ACTIVATION", "tanh")
    
    print(f"Environment config: num_agents={num_agents}, num_landmarks={num_landmarks}")
    
    # Create environment
    env_kwargs = {"num_agents": num_agents, "num_landmarks": num_landmarks}
    env = LogWrapper(jaxmarl.make(env_name, max_steps=max_steps, **env_kwargs))
    
    # Create network
    network = ActorCritic(env.action_space(env.agents[0]).n, activation=activation)
    
    # Initialize RNG
    rng = jax.random.PRNGKey(seed)
    num_envs = 1
    num_actors = env.num_agents * num_envs
    
    # Run simulation
    all_state_sequences = []
    
    for episode_idx in range(num_episodes):
        print(f"Running episode {episode_idx + 1}/{num_episodes}...")
        
        # Reset environment with vmap (even for single env)
        rng, reset_rng = jax.random.split(rng)
        rngs = jax.random.split(reset_rng, num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0,))(rngs)
        
        # Collect states
        state_seq = [env_state.env_state]
        
        for step in range(max_steps):
            # Batchify observations matching training format
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            obs_batch = obs_batch.transpose(1, 0, 2).reshape(-1, obs_batch.shape[-1])
            
            # Get action from policy (feedforward - no hidden state)
            pi, _ = network.apply(params, obs_batch)
            
            # Sample actions
            rng, action_rng = jax.random.split(rng)
            actions = pi.sample(seed=action_rng)
            
            # Reshape actions for environment (num_envs, num_agents)
            actions = actions.reshape(num_envs, env.num_agents)
            act_dict = {a: np.array(actions[:, i]) for i, a in enumerate(env.agents)}
            
            # Step environment with vmap
            rng_step = jax.random.split(action_rng, num_envs)
            obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                rng_step, env_state, act_dict
            )
            
            state_seq.append(env_state.env_state)
            
            # Check if episode is done
            if done["__all__"][0]:
                print(f"Episode {episode_idx + 1} finished at step {step + 1}")
                break
        
        all_state_sequences.append(state_seq)
    
    # Visualize
    print("Creating visualization...")
    try:
        from baselines.Custom.simple_spread_visualizer import MPEVisualizer
    except ImportError as e:
        raise ImportError(
            "MPEVisualizer not found. Make sure baselines.Custom.simple_spread_visualizer is available."
        ) from e
    
    # Create base environment for visualization
    env_base = jaxmarl.make(env_name, **env_kwargs)
    viz = MPEVisualizer(env_base, all_state_sequences[0])
    anim = viz.animate()
    
    print("Visualization complete!")
    return anim


if __name__ == "__main__":
    main()