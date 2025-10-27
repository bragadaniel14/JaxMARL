"""
Based on PureJaxRL Implementation of PPO.

Note, this file will only work for MPE environments with homogenous agents (e.g. Simple Spread).

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

import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper

import wandb
import functools


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
    
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        name=config["RUN_NAME"],
        tags=["IPPO", "RNN"],
        config=config,
        mode=config["WANDB_MODE"],
        reinit=True  # Allow multiple wandb.init() calls in same process
    )
    
    try:
        rng = jax.random.PRNGKey(config["SEED"])
        train_jit = jax.jit(make_train(config), device=jax.devices()[0])
        out = train_jit(rng)
        
        # === LOG FINAL METRICS AND CHARTS ===
        import json
        
        # Create loss charts
        updates_x = jnp.arange(out["metrics"]["total_loss"][0].shape[0])
        loss_table = jnp.stack([
            updates_x, 
            out["metrics"]["total_loss"].mean(axis=0), 
            out["metrics"]["actor_loss"].mean(axis=0), 
            out["metrics"]["critic_loss"].mean(axis=0), 
            out["metrics"]["entropy"].mean(axis=0), 
            out["metrics"]["ratio"].mean(axis=0)
        ], axis=1)
        loss_table_wandb = wandb.Table(
            data=loss_table.tolist(), 
            columns=["updates", "total_loss", "actor_loss", "critic_loss", "entropy", "ratio"]
        )
        
        # Create returns chart
        returns_updates_x = jnp.arange(out["metrics"]["returned_episode_returns"][0].shape[0])
        returns_table = jnp.stack([
            returns_updates_x, 
            out["metrics"]["returned_episode_returns"].mean(axis=0)
        ], axis=1)
        returns_table_wandb = wandb.Table(
            data=returns_table.tolist(), 
            columns=["updates", "returns"]
        )
        
        # Log charts and final metrics
        wandb.log({
            "charts/total_loss": wandb.plot.line(loss_table_wandb, "updates", "total_loss", title="Total Loss vs Updates"),
            "charts/actor_loss": wandb.plot.line(loss_table_wandb, "updates", "actor_loss", title="Actor Loss vs Updates"),
            "charts/critic_loss": wandb.plot.line(loss_table_wandb, "updates", "critic_loss", title="Critic Loss vs Updates"),
            "charts/entropy": wandb.plot.line(loss_table_wandb, "updates", "entropy", title="Entropy vs Updates"),
            "charts/ratio": wandb.plot.line(loss_table_wandb, "updates", "ratio", title="Ratio vs Updates"),
            "charts/returns": wandb.plot.line(returns_table_wandb, "updates", "returns", title="Returns vs Updates"),
            "final/returns": out["metrics"]["returned_episode_returns"][:, -1].mean(),
            "final/total_loss": out["metrics"]["total_loss"][:, -1].mean(),
        })
        
        # === SAVE FINAL MODEL PARAMETERS TO WANDB ===
        final_train_state = out["runner_state"][0][0]  # (train_state, env_state, ...)
        params_bytes = flax.serialization.to_bytes(final_train_state.params)
        
        # Save model params
        with open("final_model_params.msgpack", "wb") as f:
            f.write(params_bytes)
        wandb.save("final_model_params.msgpack")  # Upload to W&B run folder
        
        # === SAVE CONFIGURATION AS JSON ===
        config_to_save = {
            "num_agents": config.get("num_agents", 3),
            "num_landmarks": config.get("num_landmarks", 3),
            "ENV_NAME": config.get("ENV_NAME"),
            "ENV_KWARGS": config.get("ENV_KWARGS", {}),
            "SEED": config.get("SEED"),
            "TOTAL_TIMESTEPS": config.get("TOTAL_TIMESTEPS"),
            "NUM_ENVS": config.get("NUM_ENVS"),
            "LR": config.get("LR"),
            "RUN_NAME": config.get("RUN_NAME"),
            "full_config": config  # Save entire config for reference
        }
        with open("run_config.json", "w") as f:
            json.dump(config_to_save, f, indent=2)
        wandb.save("run_config.json")  # Upload to W&B run folder
        
        # Log final metrics summary
        wandb.summary["num_agents"] = config.get("num_agents", 3)
        wandb.summary["num_landmarks"] = config.get("num_landmarks", 3)
        wandb.summary["final_returns"] = float(out["metrics"]["returned_episode_returns"][:, -1].mean())
        wandb.summary["final_total_loss"] = float(out["metrics"]["total_loss"][:, -1].mean())
        
        return out
    finally:
        # Always finish the wandb run to clean up properly
        wandb.finish()

def load_agent_from_wandb(wandb_run_path, config=None):
    """
    Load a trained IPPO RNN MPE agent from a wandb run folder.
    
    Args:
        wandb_run_path: Path to wandb run folder (e.g., "wandb/offline-run-20251027_174154-ohjc4ruk")
                       or just the run ID (e.g., "ohjc4ruk")
        config: Optional config dict. If None, will try to load from run_config.json
        
    Returns:
        tuple: (params, config, network) where:
            - params: Loaded model parameters
            - config: Configuration dict with environment settings
            - network: ActorCriticRNN network instance
    """
    from pathlib import Path
    import json
    
    # Handle both full path and just run ID
    run_path = Path(wandb_run_path)
    if not run_path.exists():
        # Try to find by ID
        wandb_dir = Path("wandb")
        matching_runs = list(wandb_dir.glob(f"*-{wandb_run_path}"))
        if not matching_runs:
            raise FileNotFoundError(f"Could not find run with ID {wandb_run_path}")
        run_path = matching_runs[0]
    
    # Load model parameters
    params_path = run_path / "files" / "final_model_params.msgpack"
    if not params_path.exists():
        raise FileNotFoundError(f"Model parameters not found at {params_path}")
    
    with open(params_path, "rb") as f:
        params = flax.serialization.from_bytes(None, f.read())
    
    # Load config if not provided
    if config is None:
        config_path = run_path / "files" / "run_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = json.load(f)
                config = config_data.get("full_config", config_data)
        else:
            raise FileNotFoundError(
                f"Configuration not found at {config_path}. "
                "Please provide config manually or ensure run_config.json exists."
            )
    
    # Create environment to get action space
    env_kwargs = config.get("ENV_KWARGS", {})
    env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
    
    # Create network
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config)
    
    return params, config, network


def run_and_visualize(wandb_run_path, config=None, num_steps=100, max_env_steps=50, 
                      seed=42, save_animation=None, return_animation=True):
    """
    Load an agent from wandb, run a simulation, and visualize the results.
    
    Args:
        wandb_run_path: Path to wandb run folder or run ID
        config: Optional config dict. If None, will load from run_config.json
        num_steps: Number of simulation steps to run
        max_env_steps: Maximum steps per episode for the environment
        seed: Random seed for reproducibility
        save_animation: Optional filename to save animation (e.g., "animation.gif")
        return_animation: If True, returns HTML animation object for notebooks
        
    Returns:
        HTML animation object if return_animation=True, else None
    """
    from IPython.display import HTML
    
    # Load the agent
    print(f"Loading agent from {wandb_run_path}...")
    params, config, network = load_agent_from_wandb(wandb_run_path, config)
    
    env_kwargs = config.get("ENV_KWARGS", {})
    num_agents = env_kwargs.get("num_agents", 3)
    num_landmarks = env_kwargs.get("num_landmarks", 3)
    
    print(f"Configuration: {num_agents} agents, {num_landmarks} landmarks")
    
    # Create environment
    from jaxmarl.wrappers.baselines import MPELogWrapper
    env = MPELogWrapper(jaxmarl.make(config["ENV_NAME"], max_steps=max_env_steps, **env_kwargs))
    
    # Initialize environment
    config_sim = config.copy()
    config_sim['NUM_ENVS'] = 1
    rng = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rng, config_sim["NUM_ENVS"])
    obs, env_state = jax.vmap(env.reset, in_axes=(0,))(rngs)
    
    # Initialize network hidden state
    num_actors = env.num_agents * config_sim["NUM_ENVS"]
    hstate = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])
    
    # Run simulation
    print(f"Running simulation for {num_steps} steps...")
    state_seq = [env_state.env_state]
    
    for step in range(num_steps):
        # Batchify observations for RNN
        obs_batch = jnp.stack([obs[a] for a in env.agents])
        obs_batch = obs_batch.transpose(1, 0, 2).reshape(-1, obs_batch.shape[-1])
        dones = jnp.zeros((num_actors,), dtype=jnp.float32)
        ac_in = (obs_batch[None, :, :], dones[None, :])  # add time dim for scan
        
        # Get actions from network
        hstate, pi, _ = network.apply(params, hstate, ac_in)
        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng)
        
        # Unbatch actions for environment
        actions = actions.reshape(config_sim["NUM_ENVS"], env.num_agents)
        act_dict = {a: np.array(actions[:, i]) for i, a in enumerate(env.agents)}
        
        # Step environment
        rng_step = jax.random.split(_rng, config_sim["NUM_ENVS"])
        obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
            rng_step, env_state, act_dict
        )
        
        state_seq.append(env_state.env_state)
    
    print(f"Simulation complete. Creating visualization...")
    
    # Visualize
    try:
        from baselines.Custom.simple_spread_visualizer import MPEVisualizer
    except ImportError:
        # Fallback if visualizer is in different location
        import sys
        sys.path.insert(0, "baselines/Custom")
        from simple_spread_visualizer import MPEVisualizer
    
    env_viz = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
    viz = MPEVisualizer(env_viz, state_seq)
    
    if save_animation:
        print(f"Saving animation to {save_animation}...")
        ani = viz.animate(save_fname=save_animation, view=False)
        print(f"Animation saved!")
    
    if return_animation:
        return viz.animate()
    else:
        viz.animate(view=False)
        return None


if __name__ == "__main__":
    main()
