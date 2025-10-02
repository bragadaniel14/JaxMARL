from baselines.Custom import ippo_ff_mpe, ippo_rnn_mpe,  mappo_rnn_mpe
import importlib.resources as resources
import yaml
from omegaconf import OmegaConf
import argparse
import jax




if __name__ == "__main__":
    print(f"We are dealing with device {jax.devices()}")
    algo_and_config_map = {
        "ippo_ff_mpe": (ippo_ff_mpe, OmegaConf.load(resources.files("baselines.Custom.config") / "ippo_ff_mpe_simple_spread.yaml")),
        "ippo_rnn_mpe": (ippo_rnn_mpe, OmegaConf.load(resources.files("baselines.IPPO.config") / "ippo_rnn_mpe.yaml")),
        "mappo_rnn_mpe": (mappo_rnn_mpe, OmegaConf.load(resources.files("baselines.MAPPO.config") / "mappo_homogenous_rnn_mpe.yaml"))
    }

    parser = argparse.ArgumentParser(description="A simple starter script")
    parser.add_argument("--algo_type", type=str, default="ippo_ff_mpe")
    args = parser.parse_args()

    algo, config = algo_and_config_map[args.algo_type]

    algo.main(config)

    print(f"Finished rujnning algo {args.algo_type}")
