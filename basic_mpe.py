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
        "ippo_rnn_mpe": (ippo_rnn_mpe, OmegaConf.load(resources.files("baselines.Custom.config") / "ippo_rnn_mpe_simple_spread.yaml")),
        "mappo_rnn_mpe": (mappo_rnn_mpe, OmegaConf.load(resources.files("baselines.Custom.config") / "mappo_homogenous_rnn_mpe_simple_spread.yaml"))
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_type", type=str, default="ippo_ff_mpe")
    parser.add_argument("--script_name", type=str, default=None)
    args = parser.parse_args()
    algo, config = algo_and_config_map[args.algo_type]
    
    if args.script_name:
        config = OmegaConf.load(resources.files("baselines.Custom.config") / args.script_name)

    print(f"Running MPE Training with args {args}, and config {config}")
    algo.main(config)

    print(f"Finished running algo {args.algo_type}")
