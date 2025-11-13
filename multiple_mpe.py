from baselines.Custom import ippo_ff_mpe, ippo_rnn_mpe,  mappo_rnn_mpe
import importlib.resources as resources
import yaml
from omegaconf import OmegaConf
import argparse
import jax
import time
import logging




if __name__ == "__main__":
    algo_and_config_map = {
        "ippo_ff_mpe": (ippo_ff_mpe, OmegaConf.load(resources.files("baselines.Custom.config") / "ippo_ff_mpe_simple_spread.yaml")),
        "ippo_rnn_mpe": (ippo_rnn_mpe, OmegaConf.load(resources.files("baselines.Custom.config") / "ippo_rnn_mpe_simple_spread.yaml")),
        "mappo_rnn_mpe": (mappo_rnn_mpe, OmegaConf.load(resources.files("baselines.Custom.config") / "mappo_rnn_mpe_simple_spread.yaml"))
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_type", type=str, default="ippo_ff_mpe")
    parser.add_argument("--wandb_folder", type=str, default=None)
    parser.add_argument("--log_file", type=str, default="training.log")
    parser.add_argument("--max_agents", type=int, default=5, help="Range of number of agents to run (inclusive, exclusive)")
    parser.add_argument("--max_landmarks", type=int, default=5, help="Range of number of landmarks to run (inclusive, exclusive)")
    parser.add_argument("--min_agents", type=int, default=1, help="Range of number of agents to run (inclusive, exclusive)")
    parser.add_argument("--min_landmarks", type=int, default=1, help="Range of number of agents to run (inclusive, exclusive)")

    args = parser.parse_args()
    algo, config = algo_and_config_map[args.algo_type]


    # Set up logging
    logging.basicConfig(filename=args.log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"We are dealing with device {jax.devices()}")
    logging.info(f"We are dealing with device {jax.devices()}")    

    for num_agents in range(args.min_agents, args.max_agents+1):
        for num_landmarks in range(args.min_landmarks, args.max_landmarks+1):
            start = time.time()
            logging.info(f"Start to run {num_agents} {num_landmarks}")
            print(f"Start to run {num_agents} {num_landmarks}")
            config["ENV_KWARGS"]["num_agents"] = num_agents
            config["ENV_KWARGS"]["num_landmarks"] = num_landmarks
            config['RUN_NAME'] = f"{args.algo_type}-spread_{num_landmarks}_landmarks_{num_agents}_agents" 
            if args.wandb_folder is not None:
                config['WANDB_DIR'] = f"{args.wandb_folder}/agents{num_agents}-landmarks{num_landmarks}"
            print(f"Running MPE Training with args {args}, and config {config}")
            logging.info(f"Running MPE Training with args {args}, and config {config}")
            algo.main(config)
            print(f"Finished in {time.time()-start}")
            logging.info(f"Finished in {time.time()-start}")
            

    print(f"Finished running algo {args.algo_type}")
