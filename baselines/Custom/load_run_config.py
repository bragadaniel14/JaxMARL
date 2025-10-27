"""
Helper script to load configuration from WandB offline runs.

Usage:
    python load_run_config.py <run_folder_or_id>
    
Example:
    python load_run_config.py wandb/offline-run-20251027_174154-ohjc4ruk
    python load_run_config.py ohjc4ruk
"""

import json
import os
import sys
from pathlib import Path


def load_config_from_run(run_path_or_id):
    """
    Load run_config.json from a wandb run folder.
    
    Args:
        run_path_or_id: Either full path to run folder or just the run ID
        
    Returns:
        dict: Configuration dictionary
    """
    # If it's just an ID, search for it in wandb folder
    if not os.path.exists(run_path_or_id):
        wandb_dir = Path("wandb")
        matching_runs = list(wandb_dir.glob(f"*-{run_path_or_id}"))
        if not matching_runs:
            raise FileNotFoundError(f"Could not find run with ID {run_path_or_id}")
        run_path = matching_runs[0]
    else:
        run_path = Path(run_path_or_id)
    
    # Load the config
    config_file = run_path / "files" / "run_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"run_config.json not found in {run_path}")
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    return config


def print_config_summary(config):
    """Print a nice summary of the configuration."""
    print("=" * 60)
    print("RUN CONFIGURATION")
    print("=" * 60)
    print(f"Run Name:        {config.get('RUN_NAME')}")
    print(f"Num Agents:      {config.get('num_agents')}")
    print(f"Num Landmarks:   {config.get('num_landmarks')}")
    print(f"Environment:     {config.get('ENV_NAME')}")
    print(f"Total Timesteps: {config.get('TOTAL_TIMESTEPS')}")
    print(f"Num Envs:        {config.get('NUM_ENVS')}")
    print(f"Learning Rate:   {config.get('LR')}")
    print(f"Seed:            {config.get('SEED')}")
    print("=" * 60)
    print(f"ENV_KWARGS: {json.dumps(config.get('ENV_KWARGS', {}), indent=2)}")
    print("=" * 60)


def find_runs_by_config(num_agents=None, num_landmarks=None, wandb_dir="wandb"):
    """
    Find all runs matching specific agent/landmark configuration.
    
    Args:
        num_agents: Filter by number of agents
        num_landmarks: Filter by number of landmarks
        wandb_dir: Path to wandb directory
        
    Returns:
        list: List of (run_folder, config) tuples matching the criteria
    """
    wandb_path = Path(wandb_dir)
    matching_runs = []
    
    for run_folder in wandb_path.glob("offline-run-*"):
        config_file = run_folder / "files" / "run_config.json"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                
                # Check if it matches criteria
                matches = True
                if num_agents is not None and config.get("num_agents") != num_agents:
                    matches = False
                if num_landmarks is not None and config.get("num_landmarks") != num_landmarks:
                    matches = False
                
                if matches:
                    matching_runs.append((run_folder, config))
            except Exception as e:
                print(f"Warning: Could not read config from {run_folder}: {e}")
    
    return matching_runs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_run_config.py <run_folder_or_id>")
        print("\nOr to search:")
        print("  python load_run_config.py --search --agents 3 --landmarks 4")
        sys.exit(1)
    
    if sys.argv[1] == "--search":
        # Parse search arguments
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--search", action="store_true")
        parser.add_argument("--agents", type=int, help="Number of agents")
        parser.add_argument("--landmarks", type=int, help="Number of landmarks")
        args = parser.parse_args()
        
        matching = find_runs_by_config(args.agents, args.landmarks)
        print(f"\nFound {len(matching)} matching runs:")
        for run_folder, config in matching:
            print(f"\n{run_folder.name}")
            print(f"  Agents: {config.get('num_agents')}, Landmarks: {config.get('num_landmarks')}")
            print(f"  Run: {config.get('RUN_NAME')}")
    else:
        # Load and display specific run
        run_identifier = sys.argv[1]
        config = load_config_from_run(run_identifier)
        print_config_summary(config)
        
        # Print full config path for reference
        print(f"\nFull config available as 'full_config' key in JSON")
        print(f"Model params: {Path(run_identifier).resolve()}/files/final_model_params.msgpack")
