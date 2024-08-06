#! .venv/bin/python

import argparse
from pathlib import Path
from typing import Dict

from disease_pred.train.multi_dldl import MultiDLDLTrainer
from disease_pred.utils import expand_params, load_params

parser = argparse.ArgumentParser(
    description="Train MultiDLDL prediction models. If 'seed' or/and 'data_norm' are lists of multiple entries, all combinations are trained.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-c", "--config", type=str, help="path to configuration file", required=True)
parser.add_argument("-g", "--gpus", type=int, nargs="+", help="List of GPUs used for training.", default=[])
parser.add_argument("-nw", "--nworkers", type=int, help="Number of workers.", default=4)
parser.add_argument("-chkpt", "--checkpoint", type=str, help="Path to checkpoint for resuming training.", default=None)
parser.add_argument(
    "-d",
    "--dev",
    dest="dev",
    action="store_true",
    help="If set, a fast dev run is performed (train and val of only 1 batch without logging).",
)
args = parser.parse_args()


def run(config: Dict):
    config["training"].update({"gpus": args.gpus, "n_workers": args.nworkers, "persistent_workers": False, "pin_memory": True})
    trainer = MultiDLDLTrainer(config, args.dev)

    trainer.run(args.checkpoint)

    return None


if __name__ == "__main__":
    config_path = Path(args.config)
    if config_path.exists():
        # load configuration file and expand grid search parameters
        configs, _parameter_combinations = expand_params(load_params(args.config))

    else:
        raise FileNotFoundError(config_path)

    for config in configs:
        run(config)
