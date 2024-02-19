import os
import argparse

import torch
import torch.optim as optim

from src.utils import Logger, Setting, load_model, load_config
from src.dataloader import DataLoader
from src.model import MultiDAE, MultiVAE
from src.trainer import run, inference


def main(args):

    ##### Load config
    print("##### Load config ...")
    config = load_config(args)
    Setting.seed_everything(config["seed"])
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    ##### Load Data
    print("##### Load Data ...")
    loader = DataLoader(config["data_path"])

    ##### Load Model
    print("##### Load Model ...")
    with open(
        os.path.join(
            config["model_save_path"], f"{config['model']}_V_{config['config_ver']}.pt"
        ),
        "rb",
    ) as f:
        model = torch.load(f)

    ##### Inference
    inference(config, model, loader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    #### Environment Settings
    arg("--model", "-m", type=str, help="select model")
    arg("--config_ver", "-c", type=str, default="0", help="veresion of experiments")

    args = parser.parse_args()
    main(args)
