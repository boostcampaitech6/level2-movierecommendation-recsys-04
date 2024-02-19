import os
import argparse

import torch
import torch.optim as optim

from .src.utils import Logger, Setting, load_model, load_config
from .src.dataloader import DataLoader
from .src.model import MultiDAE, MultiVAE
from .src.trainer import run, inference


def main(args):

    ##### Load config
    print("##### Load config ...")
    config = load_config(args.config)
    Setting.seed_everything(config.seed)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    ##### Load Data
    print("##### Load Data ...")
    loader = DataLoader(args.data)
    n_items = loader.load_n_items()
    train_data = loader.load_data("train")
    vad_data_tr, vad_data_te = loader.load_data("validation")
    test_data_tr, test_data_te = loader.load_data("test")

    ##### Load Model
    print("##### Load Model ...")
    with open(
        os.path.join(
            config.model_save_path, f"{config.model}_V_{config.config_ver}.pt"
        ),
        "rb",
    ) as f:
        model = torch.load(f)

    ##### Inference
    inference(
        config, model, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    #### Environment Settings
    arg("--model", "-m", type=str, defalut="nomodel", help="select model")
    arg("--config_ver", "-c", type=str, default="0", help="veresion of experiments")

    args = parser.parse_args()
    main(args)
