import os
import argparse

import torch
import torch.optim as optim
import pandas as pd

from .src.utils import Logger, Setting, load_model, load_config
from .src.dataloader import DataLoader, DataPreprocess
from .src.model import MultiDAE, MultiVAE
from .src.trainer import run


def main(args):
    ##### Load config
    print("##### Load config ...")
    config = load_config(args.config)
    Setting.seed_everything(config.seed)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    ##### Data Preprocess
    print("##### Data Preprocess ...")
    DataPreprocess(config.data_path)

    ##### Load DataLoader
    print("##### Load DataLoader ...")
    loader = DataLoader(config.data_path)
    n_items = loader.load_n_items()
    train_data = loader.load_data("train")
    vad_data_tr, vad_data_te = loader.load_data("validation")
    test_data_tr, test_data_te = loader.load_data("test")

    ##### Load Model
    print("##### Load Model ...")
    p_dims = [200, 600, n_items]
    model = load_model(args, config, p_dims)

    ##### Training
    run(config, model, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    #### Environment Settings
    arg("--model", "-m", type=str, defalut="nomodel", help="select model")
    arg("--config_ver", "-c", type=str, default="0", help="veresion of experiments")

    args = parser.parse_args()
    main(args)
