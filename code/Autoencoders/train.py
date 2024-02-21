import os
import argparse

import torch

from src.utils import Logger, Setting, load_model, load_config
from src.dataloader import DataLoader, Data_Preprocess
from src.model import MultiDAE, MultiVAE
from src.trainer import run


def main(args):
    ##### Load config
    print("##### Load config ...")
    config = load_config(args)
    Setting.seed_everything(config["seed"])
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    ##### Data Preprocess
    print("##### Data Preprocess ...")
    if args.preprocess == True:  # seed가 바뀌면 반드시 preprocess를 해야 합니다.
        Data_Preprocess(config)

    ##### Load DataLoader
    print("##### Load DataLoader ...")
    loader = DataLoader(config["data_path"])
    n_items = loader.load_n_items()
    train_data = loader.load_data("train")
    vad_data_tr, vad_data_te = loader.load_data("validation")
    test_data_tr, test_data_te = loader.load_data("test")

    ##### Load Model
    print("##### Load Model ...")
    model = load_model(args, config, n_items)

    ##### Training
    run(config, model, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    #### Environment Settings
    arg("--model", "-m", type=str, help="select model")
    arg("--config_ver", "-c", type=str, help="veresion of experiments")
    arg("--preprocess", "-p", default=False, type=str, help="Data Data preprocessing ")

    args = parser.parse_args()
    main(args)
