import argparse
import os
import shutil

import pandas as pd
import torch
import random
import numpy as np
from tqdm import tqdm

from libs.builder import build_all, BuildMode, BuildObjects, _load_yaml
from libs.lightning_module import LightningModule
from pytorch_lightning.trainer.states import TrainerStatus


def create_dir_for_file_if_needed(file_name: str):
    dir = os.path.dirname(file_name)
    os.makedirs(dir, exist_ok=True)


def get_model_paths(opt):
    path = opt["model"]["weights"]
    if "cross_val_parts" not in opt:
        return [path]

    parts = opt["cross_val_parts"]
    cross_val_repeats = opt["cross_val_repeats"]
    result = []
    path_split = path.split(".")
    for split in range(parts*cross_val_repeats):
        result.append(
            ".".join(path_split[:-1]) + "-" + str(split) + "." + path_split[-1]
        )
    return result


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Load weights from {path}")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_predict.yaml")
    args = parser.parse_args()
    return args


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def predict(config_path: str):
    np.seterr(all='raise')

    opt = _load_yaml(config_path)
    model_paths = get_model_paths(opt)
    opt["model"].pop("weights")

    build_dict = build_all(BuildMode.INFERENCE, opt)
    model: LightningModule = build_dict[BuildObjects.MODEL]
    dataloader = build_dict[BuildObjects.DATALOADERS][0]
    device = get_device()

    cv_predictions = []
    model.to(device)
    model.eval()
    i = 0
    with torch.no_grad():
        for path in tqdm(model_paths, ""):
            load_model(model, path)
            predictions = []
            for batch in tqdm(dataloader):
                x = batch["X"].to(device)
                scale = batch["volume_scale_factor"].to(device)
                pred = model(x)["out"]
                pred = pred * scale.unsqueeze(-1)
                pred = pred.cpu().numpy()

                predictions.append(pred)

            predictions = np.concatenate(predictions, axis=0)
            cv_predictions.append(predictions)

    cv_predictions = np.stack(cv_predictions, axis=0)
    cv_predictions = np.median(cv_predictions, axis=0)

    out_filename = opt["predict"]["path"]
    create_dir_for_file_if_needed(out_filename)

    submission = pd.read_csv('data/submission_format.csv')
    submission['volume_50'] = cv_predictions[:, 0]
    submission['volume_10'] = cv_predictions[:, 1]
    submission['volume_90'] = cv_predictions[:, 2]

    submission.to_csv(out_filename, index=False, float_format='%.3f')


if __name__ == "__main__":
    args = arguments()
    predict(args.config)
