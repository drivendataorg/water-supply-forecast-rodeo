import argparse
import os
import pickle

import yaml
from typing import Optional, Dict, Any, List, Tuple
from copy import deepcopy

import pytorch_lightning
import torch
import random
import numpy as np
from tqdm import tqdm

from libs.data import rawdata
from libs.data.NormalizedData import NormalizedData
from libs.data import timeseries

from libs.builder import build_all, BuildMode, BuildObjects
from libs.lightning_module import LightningModule
from pytorch_lightning.trainer.states import TrainerStatus


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dir_for_file_if_needed(file_name: str):
    dir = os.path.dirname(file_name)
    os.makedirs(dir, exist_ok=True)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_mlp_sum_res.yaml")
    return parser.parse_args()


def final_kfold_train_test_splits(all_years: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    splits = []
    test_years = np.arange(2004, 2023+1)

    for test_year in test_years:
        train_years = all_years[all_years!=test_year]
        splits.append((train_years, test_year))

    return splits


def get_split_weights_path(target_path, year):
    split = target_path.split(".")
    target_path = ".".join(split[:-1]) + "-" + str(year) + "." + split[-1]
    return target_path


def lightning_train(
        cfg: Dict[str, Any],
        train_years: np.ndarray,
        test_year: int,
        cache_timeseries: bool = True
) -> None:
    norm_path = get_split_weights_path(cfg["norm_path"], test_year)
    timeseries_cache_path = get_split_weights_path(cfg["timeseries_cache_path"], test_year)
    create_dir_for_file_if_needed(norm_path)
    create_dir_for_file_if_needed(timeseries_cache_path)

    if os.path.exists(timeseries_cache_path) and cache_timeseries:
        with open(timeseries_cache_path, "rb") as f:
            train_timeseries = pickle.load(f)
    else:
        # load and prepare data by train_years
        raw = rawdata.readAllDataForYears(train_years)

        normData = NormalizedData()
        normData.setRawData(raw)
        if os.path.exists(norm_path):
            normData.loadNormalizationModelsFromFile(norm_path)
        else:
            normData.initNormalization()
            normData.saveNormalizationModelsToFile(norm_path)

        train_timeseries = timeseries.getTimeseries(normData, 'train')
        if cache_timeseries:
            with open(timeseries_cache_path, "wb") as f:
                pickle.dump(train_timeseries, f)

    # put all timeseries to config to use it in train/val split
    cfg["full_train_timeseries"] = train_timeseries
    cfg["all_train_years"] = train_years
    cfg["train"]["weights_path"] = get_split_weights_path(cfg["train"]["weights_path"], test_year)

    seed_everything()

    # create all training related objects
    build_dict = build_all(BuildMode.LIGHTNING, cfg)
    trainer: pytorch_lightning.Trainer = build_dict[BuildObjects.TRAINER]
    model: LightningModule = build_dict[BuildObjects.MODEL]
    dataloaders = build_dict[BuildObjects.DATALOADERS]

    # Train it
    trainer.fit(model, dataloaders[0], dataloaders[1] if len(dataloaders) > 1 else None)
    if trainer.state.status != TrainerStatus.FINISHED:
        raise InterruptedError()

    # save resulting weights
    if trainer.is_global_zero:
        if BuildObjects.CHECKPOINT_CALLBACK in build_dict and BuildObjects.WEIGHT_PATH in build_dict:
            path = build_dict[BuildObjects.CHECKPOINT_CALLBACK].best_model_path
            target_path = build_dict[BuildObjects.WEIGHT_PATH]

            print(f"Recover checkpoint {path}")
            model.load_state_dict(torch.load(str(path))["state_dict"])
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            torch.save(model.model.state_dict(), target_path)
            print(f"Weights saved to the {target_path}")
        elif BuildObjects.WEIGHT_PATH in build_dict:
            target_path = build_dict[BuildObjects.WEIGHT_PATH]

            torch.save(model.model.state_dict(), target_path)
            print(f"Weights saved to the {target_path}")



def train(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    original_config = deepcopy(config)

    all_years = rawdata.getAllYears()
    # All train/test splits as Leave one year out
    year_folds = final_kfold_train_test_splits(all_years)

    for train_years, test_year in year_folds:
        config = deepcopy(original_config)

        print("="*20)
        print(f"Train with the {test_year} leaved out")
        lightning_train(config, train_years, test_year)


if __name__ == "__main__":
    args = arguments()
    train(args.config)
