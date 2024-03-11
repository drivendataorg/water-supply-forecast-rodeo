import argparse
import os

import pandas as pd
import yaml
from typing import Optional, Dict, Any, List
from copy import deepcopy

import pytorch_lightning
import torch
import random
import numpy as np
from tqdm import tqdm

from libs.builder import build_all, BuildMode, BuildObjects
from libs.data.datasets import TimeSeriesSample
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
    parser.add_argument("--config", default="configs/mlp_sum_res_div.yaml")
    return parser.parse_args()


def get_cv_path(opt, fold, target_path):
    split = target_path.split(".")
    target_path = ".".join(split[:-1]) + "-" + str(fold) + "." + split[-1]
    return target_path


def lightning_train(opt: Dict[str, Any]):
    seed_everything()
    opt["fixed_snotel"] = True
    build_dict = build_all(BuildMode.LIGHTNING, opt)
    trainer: pytorch_lightning.Trainer = build_dict[BuildObjects.TRAINER]
    model: LightningModule = build_dict[BuildObjects.MODEL]
    dataloaders = build_dict[BuildObjects.DATALOADERS]

    trainer.fit(model, dataloaders[0], dataloaders[1] if len(dataloaders) > 1 else None)
    if trainer.state.status != TrainerStatus.FINISHED:
        raise InterruptedError()

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

        # save val predictions
        if "val_prediction" in opt["train"]:
            prediction_path = opt["train"]["val_prediction"]
            predictions = []
            targets = []
            model.model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloaders[1]):
                        x = batch["X"].to(model.device)
                        scale = batch["volume_scale_factor"].to(model.device)
                        pred = model.model(x)["out"]
                        pred = pred * scale.unsqueeze(-1)
                        pred = pred.cpu().numpy()
                        predictions.append(pred)
                        targets.append(batch["target"] * batch["volume_scale_factor"])
            predictions = np.concatenate(predictions, axis=0)
            targets = np.concatenate(targets)
            samples: List[TimeSeriesSample] = dataloaders[1].dataset.timeseries_samples

            def doy_from_sample(sample: TimeSeriesSample) -> int:
                res = [0]
                if sample.daily_volumes_doy is not None and len(sample.daily_volumes_doy) > 0:
                    res.append(sample.daily_volumes_doy[-1])
                if sample.monthly_volumes_doy is not None and len(sample.monthly_volumes_doy) > 0:
                    res.append(sample.monthly_volumes_doy[-1])
                if sample.snotel_doy is not None and len(sample.snotel_doy) > 0:
                    res.append(sample.snotel_doy[-1])
                return np.max(res)
            doys = [doy_from_sample(s) for s in samples]

            df = pd.DataFrame({
                "site_id": [s.site_id for s in samples],
                "year": [s.forecast_year for s in samples],
                "target": targets,
                "doy": doys,
                "volume_50": predictions[:, 0],
                "volume_10": predictions[:, 1],
                "volume_90": predictions[:, 2],
            })
            df.to_csv(prediction_path, index=False)


def train(config_path: str):
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)
    original_opt = deepcopy(opt)

    cross_val_parts = opt.get("cross_val_parts", None)
    cross_val_repeats = opt.get("cross_val_repeats", 1)
    folds = list(range(cross_val_parts*cross_val_repeats)) if cross_val_parts else [0]
    for fold in folds:
        opt = deepcopy(original_opt)
        opt["cv_fold_pos"] = fold
        if len(folds) > 1:
            opt["train"]["weights_path"] = get_cv_path(opt, fold, opt["train"]["weights_path"])
            if "val_prediction" in opt["train"]:
                opt["train"]["val_prediction"] = get_cv_path(opt, fold, opt["train"]["val_prediction"])

        lightning_train(opt)


if __name__ == "__main__":
    args = arguments()
    train(args.config)
