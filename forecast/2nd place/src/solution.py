import os
import sys
from datetime import date
from pathlib import Path
from typing import Hashable, Any, Dict
from enum import Enum
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

from libs.builder import build_all, BuildMode, BuildObjects, _load_yaml
from libs.lightning_module import LightningModule


class Prepared(Enum):
    DF = 1


def get_model_paths(opt: Dict[str, Any], base_dir: str):
    path = opt["model"]["weights"]
    if "cross_val_parts" not in opt:
        return [os.path.join(base_dir, path)]

    parts = opt["cross_val_parts"]
    cross_val_repeats = opt.get("cross_val_repeats", 1)
    result = []
    path_split = path.split(".")
    for split in range(parts*cross_val_repeats):
        result.append(
            os.path.join(base_dir, ".".join(path_split[:-1]) + "-" + str(split) + "." + path_split[-1])
        )
    return result


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    # print(f"Load weights from {path}")


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def predict(
    site_id: str,
    issue_date: str,
    assets: dict[Hashable, Any],
    src_dir: Path,
    data_dir: Path,
    preprocessed_dir: Path,
) -> tuple[float, float, float]:
    """A function that generates a forecast for a single site on a single issue
    date. This function will be called for each site and each issue date in the
    test set.

    Args:
        site_id (str): the ID of the site being forecasted.
        issue_date (str): the issue date of the site being forecasted in
            'YYYY-MM-DD' format.
        assets (dict[Hashable, Any]): a dictionary of any assets that you may
            have loaded in the 'preprocess' function. See next section.
        src_dir (Path): path to the directory that your submission ZIP archive
            contents are unzipped to.
        data_dir (Path): path to the mounted data drive.
        preprocessed_dir (Path): path to a directory where you can save any
            intermediate outputs for later use.
    Returns:
        tuple[float, float, float]: forecasted values for the seasonal water
            supply. The three values should be (0.10 quantile, 0.50 quantile,
            0.90 quantile).
    """

    opt = _load_yaml(str(src_dir / "configs/forecast_cv_mlp_sumres_predict.yaml"))

    # !!!!
    opt["forecast_test_data"]["site_id"] = site_id
    opt["forecast_test_data"]["issue_date"] = date.fromisoformat(issue_date)

    model_paths = get_model_paths(opt, str(src_dir))
    opt["model"].pop("weights")
    opt["base_dir"] = str(data_dir)
    opt["src_dir"] = str(src_dir)

    build_dict = build_all(BuildMode.INFERENCE, opt)
    model: LightningModule = build_dict[BuildObjects.MODEL]
    dataloader = build_dict[BuildObjects.DATALOADERS][0]
    device = get_device()
    model.to(device)
    model.eval()

    cv_predictions = []
    with torch.no_grad():
        for path in tqdm(model_paths, "", disable=True):
            load_model(model, path)
            predictions = []
            for batch in tqdm(dataloader, disable=True):
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

    result = cv_predictions[0][1], cv_predictions[0][0], cv_predictions[0][2]
    print(site_id, issue_date, result)
    assert result[0] >= 0 and result[1] >= 0 and result[2] >= 0, "Prediction of negative values. Some inputs are incorrect"
    assert result[0] <= result[1] and result[1] <= result[2], "Percentile is in wrong order. Some inputs are incorrect"
    return result


def preprocess(
    src_dir: Path, data_dir: Path, preprocessed_dir: Path
) -> dict[Hashable, Any]:
    """An optional function that performs setup or processing.

    Args:
        src_dir (Path): path to the directory that your submission ZIP archive
            contents are unzipped to.
        data_dir (Path): path to the mounted data drive.
        preprocessed_dir (Path): path to a directory where you can save any
            intermediate outputs for later use.

    Returns:
        (dict[Hashable, Any]): a dictionary containing any assets you want to
            hold in memory that will be passed to to your 'predict' function as
            the keyword argument 'assets'.
    """

    print("Data files:")
    print(os.listdir(str(data_dir)))
    print("--------------------")

    return {}
