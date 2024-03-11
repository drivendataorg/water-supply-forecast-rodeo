import os
import sys
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
    print(f"Load weights from {path}")


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

    df: pd.DataFrame = assets[Prepared.DF]

    site_df = df[(df["site_id"] == site_id) & (df["issue_date"] == issue_date)]
    if len(site_df) == 0:
        print(f"Not found site_id and date in prediction: {site_id}, {issue_date}")
    row = site_df.iloc[0]

    return row["volume_10"], row["volume_50"], row["volume_90"]


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

    opt = _load_yaml(str(src_dir / "configs/cv_mlp_sumres_predict.yaml"))
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

    submission = pd.read_csv(str(data_dir / 'submission_format.csv'))
    submission['volume_50'] = cv_predictions[:, 0]
    submission['volume_10'] = cv_predictions[:, 1]
    submission['volume_90'] = cv_predictions[:, 2]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(submission.head(40))

    return {
        Prepared.DF: submission
    }
