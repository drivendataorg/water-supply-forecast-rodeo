import numpy as np
import pandas as pd
import argparse
import os
import yaml
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, Any
import torch

from libs.data import rawdata
from libs.data.NormalizedData import NormalizedData
from libs.data import timeseries
from libs.builder import build_all, BuildMode, BuildObjects, _load_yaml
from libs.models.model_factory import create_model
from libs.data.datasets import create_dataloaders


def get_split_weights_path(target_path, year):
    split = target_path.split(".")
    target_path = ".".join(split[:-1]) + "-" + str(year) + "." + split[-1]
    return target_path


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    cfg = deepcopy(cfg)
    cfg.pop("weights")
    model = create_model(cfg)
    model.to(get_device())
    model.eval()
    return model


def lightning_inference(
        cfg: Dict[str, Any],
        test_year: int,
        model: torch.nn.Module
) -> pd.DataFrame:
    cfg = deepcopy(cfg)

    # Read data and normalization
    raw = rawdata.readAllDataForYears(test_year)
    normData = NormalizedData()
    normData.setRawData(raw)

    norm_path = get_split_weights_path(cfg["norm_path"], test_year)
    normData.loadNormalizationModelsFromFile(norm_path)
    samples = timeseries.getTestTimeseriesForecastYear(normData, forecast_year=test_year)
    cfg["test_timeseries"] = samples
    dataloader = create_dataloaders(cfg)[0]

    # prepare model
    weights_path = get_split_weights_path(cfg["model"]["weights"], test_year)
    model.load_state_dict(torch.load(weights_path))
    device = get_device()

    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x = batch["X"].to(device)
            scale = batch["volume_scale_factor"].to(device)
            pred = model(x)["out"]
            pred = pred * scale.unsqueeze(-1)
            pred = pred.cpu().numpy()

            predictions.append(pred)

        predictions = np.concatenate(predictions, axis=0)

    site_ids = [s.site_id for s in samples]
    issue_dates = [s.issue_date for s in samples]

    result = pd.DataFrame({'site_id': site_ids,
                           'issue_date': issue_dates,
                           'volume_10': predictions[:, 1],
                           'volume_50': predictions[:, 0],
                           'volume_90': predictions[:, 2]
                           })
    return result


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_mlp_sum_res_predict.yaml")
    return parser.parse_args()


def create_dir_for_file_if_needed(file_name: str):
    dir = os.path.dirname(file_name)
    os.makedirs(dir, exist_ok=True)


def inference(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    original_config = deepcopy(config)

    model = get_model(original_config["model"])
    test_years = np.arange(2004, 2023 + 1)
    results = []
    for test_year in test_years:
        config = deepcopy(original_config)

        print("="*20)
        print(f"Inference for the {test_year} year")
        ds = lightning_inference(config, test_year, model)
        results.append(ds)

    results = pd.concat(results)
    submission_format = pd.read_csv('data/final_stage/cross_validation_submission_format.csv').drop(columns=['volume_10','volume_50','volume_90'])
    submission_format['order'] = np.arange(0, len(submission_format))

    submission = submission_format.merge(results, on=['site_id', 'issue_date']).sort_values('order')
    submission = submission[['site_id', 'issue_date', 'volume_10', 'volume_50', 'volume_90']].copy()

    out_filename = original_config["predict"]["path"]

    create_dir_for_file_if_needed(out_filename)
    submission.to_csv(out_filename, index=False, float_format='%.3f')


if __name__ == "__main__":
    args = arguments()
    inference(args.config)
