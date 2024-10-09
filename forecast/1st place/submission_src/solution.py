import os
import sys
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).parents[0]))

from wsfr_read.sites import read_metadata

import generate_predictions
from features.feature_engineering import generate_feature_dataset

def update_metadata(preprocessed_dir: Path) -> None:
    metadata = read_metadata()

    metadata.loc['ruedi_reservoir_inflow', 'usgs_id'] = '09080400'
    metadata.loc['fontenelle_reservoir_inflow', 'usgs_id'] = '09211200'
    metadata.loc['american_river_folsom_lake', 'usgs_id'] = '11446500'
    metadata.loc['skagit_ross_reservoir', 'usgs_id'] = '12181000'
    metadata.loc['skagit_ross_reservoir', 'drainage_area'] = 999.0
    metadata.loc['boysen_reservoir_inflow', 'usgs_id'] = '06279500'

    metadata.to_csv(preprocessed_dir / 'metadata.csv')


def preprocess(src_dir: Path, data_dir: Path, preprocessed_dir: Path) -> dict[Hashable, Any]:
    submission_file = 'submission_format.csv'
    is_smoke = os.getenv("IS_SMOKE")
    if is_smoke:
        submission_file = 'smoke_submission_format.csv'
        logger.info("Running in smoke test mode.")

    update_metadata(preprocessed_dir)


    submission = pd.read_csv(data_dir / submission_file)
    issue_date = submission.loc[0]['issue_date']
    logger.info(f"Predicting as of issue_date: {issue_date}")
    test_features = generate_feature_dataset(src_dir, data_dir, preprocessed_dir, submission, issue_date)
    submission_final = generate_predictions.generate_predictions(src_dir, preprocessed_dir, test_features, issue_date)

    prediction_dict = submission_final.groupby(
        ['site_id']
    )[['issue_date', 'volume_10', 'volume_50', 'volume_90']].apply(
        lambda x: x.set_index('issue_date').to_dict(orient='index')).to_dict()
    return prediction_dict


def predict(
        site_id: str,
        issue_date: str,
        assets: dict[Any, Any],
        src_dir: Path,
        data_dir: Path,
        preprocessed_dir: Path,
) -> tuple[float, float, float]:
    preds = assets[site_id][issue_date]
    return preds['volume_10'], preds['volume_50'], preds['volume_90']
