import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import pandas as pd
from loguru import logger

from features.acis import generate_acis_deviation
from features.cdec_deviation import generate_cdec_deviation
from features.monthly_naturalized_flow import generate_monthly_naturalized_flow
from features.snotel_deviation import generate_snotel_deviation
from features.streamflow_deviation import generate_streamflow_deviation
from features.ua_swann_deviation import generate_ua_swann_deviation


def generate_feature_dataset(src_dir: Path,
                             data_dir: Path,
                             preprocessed_dir: Path,
                             submission: pd.DataFrame,
                             issue_date: str) -> pd.DataFrame:
    logger.info("Generating feature datasets.")

    # Generate base dataset
    metadata = pd.read_csv(preprocessed_dir / 'metadata.csv', dtype={"usgs_id": "string"})

    submission['year'] = pd.to_datetime(submission['issue_date']).dt.year
    submission['month'] = pd.to_datetime(submission['issue_date']).dt.month
    submission['day'] = pd.to_datetime(submission['issue_date']).dt.day
    submission['day_of_year'] = pd.to_datetime(submission['issue_date']).dt.day_of_year

    submission.to_csv(preprocessed_dir / 'submission.csv')

    issue_year = pd.to_datetime(issue_date).year

    site_list = []
    date_list = []

    for month in [1, 2, 3, 4, 5, 6, 7]:
        for day in [1, 8, 15, 22]:
            for idx, rw in metadata.iterrows():
                site_list.append(rw['site_id'])
                date_list.append(f"{issue_year}-{month:02d}-{day:02d}")

    pred_idx = pd.DataFrame({'site_id': site_list, 'issue_date': date_list})
    pred_idx['year'] = pd.to_datetime(pred_idx['issue_date']).dt.year
    pred_idx['month'] = pd.to_datetime(pred_idx['issue_date']).dt.month
    pred_idx['day'] = pd.to_datetime(pred_idx['issue_date']).dt.day
    pred_idx['day_of_year'] = pd.to_datetime(pred_idx['issue_date']).dt.day_of_year

    test_features = pred_idx[['site_id', 'issue_date', 'year', 'month', 'day', 'day_of_year']]
    metadata_keep_cols = ['site_id', 'elevation', 'latitude', 'longitude', 'drainage_area',
                          'season_start_month', 'season_end_month']
    test_features = pd.merge(test_features,
                             metadata[metadata_keep_cols],
                             on=['site_id'], how='left')

    logger.info(f"test_features_shape: {test_features.shape}")

    site_elevations = pd.read_csv(src_dir / 'feature_parameters/site_elevations.csv')
    test_features = pd.merge(test_features, site_elevations, on='site_id')

    test_features = generate_monthly_naturalized_flow(src_dir, data_dir, test_features, metadata, issue_date)

    # UA Swann features
    test_features = generate_ua_swann_deviation(src_dir, test_features, issue_date)

    # Acis climate deviation features
    test_features = generate_acis_deviation(src_dir, test_features, issue_date)

    # Site streamflow deviation features
    test_features = generate_streamflow_deviation(src_dir, preprocessed_dir, test_features)

    # SWE deviation features
    test_features = generate_snotel_deviation(src_dir, data_dir, test_features)
    test_features = generate_cdec_deviation(src_dir, data_dir, test_features)

    test_features['combined_swe_deviation_30'] = test_features[['snotel_wteq_deviation_30', 'cdec_deviation_30']].mean(
        axis=1)
    test_features['combined_swe_deviation_0'] = test_features[['snotel_wteq_deviation_0', 'cdec_deviation_0']].mean(
        axis=1)

    logger.info(f"test_features_shape: {test_features.shape}")
    logger.info(test_features.columns)
    test_features.to_csv(preprocessed_dir / 'test_features.csv', index=False)

    return test_features
