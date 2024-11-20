import ssl
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from features.feature_utils import SITE_BASINS, clean_and_merge_features
from loguru import logger

ssl._create_default_https_context = ssl._create_unverified_context


def generate_ua_swann_deviation(preprocessed_dir: Path,
                                df_features: pd.DataFrame,
                                cv_year: Optional[int] = None) -> pd.DataFrame:
    logger.info('Generating UA Swann deviation features.')
    ua_swann_columns = ['acc_water', 'ua_swe', 'ua_swe_deviation', 'acc_water_deviation']
    ua_swann_file = preprocessed_dir / 'train_ua_swann.csv'
    if ua_swann_file.is_file():
        logger.info('UA Swann file already exists, pulling existing.')
        ua_swann_deviation = pd.read_csv(ua_swann_file)
    else:
        swann_base_url = 'https://climate.arizona.edu/snowview/csv/Download/Watersheds/'
        ua_swann = []
        for site_id in df_features.site_id.unique():
            site_basin = SITE_BASINS[site_id]
            site_file_name = f'{site_basin}.csv'
            site_ua_swann = pd.read_csv(f'{swann_base_url}{site_file_name}')
            site_ua_swann.columns = ['date', 'acc_water', 'ua_swe']
            site_ua_swann['date'] = pd.to_datetime(site_ua_swann['date'])
            site_ua_swann['site_id'] = site_id
            ua_swann.append(site_ua_swann)

        ua_swann_pd = pd.concat(ua_swann)

        # Set issue_date to be one day ahead, so we only join with past data
        ua_swann_pd['issue_date'] = (ua_swann_pd['date'] + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')

        ua_swann_deviation = ua_swann_pd
        ua_swann_deviation['month_day'] = ua_swann_deviation['date'].dt.strftime('%m%d')
        ua_swann_deviation['month'] = ua_swann_deviation['date'].dt.month
        ua_swann_deviation['year'] = ua_swann_deviation['date'].dt.year
        ua_swann_deviation['forecast_year'] = ua_swann_deviation.apply(
            lambda rw: rw['year'] if rw['month'] < 8 else rw['year'] + 1,
            axis=1)
        ua_swann_deviation.to_csv(ua_swann_file, index=False)

    ua_swann_deviation['date'] = pd.to_datetime(ua_swann_deviation['date'])

    # Holdout data from the forecast year when computing mean and std to avoid data leakage
    if cv_year:
        ua_swann_deviation = ua_swann_deviation[ua_swann_deviation['forecast_year'] != cv_year]

    ua_swann_deviation = ua_swann_deviation.sort_values(['site_id', 'date'])

    grouped_ua_swann = ua_swann_deviation.groupby(['site_id', 'month_day'])[['acc_water', 'ua_swe']].agg(
        [np.mean, np.std]).reset_index()
    grouped_ua_swann.columns = ['site_id', 'month_day', 'acc_water_mean', 'acc_water_std', 'ua_swe_mean', 'ua_swe_std']
    ua_swann_deviation = pd.merge(ua_swann_deviation, grouped_ua_swann, on=['site_id', 'month_day'])

    acc_water = ua_swann_deviation['acc_water']
    acc_water_mean = ua_swann_deviation['acc_water_mean']
    acc_water_std = ua_swann_deviation['acc_water_std']
    ua_swann_deviation['acc_water_deviation'] = (acc_water - acc_water_mean) / acc_water_std

    ua_swe = ua_swann_deviation['ua_swe']
    ua_swe_mean = ua_swann_deviation['ua_swe_mean']
    ua_swe_std = ua_swann_deviation['ua_swe_std']
    ua_swann_deviation['ua_swe_deviation'] = (ua_swe - ua_swe_mean) / ua_swe_std
    ua_swann_deviation = ua_swann_deviation[
        ['issue_date', 'site_id', 'acc_water', 'ua_swe', 'ua_swe_deviation', 'acc_water_deviation']]

    df_features = clean_and_merge_features(df_features, ua_swann_deviation, ua_swann_columns, ['site_id', 'issue_date'])

    return df_features
