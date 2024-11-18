from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from features.feature_utils import clean_and_merge_features
from wsfr_download_train.usgs_streamflow import ALTERNATIVE_USGS_IDS, download_alternative_usgs_streamflow
from wsfr_read.streamflow import read_usgs_streamflow_data

MEAN_DISCHARGE_RAW_COL = '00060_Mean'
MEAN_DISCHARGE_READABLE_COL = 'discharge_cfs_mean'


def generate_streamflow_deviation(preprocessed_dir: Path,
                                  df_features: pd.DataFrame,
                                  cv_year: Optional[int] = None) -> pd.DataFrame:
    streamflow_file = preprocessed_dir / 'train_streamflow.csv'
    if streamflow_file.is_file():
        logger.info('Streamflow file already exists, pulling existing.')
        streamflow_deviation = pd.read_csv(streamflow_file)
    else:
        forecast_years = df_features['year'].unique()
        download_alternative_usgs_streamflow(forecast_years,
                                             preprocessed_dir,
                                             skip_existing=True)

        logger.info('Generating streamflow deviation features.')
        df_list = []
        for site_id in df_features['site_id'].unique():
            if site_id in ALTERNATIVE_USGS_IDS.keys():
                for year in df_features['year'].unique():
                    try:
                        streamflow_data_dir = preprocessed_dir / 'usgs_streamflow' / 'alternative'
                        df = _read_usgs_streamflow_data(site_id,
                                                        f'{year}-07-23',
                                                        year,
                                                        streamflow_data_dir).replace(-999999.0, np.nan)
                        df['site_id'] = site_id
                        df_list.append(df)
                    except Exception as e:
                        print(e)
            else:
                for year in df_features['year'].unique():
                    try:
                        df = read_usgs_streamflow_data(site_id,
                                                       f'{year}-07-23').replace(-999999.0, np.nan)
                        df['site_id'] = site_id
                        df_list.append(df)
                    except Exception as e:
                        print(e)

        streamflow_deviation = pd.concat(df_list)
        streamflow_deviation['datetime'] = pd.to_datetime(streamflow_deviation['datetime'])
        streamflow_deviation['month_day'] = streamflow_deviation['datetime'].dt.strftime('%m%d')
        streamflow_deviation['year'] = streamflow_deviation['datetime'].dt.year
        streamflow_deviation['month'] = streamflow_deviation['datetime'].dt.month
        streamflow_deviation['forecast_year'] = streamflow_deviation.apply(
            lambda rw: rw['year'] if rw['month'] < 8 else rw['year'] + 1,
            axis=1)
        streamflow_deviation.to_csv(streamflow_file, index=False)

    streamflow_deviation['datetime'] = pd.to_datetime(streamflow_deviation['datetime'])

    # Holdout data from the forecast year when computing mean and std to avoid data leakage
    if cv_year:
        streamflow_deviation = streamflow_deviation[streamflow_deviation['forecast_year'] != cv_year]

    streamflow_deviation = streamflow_deviation.sort_values(['site_id', 'datetime'])

    grouped_streamflow = streamflow_deviation.groupby(['site_id', 'month_day'])['discharge_cfs_mean'].agg(
        [np.mean, np.std]).reset_index()
    grouped_streamflow.columns = ['site_id', 'month_day', 'discharge_mean', 'discharge_std']
    streamflow_deviation = pd.merge(streamflow_deviation, grouped_streamflow, on=['site_id', 'month_day'])

    discharge = streamflow_deviation['discharge_cfs_mean']
    discharge_mean = streamflow_deviation['discharge_mean']
    discharge_std = streamflow_deviation['discharge_std']
    streamflow_deviation['streamflow_deviation'] = (discharge - discharge_mean) / discharge_std
    streamflow_deviation = streamflow_deviation.sort_values(['site_id', 'datetime'])
    streamflow_deviation_30 = streamflow_deviation.groupby(
        ['forecast_year', 'site_id']
    )['streamflow_deviation'].rolling(window=30, min_periods=1).agg(
        [np.mean]).reset_index(level=[0, 1], drop=True)
    streamflow_deviation_30.columns = ['streamflow_deviation_30_mean']
    streamflow_deviation = streamflow_deviation.join(streamflow_deviation_30)
    streamflow_deviation_season = streamflow_deviation.groupby(
        ['forecast_year', 'site_id']
    )['streamflow_deviation'].rolling(window=180, min_periods=1).agg(
        [np.mean]).reset_index(level=[0, 1], drop=True)
    streamflow_deviation_season.columns = ['streamflow_deviation_season_mean']
    streamflow_deviation = streamflow_deviation.join(streamflow_deviation_season)
    # Set issue_date to be one day ahead, so we only join with past data
    streamflow_deviation['issue_date'] = (streamflow_deviation['datetime'] + pd.Timedelta(days=1)).dt.strftime(
        '%Y-%m-%d')

    streamflow_columns = ['streamflow_deviation_30_mean', 'streamflow_deviation_season_mean']

    df_features = clean_and_merge_features(df_features, streamflow_deviation, streamflow_columns,
                                           ['site_id', 'issue_date'])

    return df_features


def _read_usgs_streamflow_data(
        site_id: str,
        issue_date: str,
        forecast_year: int,
        streamflow_data_dir: Path,
) -> pd.DataFrame:
    '''Read USGS daily mean streamflow data for a given forecast site as of a given forecast issue
    date.

    Args:
        site_id (str): Identifier for forecast site
        issue_date (str | datetime.date | pd.Timestamp): Date that forecast is being issued for

    Returns:
        pd.DateFrame: dateframe with columns ['datetime', 'discharge_cfs_mean']
    '''

    issue_date = pd.to_datetime(issue_date)
    fy_dir = streamflow_data_dir / f'FY{forecast_year}'
    fy_dir.mkdir(exist_ok=True, parents=True)
    path = fy_dir / f'{site_id}.csv'
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df[df['datetime'].dt.date < issue_date.date()][['datetime', MEAN_DISCHARGE_RAW_COL, 'site_no']]
    df = df.rename(columns={MEAN_DISCHARGE_RAW_COL: MEAN_DISCHARGE_READABLE_COL})
    return df.copy()
