from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from wsfr_read.streamflow import read_usgs_streamflow_data

from wsfr_download_train.usgs_streamflow import download_alternative_usgs_streamflow, ALTERNATIVE_USGS_IDS


MEAN_DISCHARGE_RAW_COL = "00060_Mean"
MEAN_DISCHARGE_READABLE_COL = "discharge_cfs_mean"

def generate_streamflow_deviation(preprocessed_dir: Path,
                                  df_features: pd.DataFrame) -> pd.DataFrame:
    forecast_years = df_features['year'].unique()
    download_alternative_usgs_streamflow(forecast_years,
                                         preprocessed_dir,
                                         skip_existing=True)

    logger.info("Generating streamflow deviation features.")
    df_list = []
    for site_id in df_features['site_id'].unique():
        if site_id in ALTERNATIVE_USGS_IDS.keys():
            for year in df_features['year'].unique():
                try:
                    df = _read_usgs_streamflow_data(site_id,
                                                    f'{year}-07-23',
                                                    year,
                                                    preprocessed_dir).replace(-999999.0, np.nan)
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
    streamflow_deviation.to_csv(preprocessed_dir / 'train_streamflow.csv', index=False)
    streamflow_deviation['datetime'] = pd.to_datetime(streamflow_deviation['datetime'])
    streamflow_deviation['month_day'] = streamflow_deviation['datetime'].dt.strftime("%m%d")
    grouped_streamflow = streamflow_deviation.groupby(['site_id', 'month_day'])['discharge_cfs_mean'].agg(
        [np.mean, np.std]).reset_index()
    grouped_streamflow.columns = ['site_id', 'month_day', 'discharge_mean', 'discharge_std']
    grouped_streamflow.to_csv(preprocessed_dir / 'streamflow_parameters.csv', index=False)
    streamflow_deviation = pd.merge(streamflow_deviation, grouped_streamflow, on=['site_id', 'month_day'])

    discharge = streamflow_deviation['discharge_cfs_mean']
    discharge_mean = streamflow_deviation['discharge_mean']
    discharge_std = streamflow_deviation['discharge_std']
    streamflow_deviation['discharge_deviation'] = (discharge - discharge_mean) / discharge_std
    streamflow_deviation = streamflow_deviation.sort_values(['site_id', 'datetime'])

    def rolling_streamflow_deviation(rw, window=30, agg='mean'):
        rw_year = rw['year']
        sub_df = streamflow_deviation[(streamflow_deviation['site_id'] == rw['site_id']) &
                                (streamflow_deviation['datetime'] < rw['issue_date']) &
                                (streamflow_deviation['datetime'] >= f"{rw_year - 1}-10-01")]

        rw[f'streamflow_deviation_{window}_{agg}'] = sub_df[-window:]['discharge_deviation'].agg(agg)
        rw[f'streamflow_deviation_season_{agg}'] = sub_df['discharge_deviation'].agg(agg)

        return rw

    df_features = df_features.apply(lambda rw: rolling_streamflow_deviation(rw), axis=1)
    logger.info(df_features[['streamflow_deviation_30_mean', 'streamflow_deviation_season_mean']].describe())

    return df_features

def _read_usgs_streamflow_data(
        site_id: str,
        issue_date: str,
        forecast_year: int,
        preprocessed_dir: Path,
) -> pd.DataFrame:
    """Read USGS daily mean streamflow data for a given forecast site as of a given forecast issue
    date.

    Args:
        site_id (str): Identifier for forecast site
        issue_date (str | datetime.date | pd.Timestamp): Date that forecast is being issued for

    Returns:
        pd.DateFrame: dateframe with columns ["datetime", "discharge_cfs_mean"]
    """

    issue_date = pd.to_datetime(issue_date)
    fy_dir = preprocessed_dir / "usgs_streamflow" / "alternative" / f"FY{forecast_year}"
    fy_dir.mkdir(exist_ok=True, parents=True)
    path = fy_dir / f"{site_id}.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df[df["datetime"].dt.date < issue_date.date()][["datetime", MEAN_DISCHARGE_RAW_COL]]
    df = df.rename(columns={MEAN_DISCHARGE_RAW_COL: MEAN_DISCHARGE_READABLE_COL})
    return df.copy()
