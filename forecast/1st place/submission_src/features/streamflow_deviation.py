from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from wsfr_read.streamflow import read_usgs_streamflow_data

from wsfr_download.usgs_streamflow import download_alternative_usgs_streamflow, ALTERNATIVE_USGS_IDS


MEAN_DISCHARGE_RAW_COL = "00060_Mean"
MEAN_DISCHARGE_READABLE_COL = "discharge_cfs_mean"

def generate_streamflow_deviation(src_dir: Path,
                                  preprocessed_dir: Path,
                                  test_features: pd.DataFrame) -> pd.DataFrame:
    forecast_years = test_features['year'].unique()
    download_alternative_usgs_streamflow(forecast_years,
                                         preprocessed_dir,
                                         skip_existing=False)

    logger.info("Generating streamflow deviation features.")
    train_streamflow = pd.read_csv(src_dir / 'feature_parameters/train_streamflow.csv')
    use_alternative = True
    df_list = [train_streamflow]
    for site_id in test_features['site_id'].unique():
        if use_alternative and (site_id in ALTERNATIVE_USGS_IDS.keys()):
            for year in test_features['year'].unique():
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
            for year in test_features['year'].unique():
                try:
                    df = read_usgs_streamflow_data(site_id,
                                                   f'{year}-07-23').replace(-999999.0, np.nan)
                    df['site_id'] = site_id
                    df_list.append(df)
                except Exception as e:
                    print(e)

    streamflow_deviation = pd.concat(df_list)
    streamflow_deviation['datetime'] = pd.to_datetime(streamflow_deviation['datetime'])
    streamflow_max_date = streamflow_deviation['datetime'].max()
    logger.info(f"Streamflow max date: {streamflow_max_date}")
    streamflow_deviation['month_day'] = streamflow_deviation['datetime'].dt.strftime("%m%d")
    grouped_streamflow = streamflow_deviation.groupby(['site_id', 'month_day'])['discharge_cfs_mean'].agg(
        [np.mean, np.std]).reset_index()
    grouped_streamflow.columns = ['site_id', 'month_day', 'discharge_mean', 'discharge_std']
    streamflow_deviation = pd.merge(streamflow_deviation, grouped_streamflow, on=['site_id', 'month_day'])

    discharge = streamflow_deviation['discharge_cfs_mean']
    discharge_mean = streamflow_deviation['discharge_mean']
    discharge_std = streamflow_deviation['discharge_std']
    streamflow_deviation['discharge_deviation'] = (discharge - discharge_mean) / discharge_std
    streamflow_deviation.to_csv(preprocessed_dir / 'streamflow_deviation.csv', index=False)

    def rolling_streamflow_deviation(rw, window=30, agg='mean'):
        rw_year = rw['year']
        sub_df = streamflow_deviation[(streamflow_deviation['site_id'] == rw['site_id']) &
                                (streamflow_deviation['datetime'] < rw['issue_date']) &
                                (streamflow_deviation['datetime'] >= f"{rw_year - 1}-10-01")]

        rw[f'streamflow_deviation_{window}_{agg}'] = sub_df[-window:]['discharge_deviation'].agg(agg)
        rw[f'streamflow_deviation_season_{agg}'] = sub_df['discharge_deviation'].agg(agg)

        return rw

    test_features = test_features.apply(lambda rw: rolling_streamflow_deviation(rw), axis=1)
    test_features['streamflow_deviation_30_mean'] = test_features['streamflow_deviation_30_mean'].fillna(
        test_features['streamflow_deviation_season_mean']
    )
    logger.info(test_features.groupby('site_id')[['streamflow_deviation_30_mean',
                               'streamflow_deviation_season_mean']].mean().round(3))

    return test_features

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
