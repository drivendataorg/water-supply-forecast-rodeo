from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def generate_cdec_deviation(data_dir: Path,
                            preprocessed_dir: Path,
                            df_features: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating CDEC deviation features.")

    sites_to_cdec_stations = pd.read_csv(data_dir / 'cdec/sites_to_cdec_stations.csv')

    df_list = []
    # Concatenate all the CDEC year files into single station files for all years
    for station_id in sites_to_cdec_stations['station_id'].unique():
        for year in df_features['year'].unique():
            try:
                cdec = pd.read_csv(f'{data_dir}/cdec/FY{year}/{station_id}.csv')
                cdec = cdec[(cdec['SENSOR_NUM'].isin([82])) & (cdec['value'] != -9999.0)]
                cdec['station_id'] = station_id
                df_list.append(cdec[['station_id', 'SENSOR_NUM', 'date', 'value', 'dataFlag']])
            except:
                pass

    all_cdec = pd.concat(df_list)

    def negative_to_zero(rw):
        if rw['value'] < 0:
            rw['value'] = 0.0
        return rw

    all_cdec = all_cdec.apply(lambda rw: negative_to_zero(rw), axis=1)
    all_cdec = all_cdec[all_cdec['value'] < 200]
    all_cdec.to_csv(preprocessed_dir / 'train_cdec.csv', index=False)
    all_cdec["month_day"] = pd.to_datetime(all_cdec["date"]).dt.strftime("%m%d")
    cdec_grouped = all_cdec.groupby(["month_day", "station_id"])["value"].agg([np.mean, np.std]).reset_index()
    cdec_grouped.columns = ['month_day', 'station_id', 'cdec_mean', 'cdec_std']
    logger.info(cdec_grouped.describe())
    # The mean and std for each cdec station in the training dataset
    cdec_grouped.to_csv(preprocessed_dir / 'cdec_parameters.csv', index=False)

    all_dfs = []

    for site_id in sites_to_cdec_stations['site_id'].unique():
        relevant_stations = sites_to_cdec_stations[sites_to_cdec_stations['site_id'] == site_id][
            ['station_id']]
        cdec = pd.merge(all_cdec, relevant_stations, on='station_id')
        cdec = pd.merge(cdec, cdec_grouped, on=["month_day", "station_id"])
        cdec['cdec_deviation'] = (cdec['value'] - cdec['cdec_mean']) / cdec['cdec_std']
        cdec['cdec_deviation'] = cdec['cdec_deviation'].replace(np.inf, np.nan).replace(-np.inf, np.nan)
        site_cdec = cdec.groupby('date')[['cdec_deviation']].mean().reset_index()
        site_cdec['site_id'] = site_id
        all_dfs.append(site_cdec)

    all_cdec_deviation = pd.concat(all_dfs)
    all_cdec_deviation = all_cdec_deviation.sort_values(['site_id', 'date'])
    all_cdec_deviation['date'] = pd.to_datetime(all_cdec_deviation['date'])

    def cdec_deviation_for_site(rw, window: int = 30):
        rw_year = rw['year']
        element_cols = ['cdec_deviation']
        element_means = all_cdec_deviation[
                            (all_cdec_deviation['site_id'] == rw['site_id']) &
                            (all_cdec_deviation['date'] >= f"{rw_year - 1}-10-01") &
                            (all_cdec_deviation['date'] < rw['issue_date'])
                            ][-window:][element_cols].mean()

        for idx, col in enumerate(element_cols):
            rw[f'{col}_{window}'] = element_means[idx]

        return rw

    df_features = df_features.apply(lambda rw: cdec_deviation_for_site(rw, window=0), axis=1)
    df_features = df_features.apply(lambda rw: cdec_deviation_for_site(rw, window=30), axis=1)

    logger.info(df_features[['cdec_deviation_0', 'cdec_deviation_30']].describe())

    return df_features
