from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def generate_cdec_deviation(src_dir: Path,
                            data_dir: Path,
                            test_features: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating CDEC deviation features.")

    sites_to_cdec_stations = pd.read_csv(data_dir / 'cdec/sites_to_cdec_stations.csv')
    train_cdec = pd.read_csv(src_dir / f'feature_parameters/train_cdec.csv')

    df_list = [train_cdec]

    def negative_to_zero(rw):
        if rw['value'] < 0:
            rw['value'] = 0.0
        return rw

    # Concatenate all the CDEC year files into single station files for all years
    for station_id in train_cdec['station_id'].unique():
        for year in test_features['year'].unique():
            try:
                cdec = pd.read_csv(f'{data_dir}/cdec/FY{year}/{station_id}.csv')
                cdec = cdec[(cdec['SENSOR_NUM'].isin([82])) & (cdec['value'] != -9999.0)]
                cdec = cdec[cdec['value'] < 200]
                cdec = cdec.apply(lambda rw: negative_to_zero(rw), axis=1)
                cdec['station_id'] = station_id
                df_list.append(cdec[['station_id', 'SENSOR_NUM', 'date', 'value']])
            except:
                pass

    all_cdec = pd.concat(df_list)
    cdec_max_date = all_cdec['date'].max()
    logger.info(f"Cdec max date: {cdec_max_date}")

    all_dfs = []
    for site_id in sites_to_cdec_stations['site_id'].unique():
        relevant_stations = sites_to_cdec_stations[sites_to_cdec_stations['site_id'] == site_id][
            ['station_id']]
        cdec = pd.merge(all_cdec, relevant_stations, on='station_id')
        cdec["month_day"] = pd.to_datetime(cdec["date"]).dt.strftime("%m%d")
        cdec_grouped = cdec.groupby(["month_day", "station_id"])["value"].agg([np.mean, np.std]).reset_index()
        cdec_grouped.columns = ['month_day', 'station_id', 'cdec_mean', 'cdec_std']
        cdec = pd.merge(cdec, cdec_grouped, on=["month_day", "station_id"])
        cdec['cdec_deviation'] = (cdec['value'] - cdec['cdec_mean']) / cdec['cdec_std']
        cdec['cdec_deviation'] = cdec['cdec_deviation'].replace(np.inf, np.nan).replace(-np.inf, np.nan)
        site_cdec = cdec.groupby('date')[['cdec_deviation']].mean().reset_index()
        site_cdec['site_id'] = site_id
        all_dfs.append(site_cdec)

    all_cdec_grouped = pd.concat(all_dfs)
    all_cdec_grouped['date'] = pd.to_datetime(all_cdec_grouped['date'])

    def cdec_deviation_for_site(rw, window: int = 30):
        rw_year = rw['year']
        element_cols = ['cdec_deviation']
        element_means = all_cdec_grouped[
                            (all_cdec_grouped['site_id'] == rw['site_id']) &
                            (all_cdec_grouped['date'] >= f"{rw_year - 1}-10-01") &
                            (all_cdec_grouped['date'] < rw['issue_date'])
                            ][-window:][element_cols].mean()

        for idx, col in enumerate(element_cols):
            rw[f'{col}_{window}'] = element_means[idx]

        return rw

    test_features = test_features.apply(lambda rw: cdec_deviation_for_site(rw, window=0), axis=1)
    test_features = test_features.apply(lambda rw: cdec_deviation_for_site(rw, window=30), axis=1)

    logger.info(test_features.groupby('site_id')[['cdec_deviation_0', 'cdec_deviation_30']].mean().round(3))

    return test_features
