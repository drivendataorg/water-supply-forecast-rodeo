from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def generate_snotel_deviation(data_dir: Path,
                              preprocessed_dir: Path,
                              df_features: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating Snotel deviation features.")

    sites_to_snotel_stations = pd.read_csv(data_dir / 'snotel/sites_to_snotel_stations.csv')
    sites_to_snotel_stations['station_triplet'] = sites_to_snotel_stations['stationTriplet'].str.replace(":", "_")

    df_list = []
    # Concatenate all the Snotel year files into single station files for all years
    for station_triplet in sites_to_snotel_stations['station_triplet'].unique():
        for year in df_features['year'].unique():
            try:
                df = pd.read_csv(data_dir / f'snotel/FY{year}/{station_triplet}.csv')
                df['station_triplet'] = station_triplet
                df_list.append(df[['station_triplet', 'date', 'WTEQ_DAILY']])
            except:
                pass

    all_snotel = pd.concat(df_list)
    all_snotel.to_csv(preprocessed_dir / 'train_snotel.csv', index=False)
    all_snotel["month_day"] = pd.to_datetime(all_snotel["date"]).dt.strftime("%m%d")
    snotel_grouped = all_snotel.groupby(["month_day", "station_triplet"])[['WTEQ_DAILY']].agg(
        [np.mean, np.std]).reset_index()
    snotel_grouped.columns = ['month_day', 'station_triplet', 'wteq_mean', 'wteq_std']
    logger.info(snotel_grouped.describe())

    all_dfs = []
    for site_id in sites_to_snotel_stations['site_id'].unique():
        relevant_stations = sites_to_snotel_stations[sites_to_snotel_stations['site_id'] == site_id][
            ['station_triplet']]
        snotel = pd.merge(all_snotel, relevant_stations, on='station_triplet')
        snotel = pd.merge(snotel, snotel_grouped, on=["month_day", "station_triplet"])
        snotel['snotel_wteq_deviation'] = (snotel['WTEQ_DAILY'] - snotel['wteq_mean']) / snotel['wteq_std']
        snotel['snotel_wteq_deviation'] = snotel['snotel_wteq_deviation'].replace(np.inf, np.nan).replace(-np.inf, np.nan)
        snotel = snotel.dropna(subset='snotel_wteq_deviation')
        site_snotel = snotel.groupby('date')['snotel_wteq_deviation'].mean().reset_index()
        site_snotel['site_id'] = site_id
        all_dfs.append(site_snotel)

    all_snotel_deviation = pd.concat(all_dfs)
    all_snotel_deviation = all_snotel_deviation.sort_values(['site_id', 'date'])

    def snotel_deviation_for_site(rw, window: int = 30):
        rw_year = rw['year']

        wteq = all_snotel_deviation[
                            (all_snotel_deviation['site_id'] == rw['site_id']) &
                            (all_snotel_deviation['date'] >= f"{rw_year - 1}-10-01") &
                            (all_snotel_deviation['date'] < rw['issue_date'])
                            ][-window:]['snotel_wteq_deviation'].mean()

        rw[f'snotel_wteq_deviation_{window}'] = wteq

        return rw

    df_features = df_features.apply(lambda rw: snotel_deviation_for_site(rw, window=0), axis=1)
    df_features = df_features.apply(lambda rw: snotel_deviation_for_site(rw, window=30), axis=1)
    logger.info(df_features[['snotel_wteq_deviation_0', 'snotel_wteq_deviation_30']].describe())

    return df_features
