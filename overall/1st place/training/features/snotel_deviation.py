from pathlib import Path
from typing import List

import pandas as pd
from features.feature_utils import compute_deviation
from loguru import logger


def generate_snotel_deviation(data_dir: Path,
                              preprocessed_dir: Path,
                              df_features: pd.DataFrame,
                              snow_pack_windows: List,
                              cv_year: int,
                              test_features: pd.DataFrame) -> pd.DataFrame:
    logger.info('Generating Snotel deviation features.')

    sites_to_snotel_stations = pd.read_csv(data_dir / 'snotel/sites_to_snotel_stations.csv')
    sites_to_snotel_stations['station_triplet'] = sites_to_snotel_stations['stationTriplet'].str.replace(':', '_')

    snotel_file = preprocessed_dir / 'train_snotel.csv'
    if snotel_file.is_file():
        logger.info('Snotel file already exists, pulling existing.')
        all_snotel = pd.read_csv(snotel_file)
    else:
        df_list = []
        # Concatenate all the Snotel year files into single station files for all years
        for idx, rw in sites_to_snotel_stations.iterrows():
            station_triplet = rw['station_triplet']
            site_id = rw['site_id']
            for year in df_features['year'].unique():
                try:
                    df = pd.read_csv(data_dir / f'snotel/FY{year}/{station_triplet}.csv')
                    df['station_triplet'] = station_triplet
                    df['site_id'] = site_id
                    df_list.append(df)
                except:
                    pass

        all_snotel = pd.concat(df_list)
        all_snotel.columns = ['date', 'prec', 'snwd', 'tavg', 'tmax', 'tmin', 'wteq', 'station_triplet', 'site_id']

        all_snotel['date'] = pd.to_datetime(all_snotel['date'])
        all_snotel['year'] = all_snotel['date'].dt.year
        all_snotel['month'] = all_snotel['date'].dt.month
        all_snotel['forecast_year'] = all_snotel.apply(lambda rw: rw['year'] if rw['month'] < 8 else rw['year'] + 1,
                                                       axis=1)
        all_snotel['day_of_year'] = all_snotel['date'].dt.day_of_year
        all_snotel['month_day'] = all_snotel['date'].dt.strftime('%m%d')
        all_snotel.to_csv(snotel_file, index=False)

    applicable_sites = df_features['site_id'].unique()

    df_features, test_features = compute_deviation(
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        cv_year=cv_year,
        input_col='wteq',
        output_col='snotel_wteq_deviation',
        pivot_col='station_triplet',
        base_data=all_snotel,
        df_features=df_features,
        applicable_sites=applicable_sites,
        rolling_windows=snow_pack_windows,
        min_num_locations=15,
        thresh=20,
        test_features=test_features,
        agg_months=[1, 2, 3, 4, 5, 6, 7, 11, 12],
        sort_ascending=False
    )

    df_features, test_features = compute_deviation(
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        cv_year=cv_year,
        input_col='prec',
        output_col='snotel_prec_deviation',
        pivot_col='station_triplet',
        base_data=all_snotel,
        df_features=df_features,
        applicable_sites=applicable_sites,
        rolling_windows=snow_pack_windows,
        min_num_locations=15,
        thresh=20,
        test_features=test_features,
        agg_months=[1, 2, 3, 4, 5, 6, 7],
        sort_ascending=False
    )

    df_features, test_features = compute_deviation(
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        cv_year=cv_year,
        input_col='tmax',
        output_col='snotel_tmax_deviation',
        pivot_col='station_triplet',
        base_data=all_snotel,
        df_features=df_features,
        applicable_sites=applicable_sites,
        rolling_windows=snow_pack_windows,
        min_num_locations=10,
        thresh=20,
        test_features=test_features,
        agg_months=[1, 2, 3, 4, 5, 6, 7],
        sort_ascending=True
    )

    return df_features, test_features
