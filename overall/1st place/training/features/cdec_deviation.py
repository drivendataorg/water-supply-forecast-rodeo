from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from features.feature_utils import compute_deviation


def generate_cdec_deviation(data_dir: Path,
                            preprocessed_dir: Path,
                            df_features: pd.DataFrame,
                            snow_pack_windows: List,
                            cv_year: int,
                            test_features: pd.DataFrame) -> pd.DataFrame:
    logger.info('Generating CDEC deviation features.')

    sites_to_cdec_stations = pd.read_csv(data_dir / 'cdec/sites_to_cdec_stations.csv')
    cdec_file = preprocessed_dir / 'train_cdec.csv'
    if cdec_file.is_file():
        logger.info('cdec file already exists, pulling existing.')
        all_cdec = pd.read_csv(cdec_file)
    else:
        df_list = []
        # Concatenate all the CDEC year files into single station files for all years
        for idx, rw in sites_to_cdec_stations.iterrows():
            station_id = rw['station_id']
            site_id = rw['site_id']
            for year in df_features['year'].unique():
                try:
                    cdec = pd.read_csv(f'{data_dir}/cdec/FY{year}/{station_id}.csv')
                    cdec = cdec[(cdec['SENSOR_NUM'].isin([82])) & (cdec['value'] != -9999.0)]
                    cdec['station_id'] = station_id
                    cdec['site_id'] = site_id
                    df_list.append(cdec[['site_id', 'station_id', 'SENSOR_NUM', 'date', 'value', 'dataFlag']])
                except:
                    pass

        all_cdec = pd.concat(df_list)

        def negative_to_zero(rw):
            if rw['value'] < 0:
                rw['value'] = 0.0
            return rw

        all_cdec = all_cdec.apply(lambda rw: negative_to_zero(rw), axis=1)
        all_cdec = all_cdec[all_cdec['value'] < 200]
        all_cdec['cdec'] = all_cdec['value']
        all_cdec['date'] = pd.to_datetime(all_cdec['date'])
        all_cdec['day_of_year'] = all_cdec['date'].dt.day_of_year
        all_cdec['year'] = all_cdec['date'].dt.year
        all_cdec['month'] = all_cdec['date'].dt.month
        all_cdec['month_day'] = all_cdec['date'].dt.strftime('%m%d')
        all_cdec['forecast_year'] = all_cdec.apply(lambda rw: rw['year'] if rw['month'] < 8 else rw['year'] + 1,
                                                   axis=1)
        all_cdec.to_csv(preprocessed_dir / 'train_cdec.csv', index=False)

    applicable_sites = sites_to_cdec_stations['site_id'].unique()
    
    df_features, test_features = compute_deviation(
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        cv_year=cv_year,
        input_col='cdec',
        output_col='cdec_deviation',
        pivot_col='station_id',
        base_data=all_cdec,
        df_features=df_features,
        applicable_sites=applicable_sites,
        rolling_windows=snow_pack_windows,
        min_num_locations=7,
        thresh=15,
        test_features=test_features,
        agg_months=[1, 2, 3, 4, 5, 6, 7, 11, 12],
        sort_ascending=False
    )

    return df_features, test_features
