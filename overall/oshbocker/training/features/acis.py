import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
from features.feature_utils import SITE_BASINS, compute_deviation
from loguru import logger


def generate_acis_deviation(data_dir: Path,
                            preprocessed_dir: Path,
                            df_features: pd.DataFrame,
                            acis_windows: List,
                            cv_year: int,
                            test_features: pd.DataFrame) -> pd.DataFrame:
    logger.info('Generating acis climate features.')

    precip_file = preprocessed_dir / 'train_acis_precip.csv'
    maxt_file = preprocessed_dir / 'train_acis_maxt.csv'
    if precip_file.is_file() and maxt_file.is_file():
        logger.info('Acis files already exists, pulling existing.')
        acis_precip = pd.read_csv(precip_file)
        acis_precip['date'] = pd.to_datetime(acis_precip['date'])
        acis_maxt = pd.read_csv(maxt_file)
        acis_maxt['date'] = pd.to_datetime(acis_maxt['date'])
    else:
        all_precips = []
        all_temps = []
        for site_id in df_features['site_id'].unique():
            site_list = []
            date_list = []
            station_list = []
            precip_list = []
            maxt_list = []
            for year in df_features['year'].unique():
                sdate = f'{year - 1}-10-01'
                edate = f'{year}-07-21'
                input_dict = {
                    'basin': SITE_BASINS[site_id],
                    'sdate': sdate,
                    'edate': edate,
                    'meta': 'name, sids',
                    'elems': [{
                        'name': 'pcpn',
                        'interval': 'dly',
                        'duration': 1,
                        'reduce': {'reduce': 'sum'},
                        'maxmissing': 3
                    }, {
                        'name': 'maxt',
                        'interval': 'dly',
                        'duration': 1,
                        'reduce': {'reduce': 'sum'},
                        'maxmissing': 3
                    }]
                }
                dates = list(pd.date_range(sdate, edate, freq='D'))
                sites = [site_id] * len(dates)
                params = {'params': json.dumps(input_dict)}
                headers = {'Accept': 'application/json'}
                req = requests.post('http://data.rcc-acis.org/MultiStnData', data=params, headers=headers)
                response = req.json()
                acis_data = response['data']

                for v in range(0, len(acis_data)):
                    site_list += sites
                    date_list += dates
                    stations = [acis_data[v]['meta']['name']] * len(dates)
                    station_list += stations
                    precip_list += list(np.array(acis_data[v]['data'])[:, 0])
                    maxt_list += list(np.array(acis_data[v]['data'])[:, 1])
            acis_precip = pd.DataFrame({
                'site_id': site_list,
                'date': date_list,
                'station': station_list,
                'precip': precip_list,
            })
            all_precips.append(acis_precip)
            acis_maxt = pd.DataFrame({
                'site_id': site_list,
                'date': date_list,
                'station': station_list,
                'maxt': maxt_list,
            })
            all_temps.append(acis_maxt)

        acis_precip = pd.concat(all_precips)
        acis_precip['date'] = pd.to_datetime(acis_precip['date'])
        acis_precip['month'] = acis_precip['date'].dt.month
        acis_precip['year'] = acis_precip['date'].dt.year
        acis_precip['forecast_year'] = acis_precip.apply(lambda rw: rw['year'] if rw['month'] < 8 else rw['year'] + 1,
                                                         axis=1)
        acis_precip['month_day'] = acis_precip['date'].dt.strftime('%m%d')
        acis_precip['day_of_year'] = acis_precip['date'].dt.day_of_year
        acis_precip['precip'] = acis_precip['precip'].replace('T', 0).replace('M', np.nan)
        acis_precip['precip'] = acis_precip['precip'].astype(float)

        acis_precip.to_csv(precip_file, index=False)

        acis_maxt = pd.concat(all_temps)
        acis_maxt['date'] = pd.to_datetime(acis_maxt['date'])
        acis_maxt['month'] = acis_maxt['date'].dt.month
        acis_maxt['year'] = acis_maxt['date'].dt.year
        acis_maxt['forecast_year'] = acis_maxt.apply(lambda rw: rw['year'] if rw['month'] < 8 else rw['year'] + 1,
                                                     axis=1)
        acis_maxt['month_day'] = acis_maxt['date'].dt.strftime('%m%d')
        acis_maxt['day_of_year'] = acis_maxt['date'].dt.day_of_year
        acis_maxt['maxt'] = acis_maxt['maxt'].replace('T', 0).replace('M', np.nan)
        acis_maxt['maxt'] = acis_maxt['maxt'].astype(float)

        acis_maxt.to_csv(maxt_file, index=False)

    logger.info('Generating acis precipitation features.')

    applicable_sites = df_features['site_id'].unique()

    df_features, test_features = compute_deviation(
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        cv_year=cv_year,
        input_col='precip',
        output_col='precip_deviation',
        pivot_col='station',
        base_data=acis_precip,
        df_features=df_features,
        applicable_sites=applicable_sites,
        rolling_windows=acis_windows,
        min_num_locations=5,
        thresh=15,
        test_features=test_features,
        agg_months=[1, 2, 3, 4, 5, 6, 7],
        sort_ascending=False
    )

    logger.info('Generating acis temperature features.')

    df_features, test_features = compute_deviation(
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        cv_year=cv_year,
        input_col='maxt',
        output_col='maxt_deviation',
        pivot_col='station',
        base_data=acis_maxt,
        df_features=df_features,
        applicable_sites=applicable_sites,
        rolling_windows=acis_windows,
        min_num_locations=5,
        thresh=15,
        test_features=test_features,
        agg_months=[1, 2, 3, 4, 5, 6, 7],
        sort_ascending=True
    )

    return df_features, test_features
