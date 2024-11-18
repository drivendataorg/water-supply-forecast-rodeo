#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parents[0]))

from loguru import logger
import pandas as pd


def main(data_dir: Path,
         preprocessed_dir: Path,
         train_since: int = 1960):
    logger.info(f'Beginning generation of train features from {train_since}.')

    update_metadata(data_dir, preprocessed_dir)

    metadata = pd.read_csv(preprocessed_dir / 'metadata.csv', dtype={'usgs_id': 'string'})

    train_features_file = preprocessed_dir / 'train_features.csv'
    if train_features_file.is_file():
        logger.info('train features file exists, pulling existing.')
        train_features = pd.read_csv(train_features_file)
    else:
        logger.info('train features file does not exist, generate from scratch.')
        prior_train = pd.read_csv(DATA_DIR / 'prior_historical_labels.csv')
        cv_train = pd.read_csv(DATA_DIR / 'cross_validation_labels.csv')
        train = pd.concat([prior_train, cv_train])

        site_list = []
        date_list = []
        year_list = []
        month_list = []
        day_list = []
        for year in train['year'].unique():
            for idx, rw in metadata.iterrows():
                for month in [1, 2, 3, 4, 5, 6, 7]:
                    for day in [1, 8, 15, 22]:
                        site_list.append(rw['site_id'])
                        date_list.append(f'{year}-{month:02d}-{day:02d}')
                        year_list.append(year)
                        month_list.append(month)
                        day_list.append(day)

        train_idx = pd.DataFrame(
            {'site_id': site_list, 'issue_date': date_list, 'year': year_list, 'month': month_list, 'day': day_list})

        train_features = pd.merge(train_idx, train, on=['site_id', 'year'], how='left')
        train_features['day_of_year'] = pd.to_datetime(train_features['issue_date']).dt.day_of_year

        train_features = train_features[train_features['year'] > train_since]
        print(f'train_features shape, {train_features.shape}')
        metadata_keep_cols = ['site_id', 'elevation', 'latitude', 'longitude', 'drainage_area',
                              'season_start_month', 'season_end_month']
        train_features = pd.merge(train_features,
                                  metadata[metadata_keep_cols],
                                  on=['site_id'], how='left')

        # Turn site_id into categorical integer
        train_features['site_cat'] = pd.factorize(train_features['site_id'])[0]

    site_elevations = generate_elevations(data_dir, preprocessed_dir)
    elevation_cols = ['elevation_means', 'elevation_stds', 'southern_gradient_rates', 'eastern_gradient_rates']

    train_features = clean_and_merge_features(train_features, site_elevations, elevation_cols, ['site_id'])

    cv_test = pd.read_csv(DATA_DIR / 'cross_validation_labels.csv')
    try:
        all_lat_lon = pd.read_csv(preprocessed_dir / 'train_lat_lon_pdsi.csv')
    except:
        all_lat_lon = None
    cv_dfs = []
    cv_test_dfs = []
    cv_splits = list(range(2004, 2024))
    for cv_year in cv_splits:
        test = cv_test[cv_test['year'] == cv_year]
        test_features = generate_base_test_data(data_dir, preprocessed_dir, cv_year, metadata, test)
        cv_features, test_features = generate_dynamic_cv_features(data_dir,
                                                                  preprocessed_dir,
                                                                  metadata,
                                                                  train_features,
                                                                  test_features,
                                                                  cv_year,
                                                                  all_lat_lon)
        cv_features['cv_year'] = cv_year
        cv_dfs.append(cv_features)
        cv_test_dfs.append(test_features)

    cv_features = pd.concat(cv_dfs)
    cv_features.to_csv(preprocessed_dir / 'cv_features.csv', index=False)

    logger.info(cv_features.columns)
    logger.info(cv_features.shape)

    cv_test_features = pd.concat(cv_test_dfs)
    cv_test_features.to_csv(preprocessed_dir / 'cv_test_features.csv', index=False)
    logger.info(cv_test_features.shape)


def generate_base_test_data(data_dir: Path,
                            preprocessed_dir: Path,
                            year: int,
                            metadata: pd.DataFrame,
                            test: pd.DataFrame):
    site_list = []
    date_list = []
    year_list = []
    month_list = []
    day_list = []
    for idx, rw in metadata.iterrows():
        for month in [1, 2, 3, 4, 5, 6, 7]:
            for day in [1, 8, 15, 22]:
                site_list.append(rw['site_id'])
                date_list.append(f'{year}-{month:02d}-{day:02d}')
                year_list.append(year)
                month_list.append(month)
                day_list.append(day)

    test_idx = pd.DataFrame(
        {'site_id': site_list, 'issue_date': date_list, 'year': year_list, 'month': month_list, 'day': day_list})

    test_features = pd.merge(test_idx, test, on=['site_id', 'year'], how='left')
    test_features['day_of_year'] = pd.to_datetime(test_features['issue_date']).dt.day_of_year

    print(f'test_features shape, {test_features.shape}')
    metadata_keep_cols = ['site_id', 'elevation', 'latitude', 'longitude', 'drainage_area',
                          'season_start_month', 'season_end_month']
    test_features = pd.merge(test_features,
                             metadata[metadata_keep_cols],
                             on=['site_id'], how='left')

    # Turn site_id into categorical integer
    test_features['site_cat'] = pd.factorize(test_features['site_id'])[0]

    site_elevations = generate_elevations(data_dir, preprocessed_dir)

    test_features = pd.merge(test_features, site_elevations, on='site_id')

    return test_features


def generate_dynamic_cv_features(data_dir: Path,
                                 preprocessed_dir: Path,
                                 metadata: pd.DataFrame,
                                 train_features: pd.DataFrame,
                                 test_features: pd.DataFrame,
                                 cv_year: Optional[int] = None,
                                 all_lat_lon: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    logger.info(f'Beginning generation of dynamic time features for {cv_year}.')

    # Booleans to determine whether to generate specific features
    naturalized_flow_features = True
    pdsi_features = True
    acis_features = True
    snotel_features = True
    cdec_features = True
    streamflow_features = True
    ua_swann_features = True

    snow_pack_windows = [10, 20, 180]
    acis_windows = [10, 30, 180]

    if naturalized_flow_features:
        train_features = generate_monthly_naturalized_flow(data_dir=data_dir,
                                                           preprocessed_dir=preprocessed_dir,
                                                           df_features=train_features,
                                                           metadata=metadata,
                                                           cv_year=cv_year)
        test_features = generate_monthly_naturalized_flow(data_dir=data_dir,
                                                           preprocessed_dir=preprocessed_dir,
                                                           df_features=test_features,
                                                           metadata=metadata,
                                                           cv_year=None)
        test_features = test_features[test_features['year'] == cv_year]
        logger.info(f'{train_features.shape}, {test_features.shape}')

    if pdsi_features:
        # Passing in all_lat_lon file because it takes a minute to read
        train_features, test_features = generate_drought_deviation(data_dir=data_dir,
                                                                   preprocessed_dir=preprocessed_dir,
                                                                   df_features=train_features,
                                                                   cv_year=cv_year,
                                                                   test_features=test_features,
                                                                   all_lat_lon=all_lat_lon)
        logger.info(f'{train_features.shape}, {test_features.shape}')

    if acis_features:
        train_features, test_features = generate_acis_deviation(data_dir=data_dir,
                                                                preprocessed_dir=preprocessed_dir,
                                                                df_features=train_features,
                                                                acis_windows=acis_windows,
                                                                cv_year=cv_year,
                                                                test_features=test_features)
        logger.info(f'{train_features.shape}, {test_features.shape}')

    if snotel_features:
        train_features, test_features = generate_snotel_deviation(data_dir=data_dir,
                                                                  preprocessed_dir=preprocessed_dir,
                                                                  df_features=train_features,
                                                                  snow_pack_windows=snow_pack_windows,
                                                                  cv_year=cv_year,
                                                                  test_features=test_features)
        logger.info(f'{train_features.shape}, {test_features.shape}')

    if cdec_features:
        train_features, test_features = generate_cdec_deviation(data_dir=data_dir,
                                                                preprocessed_dir=preprocessed_dir,
                                                                df_features=train_features,
                                                                snow_pack_windows=snow_pack_windows,
                                                                cv_year=cv_year,
                                                                test_features=test_features)
        logger.info(f'{train_features.shape}, {test_features.shape}')

    if streamflow_features:
        train_features = generate_streamflow_deviation(preprocessed_dir=preprocessed_dir,
                                                       df_features=train_features,
                                                       cv_year=cv_year)
        test_features = generate_streamflow_deviation(preprocessed_dir=preprocessed_dir,
                                                       df_features=test_features,
                                                       cv_year=None)
        logger.info(f'{train_features.shape}, {test_features.shape}')

    if ua_swann_features:
        train_features = generate_ua_swann_deviation(preprocessed_dir=preprocessed_dir,
                                                     df_features=train_features,
                                                     cv_year=cv_year)
        test_features = generate_ua_swann_deviation(preprocessed_dir=preprocessed_dir,
                                                    df_features=test_features,
                                                    cv_year=None)
        logger.info(f'{train_features.shape}, {test_features.shape}')

    for window in snow_pack_windows:
        train_features[f'combined_swe_deviation_{window}'] = train_features[
            [f'snotel_wteq_deviation_{window}', f'cdec_deviation_{window}']].mean(
            axis=1)
        test_features[f'combined_swe_deviation_{window}'] = test_features[
            [f'snotel_wteq_deviation_{window}', f'cdec_deviation_{window}']].mean(
            axis=1)

    return train_features, test_features


def update_metadata(data_dir: Path,
                    preprocessed_dir: Path) -> None:
    metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={'usgs_id': 'string'})
    metadata = metadata.set_index('site_id')

    metadata.loc['ruedi_reservoir_inflow', 'usgs_id'] = '09080400'
    metadata.loc['fontenelle_reservoir_inflow', 'usgs_id'] = '09211200'
    metadata.loc['american_river_folsom_lake', 'usgs_id'] = '11446500'
    metadata.loc['skagit_ross_reservoir', 'usgs_id'] = '12181000'
    metadata.loc['skagit_ross_reservoir', 'drainage_area'] = 999.0
    metadata.loc['boysen_reservoir_inflow', 'usgs_id'] = '06279500'
    metadata.loc['boise_r_nr_boise', 'usgs_id'] = '13185000'
    metadata.loc['sweetwater_r_nr_alcova', 'usgs_id'] = '06235500'

    metadata.to_csv(preprocessed_dir / 'metadata.csv')
    metadata.to_csv(data_dir / 'metadata.csv')


if __name__ == '__main__':
    from pathlib import Path

    import os
    import sys
    import warnings

    sys.path.append(str(Path(__file__).parent.resolve()))

    DATA_DIR = Path.cwd() / 'training/train_data'
    os.environ['WSFR_DATA_ROOT'] = str(DATA_DIR)
    PREPROCESSED_DIR = Path.cwd() / 'training/preprocessed_data'

    warnings.filterwarnings('ignore')

    from features.acis import generate_acis_deviation
    from features.cdec_deviation import generate_cdec_deviation
    from features.drought_deviation import generate_drought_deviation
    from features.glo_elevations import generate_elevations
    from features.feature_utils import clean_and_merge_features
    from features.monthly_naturalized_flow import generate_monthly_naturalized_flow
    from features.snotel_deviation import generate_snotel_deviation
    from features.streamflow_deviation import generate_streamflow_deviation
    from features.ua_swann_deviation import generate_ua_swann_deviation


    train_since = 1960
    main(DATA_DIR, PREPROCESSED_DIR, train_since)
