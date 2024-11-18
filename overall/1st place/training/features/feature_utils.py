from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from loguru import logger

SITE_BASINS = {
    'libby_reservoir_inflow': '17010101',
    'american_river_folsom_lake': '18020111',
    'ruedi_reservoir_inflow': '14010004',
    'skagit_ross_reservoir': '17110005',
    'snake_r_nr_heise': '17040104',
    'missouri_r_at_toston': '10030101',
    'san_joaquin_river_millerton_reservoir': '18040001',
    'hungry_horse_reservoir_inflow': '17010209',
    'boysen_reservoir_inflow': '10080005',
    'boise_r_nr_boise': '17050114',
    'fontenelle_reservoir_inflow': '14040101',
    'yampa_r_nr_maybell': '14050002',
    'owyhee_r_bl_owyhee_dam': '17050110',
    'detroit_lake_inflow': '17090005',
    'merced_river_yosemite_at_pohono_bridge': '18040008',
    'stehekin_r_at_stehekin': '17020009',
    'pueblo_reservoir_inflow': '11020002',
    'green_r_bl_howard_a_hanson_dam': '17110013',
    'animas_r_at_durango': '14080104',
    'colville_r_at_kettle_falls': '17020003',
    'dillon_reservoir_inflow': '14010002',
    'sweetwater_r_nr_alcova': '10180006',
    'weber_r_nr_oakley': '16020101',
    'taylor_park_reservoir_inflow': '14020001',
    'virgin_r_at_virtin': '15010008',
    'pecos_r_nr_pecos': '13060001',
}


def compute_deviation(data_dir: Path,
                      preprocessed_dir: Path,
                      cv_year: int,
                      input_col: str,
                      output_col: str,
                      pivot_col: str,
                      base_data: pd.DataFrame,
                      df_features: pd.DataFrame,
                      applicable_sites: List,
                      rolling_windows: List,
                      agg_months: List,
                      min_num_locations: int,
                      thresh: int,
                      test_features: Optional[pd.DataFrame] = None,
                      sort_ascending: Optional[bool] = False,
                      ffill: Optional[bool] = False) -> pd.DataFrame:
    """
    For the given input column find the most correlated location measurements from the pivot column with each site using
    only the holdout train years, include a minimum number of locations. Compute the z-score for each measurement.
    Create rolling window features corresponding to the rolling_windows.
    """
    base_data['date'] = pd.to_datetime(base_data['date'])
    prior_train = pd.read_csv(data_dir / 'prior_historical_labels.csv')
    cv_train = pd.read_csv(data_dir / 'cross_validation_labels.csv')
    train = pd.concat([prior_train, cv_train])
    train = train.rename(columns={'year': 'forecast_year'})

    # Holdout feature data from the forecast year when computing mean and std for train features to avoid data leakage
    train_data = base_data[base_data['forecast_year'] != cv_year].copy(deep=True)
    # Always hold out current year label
    train = train[train['forecast_year'] != cv_year]

    # test_data needs current year feature data
    all_test_feature = base_data.sort_values([pivot_col, 'date']).copy(deep=True)
    sub_test_feature = all_test_feature[all_test_feature['month'].isin(agg_months)]
    sub_test_feature = sub_test_feature[sub_test_feature['forecast_year'] == cv_year]

    all_test_dfs = []

    train.set_index('forecast_year', inplace=True)

    all_feature = train_data.sort_values([pivot_col, 'date'])
    sub_feature = all_feature[all_feature['month'].isin(agg_months)]

    feature_grouped = sub_feature.groupby(['month_day', pivot_col])[input_col].agg(
        [np.mean, np.std]).reset_index()
    feature_grouped.columns = ['month_day', pivot_col, f'{input_col}_mean', f'{input_col}_std']
    logger.info(feature_grouped[[f'{input_col}_mean', f'{input_col}_std']].describe())

    feature_columns = [f'{output_col}_{window}' for window in rolling_windows]

    all_dfs = []
    for site_id in applicable_sites:
        feature_year = sub_feature.groupby([pivot_col, 'forecast_year'])[input_col].mean().reset_index()
        input_col_pivot = feature_year.pivot_table(index='forecast_year', columns=pivot_col, values=input_col)
        input_col_pivot = input_col_pivot[input_col_pivot.index >= 1990]
        input_col_pivot.dropna(thresh=thresh, axis=1, inplace=True)
        site_input_col_pivot = train[train['site_id'] == site_id].join(input_col_pivot, how='inner')
        site_input_col_pivot.drop(columns=['site_id'], inplace=True)
        site_feature_corr = site_input_col_pivot.corr()['volume'].sort_values(ascending=sort_ascending)
        site_feature_corr.drop('volume', inplace=True)
        site_feature_high_corr = site_feature_corr.iloc[:min_num_locations]
        feature_corr_dir = preprocessed_dir / f'feature_corrs/{cv_year}/{site_id}'
        feature_corr_dir.mkdir(parents=True, exist_ok=True)
        feature_corr_df = site_feature_high_corr.to_frame(name='volume_corr').reset_index(names=pivot_col)
        feature_corr_df.to_csv(feature_corr_dir / f'{output_col}_corr.csv', index=False)
        relevant_locations = pd.DataFrame(site_feature_high_corr.index, columns=[pivot_col])

        # Create train features using the measurements from the most correlated locations with training streamflow
        site_feature = site_deviation_features(sub_feature,
                                               relevant_locations,
                                               feature_grouped,
                                               input_col,
                                               output_col,
                                               pivot_col,
                                               rolling_windows,
                                               site_id)

        # Create test features using the same list of locations that were used for the train features
        site_test_feature = site_deviation_features(sub_test_feature,
                                                    relevant_locations,
                                                    feature_grouped,
                                                    input_col,
                                                    output_col,
                                                    pivot_col,
                                                    rolling_windows,
                                                    site_id)
        all_test_dfs.append(site_test_feature)
        all_dfs.append(site_feature)

    df_features = integrate_features(df_features,
                                     all_dfs,
                                     feature_columns,
                                     ffill)

    test_features = integrate_features(test_features,
                                       all_test_dfs,
                                       feature_columns,
                                       ffill)

    return df_features, test_features


def site_deviation_features(sub_feature: pd.DataFrame,
                            relevant_locations: pd.DataFrame,
                            feature_grouped: pd.DataFrame,
                            input_col: str,
                            output_col: str,
                            pivot_col: str,
                            rolling_windows: List,
                            site_id: str) -> pd.DataFrame:
    """
    Create z-score deviation features using the measurements from locations most correlated with training streamflow.
    """
    feature = pd.merge(sub_feature, relevant_locations, on=pivot_col)
    feature = pd.merge(feature, feature_grouped, on=['month_day', pivot_col])
    feature[output_col] = (feature[input_col] - feature[f'{input_col}_mean']) / feature[f'{input_col}_std']
    feature[output_col] = feature[output_col].replace(np.inf, np.nan).replace(-np.inf, np.nan)
    site_feature = feature.groupby(['date', 'forecast_year'])[output_col].mean().reset_index()
    site_feature['site_id'] = site_id
    site_feature = site_feature.sort_values(['site_id', 'date'])
    for window in rolling_windows:
        feature_window = site_rolling_features(site_feature, output_col, window)
        site_feature = site_feature.join(feature_window)

    return site_feature


def site_rolling_features(site_df, col_name, window):
    site_window = site_df.groupby(
        ['forecast_year', 'site_id']
    )[col_name].rolling(window=window, min_periods=1).agg(
        [np.mean]).reset_index(level=[0, 1], drop=True)
    site_window.columns = [f'{col_name}_{window}']
    return site_window


def integrate_features(df_features: pd.DataFrame,
                       all_dfs: List[pd.DataFrame],
                       feature_columns: List,
                       ffill: bool) -> pd.DataFrame:
    all_feature_deviation = pd.concat(all_dfs)

    # Set issue_date for features to be one day ahead, so we only join with past data avoiding data leakage
    all_feature_deviation['issue_date'] = (all_feature_deviation['date'] + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')

    if ffill:
        # Forward fill (from the past) missing observations for non-daily data
        all_feature_deviation = pd.merge(df_features[['site_id', 'issue_date']],
                                         all_feature_deviation[['site_id', 'issue_date'] + feature_columns],
                                         on=['site_id', 'issue_date'],
                                         how='outer')
        all_feature_deviation['forecast_year'] = pd.to_datetime(all_feature_deviation['issue_date']).dt.year
        all_feature_deviation = all_feature_deviation.sort_values(['site_id', 'issue_date'])
        for col in feature_columns:
            all_feature_deviation[col] = all_feature_deviation.groupby(['site_id', 'forecast_year']).ffill()[col]

    df_features = clean_and_merge_features(df_features, all_feature_deviation, feature_columns,
                                           ['site_id', 'issue_date'])

    return df_features


def clean_and_merge_features(main_df, features, feature_cols, join_cols):
    feature_subset = features[join_cols + feature_cols]
    for col in feature_cols:
        try:
            main_df = main_df.drop(col, axis=1)
        except:
            logger.info(f"Trying to drop {col} column that don't exist.")

    main_df = pd.merge(main_df, feature_subset, on=join_cols, how='left')

    logger.info(main_df[feature_cols].describe())

    return main_df
