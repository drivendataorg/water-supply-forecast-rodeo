from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def generate_monthly_naturalized_flow(data_dir: Path,
                                      preprocessed_dir: Path,
                                      df_features: pd.DataFrame,
                                      metadata: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating monthly naturalized flow features.")
    naturalized_columns = ['month_volume', 'prev_month_volume', 'monthly_volume_rolling',
                           'season_monthly_volume_rolling', 'prev_monthly_volume_rolling',
                           'prev_season_monthly_volume_rolling', 'nat_flow_deviation_season',
                           'nat_flow_deviation_recent']
    try:
        df_features = df_features.drop(naturalized_columns, axis=1)
    except:
        logger.info("Trying to drop naturalized monthly flow columns that don't exist.")
    df_monthly_naturalized_flow = pd.read_csv(data_dir / 'forecast_train_monthly_naturalized_flow.csv')
    df_monthly_naturalized_flow.columns = ['site_id', 'forecast_year', 'year', 'month', 'month_volume']
    site_list = []
    year_list = []
    month_list = []
    for year in df_features['year'].unique():
        for idx, rw in metadata.iterrows():
            for month in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]:
                site_list.append(rw['site_id'])
                year_list.append(year)
                month_list.append(month)

    df_idx = pd.DataFrame(
        {'site_id': site_list, 'forecast_year': year_list, 'month': month_list})
    df_monthly_naturalized_flow = pd.merge(df_idx,
                                           df_monthly_naturalized_flow,
                                           on=['site_id', 'forecast_year', 'month'],
                                           how='left')
    df_monthly_naturalized_flow['year'] = df_monthly_naturalized_flow.apply(
        lambda rw: rw['forecast_year'] if rw['month'] < 10 else rw['forecast_year'] - 1,
        axis=1).astype(int)
    df_monthly_naturalized_flow = df_monthly_naturalized_flow.sort_values(['site_id', 'year', 'month'])
    logger.info(f"test_monthly_naturalized_flow_shape: {df_monthly_naturalized_flow.shape}")
    monthly_group = df_monthly_naturalized_flow.groupby(['site_id', 'month'])[['month_volume']].agg(
        [np.mean, np.std]).reset_index()
    monthly_group.columns = ['site_id', 'month', 'month_volume_mean', 'month_volume_std']
    monthly_group.to_csv(preprocessed_dir / 'train_monthly_nat_group.csv', index=False)
    df_monthly_naturalized_flow = pd.merge(df_monthly_naturalized_flow, monthly_group, on=['site_id', 'month'])
    nat_monthly_flow = df_monthly_naturalized_flow['month_volume']
    nat_monthly_flow_mean = df_monthly_naturalized_flow['month_volume_mean']
    nat_monthly_flow_std = df_monthly_naturalized_flow['month_volume_std']
    month_nat_flow_deviation = (nat_monthly_flow - nat_monthly_flow_mean) / nat_monthly_flow_std
    df_monthly_naturalized_flow['month_nat_flow_deviation'] = month_nat_flow_deviation

    no_monthly_naturalized_flow_data = set(df_features.site_id.unique()).difference(
        set(df_monthly_naturalized_flow.site_id.unique()))

    def keep_season_volume(rw):
        if (rw['month'] >= rw['season_start_month']) and (rw['month'] <= rw['season_end_month']):
            return rw['month_volume']
        else:
            return 0

    def fill_null_month_volume(rw):
        if pd.notnull(rw['month_volume']):
            return rw['month_volume']
        else:
            values = monthly_group[(monthly_group['site_id'] == rw['site_id']) &
                                   (monthly_group['month'] == rw['month'])]['month_volume_mean'].values
            if values:
                return values[0]
            else:
                return np.nan

    df_idx_month = df_features[['site_id', 'year', 'month']].drop_duplicates()
    df_monthly_naturalized_flow['month_volume'] = df_monthly_naturalized_flow.apply(
        lambda rw: fill_null_month_volume(rw), axis=1)
    df_monthly_naturalized_flow['prev_month_volume'] = \
        df_monthly_naturalized_flow.groupby(['site_id', 'forecast_year']).shift()['month_volume']
    df_month = pd.merge(df_idx_month, df_monthly_naturalized_flow, on=['site_id', 'year', 'month'], how='left')
    df_month = pd.merge(df_month, metadata[['site_id', 'season_start_month', 'season_end_month']])
    df_month['season_month_volume'] = df_month.apply(lambda rw: keep_season_volume(rw), axis=1)
    # df_month['forecast_year'] = df_month.apply(lambda rw: rw['year'] if rw['month'] < 10 else rw['forecast_year'],
    #                                            axis=1).astype(int)
    df_month['prev_season_month_volume'] = df_month.groupby(['site_id', 'forecast_year']).shift()[
        'season_month_volume']

    # Get the rolling monthly naturalized flow within the forecast season
    df_monthly_rolling = df_month.groupby(['site_id', 'year']).rolling(window=5, min_periods=0).sum()[
        ['month_volume', 'prev_month_volume', 'season_month_volume',
         'prev_season_month_volume']].reset_index().set_index('level_2')
    df_monthly_rolling.columns = ['site_id', 'year', 'monthly_volume_rolling', 'prev_monthly_volume_rolling',
                                  'season_monthly_volume_rolling', 'prev_season_monthly_volume_rolling']
    df_monthly_rolling = df_monthly_rolling[
        ['monthly_volume_rolling', 'prev_monthly_volume_rolling', 'season_monthly_volume_rolling',
         'prev_season_monthly_volume_rolling']]

    monthly_cols = ['site_id', 'year', 'month', 'month_volume', 'prev_month_volume', 'monthly_volume_rolling',
                    'season_monthly_volume_rolling', 'prev_monthly_volume_rolling',
                    'prev_season_monthly_volume_rolling']
    df_rolling = df_month.join(df_monthly_rolling, how='left')
    df_rolling = df_rolling.drop(['year'], axis=1)
    df_rolling = df_rolling.rename(columns={'forecast_year': 'year'})
    df_features = pd.merge(df_features,
                           df_rolling[monthly_cols],
                           on=['site_id', 'year', 'month'],
                           how='left')

    def get_last_monthly_volume(rw):
        if (rw['month'] == rw['season_end_month']) and (rw['site_id'] not in no_monthly_naturalized_flow_data):
            rw['month_volume'] = rw['volume'] - rw['season_monthly_volume_rolling']

        return rw

    monthly_quantile_10 = df_features.groupby(['site_id', 'month'])['month_volume'].quantile(0.1).reset_index()
    monthly_quantile_10.to_csv(preprocessed_dir / 'monthly_quantile_10.csv', index=False)

    def adjust_negative_volume(rw):
        if rw['month_volume'] < 0:
            rw['month_volume'] = monthly_quantile_10[
                (monthly_quantile_10['site_id'] == rw['site_id']) & (monthly_quantile_10['month'] == rw['month'])][
                'month_volume'].values[0]

        return rw

    def nat_flow_deviation_for_site(rw):
        rw['nat_flow_deviation_season'] = df_monthly_naturalized_flow[
            (df_monthly_naturalized_flow['site_id'] == rw['site_id']) &
            (df_monthly_naturalized_flow['forecast_year'] == rw['year']) &
            ((df_monthly_naturalized_flow['month'] < rw['month']) |
             (df_monthly_naturalized_flow['month'] >= 10))
            ]['month_nat_flow_deviation'].mean()

        recent_month_map = {
            1: [11, 12],
            2: [12, 1],
            3: [1, 2],
            4: [2, 3],
            5: [3, 4],
            6: [4, 5],
            7: [5, 6]
        }

        rw['nat_flow_deviation_recent'] = df_monthly_naturalized_flow[
            (df_monthly_naturalized_flow['site_id'] == rw['site_id']) &
            (df_monthly_naturalized_flow['forecast_year'] == rw['year']) &
            (df_monthly_naturalized_flow['month'].isin(recent_month_map[rw['month']]))
            ]['month_nat_flow_deviation'].mean()
        return rw

    df_features = df_features.apply(lambda rw: get_last_monthly_volume(rw), axis=1)
    df_features = df_features.apply(lambda rw: adjust_negative_volume(rw), axis=1)
    df_features = df_features.apply(lambda rw: nat_flow_deviation_for_site(rw), axis=1)

    logger.info(df_features.groupby('site_id')[
                    ['month_volume', 'prev_month_volume', 'nat_flow_deviation_season', 'nat_flow_deviation_recent']
                ].mean())

    return df_features
