from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def generate_monthly_naturalized_flow(data_dir: Path,
                                      preprocessed_dir: Path,
                                      df_features: pd.DataFrame,
                                      metadata: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating monthly naturalized flow features.")
    df_monthly_naturalized_flow = pd.read_csv(data_dir / 'train_monthly_naturalized_flow.csv')
    df_monthly_naturalized_flow.columns = ['site_id', 'forecast_year', 'year', 'month', 'month_volume']

    no_monthly_naturalized_flow_data = set(df_features.site_id.unique()).difference(
        set(df_monthly_naturalized_flow.site_id.unique()))

    def keep_season_volume(rw):
        if (rw['month'] >= rw['season_start_month']) and (rw['month'] <= rw['season_end_month']):
            return rw['month_volume']
        else:
            return 0

    monthly_group = df_monthly_naturalized_flow.groupby(['site_id', 'month'], as_index=False).mean()[
        ['site_id', 'month', 'month_volume']]
    monthly_group.to_csv(preprocessed_dir / 'train_monthly_nat_group.csv', index=False)

    def fill_null_month_volume(rw):
        if pd.notnull(rw['month_volume']):
            return rw['month_volume']
        else:
            values = monthly_group[(monthly_group['site_id'] == rw['site_id']) &
                                   (monthly_group['month'] == rw['month'])]['month_volume'].values
            if values:
                return values[0]
            else:
                return np.nan

    df_idx_month = df_features[['site_id', 'year', 'month']].drop_duplicates()
    df_month = pd.merge(df_idx_month, df_monthly_naturalized_flow, on=['site_id', 'year', 'month'], how='left')
    df_month['month_volume'] = df_month.apply(lambda rw: fill_null_month_volume(rw), axis=1)
    df_month = pd.merge(df_month, metadata[['site_id', 'season_start_month', 'season_end_month']])
    df_month['season_month_volume'] = df_month.apply(lambda rw: keep_season_volume(rw), axis=1)
    df_month['forecast_year'] = df_month.apply(lambda rw: rw['year'] if rw['month'] < 10 else rw['forecast_year'],
                                                   axis=1).astype(int)
    df_month['prev_month_volume'] = df_month.groupby(['site_id', 'forecast_year']).shift()['month_volume']
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

    df_features = df_features.apply(lambda rw: get_last_monthly_volume(rw), axis=1)
    df_features = df_features.apply(lambda rw: adjust_negative_volume(rw), axis=1)

    logger.info(df_features[['month_volume', 'prev_month_volume']].describe())

    return df_features
