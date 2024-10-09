from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def generate_monthly_naturalized_flow(src_dir: Path,
                                      data_dir: Path,
                                      test_features: pd.DataFrame,
                                      metadata: pd.DataFrame,
                                      issue_date: str) -> pd.DataFrame:
    logger.info("Generating monthly naturalized flow features.")
    test_monthly_naturalized_flow = pd.read_csv(data_dir / 'test_monthly_naturalized_flow.csv')
    logger.info(f"test_monthly_naturalized_flow_shape: {test_monthly_naturalized_flow.shape}")
    site_list = []
    year_list = []
    month_list = []
    for year in test_features['year'].unique():
        for idx, rw in metadata.iterrows():
            for month in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]:
                site_list.append(rw['site_id'])
                year_list.append(year)
                month_list.append(month)

    test_idx = pd.DataFrame(
        {'site_id': site_list, 'forecast_year': year_list, 'month': month_list})
    test_monthly_naturalized_flow = pd.merge(test_idx,
                                             test_monthly_naturalized_flow,
                                             on=['site_id', 'forecast_year', 'month'],
                                             how='left')
    logger.info(f"test_monthly_naturalized_flow_shape: {test_monthly_naturalized_flow.shape}")
    logger.info(test_monthly_naturalized_flow.columns)
    test_monthly_naturalized_flow = test_monthly_naturalized_flow.sort_values(['site_id', 'year', 'month'])
    logger.info(test_monthly_naturalized_flow.dropna(subset='volume').groupby(
        ['site_id'])[['month', 'volume']].last().round(3))


    test_monthly_naturalized_flow.columns = ['site_id', 'forecast_year', 'month', 'year', 'month_volume']
    monthly_group = pd.read_csv(src_dir / 'feature_parameters/train_monthly_nat_group.csv')
    test_monthly_naturalized_flow = pd.merge(test_monthly_naturalized_flow,
                                             monthly_group,
                                             on=['site_id', 'month'],
                                             how='left')
    nat_monthly_flow = test_monthly_naturalized_flow['month_volume']
    nat_monthly_flow_mean = test_monthly_naturalized_flow['month_volume_mean']
    nat_monthly_flow_std = test_monthly_naturalized_flow['month_volume_std']
    month_nat_flow_deviation = (nat_monthly_flow - nat_monthly_flow_mean) / nat_monthly_flow_std
    test_monthly_naturalized_flow['month_nat_flow_deviation'] = month_nat_flow_deviation

    # def keep_season_volume(rw):
    #     if (rw['month'] >= rw['season_start_month']) and (rw['month'] <= rw['season_end_month']):
    #         return rw['month_volume']
    #     else:
    #         return 0
    #
    # def fill_null_month_volume(rw):
    #     if pd.notnull(rw['month_volume']):
    #         return rw['month_volume']
    #     else:
    #         values = monthly_group[(monthly_group['site_id'] == rw['site_id']) &
    #                                (monthly_group['month'] == rw['month'])]['month_volume'].values
    #         if values:
    #             return values[0]
    #         else:
    #             return np.nan

    # test_idx_month = test_features[['site_id', 'year', 'month']].drop_duplicates()
    # test_monthly_naturalized_flow['month_volume'] = test_monthly_naturalized_flow.apply(
    #     lambda rw: fill_null_month_volume(rw), axis=1)
    test_monthly_naturalized_flow = test_monthly_naturalized_flow.sort_values(['site_id', 'year', 'month'])
    test_monthly_naturalized_flow['prev_month_volume'] = \
        test_monthly_naturalized_flow.groupby(['site_id', 'forecast_year']).shift()['month_volume']
    # test_month = pd.merge(test_idx_month, test_monthly_naturalized_flow, on=['site_id', 'year', 'month'], how='left')
    # test_month = pd.merge(test_month, metadata[['site_id', 'season_start_month', 'season_end_month']])
    # test_month['season_month_volume'] = test_month.apply(lambda rw: keep_season_volume(rw), axis=1)
    # test_month['forecast_year'] = test_month.apply(lambda rw: rw['year'] if rw['month'] < 10 else rw['forecast_year'],
    #                                                axis=1).astype(int)
    # test_month['prev_season_month_volume'] = test_month.groupby(['site_id', 'forecast_year']).shift()[
    #     'season_month_volume']
    #
    # # Get the rolling monthly naturalized flow within the forecast season
    # test_monthly_rolling = test_month.groupby(['site_id', 'year']).rolling(window=5, min_periods=0).sum()[
    #     ['month_volume', 'prev_month_volume', 'season_month_volume',
    #      'prev_season_month_volume']].reset_index().set_index('level_2')
    # test_monthly_rolling.columns = ['site_id', 'year', 'monthly_volume_rolling', 'prev_monthly_volume_rolling',
    #                                 'season_monthly_volume_rolling', 'prev_season_monthly_volume_rolling']
    # test_monthly_rolling = test_monthly_rolling[
    #     ['monthly_volume_rolling', 'prev_monthly_volume_rolling', 'season_monthly_volume_rolling',
    #      'prev_season_monthly_volume_rolling']]
    #
    # monthly_cols = ['site_id', 'year', 'month', 'month_volume', 'prev_month_volume', 'monthly_volume_rolling',
    #                 'prev_monthly_volume_rolling', 'prev_season_monthly_volume_rolling']
    # test_rolling = test_month.join(test_monthly_rolling, how='left')
    # test_features = pd.merge(test_features,
    #                          test_rolling[monthly_cols],
    #                          on=['site_id', 'year', 'month'],
    #                          how='left')
    monthly_cols = ['site_id', 'year', 'month', 'month_volume', 'prev_month_volume']
    test_monthly_naturalized_flow = test_monthly_naturalized_flow.drop(['year'], axis=1)
    test_monthly_naturalized_flow = test_monthly_naturalized_flow.rename(columns={'forecast_year': 'year'})
    test_features = pd.merge(test_features,
                             test_monthly_naturalized_flow[monthly_cols],
                             on=['site_id', 'year', 'month'],
                             how='left')

    def nat_flow_deviation_for_site(rw):
        recent_month_map = {
            1: [11, 12],
            2: [12, 1],
            3: [1, 2],
            4: [2, 3],
            5: [3, 4],
            6: [4, 5],
            7: [5, 6]
        }

        rw['nat_flow_deviation_recent'] = test_monthly_naturalized_flow[
            (test_monthly_naturalized_flow['site_id'] == rw['site_id']) &
            (test_monthly_naturalized_flow['year'] == rw['year']) &
            (test_monthly_naturalized_flow['month'].isin(recent_month_map[rw['month']]))
            ]['month_nat_flow_deviation'].mean()
        return rw

    test_features = test_features.apply(lambda rw: nat_flow_deviation_for_site(rw), axis=1)

    logger.info(test_features[test_features['issue_date'] == issue_date][
                    ['site_id', 'prev_month_volume']
                ].round(3))

    return test_features
