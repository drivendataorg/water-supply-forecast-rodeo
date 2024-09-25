from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
from loguru import logger


def generate_predictions(src_dir: Path,
                         preprocessed_dir: Path,
                         test_features: pd.DataFrame,
                         issue_date: str) -> pd.DataFrame:
    logger.info("Generating yearly predictions.")

    yearly_feature_cols = ['site_id', 'latitude', 'longitude', 'elevation', 'elevation_stds',
                           'day_of_year', 'streamflow_deviation_30_mean', 'streamflow_deviation_season_mean',
                           'precip_deviation_season_mean', 'combined_swe_deviation_0', 'ua_swe_deviation',
                           'acc_water_deviation']

    cat_10_yearly_model = cb.CatBoostRegressor()
    cat_10_yearly_model.load_model(src_dir / 'models/cat_10_yearly_model.txt')

    cat_50_yearly_model = cb.CatBoostRegressor()
    cat_50_yearly_model.load_model(src_dir / 'models/cat_50_yearly_model.txt')

    cat_90_yearly_model = cb.CatBoostRegressor()
    cat_90_yearly_model.load_model(src_dir / 'models/cat_90_yearly_model.txt')

    preds_10_yearly = cat_10_yearly_model.predict(test_features[yearly_feature_cols])
    preds_50_yearly = cat_50_yearly_model.predict(test_features[yearly_feature_cols])
    preds_90_yearly = cat_90_yearly_model.predict(test_features[yearly_feature_cols])

    preds_10_yearly = np.exp(preds_10_yearly)
    preds_50_yearly = np.exp(preds_50_yearly)
    preds_90_yearly = np.exp(preds_90_yearly)

    test_features['volume_10_yr'] = preds_10_yearly
    test_features['volume_50_yr'] = preds_50_yearly
    test_features['volume_90_yr'] = preds_90_yearly

    submission_yearly = test_features[['site_id', 'issue_date', 'volume_10_yr', 'volume_50_yr', 'volume_90_yr']]

    logger.info("Generating monthly predictions.")

    monthly_test = test_features[
        ['site_id', 'year', 'month', 'month_volume', 'season_start_month', 'season_end_month']]
    monthly_test = monthly_test[
        (monthly_test['month'] >= monthly_test['season_start_month']) &
        (monthly_test['month'] <= monthly_test['season_end_month'])]
    monthly_test = monthly_test.groupby(['site_id', 'year', 'month'])['month_volume'].mean().reset_index()
    no_monthly_data = ['american_river_folsom_lake', 'merced_river_yosemite_at_pohono_bridge',
                       'san_joaquin_river_millerton_reservoir']
    monthly_test = monthly_test[~monthly_test['site_id'].isin(no_monthly_data)]
    monthly_test.columns = ['site_id', 'year', 'pred_month', 'month_volume_observed']

    logger.info(f"monthly_test_shape: {monthly_test.shape}")

    monthly_test_features = pd.merge(test_features, monthly_test, on=['site_id', 'year'])
    # test_features = pd.merge(test_features, monthly_test, on=['site_id', 'year'])
    # monthly_test_features = test_features[
    #     pd.to_datetime(test_features['issue_date']).dt.month <= test_features['pred_month']]

    monthly_feature_cols = ['site_id', 'pred_month', 'latitude', 'longitude', 'elevation', 'elevation_stds',
                            'day_of_year', 'streamflow_deviation_30_mean',
                            'precip_deviation_30_mean', 'maxt_deviation_30_mean', 'combined_swe_deviation_30',
                            'ua_swe_deviation', 'acc_water_deviation']

    cat_10_monthly_model = cb.CatBoostRegressor()
    cat_10_monthly_model.load_model(src_dir / 'models/cat_10_monthly_model.txt')

    cat_50_monthly_model = cb.CatBoostRegressor()
    cat_50_monthly_model.load_model(src_dir / 'models/cat_50_monthly_model.txt')

    cat_90_monthly_model = cb.CatBoostRegressor()
    cat_90_monthly_model.load_model(src_dir / 'models/cat_90_monthly_model.txt')

    preds_10_monthly = cat_10_monthly_model.predict(monthly_test_features[monthly_feature_cols])
    preds_50_monthly = cat_50_monthly_model.predict(monthly_test_features[monthly_feature_cols])
    preds_90_monthly = cat_90_monthly_model.predict(monthly_test_features[monthly_feature_cols])

    preds_10_monthly = np.exp(preds_10_monthly)
    preds_50_monthly = np.exp(preds_50_monthly)
    preds_90_monthly = np.exp(preds_90_monthly)

    monthly_test_features['volume_10_mth'] = preds_10_monthly
    monthly_test_features['volume_50_mth'] = preds_50_monthly
    monthly_test_features['volume_90_mth'] = preds_90_monthly

    result_cols = ['site_id', 'issue_date', 'month', 'pred_month', 'volume_10_mth', 'volume_50_mth', 'volume_90_mth',
                   'month_volume_observed']
    test_results = monthly_test_features[result_cols]

    logger.info(f"test_results_shape: {test_results.shape}")
    logger.info(test_results.columns)

    # For months occurring in the past use the monthly naturalized flow value if it isn't null
    def use_observed_monthly_flow(rw):
        return (rw['pred_month'] < rw['month']) and (not pd.isna(rw['month_volume_observed']))

    test_rw = pd.Series({'pred_month': 6, 'month': 7, 'month_volume_observed': np.nan})

    print("test ", use_observed_monthly_flow(test_rw))

    test_results['volume_10_mth'] = test_results.apply(
        lambda rw: rw['month_volume_observed'] if use_observed_monthly_flow(rw) else rw['volume_10_mth'], axis=1)
    test_results['volume_50_mth'] = test_results.apply(
        lambda rw: rw['month_volume_observed'] if use_observed_monthly_flow(rw) else rw['volume_50_mth'], axis=1)
    test_results['volume_90_mth'] = test_results.apply(
        lambda rw: rw['month_volume_observed'] if use_observed_monthly_flow(rw) else rw['volume_90_mth'], axis=1)
    test_results[
        ['site_id', 'issue_date', 'pred_month', 'month_volume_observed', 'volume_10_mth', 'volume_50_mth',
         'volume_90_mth']
    ].groupby(['site_id', 'issue_date']).sum().reset_index()

    grouped_result = test_results[
        ['site_id', 'issue_date', 'pred_month', 'month_volume_observed', 'volume_10_mth', 'volume_50_mth',
         'volume_90_mth']].groupby(
        ['site_id', 'issue_date']).sum().reset_index()

    mth_cols = ['site_id', 'issue_date', 'volume_10_mth', 'volume_50_mth', 'volume_90_mth']
    submission = pd.read_csv(preprocessed_dir / 'submission.csv')
    submission_monthly = pd.merge(submission[['site_id', 'issue_date', 'month']], grouped_result[mth_cols],
                                  on=['site_id', 'issue_date'])

    submission_final = pd.merge(submission_yearly, submission_monthly, on=['site_id', 'issue_date'], how='left')

    mth_pct_10 = 0.35
    yr_pct_10 = 1.0 - mth_pct_10
    mth_pct_50 = 0.45
    yr_pct_50 = 1.0 - mth_pct_50
    mth_pct_90 = 0.55
    yr_pct_90 = 1.0 - mth_pct_90

    def ensemble_monthly_yearly(rw):
        if pd.isna(rw['volume_10_mth']):
            rw['volume_10'] = rw['volume_10_yr']
        elif rw['month'] in [1, 2, 3, 4, 5, 6]:
            rw['volume_10'] = yr_pct_10 * rw['volume_10_yr'] + mth_pct_10 * rw['volume_10_mth']
        else:
            rw['volume_10'] = rw['volume_10_mth']

        if pd.isna(rw['volume_50_mth']):
            rw['volume_50'] = rw['volume_50_yr']
        elif rw['month'] in [1, 2, 3, 4, 5, 6]:
            rw['volume_50'] = yr_pct_50 * rw['volume_50_yr'] + mth_pct_50 * rw['volume_50_mth']
        else:
            rw['volume_50'] = rw['volume_50_mth']

        if pd.isna(rw['volume_90_mth']):
            rw['volume_90'] = rw['volume_90_yr']
        elif rw['month'] in [1, 2, 3, 4, 5, 6]:
            rw['volume_90'] = yr_pct_90 * rw['volume_90_yr'] + mth_pct_90 * rw['volume_90_mth']
        else:
            rw['volume_90'] = rw['volume_90_mth']

        return rw

    submission_final = submission_final.apply(lambda rw: ensemble_monthly_yearly(rw), axis=1)
    submission_final = submission_final[submission_final['issue_date'] <= issue_date]

    logger.info("Yearly model")
    submission_final = submission_final.rename(
        columns={"volume_10_yr": "yr_10", "volume_50_yr": "yr_50", "volume_90_yr": "yr_90",
                 "volume_10_mth": "mth_10", "volume_50_mth": "mth_50", "volume_90_mth": "mth_90"}
    )
    logger.info(submission_final[
                    submission_final['issue_date'] == issue_date
                    ][['site_id', 'yr_10', 'yr_50', 'yr_90']].set_index('site_id').sort_values(
        'site_id').round(decimals=1))
    logger.info("Monthly model")
    logger.info(submission_final[
                    submission_final['issue_date'] == issue_date
                    ][['site_id', 'mth_10', 'mth_50', 'mth_90']].set_index(
        'site_id').sort_values('site_id').round(decimals=1))
    output_cols = ['site_id', 'issue_date', 'volume_10', 'volume_50', 'volume_90']
    submission_final = submission_final[output_cols]
    logger.info("Combined model")
    logger.info(submission_final[
                    submission_final['issue_date'] == issue_date
                    ][['site_id', 'volume_10', 'volume_50', 'volume_90']].set_index('site_id').sort_values(
        'site_id').round(decimals=1))

    submission_final.to_csv(preprocessed_dir / 'submission_final.csv', index=False)

    return submission_final
