import numpy as np
import pandas as pd


def predict_yearly_model(df_features: pd.DataFrame,
                         feature_cols: dict,
                         model_yr_10,
                         model_yr_50,
                         model_yr_90,
                         model_type: str):
    preds_10_yearly = model_yr_10.predict(df_features[feature_cols['10']])
    preds_50_yearly = model_yr_50.predict(df_features[feature_cols['50']])
    preds_90_yearly = model_yr_90.predict(df_features[feature_cols['90']])

    preds_10_yearly = np.exp(preds_10_yearly)
    preds_50_yearly = np.exp(preds_50_yearly)
    preds_90_yearly = np.exp(preds_90_yearly)

    df_features[f'{model_type}_volume_10_yr'] = preds_10_yearly
    df_features[f'{model_type}_volume_50_yr'] = preds_50_yearly
    df_features[f'{model_type}_volume_90_yr'] = preds_90_yearly

    yr_cols = ['site_id', 'issue_date', f'{model_type}_volume_10_yr', f'{model_type}_volume_50_yr',
               f'{model_type}_volume_90_yr', 'month', 'volume']
    submission_yearly = df_features[yr_cols]

    return submission_yearly


def predict_monthly_model(val_features: pd.DataFrame,
                          feature_cols: dict,
                          model_mth_10,
                          model_mth_50,
                          model_mth_90,
                          no_monthly_data: list,
                          model_type: str):
    monthly_val = val_features[
        ['site_id', 'year', 'month', 'month_volume', 'season_start_month', 'season_end_month']]
    monthly_val = monthly_val[
        (monthly_val['month'] >= monthly_val['season_start_month']) &
        (monthly_val['month'] <= monthly_val['season_end_month'])]
    monthly_val = monthly_val.groupby(['site_id', 'year', 'month'])['month_volume'].mean().reset_index()
    monthly_val = monthly_val[~monthly_val['site_id'].isin(no_monthly_data)]
    monthly_val.columns = ['site_id', 'year', 'pred_month', 'month_volume_observed']

    monthly_val_features = pd.merge(val_features, monthly_val, on=['site_id', 'year'])

    preds_10_monthly = model_mth_10.predict(monthly_val_features[feature_cols['10']])
    preds_50_monthly = model_mth_50.predict(monthly_val_features[feature_cols['50']])
    preds_90_monthly = model_mth_90.predict(monthly_val_features[feature_cols['90']])

    preds_10_monthly = np.exp(preds_10_monthly)
    preds_50_monthly = np.exp(preds_50_monthly)
    preds_90_monthly = np.exp(preds_90_monthly)

    monthly_val_features[f'{model_type}_volume_10_mth'] = preds_10_monthly
    monthly_val_features[f'{model_type}_volume_50_mth'] = preds_50_monthly
    monthly_val_features[f'{model_type}_volume_90_mth'] = preds_90_monthly

    result_cols = ['site_id', 'issue_date', 'month', 'pred_month', f'{model_type}_volume_10_mth',
                   f'{model_type}_volume_50_mth',
                   f'{model_type}_volume_90_mth',
                   'month_volume_observed']
    test_results = monthly_val_features[result_cols]

    # For months occurring in the past use the monthly naturalized flow value if it isn't null
    def use_observed_monthly_flow(rw):
        return (rw['pred_month'] < rw['month']) and (not pd.isna(rw['month_volume_observed']))

    test_results[f'{model_type}_volume_10_mth'] = test_results.apply(
        lambda rw: rw['month_volume_observed'] if use_observed_monthly_flow(rw) else rw[f'{model_type}_volume_10_mth'],
        axis=1)
    test_results[f'{model_type}_volume_50_mth'] = test_results.apply(
        lambda rw: rw['month_volume_observed'] if use_observed_monthly_flow(rw) else rw[f'{model_type}_volume_50_mth'],
        axis=1)
    test_results[f'{model_type}_volume_90_mth'] = test_results.apply(
        lambda rw: rw['month_volume_observed'] if use_observed_monthly_flow(rw) else rw[f'{model_type}_volume_90_mth'],
        axis=1)
    test_results[
        ['site_id', 'issue_date', 'pred_month', 'month_volume_observed', f'{model_type}_volume_10_mth',
         f'{model_type}_volume_50_mth',
         f'{model_type}_volume_90_mth']
    ].groupby(['site_id', 'issue_date']).sum().reset_index()

    grouped_result = test_results[
        ['site_id', 'issue_date', 'pred_month', 'month_volume_observed', f'{model_type}_volume_10_mth',
         f'{model_type}_volume_50_mth',
         f'{model_type}_volume_90_mth']].groupby(
        ['site_id', 'issue_date']).sum().reset_index()

    mth_cols = ['site_id', 'issue_date', f'{model_type}_volume_10_mth', f'{model_type}_volume_50_mth',
                f'{model_type}_volume_90_mth']
    submission_monthly = grouped_result[mth_cols]

    return submission_monthly
