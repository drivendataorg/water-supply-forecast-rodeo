#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[0]))

from loguru import logger
from train_yearly_model import cat_yr_feature_cols, lgb_yr_feature_cols, train_yearly_catboost_model, \
    train_yearly_lgb_model
from train_monthly_model import cat_mth_feature_cols, lgb_mth_feature_cols, generate_monthly_features, \
    train_monthly_catboost_model, train_monthly_lgb_model
from predict_model import predict_yearly_model, predict_monthly_model


def main(preprocessed_dir: Path):
    logger.info('Beginning model training.')

    streamflow_features = pd.read_csv(preprocessed_dir / 'cv_features.csv')
    streamflow_features['cv_year'] = streamflow_features['cv_year'].astype(str)
    test_features = pd.read_csv(preprocessed_dir / 'cv_test_features.csv')

    min_year = 1960
    logger.info(f'Using training data as of {min_year}')
    streamflow_features = streamflow_features[streamflow_features['year'] >= min_year]

    # Take log of volume
    streamflow_features['volume_log'] = np.log(streamflow_features['volume'])

    cv_splits = list(range(2004, 2024))

    all_predictions = []

    for year in cv_splits:
        logger.info(f'Running cross validation for year {year}.')
        train_features = streamflow_features[(streamflow_features['cv_year'] == str(year)) &
                                             (streamflow_features['year'] != year)]
        val_features = test_features[(test_features['year'] == year)]
        print(f'{train_features.shape}, {val_features.shape}')

        no_monthly_data = ['american_river_folsom_lake', 'merced_river_yosemite_at_pohono_bridge']
        monthly_train_features, monthly_train_labels = generate_monthly_features(train_features, no_monthly_data)

        # Train Catboost models
        cat_mdl_yr_10, cat_mdl_yr_50, cat_mdl_yr_90 = train_yearly_catboost_model(
            train_features=train_features,
            feature_cols=cat_yr_feature_cols,
            preprocessed_dir=preprocessed_dir,
            cv_year=year)

        cat_mdl_mth_10, cat_mdl_mth_50, cat_mdl_mth_90 = train_monthly_catboost_model(
            train_features=monthly_train_features,
            train_labels=monthly_train_labels,
            feature_cols=cat_mth_feature_cols,
            preprocessed_dir=preprocessed_dir,
            cv_year=year)

        # Train LightGBM models
        lgb_mdl_yr_10, lgb_mdl_yr_50, lgb_mdl_yr_90 = train_yearly_lgb_model(
            train_features=train_features,
            feature_cols=lgb_yr_feature_cols,
            preprocessed_dir=preprocessed_dir,
            cv_year=year)

        lgb_mdl_mth_10, lgb_mdl_mth_50, lgb_mdl_mth_90 = train_monthly_lgb_model(
            train_features=monthly_train_features,
            train_labels=monthly_train_labels,
            feature_cols=lgb_mth_feature_cols,
            preprocessed_dir=preprocessed_dir,
            cv_year=year)

        # Predict models on held out test dataset
        cat_pred_yr = predict_yearly_model(val_features,
                                           cat_yr_feature_cols,
                                           cat_mdl_yr_10,
                                           cat_mdl_yr_50,
                                           cat_mdl_yr_90,
                                           'cat')

        score_streamflow(cat_pred_yr, 'cat_', '_yr')

        cat_pred_mth = predict_monthly_model(val_features,
                                             cat_mth_feature_cols,
                                             cat_mdl_mth_10,
                                             cat_mdl_mth_50,
                                             cat_mdl_mth_90,
                                             no_monthly_data,
                                             'cat')

        cat_mth_result = pd.merge(cat_pred_mth, val_features, on=['site_id', 'issue_date'], how='left')
        score_streamflow(cat_mth_result, 'cat_', '_mth')

        lgb_pred_yr = predict_yearly_model(val_features,
                                           lgb_yr_feature_cols,
                                           lgb_mdl_yr_10,
                                           lgb_mdl_yr_50,
                                           lgb_mdl_yr_90,
                                           'lgb')
        score_streamflow(lgb_pred_yr, 'lgb_', '_yr')

        lgb_pred_mth = predict_monthly_model(val_features,
                                             lgb_mth_feature_cols,
                                             lgb_mdl_mth_10,
                                             lgb_mdl_mth_50,
                                             lgb_mdl_mth_90,
                                             no_monthly_data,
                                             'lgb')
        mth_result = pd.merge(lgb_pred_mth, val_features, on=['site_id', 'issue_date'], how='left')
        score_streamflow(mth_result, 'lgb_', '_mth')

        # Ensemble the yearly and monthly models for Catboost and LightGBM
        submission_cat = pd.merge(cat_pred_yr, cat_pred_mth, on=['site_id', 'issue_date'], how='left')
        submission_lgb = pd.merge(lgb_pred_yr, lgb_pred_mth, on=['site_id', 'issue_date'], how='left')
        cat_result = submission_cat.apply(lambda rw: ensemble_monthly_yearly(rw, 'cat'), axis=1)
        lgb_result = submission_lgb.apply(lambda rw: ensemble_monthly_yearly(rw, 'lgb'), axis=1)

        # Ensemble the Catboost and LightGBM models
        cat_cols = ['site_id', 'issue_date', 'cat_volume_10', 'cat_volume_50', 'cat_volume_90', 'cat_volume_10_yr',
                    'cat_volume_50_yr', 'cat_volume_90_yr', 'cat_volume_10_mth', 'cat_volume_50_mth',
                    'cat_volume_90_mth']
        ensemble_result = pd.merge(cat_result[cat_cols], lgb_result, on=['site_id', 'issue_date'], how='inner')
        cat_10_pct = 0.4
        ensemble_result['volume_10'] = (1 - cat_10_pct) * lgb_result['lgb_volume_10'] + cat_10_pct * cat_result[
            'cat_volume_10']
        cat_50_pct = 0.4
        ensemble_result['volume_50'] = (1 - cat_50_pct) * lgb_result['lgb_volume_50'] + cat_50_pct * cat_result[
            'cat_volume_50']
        cat_90_pct = 0.7
        ensemble_result['volume_90'] = (1 - cat_90_pct) * lgb_result['lgb_volume_90'] + cat_90_pct * cat_result[
            'cat_volume_90']

        score_streamflow(ensemble_result)
        all_predictions.append(ensemble_result)

    concat_sub = pd.concat(all_predictions)
    score_streamflow(concat_sub)
    final_cols = ['site_id', 'issue_date', 'volume_10', 'volume_50', 'volume_90']
    concat_sub[final_cols].to_csv(preprocessed_dir / 'final_submission.csv', index=False)


def ensemble_monthly_yearly(rw, model_type):
    lower_mth_pct = {
        1: 0.4,
        2: 0.4,
        3: 0.4,
        4: 0.6,
        5: 0.7,
        6: 0.8,
        7: 1
    }

    median_mth_pct = {
        1: 0.4,
        2: 0.4,
        3: 0.4,
        4: 0.4,
        5: 0.4,
        6: 0.7,
        7: 1
    }

    upper_mth_pct = {
        1: 0.4,
        2: 0.5,
        3: 0.6,
        4: 0.6,
        5: 0.6,
        6: 0.8,
        7: 1
    }

    if pd.isna(rw[f'{model_type}_volume_10_mth']):
        rw[f'{model_type}_volume_10'] = rw[f'{model_type}_volume_10_yr']
    else:
        mth_pct_10 = lower_mth_pct[rw['month']]
        yr_pct_10 = 1 - mth_pct_10
        rw[f'{model_type}_volume_10'] = yr_pct_10 * rw[f'{model_type}_volume_10_yr'] + mth_pct_10 * rw[
            f'{model_type}_volume_10_mth']

    if pd.isna(rw[f'{model_type}_volume_50_mth']):
        rw[f'{model_type}_volume_50'] = rw[f'{model_type}_volume_50_yr']
    else:
        mth_pct_50 = median_mth_pct[rw['month']]
        yr_pct_50 = 1 - mth_pct_50
        rw[f'{model_type}_volume_50'] = yr_pct_50 * rw[f'{model_type}_volume_50_yr'] + mth_pct_50 * rw[
            f'{model_type}_volume_50_mth']

    if pd.isna(rw[f'{model_type}_volume_90_mth']):
        rw[f'{model_type}_volume_90'] = rw[f'{model_type}_volume_90_yr']
    else:
        mth_pct_90 = upper_mth_pct[rw['month']]
        yr_pct_90 = 1 - mth_pct_90
        rw[f'{model_type}_volume_90'] = yr_pct_90 * rw[f'{model_type}_volume_90_yr'] + mth_pct_90 * rw[
            f'{model_type}_volume_90_mth']

    return rw


def score_streamflow(cv_result, prefix='', suffix=''):
    print(prefix + suffix)
    pinball_result = pd.Series(
        {0.1: 2 * mean_pinball_loss(cv_result['volume'], cv_result[f'{prefix}volume_10{suffix}'], alpha=0.1),
         0.5: 2 * mean_pinball_loss(cv_result['volume'], cv_result[f'{prefix}volume_50{suffix}'], alpha=0.5),
         0.9: 2 * mean_pinball_loss(cv_result['volume'], cv_result[f'{prefix}volume_90{suffix}'], alpha=0.9)})
    print(pinball_result)


if __name__ == '__main__':
    from pathlib import Path

    import sys
    import warnings
    import numpy as np
    import pandas as pd

    from sklearn.metrics import mean_pinball_loss

    sys.path.append(str(Path(__file__).parent.resolve()))

    PREPROCESSED_DIR = Path.cwd() / 'training/preprocessed_data'

    warnings.filterwarnings('ignore')

    main(PREPROCESSED_DIR)
