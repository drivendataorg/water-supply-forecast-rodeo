from pathlib import Path
from typing import Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

cat_mth_base_feature = ['site_id', 'pred_month', 'latitude', 'longitude', 'elevation', 'elevation_stds',
                        'day_of_year', 'streamflow_deviation_30_mean', 'snotel_tmax_deviation_10',
                        'precip_deviation_30', 'maxt_deviation_30', 'combined_swe_deviation_20',
                        'combined_swe_deviation_10', 'pdsi_deviation_5', 'acc_water_deviation',
                        'prev_month_volume']
cat_mth_feature_cols = {
    '10': cat_mth_base_feature,
    '50': cat_mth_base_feature,
    '90': cat_mth_base_feature
}

lgb_mth_base_feature = ['site_cat', 'pred_month', 'latitude', 'longitude', 'elevation', 'elevation_stds',
                        'day_of_year', 'precip_deviation_30', 'streamflow_deviation_30_mean', 'maxt_deviation_30',
                        'snotel_tmax_deviation_10', 'acc_water_deviation',
                        'combined_swe_deviation_10', 'snotel_prec_deviation_10', 'snotel_prec_deviation_20']

lgb_mth_feature_cols = {
    '10': lgb_mth_base_feature + ['precip_deviation_10', 'maxt_deviation_10', 'pdsi_deviation_2'],
    '50': lgb_mth_base_feature + ['prev_month_volume', 'combined_swe_deviation_20', 'pdsi_deviation_5'],
    '90': lgb_mth_base_feature + ['prev_month_volume', 'combined_swe_deviation_20', 'pdsi_deviation_2']
}


def train_monthly_catboost_model(
        train_features: pd.DataFrame,
        train_labels: pd.DataFrame,
        feature_cols: dict,
        preprocessed_dir: Path,
        cv_year: int
) -> Tuple[
    cb.CatBoostRegressor,
    cb.CatBoostRegressor,
    cb.CatBoostRegressor
]:
    logger.info("Beginning training of monthly catboost model.")

    cat_10 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.1',
                                  iterations=1400,
                                  depth=7,
                                  random_seed=42,
                                  verbose=False)
    cat_50 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.5',
                                  iterations=1200,
                                  depth=7,
                                  random_seed=42,
                                  verbose=False)
    cat_90 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.9',
                                  iterations=1400,
                                  depth=7,
                                  random_seed=42,
                                  verbose=False)

    cat_10_model = cat_10.fit(train_features[feature_cols['10']], train_labels, cat_features=[0, 1])
    cat_50_model = cat_50.fit(train_features[feature_cols['50']], train_labels, cat_features=[0, 1])
    cat_90_model = cat_90.fit(train_features[feature_cols['90']], train_labels, cat_features=[0, 1])

    cv_model_dir = preprocessed_dir / f'models/{cv_year}'
    cv_model_dir.mkdir(parents=True, exist_ok=True)

    cat_10_model.save_model(cv_model_dir / 'cat_10_monthly_model.txt')
    cat_50_model.save_model(cv_model_dir / 'cat_50_monthly_model.txt')
    cat_90_model.save_model(cv_model_dir / 'cat_90_monthly_model.txt')

    logger.info("Finished training monthly catboost model.")
    return cat_10_model, cat_50_model, cat_90_model


def train_monthly_lgb_model(
        train_features: pd.DataFrame,
        train_labels: pd.DataFrame,
        feature_cols: dict,
        preprocessed_dir: Path,
        cv_year: int
) -> Tuple[
    lgb.LGBMRegressor,
    lgb.LGBMRegressor,
    lgb.LGBMRegressor
]:
    logger.info("Beginning training of monthly lightgbm model.")

    SEED = 42
    model_10_params = {
        'objective': 'quantile',
        'metric': 'quantile',
        'alpha': 0.1,
        'max_bin': 120,
        'num_leaves': 8,
        'min_data_in_leaf': 15,
        'learning_rate': 0.035,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbosity': 0,
        'seed': SEED,
    }

    model_50_params = {
        'objective': 'quantile',
        'metric': 'quantile',
        'alpha': 0.5,
        'max_bin': 120,
        'num_leaves': 8,
        'min_data_in_leaf': 17,
        'learning_rate': 0.035,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbosity': 0,
        'seed': SEED,
    }

    model_90_params = {
        'objective': 'quantile',
        'metric': 'quantile',
        'alpha': 0.9,
        'max_bin': 150,
        'num_leaves': 10,
        'min_data_in_leaf': 16,
        'learning_rate': 0.040,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbosity': 0,
        'seed': SEED,
    }

    train_df_kwargs = {
        "label": train_labels,
        "categorical_feature": ["site_cat", "pred_month"],
        "free_raw_data": False,
    }
    train_df_kwargs_10 = train_df_kwargs.copy()
    train_df_kwargs_10["data"] = train_features[feature_cols['10']]
    train_df_kwargs_50 = train_df_kwargs.copy()
    train_df_kwargs_50["data"] = train_features[feature_cols['50']]
    train_df_kwargs_90 = train_df_kwargs.copy()
    train_df_kwargs_90["data"] = train_features[feature_cols['90']]

    training_10_kwargs = {
        "train_set": lgb.Dataset(**train_df_kwargs_10),
        "num_boost_round": 1200,
        "params": model_10_params
    }

    training_50_kwargs = {
        "train_set": lgb.Dataset(**train_df_kwargs_50),
        "num_boost_round": 1300,
        "params": model_50_params
    }

    training_90_kwargs = {
        "train_set": lgb.Dataset(**train_df_kwargs_90),
        "num_boost_round": 1200,
        "params": model_90_params
    }

    lgb_10_model = lgb.train(**training_10_kwargs)
    lgb_50_model = lgb.train(**training_50_kwargs)
    lgb_90_model = lgb.train(**training_90_kwargs)

    cv_model_dir = preprocessed_dir / f'models/{cv_year}'
    cv_model_dir.mkdir(parents=True, exist_ok=True)

    lgb_10_model.save_model(cv_model_dir / 'lgb_10_monthly_model.txt')
    lgb_50_model.save_model(cv_model_dir / 'lgb_50_monthly_model.txt')
    lgb_90_model.save_model(cv_model_dir / 'lgb_90_monthly_model.txt')

    logger.info("Finished training monthly lightgbm model.")
    return lgb_10_model, lgb_50_model, lgb_90_model


def generate_monthly_features(train_features, no_monthly_data):
    monthly_train_features = train_features

    monthly_labels = monthly_train_features[
        ['site_id', 'year', 'month', 'month_volume', 'season_start_month', 'season_end_month']]
    monthly_labels = monthly_labels[(monthly_labels['month'] >= monthly_labels['season_start_month']) &
                                    (monthly_labels['month'] <= monthly_labels['season_end_month'])]
    monthly_labels = monthly_labels.groupby(['site_id', 'year', 'month'])['month_volume'].mean().reset_index().dropna()
    monthly_labels.columns = ['site_id', 'year', 'pred_month', 'month_volume_label']

    monthly_train_features = pd.merge(monthly_train_features, monthly_labels, on=['site_id', 'year'])
    monthly_train_features = monthly_train_features[~monthly_train_features['site_id'].isin(no_monthly_data)]
    train_features = monthly_train_features[
        pd.to_datetime(monthly_train_features['issue_date']).dt.month <= monthly_train_features['pred_month']]

    # Take log of volume
    train_features['month_volume_log'] = np.log(train_features['month_volume_label'])

    label = 'month_volume_log'

    train_labels = train_features[[label]]

    return train_features, train_labels