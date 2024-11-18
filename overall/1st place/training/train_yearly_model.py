from pathlib import Path
from typing import Tuple

import catboost as cb
import lightgbm as lgb
import pandas as pd
from loguru import logger

cat_yr_base_feature = ['site_id', 'latitude', 'longitude', 'elevation', 'elevation_stds',
                       'day_of_year', 'streamflow_deviation_30_mean', 'streamflow_deviation_season_mean',
                       'precip_deviation_180', 'combined_swe_deviation_180',
                       'prev_month_volume', 'combined_swe_deviation_20', 'combined_swe_deviation_10',
                       'maxt_deviation_30', 'maxt_deviation_180', 'precip_deviation_30',
                       'snotel_prec_deviation_20', 'pdsi_deviation_5', 'snotel_tmax_deviation_10']
cat_yr_feature_cols = {
    '10': cat_yr_base_feature,
    '50': cat_yr_base_feature + ['acc_water_deviation'],
    '90': cat_yr_base_feature
}

lgb_yr_base_feature = ['site_cat', 'latitude', 'longitude', 'elevation', 'elevation_stds',
                       'day_of_year', 'streamflow_deviation_30_mean', 'streamflow_deviation_season_mean',
                       'combined_swe_deviation_180',
                       'prev_month_volume', 'combined_swe_deviation_10',
                       'maxt_deviation_30', 'precip_deviation_10', 'precip_deviation_30', 'precip_deviation_180',
                       'snotel_prec_deviation_20', 'snotel_tmax_deviation_10']

lgb_yr_feature_cols = {
    '10': lgb_yr_base_feature + ['snotel_prec_deviation_180', 'pdsi_deviation_5', ],
    '50': lgb_yr_base_feature + ['combined_swe_deviation_20', 'pdsi_deviation_2', ],
    '90': lgb_yr_base_feature + ['pdsi_deviation_5', 'acc_water_deviation',]
}


def train_yearly_catboost_model(
        train_features: pd.DataFrame,
        feature_cols: dict,
        preprocessed_dir: Path,
        cv_year: int
) -> Tuple[
    cb.CatBoostRegressor,
    cb.CatBoostRegressor,
    cb.CatBoostRegressor
]:
    logger.info("Beginning training of yearly catboost model.")

    train_features = train_features.dropna(subset='volume')
    label = 'volume_log'

    train_labels = train_features[[label]]

    cat_10 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.1',
                                  iterations=1300,
                                  depth=6,
                                  random_seed=42,
                                  verbose=False)
    cat_50 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.5',
                                  iterations=1300,
                                  depth=8,
                                  random_seed=42,
                                  verbose=False)
    cat_90 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.9',
                                  iterations=1000,
                                  depth=5,
                                  random_seed=42,
                                  verbose=False)

    cat_10_model = cat_10.fit(train_features[feature_cols['10']], train_labels, cat_features=[0])
    cat_50_model = cat_50.fit(train_features[feature_cols['50']], train_labels, cat_features=[0])
    cat_90_model = cat_90.fit(train_features[feature_cols['90']], train_labels, cat_features=[0])

    cv_model_dir = preprocessed_dir / f'models/{cv_year}'
    cv_model_dir.mkdir(parents=True, exist_ok=True)

    cat_10_model.save_model(cv_model_dir / 'cat_10_yearly_model.txt')
    cat_50_model.save_model(cv_model_dir / 'cat_50_yearly_model.txt')
    cat_90_model.save_model(cv_model_dir / 'cat_90_yearly_model.txt')

    logger.info("Finished training yearly catboost model.")
    return cat_10_model, cat_50_model, cat_90_model


def train_yearly_lgb_model(
        train_features: pd.DataFrame,
        feature_cols: dict,
        preprocessed_dir: Path,
        cv_year:int
) -> Tuple[
    lgb.LGBMRegressor,
    lgb.LGBMRegressor,
    lgb.LGBMRegressor
]:
    logger.info("Beginning training of yearly lightgbm model.")

    train_features = train_features.dropna(subset='volume')

    SEED = 42
    model_10_params = {
        'objective': 'quantile',
        'metric': 'quantile',
        'alpha': 0.1,
        'max_bin': 120,
        'num_leaves': 7,
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
        'num_leaves': 9,
        'min_data_in_leaf': 15,
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
        'max_bin': 140,
        'num_leaves': 7,
        'min_data_in_leaf': 15,
        'learning_rate': 0.025,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbosity': 0,
        'seed': SEED,
    }

    label = 'volume_log'
    cv_labels = train_features[[label]]
    train_df_kwargs = {
        "label": cv_labels,
        "categorical_feature": ["site_cat"],
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
        "params": model_10_params,
    }

    training_50_kwargs = {
        "train_set":lgb.Dataset(**train_df_kwargs_50),
        "num_boost_round": 1100,
        "params": model_50_params
    }

    training_90_kwargs = {
        "train_set": lgb.Dataset(**train_df_kwargs_90),
        "num_boost_round": 1000,
        "params": model_90_params
    }

    lgb_10_model = lgb.train(**training_10_kwargs)
    lgb_50_model = lgb.train(**training_50_kwargs)
    lgb_90_model = lgb.train(**training_90_kwargs)

    cv_model_dir = preprocessed_dir / f'models/{cv_year}'
    cv_model_dir.mkdir(parents=True, exist_ok=True)

    lgb_10_model.save_model(cv_model_dir / 'lgb_10_yearly_model.txt')
    lgb_50_model.save_model(cv_model_dir / 'lgb_50_yearly_model.txt')
    lgb_90_model.save_model(cv_model_dir / 'lgb_90_yearly_model.txt')

    logger.info("Finished training yearly lightgbm model.")
    return lgb_10_model, lgb_50_model, lgb_90_model
