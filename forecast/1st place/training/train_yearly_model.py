from pathlib import Path

import catboost as cb
from loguru import logger
import numpy as np
import pandas as pd


def train_yearly_model(model_dir: Path,
                       train_features: pd.DataFrame,
                       null_monthly_nat_flow: bool) -> None:
    logger.info("Beginning training of yearly catboost model.")

    train_features = train_features.dropna(subset='volume')

    def apply_null_prev_month_nat_flow(rw):
        if pd.to_datetime(rw['issue_date']).day == 1:
            rw['prev_month_volume'] = np.nan

        return rw

    if null_monthly_nat_flow:
        train_features = train_features.apply(lambda rw: apply_null_prev_month_nat_flow(rw), axis=1)

    # Take log of volume
    train_features['volume_log'] = np.log(train_features['volume'])

    label = 'volume_log'

    feature_cols = ['site_id', 'latitude', 'longitude', 'elevation', 'elevation_stds',
                    'day_of_year', 'precip_deviation_season_mean', 'streamflow_deviation_season_mean',
                    'streamflow_deviation_30_mean', 'combined_swe_deviation_0', 'ua_swe_deviation',
                    'acc_water_deviation']

    train_labels = train_features[[label]]

    cat_10 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.1', iterations=1200, depth=6, random_seed=42)
    cat_50 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.5', iterations=1200, depth=8, random_seed=42)
    cat_90 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.9', iterations=900, depth=5, random_seed=42)

    cat_10_model = cat_10.fit(train_features[feature_cols], train_labels, cat_features=[0])
    cat_50_model = cat_50.fit(train_features[feature_cols], train_labels, cat_features=[0])
    cat_90_model = cat_90.fit(train_features[feature_cols], train_labels, cat_features=[0])

    cat_10_model.save_model(model_dir / 'cat_10_yearly_model.txt')
    cat_50_model.save_model(model_dir / 'cat_50_yearly_model.txt')
    cat_90_model.save_model(model_dir / 'cat_90_yearly_model.txt')

    logger.info("Finished training and saving yearly catboost model.")
