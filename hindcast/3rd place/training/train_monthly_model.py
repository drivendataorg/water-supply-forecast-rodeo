from pathlib import Path

import catboost as cb
from loguru import logger
import numpy as np
import pandas as pd


def train_monthly_model(model_dir: Path,
                        train_features: pd.DataFrame) -> None:
    logger.info("Beginning training of monthly catboost model.")
    monthly_train_features = train_features

    no_monthly_data = ['american_river_folsom_lake', 'merced_river_yosemite_at_pohono_bridge',
                       'san_joaquin_river_millerton_reservoir']

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

    feature_cols = ['site_id', 'pred_month', 'latitude', 'longitude', 'elevation', 'elevation_stds',
                    'prev_month_volume', 'day_of_year', 'precip_deviation_30_mean', 'streamflow_deviation_30_mean',
                    'maxt_deviation_30_mean', 'combined_swe_deviation_30']

    train_labels = train_features[[label]]

    cat_10 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.1', iterations=1300, random_seed=42)
    cat_50 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.5', iterations=1100, random_seed=42)
    cat_90 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.9', iterations=1300, random_seed=42)

    cat_10_model = cat_10.fit(train_features[feature_cols], train_labels, cat_features=[0, 1])
    cat_50_model = cat_50.fit(train_features[feature_cols], train_labels, cat_features=[0, 1])
    cat_90_model = cat_90.fit(train_features[feature_cols], train_labels, cat_features=[0, 1])

    cat_10_model.save_model(model_dir / 'cat_10_monthly_model.txt')
    cat_50_model.save_model(model_dir / 'cat_50_monthly_model.txt')
    cat_90_model.save_model(model_dir / 'cat_90_monthly_model.txt')

    logger.info("Finished training and saving monthly catboost model.")
