import datetime
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from joblib import Parallel, delayed
import xgboost as xgb
from pandas import CategoricalDtype

from util import post_process_quantiles, evaluate

# Features used by the models
FEATURE_NAMES = [
    'site_encoded',
    'day_in_year',
    'antecedent_flow',
    'snotel_swe_conditional',
    'swann_swe_conditional',
    'swann_ppt_conditional',
    'swann_ppt_unaccounted',
]
# Names of categorical features
CAT_FEATURE_NAMES = ['site_id']
# The target to predict for the models
INTERMEDIATE_TARGET = 'remaining_volume'
# the target to return
TARGET = 'volume'
# The quantiles the models should be trained for
QUANTILES = [0.1, 0.5, 0.9]
# Cutoff year for training data
CUTOFF_YEAR = 1975

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the features data.

    Args:
        data_path (str): Path to the features CSV file.

    Returns:
        tuple: Preprocessed features, LOOCV data, and historical data.
    """
    features = pd.read_csv(data_path)
    features = features[features['forecast_year'] >= CUTOFF_YEAR]

    # Offsets are the cumulative in-season volumes
    offsets = features['offset_volume']
    features['remaining_volume'] = features['volume'] - offsets
    features['issue_day'] = pd.to_datetime(features['issue_date']).dt.strftime('%m-%d')
    features['month'] = pd.to_datetime(features['issue_date']).dt.strftime('%m').astype(int)
    features['day_in_month'] = pd.to_datetime(features['issue_date']).dt.strftime('%d').astype(int)
    cat_type = CategoricalDtype(categories=features['site_id'].unique(), ordered=False)
    features['site_encoded'] = features['site_id'].astype(cat_type)

    loocv_data = features[features['forecast_year'] >= 2004]
    historical_data = features[features['forecast_year'] < 2004]

    return features, loocv_data, historical_data

def train_fold(train, val, random_state):
    """
    Train XGBoost models for a single fold.

    Args:
        train (pd.DataFrame): Training data.
        val (pd.DataFrame): Validation data.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Trained models and predictions.
    """
    xgb_params = {
        "objective": "reg:quantileerror",
        'seed': random_state,
        'verbosity': 0,
        'enable_categorical': True
    }
    models = {}
    predictions = {}
    for q in QUANTILES:
        fit_params = {
            'verbose': False,
            'X': train[FEATURE_NAMES],
            'y': train[INTERMEDIATE_TARGET],
        }
        model = xgb.XGBRegressor(**xgb_params, quantile_alpha=q).fit(**fit_params)
        models[q] = model
        predictions[q] = model.predict(val[FEATURE_NAMES]) + val['offset_volume']

    predictions = pd.DataFrame(predictions)
    for col in [TARGET, 'site_id', 'issue_day']:
        predictions[col] = val[col]
    return models, predictions

def train_models(data, random_state, n_splits=10):
    """
    Train models using GroupKFold cross-validation.

    Args:
        data (pd.DataFrame): Input data.
        random_state (int): Random state for reproducibility.
        n_splits (int): Number of splits for GroupKFold.

    Returns:
        dict: Trained models and conformal prediction cutoffs.
    """
    gk_split = GroupKFold(n_splits=n_splits)

    results = Parallel(n_jobs=-1)(
        delayed(train_fold)(
            data.iloc[train_idx], data.iloc[val_idx], random_state
        ) for train_idx, val_idx in gk_split.split(data[FEATURE_NAMES], data[INTERMEDIATE_TARGET], groups=data['forecast_year'])
    )

    final_models = {q: [models[q] for models, _ in results] for q in QUANTILES}
    predictions = pd.concat([pred for _, pred in results])

    cutoffs = predictions.groupby(['site_id', 'issue_day']).apply(lambda x: {
        'lower': np.quantile(x[0.1] - x[TARGET], 0.9),
        'upper': np.quantile(x[TARGET] - x[0.9], 0.9)
    }).to_dict()

    return {'models': final_models, 'cutoffs': cutoffs}

def predict_models(models, new_data, cp_adjustments):
    """
    Make predictions using trained models and apply conformal prediction adjustments.

    Args:
        models (dict): Trained XGBoost models.
        new_data (pd.DataFrame): New data to predict on.
        cp_adjustments (dict): Conformal prediction adjustments.

    Returns:
        pd.DataFrame: Predictions with conformal prediction adjustments.
    """
    predictions = {
        q: np.mean([model.predict(new_data[FEATURE_NAMES]) for model in models[q]], axis=0) + new_data['offset_volume']
        for q in QUANTILES
    }

    predictions = pd.DataFrame(predictions)

    lower_adjustments = new_data.apply(lambda row: cp_adjustments[(row['site_id'], row['issue_day'])]['lower'], axis=1)
    upper_adjustments = new_data.apply(lambda row: cp_adjustments[(row['site_id'], row['issue_day'])]['upper'], axis=1)

    predictions[0.1] -= lower_adjustments.values
    predictions[0.9] += upper_adjustments.values

    predictions = post_process_quantiles(predictions)

    return predictions

def main():
    """
    Main function to train the XGBoost models with conformal prediction.
    """
    training_start_time = datetime.datetime.now()

    data_dir = Path('./data')
    pre_processed_dir = Path('./pre-processed/')
    features, loocv_data, historical_data = load_and_preprocess_data(pre_processed_dir / 'features.csv')
    logo = LeaveOneGroupOut()
    scores = {}
    all_preds = []
    random_state = 1

    for train_indices, test_indices in logo.split(loocv_data.volume.values, groups=loocv_data.forecast_year):
        train_data, test_data = loocv_data.iloc[train_indices], loocv_data.iloc[test_indices]
        train_data = pd.concat([train_data, historical_data], axis=0)
        results = train_models(train_data, random_state)

        current_year = test_data.forecast_year.unique()[0]
        print(f'----- {current_year} -----')

        predictions = predict_models(results['models'], test_data, results['cutoffs'])
        eval_results = evaluate(test_data[TARGET], predictions, verbose=True)
        predictions = predictions.set_index(test_data.index)
        all_preds.append(pd.concat([test_data, predictions], axis=1))
        scores[current_year] = eval_results

        if current_year == 2023:
            model_path = pre_processed_dir / f'models_{current_year}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(results, f)

    training_end_time = datetime.datetime.now()
    training_time = training_end_time - training_start_time
    minutes, seconds = divmod(int(training_time.total_seconds()), 60)
    print(f'Training took {minutes}:{seconds:02}')

    test_predictions = pd.concat(all_preds)
    test_predictions.rename({0.1: 'volume_10', 0.5: 'volume_50', 0.9: 'volume_90'}, axis=1, inplace=True)

    evaluate(test_predictions[TARGET], test_predictions[['volume_10', 'volume_50', 'volume_90']], verbose=True)

    submission = pd.read_csv(data_dir / "cross_validation_submission_format.csv")
    submission.drop(columns=['volume_10', 'volume_50', 'volume_90'], inplace=True)
    predictions_df = test_predictions[['site_id', 'issue_date', 'volume_10', 'volume_50', 'volume_90']]
    submission = submission.merge(predictions_df, on=['site_id', 'issue_date'])
    submission.to_csv(pre_processed_dir / "forecasts.csv", index=False)

if __name__ == "__main__":
    main()
