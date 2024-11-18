import pandas as pd
import lightgbm as lgb
import os
from pathlib import Path
import joblib

def train_pipeline(train_df: pd.DataFrame,
                   labels: pd.Series,
                   month: int,
                   params: list,
                   train_feat: list,
                   num_rounds_months: list,
                   categorical: list,
                   alpha: float,
                   lgb_models: list,
                   evals: int) -> list:
    """
    Train LightGBM model and append it to a list. It is a single iteration of
    given month and quantile.

    Args:
        train_df (pd.DataFrame): Train data
        labels (pd.Series): The response variable
        month (int): Month from a given issue date (starts with 1 for Jan)
        params (list): A list of model parameters for a selected month
        train_feat (list): A list of features to use for a selected month
        num_rounds_months (list): A number of estimators used in LightGBM models
            from different months. Only value from a selected month will be used
        categorical (list): A list of categorical columns
        alpha (float): Informs on which quantile should be calculated. In this
            case, it is a value from 0.1, 0.5 or 0.9
        lgb_models (list): A list of LightGBM models created so far for a given
            quantile
        evals (int): A step of number of LightGBM iterations when training
            progress is shown
    Returns:
        lgb_models (list): A list of LightGBM models created so far, including
            this iteration
    """
    train_data = lgb.Dataset(train_df.loc[train_df.month == month, train_feat],
                             label = labels[train_df[train_df.month == month].index],
                             categorical_feature = categorical)
    #Add alpha. 3 different models will be run for 0.1, 0.5 and 0.9 quantiles
    #in the training script.
    params['alpha'] = alpha

    lgb_model = lgb.train(params,
                          train_data,
                          valid_sets=[train_data],
                          num_boost_round = num_rounds_months[month - 1],
                          callbacks = [lgb.log_evaluation(evals)])
    lgb_models.append(lgb_model)
    return lgb_models


def to_categorical(categorical: list,
                   df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep categorical features in a list. LightGBM accepts only 'category'
    features for categorical columns.

    Args:
        categorical (list): Columns that should be changed to 'category' type
        df (pd.DataFrame): A DataFrame with columns to change
    Returns:
        df (pd.DataFrame): Df with columns from categorical changed to 'category'
    """
    for cat in categorical:
        df[cat] = df[cat].astype('category')
    return df


def save_models(issue_months: list,
                lgb_models_10: list,
                lgb_models_50: list,
                lgb_models_90: list):
    """
    Save LightGBM models. Models are saved separately for different quantiles
    and months.

    Args:
        issue_months (list): Different months to create models for. The months
            are based on issue dates
        lgb_models_10 (list): LightGBM models for 0.1 quantile. Should be in
            the same order as issue_months
        lgb_models_50 (list): LightGBM models for 0.5 quantile. Should be in
            the same order as issue_months
        lgb_models_90 (list): LightGBM models for 0.9 quantile. Should be in
            the same order as issue_months
    """
    #Create models\ directory if it doesn't exist
    models_path = Path('models')
    models_path.mkdir(parents = True, exist_ok = True)

    #Get today's date
    today = pd.to_datetime('today').strftime('%Y_%m_%d')
    path_save_models = os.path.join('models', today)
    #Create a folder in models\ with today's date where results will be saved.
    #If the folder was created already today, FileExistsError will be thrown. It
    #is created this way for safety, so already created models won't be replaced.
    os.mkdir(path_save_models)

    #Save models
    for month, lgb_model in zip(issue_months, lgb_models_10):
        joblib.dump(lgb_model,
                    f'{path_save_models}\lgbm_10_{month}_month_{today}.pkl')
        print(f'LightGBM model for month {month} and quantile 0.1 saved')
    for month, lgb_model in zip(issue_months, lgb_models_50):
        joblib.dump(lgb_model,
                    f'{path_save_models}\lgbm_50_{month}_month_{today}.pkl')
        print(f'LightGBM model for month {month} and quantile 0.5 saved')
    for month, lgb_model in zip(issue_months, lgb_models_90):
        joblib.dump(lgb_model,
                    f'{path_save_models}\lgbm_90_{month}_month_{today}.pkl')
        print(f'LightGBM model for month {month} and quantile 0.9 saved')


def load_models(issue_months: list,
                read_models_from_repo: bool=False,
                specific_date: str='today') -> tuple[list, list, list]:
    """
    Load LightGBM models. Models are read separately for different quantiles
    and months.

    Args:
        issue_months (list): Different months for models to load. The months
            are based on issue dates
        read_models_from_repo (bool): Condition if read 2024_01_11 models from
            the repo used in the Hindcast Stage of the competition or not.
            Defaults to False
        specific_date (str): a date for models to be loaded.
            Defaults to 'today'. For other values, use date in the format of
            YYYY_MM_DD
    Returns:
        lgb_models_10 (list): LightGBM models for 0.1 quantile. Should be in
            the same order as issue_months
        lgb_models_50 (list): LightGBM models for 0.5 quantile. Should be in
            the same order as issue_months
        lgb_models_90 (list): LightGBM models for 0.9 quantile. Should be in
            the same order as issue_months
    """
    lgb_models_10 = []
    lgb_models_50 = []
    lgb_models_90 = []

    if read_models_from_repo == True:
        today = '2024_01_11'
    elif specific_date != 'today':
        today = specific_date
    else:
        today = pd.to_datetime('today').strftime('%Y_%m_%d')

    #Get path
    path_save_models = os.path.join('models', today)
    #Iterate over paths from different months and append models to lists
    for month in issue_months:
        print(f'Load models from month {month}')
        lgb_models_10.append(
            joblib.load(f'{path_save_models}\lgbm_10_{month}_month_{today}.pkl'))
        lgb_models_50.append(
            joblib.load(f'{path_save_models}\lgbm_50_{month}_month_{today}.pkl'))
        lgb_models_90.append(
            joblib.load(f'{path_save_models}\lgbm_90_{month}_month_{today}.pkl'))
    return lgb_models_10, lgb_models_50, lgb_models_90
