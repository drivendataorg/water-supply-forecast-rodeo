from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss
import optuna
import joblib
from tqdm import tqdm

from utils import get_quantiles_from_distr, all_distr_dict

#Don't show unnecessary warnings in training
pd.options.mode.chained_assignment = None

def get_cv_folds(train: pd.DataFrame,
                 month: int,
                 years_cv: list,
                 year_range: bool) -> tuple[list, list]:
    """
    Create cross-validation folds. Get train and test indices for different
    folds.

    Args:
        train (pd.DataFrame): Train data
        month (int): Month to create folds for
        years_cv (list): A list with test years for different CV folds.
            In case of many years in one test fold, those years are specified
            in consecutive lists inside the years_cv list
        year_range (bool): Specifies if there could be many years in test data
            (True) or just one test data per fold (False)
    Returns:
        train_cv_idxs (list): Indexes of train data for consecutive folds
        test_cv_idxs (list): Indexes of test data for consecutive folds
    """
    train_cv_idxs = []
    test_cv_idxs = []

    if year_range == True:
        #Store many years in test sets
        for year in years_cv:
            train_cv_idxs.append(list(train[(~(train.year.between(year[0], year[1]))) &
                                            (train.month == month)].index))
            test_cv_idxs.append(list(train[(train.year.between(year[0], year[1])) &
                                           (train.month == month)].index))
    else:
        #Store only one year in test sets
        for year in years_cv:
            train_cv_idxs.append(list(train[(train.year != year) &
                                            (train.month == month)].index))
            test_cv_idxs.append(list(train[(train.year == year) &
                                           (train.month == month)].index))
    return train_cv_idxs, test_cv_idxs


def get_years_cv(year_range: bool) -> list:
    """
    Get years for CV test folds. There are 2 options: 2 years in a test fold
    or 1 year in a test fold.
    Keep in mind that odd years since 2005 are in the test set, so in such
    cases for 2-year test folds, they aren't included in 2 years range
    calculation (2020-2022 range is treated as 2 years as 2021 is missing).

    Args:
        year_range (bool): Specifies if there could be many years in test data
            (True) or just one test data per fold (False)
    Returns:
        years_cv (list): A list with test years for different CV folds.
            In case of many years in one test fold, those years are specified
            in consecutive lists inside the years_cv list
    """
    if year_range == True:
        #2 years in one fold
        years_cv = [[1994, 1995],
                    [1996, 1997],
                    [1998, 1999],
                    [2000, 2001],
                    [2002, 2003],
                    [2004, 2006],
                    [2008, 2010],
                    [2012, 2014],
                    [2016, 2018],
                    [2020, 2022]]
    else:
        #One year at a time
        years_cv = [2004,
                    2005,
                    2006,
                    2007,
                    2008,
                    2009,
                    2010,
                    2011,
                    2012,
                    2013,
                    2014,
                    2015,
                    2016,
                    2017,
                    2018,
                    2019,
                    2020,
                    2021,
                    2022,
                    2023]
    return years_cv


def train_cv(train: pd.DataFrame,
             labels: pd.Series,
             train_cv_idxs: list,
             test_cv_idxs: list,
             train_feat: list,
             params: dict,
             categorical: list,
             num_boost_round_start: int,
             num_boost_round_month: int,
             alpha: float,
             fold: int,
             lgb_models: dict) -> tuple[np.ndarray, dict]:
    """
    Training pipeline for given fold-quantile combination. Creates model for
    given fold for the first time or continues training previous model until
    threshold is met.

    Args:
        train (pd.DataFrame): Whole training data before dividing into CV folds
        labels (pd.Series): Labels corresponding to train data
        train_cv_idxs (list): Indexes of train data for all folds
        test_cv_idxs (list): Indexes of test data for all folds
        train_feat (list): Features to use for this month
        params (dict): LightGBM hyperparameters to use for this month
        categorical (list): Categorical features in the model
        num_boost_round_start (int): Number of estimators used in LightGBM
            model after which early stopping criterion starts (could be seen
            as the minimum number of model iterations)
        num_boost_round_month (int): Maximum number of estimators used in
            LightGBM model for this training iteration. Model is trained until
            num_boost_round_month is reached
        alpha (float): Informs on which quantile should be calculated. In this
            case, it is a value from 0.1, 0.5 or 0.9
        fold (int): Fold from the given iteration. Used to get correct train
            and test indexes and key for LightGBM model for lgb_models
        lgb_models (dict): A dictionary of LightGBM models. One of its keys
            (folds) have to be updated in this iteration
    Returns:
        preds (np.ndarray): Predictions from given quantile and fold
        lgb_models (dict): A dictionary of LightGBM models with updated fold
            values
    """
    #Add alpha to params (specify quantile)
    params['alpha'] = alpha
    #Create lgb.Datasets for given fold
    train_data = lgb.Dataset(data = train.loc[train_cv_idxs[fold],
                                              train_feat],
                             label = labels[train_cv_idxs[fold]],
                             categorical_feature = categorical)
    test_data = lgb.Dataset(data = train.loc[test_cv_idxs[fold],
                                             train_feat],
                            label = labels[test_cv_idxs[fold]],
                            reference = train_data)
    #Train model
    if num_boost_round_month == num_boost_round_start:
        #Use lgb.train for the first CV iteration (for num_boost_round_start
        #number of LightGBM boosting iters)
        lgb_model = lgb.train(params,
                              train_data,
                              valid_sets=[train_data, test_data],
                              num_boost_round = num_boost_round_start,
                              keep_training_booster = True)
        #Update dictionary with first fold result
        lgb_models[fold] = lgb_model
    else:
        #For other CV iters, update LightGBM model for early_stopping_step
        #number of iterations
        while lgb_models[fold].current_iteration() < num_boost_round_month:
            lgb_models[fold].update()
    #Get predictions
    preds = lgb_models[fold].predict(train.loc[test_cv_idxs[fold], train_feat])
    return preds, lgb_models


def nat_flow_sum_clipping(train: pd.DataFrame,
                          result_df: pd.DataFrame,
                          nat_flow_sum_col: str,
                          test_cv_idxs: list,
                          fold: int,
                          month: int,
                          residuals: bool,
                          multip_10: float,
                          multip_50: float,
                          multip_90: float,
                          multip_10_thres: float = 1.0,
                          multip_50_thres: float = 1.0,
                          multip_90_thres: float = 1.0,
                          multip_10_detroit: float = 1.0,
                          multip_50_detroit: float = 1.0,
                          multip_90_detroit: float = 1.0,
                          multip_10_thres_detroit: float = 1.0,
                          multip_50_thres_detroit: float = 1.0,
                          multip_90_thres_detroit: float = 1.0) -> pd.DataFrame:
    """
    Amendments to results based on naturalized flow. If naturalized flow since
    April (for most site_ids) up to selected month with chosen multipliers is
    greater than predicted volume for different quantiles, add a correction
    to those predicted volumes.

    Args:
        train (pd.DataFrame): Whole training data before dividing into CV folds
        result_df (pd.DataFrame): This fold results before clipping
        nat_flow_sum_col (str): A column name with naturalized flow since April.
            Depends on issue month (May/Jun/Jul)
        test_cv_idxs (list): Indexes of test data for all folds
        fold (int): Fold from the given iteration. Used to get correct train
            and test indexes
        month (int): A month to make clipping for
        residuals (bool): Informs if predictions are made for volume residuals
        multip_10 (float): A multiplier for a given nat_flow_sum for Q0.1
        multip_50 (float): A multiplier for a given nat_flow_sum for Q0.5
        multip_90 (float): A multiplier for a given nat_flow_sum for Q0.9
        multip_10_thres (float): A threshold for a given nat_flow_sum column
            for Q0.1. Change volume prediction if current volume is less than
            nat_flow_sum multiplied by this threshold
        multip_50_thres (float): A threshold for a given nat_flow_sum column
            for Q0.5. Change volume prediction if current volume is less than
            nat_flow_sum multiplied by this threshold
        multip_90_thres (float): A threshold for a given nat_flow_sum column
            for Q0.9. Change volume prediction if current volume is less than
            nat_flow_sum multiplied by this threshold
        multip_10_detroit (float): A multiplier for a given nat_flow_sum for
            Q0.1 and detroit_lake_inflow site_id. Defaults to 1.0, so it
            doesn't have to be passed for July with missing detroit site_id
        multip_50_detroit (float): A multiplier for a given nat_flow_sum for
            Q0.5 and detroit_lake_inflow site_id. Defaults to 1.0, so it
            doesn't have to be passed for July with missing detroit site_id
        multip_90_detroit (float): A multiplier for a given nat_flow_sum for
            Q0.9 and detroit_lake_inflow site_id. Defaults to 1.0, so it
            doesn't have to be passed for July with missing detroit site_id
        multip_10_thres_detroit (float): A threshold for a given nat_flow_sum
            column for Q0.1 and detroit_lake_inflow site_id. Change volume
            prediction if current volume is less than nat_flow_sum multiplied
            by this threshold. Defaults to 1.0, so it
            doesn't have to be passed for July with missing detroit site_id
        multip_50_thres_detroit (float): A threshold for a given nat_flow_sum
            column for Q0.5 and detroit_lake_inflow site_id. Change volume
            prediction if current volume is less than nat_flow_sum multiplied
            by this threshold. Defaults to 1.0, so it doesn't have to be
            passed for July with missing detroit site_id
        multip_90_thres_detroit (float): A threshold for a given nat_flow_sum
            column for Q0.9 and detroit_lake_inflow site_id. Change volume
            prediction if current volume is less than nat_flow_sum multiplied
            by this threshold. Defaults to 1.0, so it doesn't have to be
            passed for July with missing detroit site_id
    Returns:
        results_clipped (pd.DataFrame): Results with added nat_flow_sum clipping
    """
    if residuals == False:
        #Append nat flow sum column. If residuals are used, nat_flow_sum_col
        #should be already appended
        results_clipped = pd.merge(result_df,
                                   train.loc[(train.index.isin(test_cv_idxs[fold])),
                                             ['site_id', 'issue_date_no_year',
                                             nat_flow_sum_col]],
                                   how = 'left',
                                   on = ['site_id', 'issue_date_no_year'])
    else:
        results_clipped = result_df.copy()
    #Get nat_flow_sum to volume ratios
    results_clipped['volume_10_nat_flow_sum_ratio'] =\
        results_clipped.volume_10 / results_clipped[nat_flow_sum_col]
    results_clipped['volume_50_nat_flow_sum_ratio'] =\
        results_clipped.volume_50 / results_clipped[nat_flow_sum_col]
    results_clipped['volume_90_nat_flow_sum_ratio'] =\
        results_clipped.volume_90 / results_clipped[nat_flow_sum_col]
    #Multiply by multiplier if < nat_flow_sum or multiply additionally by
    #volume_nat_flow_sum_ratio if volume was greater than nat_flow_sum but
    #within boundaries for condition execution
    results_clipped['volume_10_with_mult'] =\
        results_clipped.apply(lambda x: max(x[nat_flow_sum_col] * multip_10,
                                            x[nat_flow_sum_col] * multip_10 *
                                            x.volume_10_nat_flow_sum_ratio),
                              axis = 1)
    results_clipped['volume_50_with_mult'] =\
        results_clipped.apply(lambda x: max(x[nat_flow_sum_col] * multip_50,
                                            x[nat_flow_sum_col] * multip_50 *
                                            x.volume_50_nat_flow_sum_ratio),
                              axis = 1)
    results_clipped['volume_90_with_mult'] =\
        results_clipped.apply(lambda x: max(x[nat_flow_sum_col] * multip_90,
                                            x[nat_flow_sum_col] * multip_90 *
                                            x.volume_90_nat_flow_sum_ratio),
                              axis = 1)
    #Change volume values using multipliers if threshold is within boundaries
    if month == 7:
        #Don't process detroit_lake_inflow for month 7 where it doesn't exist
        results_clipped.loc[
            results_clipped.volume_10 < results_clipped[nat_flow_sum_col] *
            multip_10_thres,
            'volume_10'] = results_clipped.volume_10_with_mult
        results_clipped.loc[
            results_clipped.volume_50 < results_clipped[nat_flow_sum_col] *
            multip_50_thres,
            'volume_50'] = results_clipped.volume_50_with_mult
        results_clipped.loc[
            results_clipped.volume_90 < results_clipped[nat_flow_sum_col] *
            multip_90_thres,
            'volume_90'] = results_clipped.volume_90_with_mult
    if month in [5, 6]:
        #For detroit site
        results_clipped['volume_10_with_mult_detroit'] =\
            results_clipped.apply(
                lambda x: max(x[nat_flow_sum_col] * multip_10_detroit,
                              x[nat_flow_sum_col] * multip_10_detroit *
                              x.volume_10_nat_flow_sum_ratio), axis = 1)
        results_clipped['volume_50_with_mult_detroit'] =\
            results_clipped.apply(
                lambda x: max(x[nat_flow_sum_col] * multip_50_detroit,
                              x[nat_flow_sum_col] * multip_50_detroit *
                              x.volume_50_nat_flow_sum_ratio), axis = 1)
        results_clipped['volume_90_with_mult_detroit'] =\
            results_clipped.apply(
                lambda x: max(x[nat_flow_sum_col] * multip_90_detroit,
                              x[nat_flow_sum_col] * multip_90_detroit *
                              x.volume_90_nat_flow_sum_ratio), axis = 1)
        #For other site ids
        results_clipped.loc[
            (results_clipped.volume_10 < results_clipped[nat_flow_sum_col] *
             multip_10_thres_detroit) &
            (results_clipped.site_id == 'detroit_lake_inflow'),
            'volume_10'] = results_clipped.volume_10_with_mult_detroit
        results_clipped.loc[
            (results_clipped.volume_10 < results_clipped[nat_flow_sum_col] *
             multip_10_thres) &
            (results_clipped.site_id != 'detroit_lake_inflow'),
            'volume_10'] = results_clipped.volume_10_with_mult

        results_clipped.loc[
            (results_clipped.volume_50 < results_clipped[nat_flow_sum_col] *
             multip_50_thres_detroit) &
            (results_clipped.site_id == 'detroit_lake_inflow'),
            'volume_50'] = results_clipped.volume_50_with_mult_detroit
        results_clipped.loc[
            (results_clipped.volume_50 < results_clipped[nat_flow_sum_col] *
             multip_50_thres) &
            (results_clipped.site_id != 'detroit_lake_inflow'),
            'volume_50'] = results_clipped.volume_50_with_mult

        results_clipped.loc[
            (results_clipped.volume_90 < results_clipped[nat_flow_sum_col] *
             multip_90_thres_detroit) &
            (results_clipped.site_id == 'detroit_lake_inflow'),
            'volume_90'] = results_clipped.volume_90_with_mult_detroit
        results_clipped.loc[
            (results_clipped.volume_90 < results_clipped[nat_flow_sum_col] *
             multip_90_thres) &
            (results_clipped.site_id != 'detroit_lake_inflow'),
            'volume_90'] = results_clipped.volume_90_with_mult
    return results_clipped


def lgbm_cv(train: pd.DataFrame,
            labels: pd.Series,
            num_boost_round: int,
            num_boost_round_start: int,
            early_stopping_rounds: int,
            early_stopping_step: int,
            issue_months: list,
            years_cv: list,
            year_range: bool,
            train_feat_dict: dict,
            params_dict: dict,
            categorical: list,
            min_max_site_id_dict: dict,
            path_distr: str,
            distr_perc_dict: dict,
            no_nat_flow_sites: bool = False) -> tuple[np.ndarray, float, float,
                                                      list, np.ndarray, float,
                                                      pd.DataFrame]:
    """
    Run LightGBM CV with early stopping, get distribution estimates and
    average the results. Perform additional clipping to model predictions.

    Args:
        train (pd.DataFrame): Whole training data before dividing into CV folds
        labels (pd.Series): Labels corresponding to train data
        num_boost_round (int): Maximum number of estimators used in LightGBM
            models. Model could be stopped earlier if early stopping is met
        num_boost_round_start (int): Number of estimators used in LightGBM
            model after which early stopping criterion starts (could be seen
            as the minimum number of model iterations)
        early_stopping_rounds (int): How many times in a row early stopping can
            be met before stopping training
        early_stopping_step (int): Number of iterations when early stopping is
            performed. 20 means that it's done every 20 iterations, i,e. after
            100, 120, 140, ... iters
        issue_months (list): Months to iterate over. Every month has its own
            model
        years_cv (list): A list with test years for different CV folds
        year_range (bool): Specifies if there could be many years in test data
            (True) or just one test data per fold (False)
        train_feat_dict (dict): Features to use for different months. Key
            indicates month and value the features
        params_dict (dict): LightGBM hyperparameters to use for different months.
            Key indicates month and value, a dictionary of hyperparameters
        categorical (list): Categorical features in the model
        min_max_site_id_dict (dict): Minimum and maximum historical volumes
            for given site_id. It contains different DataFrames for different
            LOOCV years, LOOCV years are used as the keys
        path_distr (str): Path to values of distribution estimate parameters
            per each site_id already with amendments to distributions. Different
            LOOCV years have different distribution files. They can be
            distinguished by year suffixes (for example if path is distr_final,
            file name for LOOCV year 2010 will be distr_final_2010)
        distr_perc_dict (dict): How much importance is given for distribution
            estimate. For 0.4 value, it's 40% (while LightGBM model is 60%).
            Different months use different distribution percentage (1 for
            January, 2 for February, ..., 7 for July)
        no_nat_flow_sites (bool): Indicates if it's an optimization pipeline for
            3 site_ids without naturalized flow columns. Defaults to False
    Returns:
        best_cv_early_stopping (np.ndarray): CV results from different months
            with number of model iterations for each month
        result_final_rms_avg (float): RMS CV results averaged over different
            months
        result_final_avg (float): CV results averaged over different months
        num_rounds_months (list): Number of model iterations for each month
        interval_coverage_all_months (np.ndarray): interval coverage results
            from different month for num_rounds_months iterations
        best_interval_early_stopping (float): interval coverage averaged over
            different months
        final_preds_all_months (pd.DataFrame): Predictions with best number
            of iterations from selected issue months
    """
    ###########################################################################
    #Global parameters and variables initialization
    ###########################################################################
    #Initialize empty variables

    #Initialize predictions for different folds
    final_preds_all_months = pd.DataFrame()
    cv_results_all_months = dict() #CV results from different months
    results_coverage_all_months = dict() #interval coverage from different months
    results_10 = []
    results_50 = []
    results_90 = []
    #Set site ids without naturalized flow information from the given water year
    site_ids_no_nat_flow = ['american_river_folsom_lake',
                            'san_joaquin_river_millerton_reservoir',
                            'merced_river_yosemite_at_pohono_bridge']

    ###########################################################################
    #Iterate over months. Train one month at a time
    ###########################################################################
    for month_idx, month in tqdm(enumerate(issue_months)):
        print(f'\nMonth: {month}')
        #Initialize predictions for given month
        final_preds = pd.DataFrame()
        #Get distribution percentage to use for given month
        distr_perc = distr_perc_dict[month]
        #Initialize variables for the month
        #First evaluation done after num_boost_round_start iters
        num_boost_round_month = num_boost_round_start
        #Set previous value of averaged CV to infinity for first evaluation,
        #so the first evaluated value is always less
        cv_result_avg_fold_prev = np.inf
        #Initialize number of early stopping conditions met so far with 0
        num_prev_iters_better = 0
        #All [avg RMS fold result-avg fold result-number of LightGBM iterations]
        #from given month. All LGBM iters after each early_stopping_step have
        #a row in the list
        cv_results_avg_fold = []
        #Similarly for interval coverage
        results_coverage_avg_fold = []
        #Quantile 0.5 models for newest fold results
        lgb_models_50 = dict()
        #Quantile 0.1 models for newest fold results
        lgb_models_10 = dict()
        #Quantile 0.9 models for newest fold results
        lgb_models_90 = dict()

        #######################################################################
        #Start training. Train until early stopping/maximum number of iters met
        #######################################################################
        while (num_prev_iters_better < early_stopping_rounds) &\
              (num_boost_round_month <= num_boost_round):
            #Initialize variables for given iter
            results_10_clipped = []
            results_50_clipped = []
            results_90_clipped = []
            results_coverage = []
            cv_results = []
            ###################################################################
            #Iterate over different folds
            ###################################################################
            for fold, year in enumerate(years_cv):
                #Get indexes from train DataFrame for given fold's train and test
                train_cv_idxs, test_cv_idxs = get_cv_folds(train,
                                                           month,
                                                           years_cv,
                                                           year_range)
                #Choose features from given month
                train_feat = train_feat_dict[month]
                #Get params from given month
                params = params_dict[month]
                #Train/continue training model from given fold for Q0.5
                preds_50, lgb_models_50 = train_cv(train,
                                                   labels,
                                                   train_cv_idxs,
                                                   test_cv_idxs,
                                                   train_feat,
                                                   params,
                                                   categorical,
                                                   num_boost_round_start,
                                                   num_boost_round_month,
                                                   0.5,
                                                   fold,
                                                   lgb_models_50)
                #Train/continue training model from given fold for Q0.1.
                #Named preds_10_lgbm, as final preds_10 will use a weighted
                #average with distribution estimates
                preds_10_lgbm, lgb_models_10 = train_cv(train,
                                                        labels,
                                                        train_cv_idxs,
                                                        test_cv_idxs,
                                                        train_feat,
                                                        params,
                                                        categorical,
                                                        num_boost_round_start,
                                                        num_boost_round_month,
                                                        0.1,
                                                        fold,
                                                        lgb_models_10)
                #Train/continue training model from given fold for Q0.9.
                #Named preds_90_lgbm, as final preds_10 will use a weighted
                #average with distribution estimates
                preds_90_lgbm, lgb_models_90 = train_cv(train,
                                                        labels,
                                                        train_cv_idxs,
                                                        test_cv_idxs,
                                                        train_feat,
                                                        params,
                                                        categorical,
                                                        num_boost_round_start,
                                                        num_boost_round_month,
                                                        0.9,
                                                        fold,
                                                        lgb_models_90)
                #Get test rows from given fold with predictions
                result_df = train.loc[test_cv_idxs[fold], train_feat]
                result_df['volume_50'] = preds_50
                result_df['volume_10_lgbm'] = preds_10_lgbm
                result_df['volume_90_lgbm'] = preds_90_lgbm
                #Get only results from 3 site_ids without nat_flow if applicable
                if no_nat_flow_sites == True:
                    result_df =\
                        result_df[result_df.site_id.isin(site_ids_no_nat_flow)]
                #Append quantile results
                result_df = get_quantiles_from_distr(result_df,
                                                     min_max_site_id_dict[year],
                                                     all_distr_dict,
                                                     f'{path_distr}{year}')
                #Add min and max for site_id as 'max' and 'min' columns
                result_df = pd.merge(result_df,
                                     min_max_site_id_dict[year],
                                     how = 'left',
                                     left_on = 'site_id',
                                     right_index = True)
                #Change volume values greater than min (max) for site_id to
                #that min (max) value. Do it also for distribution volume.
                #Though distribution estimates shouldn't exceed maximum values,
                #do it just to be certain
                result_df.loc[result_df['volume_50'] < result_df['min'],
                              'volume_50'] = result_df['min']
                result_df.loc[result_df['volume_50'] > result_df['max'],
                              'volume_50'] = result_df['max']
                result_df.loc[result_df['volume_10_lgbm'] < result_df['min'],
                              'volume_10_lgbm'] = result_df['min']
                result_df.loc[result_df['volume_10_distr'] < result_df['min'],
                              'volume_10_distr'] = result_df['min']
                result_df.loc[result_df['volume_90_lgbm'] > result_df['max'],
                              'volume_90_lgbm'] = result_df['max']
                result_df.loc[result_df['volume_90_distr'] > result_df['max'],
                              'volume_90_distr'] = result_df['max']
                #Clipping:
                    #if volume_90 < volume_50 -> change volume_90 to volume_50
                    #if volume_50 < volume_10 -> change volume_10 to volume_50
                result_df.loc[result_df.volume_90_lgbm < result_df.volume_50,
                              'volume_90_lgbm'] = result_df.volume_50
                result_df.loc[result_df.volume_50 < result_df.volume_10_lgbm,
                              'volume_10_lgbm'] = result_df.volume_50
                result_df.loc[result_df.volume_90_distr < result_df.volume_50,
                              'volume_90_distr'] = result_df.volume_50
                result_df.loc[result_df.volume_50 < result_df.volume_10_distr,
                              'volume_10_distr'] = result_df.volume_50
                #Get weighted average from distributions and models for Q0.1
                #and Q0.9
                result_df['volume_10'] =\
                    distr_perc * result_df.volume_10_distr +\
                    (1 - distr_perc) * result_df.volume_10_lgbm
                result_df['volume_90'] =\
                    distr_perc * result_df.volume_90_distr +\
                    (1 - distr_perc) * result_df.volume_90_lgbm
                #Get quantile loss for given fold
                if no_nat_flow_sites == True:
                    #Use label indexes only for 3 site ids
                    result_10 = mean_pinball_loss(labels[result_df.index],
                                                  result_df.volume_10,
                                                  alpha = 0.1)
                    result_50 = mean_pinball_loss(labels[result_df.index],
                                                  result_df.volume_50,
                                                  alpha = 0.5)
                    result_90 = mean_pinball_loss(labels[result_df.index],
                                                  result_df.volume_90,
                                                  alpha = 0.9)
                else:
                    result_10 = mean_pinball_loss(labels[test_cv_idxs[fold]],
                                                  result_df.volume_10,
                                                  alpha = 0.1)
                    result_50 = mean_pinball_loss(labels[test_cv_idxs[fold]],
                                                  result_df.volume_50,
                                                  alpha = 0.5)
                    result_90 = mean_pinball_loss(labels[test_cv_idxs[fold]],
                                                  result_df.volume_90,
                                                  alpha = 0.9)
                #Append results from this fold
                results_10.append(result_10)
                results_50.append(result_50)
                results_90.append(result_90)

                ###############################################################
                #nat_flow_sum clipping
                ###############################################################
                results_clipped = result_df.copy()
                #Add clipping to Jul based on Apr-Jun nat_flow_sum
                if month == 7:
                    results_clipped = nat_flow_sum_clipping(
                        train = train,
                        result_df = results_clipped,
                        nat_flow_sum_col = 'nat_flow_sum_Apr_Jun',
                        test_cv_idxs = test_cv_idxs,
                        fold = fold,
                        month = 7,
                        residuals = False,
                        multip_10 = 1.1,
                        multip_50 = 1.15,
                        multip_90 = 1.2,
                        multip_10_thres = 1.0,
                        multip_50_thres = 1.05,
                        multip_90_thres = 1.05)
                #Add clipping to Jun based on Apr-May nat_flow_sum
                if month == 6:
                    results_clipped = nat_flow_sum_clipping(
                        train = train,
                        result_df = results_clipped,
                        nat_flow_sum_col = 'nat_flow_sum_Apr_May',
                        test_cv_idxs = test_cv_idxs,
                        fold = fold,
                        month = 6,
                        residuals = False,
                        multip_10 = 1.2,
                        multip_50 = 1.25,
                        multip_90 = 1.3,
                        multip_10_thres = 1.1,
                        multip_50_thres = 1.15,
                        multip_90_thres = 1.15,
                        multip_10_detroit = 1.2,
                        multip_50_detroit = 1.25,
                        multip_90_detroit = 1.3,
                        multip_10_thres_detroit = 1.1,
                        multip_50_thres_detroit = 1.15,
                        multip_90_thres_detroit = 1.15)
                #Add clipping to Jul based on Apr nat_flow_sum
                if month == 5:
                    results_clipped = nat_flow_sum_clipping(
                        train = train,
                        result_df = results_clipped,
                        nat_flow_sum_col = 'nat_flow_sum_Apr_Apr',
                        test_cv_idxs = test_cv_idxs,
                        fold = fold,
                        month = 5,
                        residuals = False,
                        multip_10 = 1.3,
                        multip_50 = 1.35,
                        multip_90 = 1.4,
                        multip_10_thres = 1.2,
                        multip_50_thres = 1.25,
                        multip_90_thres = 1.25,
                        multip_10_detroit = 1.3,
                        multip_50_detroit = 1.35,
                        multip_90_detroit = 1.4,
                        multip_10_thres_detroit = 1.2,
                        multip_50_thres_detroit = 1.25,
                        multip_90_thres_detroit = 1.25)

                ###############################################################
                #Final clipping
                ###############################################################
                #Do the final clipping to make sure that the restrictions are
                #met after taking weighted average for volume_10 and volume_90
                results_clipped.loc[results_clipped.volume_90 < results_clipped.volume_50,
                                    'volume_50'] = results_clipped.volume_90
                results_clipped.loc[results_clipped.volume_50 < results_clipped.volume_10,
                                    'volume_10'] = results_clipped.volume_50

                #Get quantile loss for given fold
                if no_nat_flow_sites == True:
                    #Use result_df for finding labels, as result_clipped have
                    #reseted indexes
                    result_10_clipped = mean_pinball_loss(
                        labels[result_df[result_df.site_id.isin(site_ids_no_nat_flow)].index],
                        results_clipped.volume_10,
                        alpha = 0.1)
                    result_50_clipped = mean_pinball_loss(
                        labels[result_df[result_df.site_id.isin(site_ids_no_nat_flow)].index],
                        results_clipped.volume_50,
                        alpha = 0.5)
                    result_90_clipped = mean_pinball_loss(
                        labels[result_df[result_df.site_id.isin(site_ids_no_nat_flow)].index],
                        results_clipped.volume_90,
                        alpha = 0.9)

                    result_coverage = interval_coverage(np.array(
                        labels[result_df[result_df.site_id.isin(site_ids_no_nat_flow)].index]),
                            np.array(results_clipped[['volume_10', 'volume_50', 'volume_90']]))
                else:
                    result_10_clipped = mean_pinball_loss(labels[test_cv_idxs[fold]],
                                                          results_clipped.volume_10,
                                                          alpha = 0.1)
                    result_50_clipped = mean_pinball_loss(labels[test_cv_idxs[fold]],
                                                          results_clipped.volume_50,
                                                          alpha = 0.5)
                    result_90_clipped = mean_pinball_loss(labels[test_cv_idxs[fold]],
                                                          results_clipped.volume_90,
                                                          alpha = 0.9)

                    result_coverage = interval_coverage(np.array(labels[test_cv_idxs[fold]]),
                                                        np.array(results_clipped[[
                                                            'volume_10', 'volume_50', 'volume_90']]))

                #Append results from this fold
                results_10_clipped.append([fold, result_10_clipped, num_boost_round_month])
                results_50_clipped.append([fold, result_50_clipped, num_boost_round_month])
                results_90_clipped.append([fold, result_90_clipped, num_boost_round_month])
                #Get competition metric
                cv_result = 2 * (result_10_clipped +
                                 result_50_clipped +
                                 result_90_clipped) / 3
                #Append the result from given fold-model iteration
                cv_results.append([cv_result])
                results_coverage.append(result_coverage)

                #Get predictions
                preds_with_volumes = results_clipped[['site_id',
                                                      'volume_10',
                                                      'volume_50',
                                                      'volume_90',
                                                      'issue_date_no_year']]
                #Append NUM_BOOST_ROUND_MONTH
                preds_with_volumes['num_boost_rounds'] = num_boost_round_month
                #Append year
                preds_with_volumes['year'] = year
                #Append predictions
                final_preds = pd.concat([final_preds, preds_with_volumes])
            #Average results over different folds for given model iteration
            cv_result_avg_fold = np.mean(cv_results)
            #Average RMS (root mean square) results over different folds for
            #given model iteration
            cv_result_rms = np.array(cv_results) ** 2
            cv_result_avg_fold_rms =\
                np.sqrt(np.sum(cv_result_rms) / len(cv_result_rms))

            #Do the same for interval coverage
            result_coverage_avg_fold = np.mean(results_coverage)
            #Keep track of early stopping condition if result is poorer than
            #in the previous early stopping check (early_stopping_step before)
            if cv_result_avg_fold_rms > cv_result_avg_fold_prev:
                num_prev_iters_better += 1
            else:
                cv_result_avg_fold_prev = cv_result_avg_fold_rms
                #If new result is better, use new result in next early stopping
                #check. Reset num_prev_iters_better to 0
                num_prev_iters_better = 0

            print(f'Avg RMS result all folds for {num_boost_round_month} trees:',
                  cv_result_avg_fold_rms)
            print(f'Avg result all folds for {num_boost_round_month} trees:',
                  cv_result_avg_fold)
            print(f'Avg interval coverage all folds for {num_boost_round_month} trees:',
                  result_coverage_avg_fold)
            #Append number of boosting iterations to average results
            cv_results_avg_fold.append([cv_result_avg_fold_rms,
                                        cv_result_avg_fold,
                                        num_boost_round_month])
            #Do the same for interval coverage
            results_coverage_avg_fold.append([result_coverage_avg_fold,
                                              num_boost_round_month])
            #Update information when next early stopping will be evaluated
            num_boost_round_month += early_stopping_step

        #######################################################################
        #Early stopping/maximum number of iterations (num_boost_round) met
        #for the selected month
        #######################################################################
        #Update results if last values weren't the best ones. Maximum value
        #is chosen as the final value
        if num_prev_iters_better != 0:
            cv_results_avg_fold = cv_results_avg_fold[:-num_prev_iters_better]
            results_coverage_avg_fold =\
                results_coverage_avg_fold[:-num_prev_iters_better]
            #Keep only predictions from best num_boost_rounds
            final_preds = final_preds[
                final_preds.num_boost_rounds == num_boost_round_month -
                (1 + num_prev_iters_better) * early_stopping_step]
        cv_results_all_months[month] = cv_results_avg_fold
        results_coverage_all_months[month] = results_coverage_avg_fold
        #Append predictions from given month
        final_preds_all_months = pd.concat([final_preds_all_months, final_preds])
    ###########################################################################
    #All months were trained. Get final results to return
    ###########################################################################
    #Get best fit per month with best number of iterations
    best_cv_early_stopping = []
    for month in cv_results_all_months.keys():
        best_cv_early_stopping.append(cv_results_all_months[month][-1])
    best_cv_early_stopping = np.array(best_cv_early_stopping)
    #Average best fits over months. detroit_lake_inflow predictions are for
    #Apr-Jun period, there aren't any values for this site_id for July, so
    #there's a need for correction

    #Add month importance. July has less importance, as 25 out of 26 site_ids
    #have values for this month
    if len(issue_months) == 7:
        month_importance = np.array([1, 1, 1, 1, 1, 1, 25/26])
    else:
        #If not all months are being evaluated, add the same weights for each
        #month for simplicity
        month_importance = np.ones(len(issue_months))

    #Sum weights
    sum_weights = np.sum(month_importance)
    #Ratio of importance to sum of all weights
    month_weights = month_importance / sum_weights
    #Sum of results for different months multiplied by weights to get
    #a weighted average over different months
    result_final_rms_avg = np.sum(best_cv_early_stopping[:, 0] * month_weights)
    result_final_avg = np.sum(best_cv_early_stopping[:, 1] * month_weights)
    #Get optimal number of rounds for each month separately
    num_rounds_months = list(best_cv_early_stopping[:, 2].astype('int'))
    #Get interval coverage from the best iteration
    best_interval_early_stopping = []
    for month in cv_results_all_months.keys():
        best_interval_early_stopping.append(results_coverage_all_months[month][-1])
    best_interval_early_stopping = np.array(best_interval_early_stopping)
    interval_coverage_all_months = best_interval_early_stopping[:, 0]
    #Get weighted average also for interval
    best_interval_early_stopping =\
        np.sum(best_interval_early_stopping[:, 0] * month_weights)
    return best_cv_early_stopping, result_final_rms_avg, result_final_avg, \
        num_rounds_months, interval_coverage_all_months, \
        best_interval_early_stopping, final_preds_all_months


def lgbm_cv_residuals(train: pd.DataFrame,
                      labels: pd.Series,
                      real_labels: pd.Series,
                      num_boost_round: int,
                      num_boost_round_start: int,
                      early_stopping_rounds: int,
                      early_stopping_step: int,
                      issue_months: list,
                      years_cv: list,
                      year_range: bool,
                      train_feat_dict: dict,
                      params_dict: dict,
                      categorical: list,
                      min_max_site_id_dict: dict,
                      path_distr: str,
                      distr_perc_dict: dict) -> tuple[np.ndarray, float, float,
                                                      list, np.ndarray, float,
                                                      pd.DataFrame]:
    """
    Run LightGBM CV with early stopping, get distribution estimates and
    average the results. Perform additional clipping to model predictions.

    Args:
        train (pd.DataFrame): Whole training data before dividing into CV folds
        labels (pd.Series): Labels corresponding to train data using volume
            residuals
        real_labels (pd.Series): Labels corresponding to train data using volume
        num_boost_round (int): Maximum number of estimators used in LightGBM
            models. Model could be stopped earlier if early stopping is met
        num_boost_round_start (int): Number of estimators used in LightGBM
            model after which early stopping criterion starts (could be seen
            as the minimum number of model iterations)
        early_stopping_rounds (int): How many times in a row early stopping can
            be met before stopping training
        early_stopping_step (int): Number of iterations when early stopping is
            performed. 20 means that it's done every 20 iterations, i,e. after
            100, 120, 140, ... iters
        issue_months (list): Months to iterate over. Every month has its own
            model
        years_cv (list): A list with test years for different CV folds
        year_range (bool): Specifies if there could be many years in test data
            (True) or just one test data per fold (False)
        train_feat_dict (dict): Features to use for different months. Key
            indicates month and value the features
        params_dict (dict): LightGBM hyperparameters to use for different months.
            Key indicates month and value, a dictionary of hyperparameters
        categorical (list): Categorical features in the model
        min_max_site_id_dict (dict): Minimum and maximum historical volumes
            for given site_id. It contains different DataFrames for different
            LOOCV years, LOOCV years are used as the keys
        path_distr (str): Path to values of distribution estimate parameters
            per each site_id already with amendments to distributions. Different
            LOOCV years have different distribution files. They can be
            distinguished by year suffixes (for example if path is distr_final,
            file name for LOOCV year 2010 will be distr_final_2010)
        distr_perc_dict (dict): How much importance is given for distribution
            estimate. For 0.4 value, it's 40% (while LightGBM model is 60%).
            Different months use different distribution percentage (1 for
            January, 2 for February, ..., 7 for July)
    Returns:
        best_cv_early_stopping (np.ndarray): CV results from different months
            with number of model iterations for each month
        result_final_rms_avg (float): RMS CV results averaged over different
            months
        result_final_avg (float): CV results averaged over different months
        num_rounds_months (list): Number of model iterations for each month
        interval_coverage_all_months (np.ndarray): interval coverage results
            from different month for num_rounds_months iterations
        best_interval_early_stopping (float): interval coverage averaged over
            different months
        final_preds_all_months (pd.DataFrame): Predictions with best number
            of iterations from selected issue months
    """
    ###########################################################################
    #Global parameters and variables initialization
    ###########################################################################
    #Initialize empty variables
    #Initialize predictions for different folds
    final_preds_all_months = pd.DataFrame()

    cv_results_all_months = dict() #CV results from different months
    results_coverage_all_months = dict() #interval coverage from different months
    results_10 = []
    results_50 = []
    results_90 = []

    ###########################################################################
    #Iterate over months. Train one month at a time
    ###########################################################################
    for month_idx, month in tqdm(enumerate(issue_months)):
        print(f'\nMonth: {month}')
        #Initialize predictions for given month
        final_preds = pd.DataFrame()
        #Get distribution percentage to use for given month
        distr_perc = distr_perc_dict[month]
        #Initialize variables for the month
        #First evaluation done after num_boost_round_start iters
        num_boost_round_month = num_boost_round_start
        #Set previous value of averaged CV to infinity for first evaluation,
        #so the first evaluated value is always less
        cv_result_avg_fold_prev = np.inf
        #Initialize number of early stopping conditions met so far with 0
        num_prev_iters_better = 0
        #All [avg RMS fold result-avg fold result-number of LightGBM iterations]
        #from given month. All LGBM iters after each early_stopping_step have
        #a row in the list
        cv_results_avg_fold = []
        #Similarly for interval coverage
        results_coverage_avg_fold = []
        #Quantile 0.5 models for newest fold results
        lgb_models_50 = dict()
        #Quantile 0.1 models for newest fold results
        lgb_models_10 = dict()
        #Quantile 0.9 models for newest fold results
        lgb_models_90 = dict()
        #Set column names for previous naturalized flow sum
        if month == 5:
            nat_flow_sum_col = 'nat_flow_sum_Apr_Apr'
        elif month == 6:
            nat_flow_sum_col = 'nat_flow_sum_Apr_May'
        elif month == 7:
            nat_flow_sum_col = 'nat_flow_sum_Apr_Jun'

        #######################################################################
        #Start training. Train until early stopping/maximum number of iters met
        #######################################################################
        while (num_prev_iters_better < early_stopping_rounds) &\
              (num_boost_round_month <= num_boost_round):
            #Initialize variables for given iter
            results_10_clipped = []
            results_50_clipped = []
            results_90_clipped = []
            results_coverage = []
            cv_results = []
            ###################################################################
            #Iterate over different folds
            ###################################################################
            for fold, year in enumerate(years_cv):
                #Get indexes from train DataFrame for given fold's train and test
                train_cv_idxs, test_cv_idxs = get_cv_folds(train,
                                                           month,
                                                           years_cv,
                                                           year_range)
                #Choose features from given month
                train_feat = train_feat_dict[month]
                #Get params from given month
                params = params_dict[month]

                #Train/continue training model from given fold for Q0.5
                preds_50, lgb_models_50 = train_cv(train,
                                                   labels,
                                                   train_cv_idxs,
                                                   test_cv_idxs,
                                                   train_feat,
                                                   params,
                                                   categorical,
                                                   num_boost_round_start,
                                                   num_boost_round_month,
                                                   0.5,
                                                   fold,
                                                   lgb_models_50)
                #Train/continue training model from given fold for Q0.1.
                #Named preds_10_lgbm, as final preds_10 will use a weighted
                #average with distribution estimates
                preds_10_lgbm, lgb_models_10 = train_cv(train,
                                                        labels,
                                                        train_cv_idxs,
                                                        test_cv_idxs,
                                                        train_feat,
                                                        params,
                                                        categorical,
                                                        num_boost_round_start,
                                                        num_boost_round_month,
                                                        0.1,
                                                        fold,
                                                        lgb_models_10)
                #Train/continue training model from given fold for Q0.9.
                #Named preds_90_lgbm, as final preds_10 will use a weighted
                #average with distribution estimates
                preds_90_lgbm, lgb_models_90 = train_cv(train,
                                                        labels,
                                                        train_cv_idxs,
                                                        test_cv_idxs,
                                                        train_feat,
                                                        params,
                                                        categorical,
                                                        num_boost_round_start,
                                                        num_boost_round_month,
                                                        0.9,
                                                        fold,
                                                        lgb_models_90)
                #Get test rows from given fold with predictions
                result_df = train.loc[test_cv_idxs[fold], train_feat]
                result_df['volume_50'] = preds_50
                result_df['volume_10_lgbm'] = preds_10_lgbm
                result_df['volume_90_lgbm'] = preds_90_lgbm
                #Roll back to volume from volume residuals
                result_df = pd.merge(result_df,
                                     train.loc[
                                         (train.index.isin(test_cv_idxs[fold])),
                                         ['site_id',
                                          'issue_date_no_year',
                                          nat_flow_sum_col]],
                                     how = 'left',
                                     on = ['site_id', 'issue_date_no_year'])
                result_df['volume_50'] = result_df['volume_50'] +\
                    result_df[nat_flow_sum_col]
                result_df['volume_10_lgbm'] = result_df['volume_10_lgbm'] +\
                    result_df[nat_flow_sum_col]
                result_df['volume_90_lgbm'] = result_df['volume_90_lgbm'] +\
                    result_df[nat_flow_sum_col]
                #Append quantile results
                result_df = get_quantiles_from_distr(result_df,
                                                     min_max_site_id_dict[year],
                                                     all_distr_dict,
                                                     f'{path_distr}{year}')
                #Add min and max for site_id as 'max' and 'min' columns
                result_df = pd.merge(result_df,
                                     min_max_site_id_dict[year],
                                     how = 'left',
                                     left_on = 'site_id',
                                     right_index = True)
                #Change volume values greater than min (max) for site_id to
                #that min (max) value. Do it also for distribution volume.
                #Though distribution estimates shouldn't exceed maximum values,
                #do it just to be certain
                result_df.loc[result_df['volume_50'] < result_df['min'],
                              'volume_50'] = result_df['min']
                result_df.loc[result_df['volume_50'] > result_df['max'],
                              'volume_50'] = result_df['max']
                result_df.loc[result_df['volume_10_lgbm'] < result_df['min'],
                              'volume_10_lgbm'] = result_df['min']
                result_df.loc[result_df['volume_10_distr'] < result_df['min'],
                              'volume_10_distr'] = result_df['min']
                result_df.loc[result_df['volume_90_lgbm'] > result_df['max'],
                              'volume_90_lgbm'] = result_df['max']
                result_df.loc[result_df['volume_90_distr'] > result_df['max'],
                              'volume_90_distr'] = result_df['max']
                #Clipping:
                    #if volume_90 < volume_50 -> change volume_90 to volume_50
                    #if volume_50 < volume_10 -> change volume_10 to volume_50
                result_df.loc[result_df.volume_90_lgbm < result_df.volume_50,
                              'volume_90_lgbm'] = result_df.volume_50
                result_df.loc[result_df.volume_50 < result_df.volume_10_lgbm,
                              'volume_10_lgbm'] = result_df.volume_50
                result_df.loc[result_df.volume_90_distr < result_df.volume_50,
                              'volume_90_distr'] = result_df.volume_50
                result_df.loc[result_df.volume_50 < result_df.volume_10_distr,
                              'volume_10_distr'] = result_df.volume_50
                #Get weighted average from distributions and models for Q0.1
                #and Q0.9
                result_df['volume_10'] =\
                    distr_perc * result_df.volume_10_distr +\
                    (1 - distr_perc) * result_df.volume_10_lgbm
                result_df['volume_90'] =\
                    distr_perc * result_df.volume_90_distr +\
                    (1 - distr_perc) * result_df.volume_90_lgbm
                #Get quantile loss for given fold. Use real labels
                result_10 = mean_pinball_loss(real_labels[test_cv_idxs[fold]],
                                              result_df.volume_10,
                                              alpha = 0.1)
                result_50 = mean_pinball_loss(real_labels[test_cv_idxs[fold]],
                                              result_df.volume_50,
                                              alpha = 0.5)
                result_90 = mean_pinball_loss(real_labels[test_cv_idxs[fold]],
                                              result_df.volume_90,
                                              alpha = 0.9)
                #Append results from this fold
                results_10.append(result_10)
                results_50.append(result_50)
                results_90.append(result_90)

                ###############################################################
                #nat_flow_sum clipping
                ###############################################################
                results_clipped = result_df.copy()
                if month == 7:
                    results_clipped = nat_flow_sum_clipping(
                        train = train,
                        result_df = results_clipped,
                        nat_flow_sum_col = nat_flow_sum_col,
                        test_cv_idxs = test_cv_idxs,
                        fold = fold,
                        month = 7,
                        residuals = True,
                        multip_10 = 1.1,
                        multip_50 = 1.15,
                        multip_90 = 1.2,
                        multip_10_thres = 1.0,
                        multip_50_thres = 1.05,
                        multip_90_thres = 1.05)
                if month == 6:
                    results_clipped = nat_flow_sum_clipping(
                        train = train,
                        result_df = results_clipped,
                        nat_flow_sum_col = nat_flow_sum_col,
                        test_cv_idxs = test_cv_idxs,
                        fold = fold,
                        month = 6,
                        residuals = True,
                        multip_10 = 1.2,
                        multip_50 = 1.25,
                        multip_90 = 1.3,
                        multip_10_thres = 1.1,
                        multip_50_thres = 1.15,
                        multip_90_thres = 1.15,
                        multip_10_detroit = 1.2,
                        multip_50_detroit = 1.25,
                        multip_90_detroit = 1.3,
                        multip_10_thres_detroit = 1.1,
                        multip_50_thres_detroit = 1.15,
                        multip_90_thres_detroit = 1.15)
                if month == 5:
                    results_clipped = nat_flow_sum_clipping(
                        train = train,
                        result_df = results_clipped,
                        nat_flow_sum_col = nat_flow_sum_col,
                        test_cv_idxs = test_cv_idxs,
                        fold = fold,
                        month = 5,
                        residuals = True,
                        multip_10 = 1.3,
                        multip_50 = 1.35,
                        multip_90 = 1.4,
                        multip_10_thres = 1.2,
                        multip_50_thres = 1.25,
                        multip_90_thres = 1.25,
                        multip_10_detroit = 1.3,
                        multip_50_detroit = 1.35,
                        multip_90_detroit = 1.4,
                        multip_10_thres_detroit = 1.2,
                        multip_50_thres_detroit = 1.25,
                        multip_90_thres_detroit = 1.25)

                ###############################################################
                #Final clipping
                ###############################################################
                #Do the final clipping to make sure that the restrictions are
                #met after taking weighted average for volume_10 and volume_90
                results_clipped.loc[results_clipped.volume_90 < results_clipped.volume_50,
                                    'volume_50'] = results_clipped.volume_90
                results_clipped.loc[results_clipped.volume_50 < results_clipped.volume_10,
                                    'volume_10'] = results_clipped.volume_50
                #Get quantile loss for given fold. Use real labels
                result_10_clipped = mean_pinball_loss(real_labels[test_cv_idxs[fold]],
                                                      results_clipped.volume_10,
                                                      alpha = 0.1)
                result_50_clipped = mean_pinball_loss(real_labels[test_cv_idxs[fold]],
                                                      results_clipped.volume_50,
                                                      alpha = 0.5)
                result_90_clipped = mean_pinball_loss(real_labels[test_cv_idxs[fold]],
                                                      results_clipped.volume_90,
                                                      alpha = 0.9)
                #Append results from this fold
                results_10_clipped.append([fold, result_10_clipped, num_boost_round_month])
                results_50_clipped.append([fold, result_50_clipped, num_boost_round_month])
                results_90_clipped.append([fold, result_90_clipped, num_boost_round_month])
                #Get competition metric
                cv_result = 2 * (result_10_clipped +
                                 result_50_clipped +
                                 result_90_clipped) / 3
                #Append the result from given fold-model iteration
                cv_results.append([cv_result])
                #Get interval coverage for given month
                different_volumes = ['volume_10', 'volume_50', 'volume_90']
                result_coverage =\
                    interval_coverage(np.array(real_labels[test_cv_idxs[fold]]),
                                      np.array(results_clipped[different_volumes]))
                results_coverage.append(result_coverage)

                #Get predictions
                preds_with_volumes = results_clipped[['site_id',
                                                      'volume_10',
                                                      'volume_50',
                                                      'volume_90',
                                                      'issue_date_no_year']]
                #Append NUM_BOOST_ROUND_MONTH
                preds_with_volumes['num_boost_rounds'] = num_boost_round_month
                #Append year
                preds_with_volumes['year'] = year
                #Append predictions
                final_preds = pd.concat([final_preds, preds_with_volumes])
            #Average results over different folds for given model iteration
            cv_result_avg_fold = np.mean(cv_results)
            #Average RMS (root mean square) results over different folds for
            #given model iteration
            cv_result_rms = np.array(cv_results) ** 2
            cv_result_avg_fold_rms =\
                np.sqrt(np.sum(cv_result_rms) / len(cv_result_rms))

            #Do the same for interval coverage
            result_coverage_avg_fold = np.mean(results_coverage)
            #Keep track of early stopping condition if result is poorer than
            #in the previous early stopping check (early_stopping_step before)
            if cv_result_avg_fold_rms > cv_result_avg_fold_prev:
                num_prev_iters_better += 1
            else:
                cv_result_avg_fold_prev = cv_result_avg_fold_rms
                #If new result is better, use new result in next early stopping
                #check. Reset num_prev_iters_better to 0
                num_prev_iters_better = 0

            print(f'Avg RMS result all folds for {num_boost_round_month} trees:',
                  cv_result_avg_fold_rms)
            print(f'Avg result all folds for {num_boost_round_month} trees:',
                  cv_result_avg_fold)
            print(f'Avg interval coverage all folds for {num_boost_round_month} trees:',
                  result_coverage_avg_fold)
            #Append number of boosting iterations to average results
            cv_results_avg_fold.append([cv_result_avg_fold_rms,
                                        cv_result_avg_fold,
                                        num_boost_round_month])
            #Do the same for interval coverage
            results_coverage_avg_fold.append([result_coverage_avg_fold,
                                              num_boost_round_month])
            #Update information when next early stopping will be evaluated
            num_boost_round_month += early_stopping_step

        #######################################################################
        #Early stopping/maximum number of iterations (num_boost_round) met
        #for the selected month
        #######################################################################
        #Update results if last values weren't the best ones. Maximum value
        #is chosen as the final value
        if num_prev_iters_better != 0:
            cv_results_avg_fold = cv_results_avg_fold[:-num_prev_iters_better]
            results_coverage_avg_fold =\
                results_coverage_avg_fold[:-num_prev_iters_better]
            #Keep only predictions from best num_boost_rounds
            final_preds = final_preds[
                final_preds.num_boost_rounds == num_boost_round_month -
                (1 + num_prev_iters_better) * early_stopping_step]
        cv_results_all_months[month] = cv_results_avg_fold
        results_coverage_all_months[month] = results_coverage_avg_fold
        #Append predictions from given month
        final_preds_all_months = pd.concat([final_preds_all_months, final_preds])
    ###########################################################################
    #All months were trained. Get final results to return
    ###########################################################################
    #Get best fit per month with best number of iterations
    best_cv_early_stopping = []
    for month in cv_results_all_months.keys():
        best_cv_early_stopping.append(cv_results_all_months[month][-1])
    best_cv_early_stopping = np.array(best_cv_early_stopping)
    #Average best fits over months. detroit_lake_inflow predictions are for
    #Apr-Jun period, there aren't any values for this site_id for July, so
    #there's a need for correction

    #Add month importance. July has less importance, as 25 out of 26 site_ids
    #have values for this month
    if len(issue_months) == 7:
        month_importance = np.array([1, 1, 1, 1, 1, 1, 25/26])
    else:
        #If not all months are being evaluated, add the same weights for each
        #month for simplicity
        month_importance = np.ones(len(issue_months))

    #Sum weights
    sum_weights = np.sum(month_importance)
    #Ratio of importance to sum of all weights
    month_weights = month_importance / sum_weights
    #Sum of results for different months multiplied by weights to get
    #a weighted average over different months
    result_final_rms_avg = np.sum(best_cv_early_stopping[:, 0] * month_weights)
    result_final_avg = np.sum(best_cv_early_stopping[:, 1] * month_weights)
    #Get optimal number of rounds for each month separately
    num_rounds_months = list(best_cv_early_stopping[:, 2].astype('int'))
    #Get interval coverage from the best iteration
    best_interval_early_stopping = []
    for month in cv_results_all_months.keys():
        best_interval_early_stopping.append(results_coverage_all_months[month][-1])
    best_interval_early_stopping = np.array(best_interval_early_stopping)
    interval_coverage_all_months = best_interval_early_stopping[:, 0]
    #Get weighted average also for interval
    best_interval_early_stopping =\
        np.sum(best_interval_early_stopping[:, 0] * month_weights)
    return best_cv_early_stopping, result_final_rms_avg, result_final_avg, \
        num_rounds_months, interval_coverage_all_months, \
        best_interval_early_stopping, final_preds_all_months


def objective(trial: optuna.trial.Trial,
              train: pd.DataFrame,
              labels: pd.Series,
              month: int,
              years_cv: list,
              year_range: bool,
              train_feat: list,
              categorical: list,
              min_max_site_id_dict: dict,
              path_distr: str,
              distr_perc_dict: dict,
              num_boost_round: int,
              num_boost_round_start: int,
              early_stopping_rounds: int,
              early_stopping_step: int,
              final_tuning: bool,
              residuals: bool,
              real_labels: pd.Series,
              no_nat_flow_sites: bool = False) -> float:
    """
    Set logic for optuna hyperparameters tuning, set range of values for
    different hyperparameters, append CV evaluation.

    Args:
        trial (optuna.trial.Trial): A process of evaluating an objective
            function. This object is passed to an objective function and
            provides interfaces to get parameter suggestion, manage the trial’s
            state, and set/get user-defined attributes of the trial
            (https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial)
        train (pd.DataFrame): Whole training data before dividing into CV folds
        labels (pd.Series): Labels corresponding to train data
        month (int): Month to optimize hyperparameters for
        years_cv (list): A list with test years for different CV folds
        year_range (bool): Specifies if there could be many years in test data
            (True) or just one test data per fold (False)
        train_feat (list): Features to use for this month
        categorical (list): Categorical features in the model
        min_max_site_id_dict (dict): Minimum and maximum historical volumes
            for given site_id. It contains different DataFrames for different
            LOOCV years, LOOCV years are used as the keys
        path_distr (str): Path to values of distribution estimate parameters
            per each site_id already with amendments to distributions. Different
            LOOCV years have different distribution files. They can be
            distinguished by year suffixes (for example if path is distr_final,
            file name for LOOCV year 2010 will be distr_final_2010)
        distr_perc_dict (dict): How much importance is given for distribution
            estimate. For 0.4 value, it's 40% (while LightGBM model is 60%).
            Different months use different distribution percentage (1 for
            January, 2 for February, ..., 7 for July)
        num_boost_round (int): Maximum number of estimators used in LightGBM
            models. Model could be stopped earlier if early stopping is met
        num_boost_round_start (int): Number of estimators used in LightGBM
            model after which early stopping criterion starts (could be seen
            as the minimum number of model iterations)
        early_stopping_rounds (int): How many times in a row early stopping can
            be met before stopping training
        early_stopping_step (int): Number of iterations when early stopping is
            performed. 20 means that it's done every 20 iterations, i,e. after
            100, 120, 140, ... iters
        final_tuning (bool): Indicates if it's an initial tuning (False) for
            the given month with a wider range of hyperparameters or final
            (True) with new range after manual examination of initial results
        residuals (bool): Informs if predictions are made for volume residuals
        real_labels (pd.Series): Labels corresponding to train data using volume
        no_nat_flow_sites (bool): Indicates if it's an optimization pipeline for
            3 site_ids without naturalized flow columns
    Returns:
        best_cv_avg (float): Score for this optimization iteration
    """
    #Set repetitive parameters created in model_params.py
    BAGGING_FREQ, OBJECTIVE, METRIC, VERBOSE, REG_ALPHA, MIN_GAIN_TO_SPLIT, \
        MIN_SUM_HESSIAN_IN_LEAF, FEATURE_FRACTION_SEED, SEED =\
        joblib.load(Path('data') / 'general_hyperparams_final.pkl')
    #Set minimal number of columns to one less than the number of columns.
    feature_fraction_min = (len(train_feat) - 1) / len(train_feat)
    if no_nat_flow_sites == False:
        if month in [1, 2, 3]:
            #Use more conservative hyperparameters for early months
            params = {'objective': OBJECTIVE,
                      'metric': METRIC,
                      'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.15),
                      'max_depth': trial.suggest_int('max_depth', 4, 9),
                      'num_leaves': trial.suggest_int('num_leaves', 16, 64),
                      'lambda_l1': REG_ALPHA,
                      'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 20.0, log = True),
                      'min_gain_to_split': MIN_GAIN_TO_SPLIT,
                      'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                      'bagging_freq': BAGGING_FREQ,
                      'bagging_seed': FEATURE_FRACTION_SEED,
                      'feature_fraction': trial.suggest_float('feature_fraction',
                                                              feature_fraction_min,
                                                              1.0,
                                                              step = 1 - feature_fraction_min),
                      'feature_fraction_seed': FEATURE_FRACTION_SEED,
                      'max_bin': trial.suggest_int('max_bin', 100, 300),
                      'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 15, 30),
                      'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
                      'verbose': VERBOSE,
                      'seed': SEED}
        #Add scenario for final tuning. Only July used final tuning
        elif month == 7 and final_tuning == True:
            params = {'objective': OBJECTIVE,
                      'metric': METRIC,
                      'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.08),
                      'max_depth': trial.suggest_int('max_depth', 8, 10),
                      'num_leaves': trial.suggest_int('num_leaves', 77, 128),
                      'lambda_l1': REG_ALPHA,
                      'lambda_l2': trial.suggest_float('lambda_l2', 0.02, 3.0, log = True),
                      'min_gain_to_split': MIN_GAIN_TO_SPLIT,
                      'subsample': trial.suggest_float('subsample', 0.75, 0.9),
                      'bagging_freq': BAGGING_FREQ,
                      'bagging_seed': FEATURE_FRACTION_SEED,
                      'feature_fraction': trial.suggest_float('feature_fraction',
                                                              feature_fraction_min,
                                                              1.0,
                                                              step = 1 - feature_fraction_min),
                      'feature_fraction_seed': FEATURE_FRACTION_SEED,
                      'max_bin': trial.suggest_int('max_bin', 150, 230),
                      'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 17, 26),
                      'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
                      'verbose': VERBOSE,
                      'seed': SEED}
        else:
            #Use wider range of values for hyperparameters tuning for later
            #months
            params = {'objective': OBJECTIVE,
                      'metric': METRIC,
                      'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                      'max_depth': trial.suggest_int('max_depth', 4, 10),
                      'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                      'lambda_l1': REG_ALPHA,
                      'lambda_l2': trial.suggest_float('lambda_l2', 0.001, 10.0, log = True),
                      'min_gain_to_split': MIN_GAIN_TO_SPLIT,
                      'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                      'bagging_freq': BAGGING_FREQ,
                      'bagging_seed': FEATURE_FRACTION_SEED,
                      'feature_fraction': trial.suggest_float('feature_fraction',
                                                              feature_fraction_min,
                                                              1.0,
                                                              step = 1 - feature_fraction_min),
                      'feature_fraction_seed': FEATURE_FRACTION_SEED,
                      'max_bin': trial.suggest_int('max_bin', 150, 300),
                      'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 15, 25),
                      'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
                      'verbose': VERBOSE,
                      'seed': SEED}
    else:
        #Use more conservative range of hyperparameters for 26 site ids training
        #with only 3 site ids without naturalized flow columns being evaluated
        params = {'objective': OBJECTIVE,
                  'metric': METRIC,
                  'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.08),
                  'max_depth': trial.suggest_int('max_depth', 6, 9),
                  'num_leaves': trial.suggest_int('num_leaves', 50, 128),
                  'lambda_l1': REG_ALPHA,
                  'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 5.0),
                  'min_gain_to_split': MIN_GAIN_TO_SPLIT,
                  'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                  'bagging_freq': BAGGING_FREQ,
                  'bagging_seed': FEATURE_FRACTION_SEED,
                  'feature_fraction': trial.suggest_float('feature_fraction',
                                                          feature_fraction_min,
                                                          1.0,
                                                          step = 1 - feature_fraction_min),
                  'feature_fraction_seed': FEATURE_FRACTION_SEED,
                  'max_bin': trial.suggest_int('max_bin', 100, 160),
                  'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 15, 30),
                  'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
                  'verbose': VERBOSE,
                  'seed': SEED}
    #Change params to params_dict dictionary
    params_dict = {month: params}
    #Change features to dictionary
    train_feat_dict = {month: train_feat}
    #Change month type to list
    month = [month]
    #Perform CV calculation
    if residuals == True:
        #CV pipeline for volume residuals calculation for 23 site ids
        best_cv_per_month, best_cv_avg_rms, best_cv_avg, num_rounds_months, \
            interval_coverage_all_months, best_interval_early_stopping, \
            final_preds =\
            lgbm_cv_residuals(train,
                              labels,
                              real_labels,
                              num_boost_round,
                              num_boost_round_start,
                              early_stopping_rounds,
                              early_stopping_step,
                              month,
                              years_cv,
                              year_range,
                              train_feat_dict,
                              params_dict,
                              categorical,
                              min_max_site_id_dict,
                              path_distr,
                              distr_perc_dict)
    else:
        #CV pipeline for volume calculation
        best_cv_per_month, best_cv_avg_rms, best_cv_avg, num_rounds_months, \
            interval_coverage_all_months, best_interval_early_stopping, \
            final_preds =\
            lgbm_cv(train,
                    labels,
                    num_boost_round,
                    num_boost_round_start,
                    early_stopping_rounds,
                    early_stopping_step,
                    month,
                    years_cv,
                    year_range,
                    train_feat_dict,
                    params_dict,
                    categorical,
                    min_max_site_id_dict,
                    path_distr,
                    distr_perc_dict,
                    no_nat_flow_sites)
    #Set additional columns to save
    trial.set_user_attr("best_result_without_rms", best_cv_avg)
    trial.set_user_attr("num_boost_rounds_best", num_rounds_months[0])
    trial.set_user_attr("interval_coverage", best_interval_early_stopping)
    return best_cv_avg_rms


def interval_coverage(actual: np.ndarray,
                      predicted: np.ndarray) -> float:
    """
    Calculates interval coverage for quantile predictions. Assumes at least two
    columns in `predicted`, and that the first column is the lower bound of
    the interval, and the last column is the upper bound of the interval.
    Taken from https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/blob/main/scoring/score.py.

    Args:
        actual (np.ndarray): Array of actual values (labels)
        predicted (np.ndarray): Array of predicted values
    Returns:
        interval_result (float): Interval coverage (proportion of predictions
            that fall within lower and upper bound)
    """
    # Use ravel to reshape to 1D arrays.
    lower = predicted[:, 0].ravel()
    upper = predicted[:, -1].ravel()
    actual = actual.ravel()
    interval_result = np.average((lower <= actual) & (actual <= upper))
    return interval_result
