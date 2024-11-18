import pandas as pd
import joblib
import pickle
from sklearn.metrics import mean_pinball_loss
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
from pathlib import Path

from train_utils import to_categorical, train_pipeline, save_models
from cv_and_hyperparams_opt import get_years_cv, lgbm_cv, lgbm_cv_residuals,\
    objective
from utils import ReadAllData

import time

start = time.time()

#Set types of training to run
RUN_FINAL_SOLUTION = True #Runs code required to obtain the final results
if RUN_FINAL_SOLUTION:
    RUN_CV = True
    RUN_HYPERPARAMS_TUNING = False
    RUN_TRAINING = False
    #Set training parameters. Jan-Apr should use False values for RESIDUALS and
    #NO_NAT_FLOW_SITES.
    #May-Jul for 23 site ids with naturalized flow history should use True value
    #for RESIDUALS and False for NO_NAT_FLOW_SITES.
    #May-Jul for 3 site ids without naturalized flow history should use True value
    #for NO_NAT_FLOW_SITES and False value for RESIDUALS

    #Set if that's a final tuning. There are 2 types, initial one, where wide
    #range of hyperparams is used to determine area where results are optimal
    #and final one (used only for July in the final solution), where based on
    #the results from the initial tuning, hyperparams space is narrowed
    FINAL_TUNING = False
    #Set if volume residuals should be used in training
    RESIDUALS = True
    #Set if training is for 3 site_ids without naturalized flow features
    NO_NAT_FLOW_SITES = True
else:
    #Make changes only if it isn't for the final solution
    RUN_CV = True
    RUN_HYPERPARAMS_TUNING = False
    RUN_TRAINING = False
    #Set training parameters. Jan-Apr should use False values for RESIDUALS and
    #NO_NAT_FLOW_SITES.
    #May-Jul for 23 site ids with naturalized flow history should use True value
    #for RESIDUALS and False for NO_NAT_FLOW_SITES.
    #May-Jul for 3 site ids without naturalized flow history should use True value
    #for NO_NAT_FLOW_SITES and False value for RESIDUALS

    #Set if that's a final tuning. There are 2 types, initial one, where wide
    #range of hyperparams is used to determine area where results are optimal
    #and final one (used only for July in the final solution), where based on
    #the results from the initial tuning, hyperparams space is narrowed
    FINAL_TUNING = False
    #Set if volume residuals should be used in training
    RESIDUALS = False
    #Set if training is for 3 site_ids without naturalized flow features
    NO_NAT_FLOW_SITES = False

#Read data
train = pd.read_pickle(Path('data') / 'train_test_final.pkl')
params_dict = joblib.load(Path('data') / 'lgbm_model_params_final.pkl')
feat_dict = joblib.load(Path('data') / 'lgbm_model_feats_final.pkl')

if NO_NAT_FLOW_SITES == True:
    #Get features
    feat_dict_no_nat_flow =\
        joblib.load(Path('data') / 'lgbm_model_feats_no_nat_flow_final.pkl')
    #Get params
    params_dict_no_nat_flow =\
        joblib.load(Path('data') / 'lgbm_model_params_no_nat_flow.pkl')
    #Set small number of hyperparams optimization iterations. It's only
    #for 3 site_ids, so it doesn't have to be bigger
    distr_perc_dict_no_nat_flow = {5: 0.25,
                                   6: 0.15,
                                   7: 0.1}

#Different models are created for different months
issue_months = [1, 2, 3, 4, 5, 6, 7]
#Number of boost rounds for following months. Taken from hyperparameters
#optimization. IF CV is run, it is overwritten with CV results
num_rounds_months = [140, 240, 400, 340, 800, 740, 1000]
#Set eval to show training results every EVAL step
EVAL = 100

#Initiate and read data for CV/hyperparams tuning
if RUN_CV == True or RUN_HYPERPARAMS_TUNING == True:
    #Read min-max values of volume per site_id
    min_max_site_id = joblib.load(Path('data') / 'min_max_site_id_dict_final.pkl')
    #Path to different distributions for different site_ids. Distributions
    #for different LOOCV years are in different files with year suffix
    PATH_DISTR = str(Path("data") / "distr" / "distr_final_")
    #Set importance for distribution in quantiles 0.1 and 0.9 calculation for
    #different months
    distr_perc_dict = {1: 0.6,
                       2: 0.5,
                       3: 0.45,
                       4: 0.3,
                       5: 0.25,
                       6: 0.15,
                       7: 0.05}
    #Use many years in a fold or not
    YEAR_RANGE = False
    #Initiate early stopping from this iteration
    NUM_BOOST_ROUND_START = 100
    #How many times in a row early stopping can be met before stopping training
    EARLY_STOPPING_ROUNDS = 3
    #Number of iterations when Early stopping is performed. 20 means that it's
    #done every 20 iterations, i,e. after 100, 120, 140, ... iters
    EARLY_STOPPING_STEP = 10
    #Get years for different CV folds
    years_cv = get_years_cv(YEAR_RANGE)

#Initiate lists for storing models for given quantile for different months
lgb_models_10 = []
lgb_models_50 = []
lgb_models_90 = []
#Change types to 'category' in categorical columns
categorical = ['site_id']
train = to_categorical(['site_id'],
                       train)
#Get labels
labels = train['volume']

#Select data concerning volume residuals
if RESIDUALS == True:
    train.loc[(train.month == 7) & ~(train.site_id.isin(['american_river_folsom_lake',
                                                         'san_joaquin_river_millerton_reservoir',
                                                         'merced_river_yosemite_at_pohono_bridge'])),
              'volume_residuals'] = train.volume - train.nat_flow_sum_Apr_Jun
    train.loc[(train.month == 6) & ~(train.site_id.isin(['american_river_folsom_lake',
                                                         'san_joaquin_river_millerton_reservoir',
                                                         'merced_river_yosemite_at_pohono_bridge'])),
              'volume_residuals'] = train.volume - train.nat_flow_sum_Apr_May
    train.loc[(train.month == 5) & ~(train.site_id.isin(['american_river_folsom_lake',
                                                         'san_joaquin_river_millerton_reservoir',
                                                         'merced_river_yosemite_at_pohono_bridge'])),
              'volume_residuals'] = train.volume - train.nat_flow_sum_Apr_Apr
    #Set site ids without naturalized flow history
    site_ids_no_nat_flow = ['american_river_folsom_lake',
                            'san_joaquin_river_millerton_reservoir',
                            'merced_river_yosemite_at_pohono_bridge']
    #Remove site_ids without naturalized flow history
    train_23 = train[~train.site_id.isin(site_ids_no_nat_flow)].reset_index(drop = True)
    #Get full volume
    labels_23 = train_23['volume']
    #Get volume residuals
    labels_residuals = train_23['volume_residuals']

#LOOCV
#Maximum number of LightGBM model iterations
NUM_BOOST_ROUND = 2000
if RUN_FINAL_SOLUTION:
    #Initialize final predictions
    preds_final = pd.DataFrame()
    #Run Jan-Apr models
    issue_months = [1, 2, 3, 4]
    best_cv_per_month, best_cv_avg_rms, best_cv_avg, num_rounds_months, \
        interval_coverage_all_months, best_interval_early_stopping, preds =\
            lgbm_cv(train,
                    labels,
                    NUM_BOOST_ROUND,
                    NUM_BOOST_ROUND_START,
                    EARLY_STOPPING_ROUNDS,
                    EARLY_STOPPING_STEP,
                    issue_months,
                    years_cv,
                    YEAR_RANGE,
                    feat_dict,
                    params_dict,
                    categorical,
                    min_max_site_id,
                    PATH_DISTR,
                    distr_perc_dict)
    #Append preds
    preds_final = pd.concat([preds_final, preds])

    #Run May-Jul models for 23 site ids with naturalized flow history
    issue_months = [5, 6, 7]
    best_cv_per_month, best_cv_avg_rms, best_cv_avg, num_rounds_months, \
        interval_coverage_all_months, best_interval_early_stopping, preds =\
            lgbm_cv_residuals(train_23,
                              labels_residuals,
                              labels_23,
                              NUM_BOOST_ROUND,
                              NUM_BOOST_ROUND_START,
                              EARLY_STOPPING_ROUNDS,
                              EARLY_STOPPING_STEP,
                              issue_months,
                              years_cv,
                              YEAR_RANGE,
                              feat_dict,
                              params_dict,
                              categorical,
                              min_max_site_id,
                              PATH_DISTR,
                              distr_perc_dict)
    #Append preds
    preds_final = pd.concat([preds_final, preds])

    #Run May-Jul models for 3 site ids without naturalized flow history
    best_cv_per_month, best_cv_avg_rms, best_cv_avg, num_rounds_months, \
        interval_coverage_all_months, best_interval_early_stopping, preds =\
            lgbm_cv(train,
                    labels,
                    NUM_BOOST_ROUND,
                    NUM_BOOST_ROUND_START,
                    EARLY_STOPPING_ROUNDS,
                    EARLY_STOPPING_STEP,
                    issue_months,
                    years_cv,
                    YEAR_RANGE,
                    feat_dict_no_nat_flow,
                    params_dict_no_nat_flow,
                    categorical,
                    min_max_site_id,
                    PATH_DISTR,
                    distr_perc_dict_no_nat_flow,
                    NO_NAT_FLOW_SITES)
    #Append preds
    preds_final = pd.concat([preds_final, preds])

    #Get submission format
    #Get cross-validation LB format for predictions
    submission = pd.read_csv(Path('data') / 'cross_validation_submission_format.csv')
    #Get issue date encoding
    with open(Path('data') / "issue_date_encoded", "rb") as fp:
        issue_date_encoded = pickle.load(fp)
    #Get month_day and year
    submission['month_day'] = submission.issue_date.str[5:]
    submission['year'] = submission.issue_date.str[:4]
    submission['year'] = submission.year.astype('int')
    #Encode issue dates to get issue_date_no_year from training
    submission['issue_date_no_year'] = submission.month_day.map(issue_date_encoded)
    #Drop volume columns that will be replaced with predictions
    submission.drop(['volume_10', 'volume_50', 'volume_90'],
                    axis = 1, inplace = True)

    #Merge submission format with predictions
    submission = pd.merge(
        submission[['site_id', 'issue_date', 'issue_date_no_year', 'year']],
        preds_final[['site_id', 'issue_date_no_year', 'year',
                     'volume_10', 'volume_50', 'volume_90']],
        how = 'left',
        on = ['site_id', 'issue_date_no_year', 'year'])
    #Remove redundant features
    submission.drop(['issue_date_no_year', 'year'], axis = 1, inplace = True)
    #Save results
    submission.to_pickle('results/submission_2024_03_28.pkl')
    submission.to_csv('results/submission_2024_03_28.csv', index=False)

    #Merge with real train volumes for validation
    dfs = ReadAllData()
    train_real_volumes = dfs.train.copy()
    submission['year'] = submission['issue_date'].str[:4].astype('int')
    #Merge predictions with real volumes
    submission_with_volumes = pd.merge(submission,
                                       train_real_volumes,
                                       how = 'left',
                                       on = ['site_id', 'year'])
    #Get quantile loss for each quantile
    result_10 = mean_pinball_loss(submission_with_volumes.volume,
                                  submission_with_volumes.volume_10,
                                  alpha = 0.1)
    result_50 = mean_pinball_loss(submission_with_volumes.volume,
                                  submission_with_volumes.volume_50,
                                  alpha = 0.5)
    result_90 = mean_pinball_loss(submission_with_volumes.volume,
                                  submission_with_volumes.volume_90,
                                  alpha = 0.9)
    #Get final metric
    cv_result = 2 * (result_10 + result_50 + result_90) / 3
    print(cv_result)
else:
    #Run LOOCV based on the selected parameters
    if RUN_CV == True:
        if NO_NAT_FLOW_SITES == False:
            if RESIDUALS == True:
                #Volume residuals CV
                best_cv_per_month, best_cv_avg_rms, best_cv_avg, num_rounds_months, \
                    interval_coverage_all_months, best_interval_early_stopping, preds =\
                        lgbm_cv_residuals(train_23,
                                          labels_residuals,
                                          labels_23,
                                          NUM_BOOST_ROUND,
                                          NUM_BOOST_ROUND_START,
                                          EARLY_STOPPING_ROUNDS,
                                          EARLY_STOPPING_STEP,
                                          issue_months,
                                          years_cv,
                                          YEAR_RANGE,
                                          feat_dict,
                                          params_dict,
                                          categorical,
                                          min_max_site_id,
                                          PATH_DISTR,
                                          distr_perc_dict)
            else:
                #Basic CV
                best_cv_per_month, best_cv_avg_rms, best_cv_avg, num_rounds_months, \
                    interval_coverage_all_months, best_interval_early_stopping, preds =\
                        lgbm_cv(train,
                                labels,
                                NUM_BOOST_ROUND,
                                NUM_BOOST_ROUND_START,
                                EARLY_STOPPING_ROUNDS,
                                EARLY_STOPPING_STEP,
                                issue_months,
                                years_cv,
                                YEAR_RANGE,
                                feat_dict,
                                params_dict,
                                categorical,
                                min_max_site_id,
                                PATH_DISTR,
                                distr_perc_dict)
        else:
            #NO_NAT_FLOW_SITES CV
            best_cv_per_month, best_cv_avg_rms, best_cv_avg, num_rounds_months, \
                interval_coverage_all_months, best_interval_early_stopping, preds =\
                    lgbm_cv(train,
                            labels,
                            NUM_BOOST_ROUND,
                            NUM_BOOST_ROUND_START,
                            EARLY_STOPPING_ROUNDS,
                            EARLY_STOPPING_STEP,
                            issue_months,
                            years_cv,
                            YEAR_RANGE,
                            feat_dict_no_nat_flow,
                            params_dict_no_nat_flow,
                            categorical,
                            min_max_site_id,
                            PATH_DISTR,
                            distr_perc_dict_no_nat_flow,
                            NO_NAT_FLOW_SITES)
        print('CV result avg over months:', best_cv_avg)
        print('CV results per month with number of iters:', best_cv_per_month)
        print('Number of iters per month:', num_rounds_months)
        print('Interval coverage per month:', interval_coverage_all_months)
        print('Average interval coverage:', best_interval_early_stopping)

        #Get cross-validation LB format for predictions
        submission = pd.read_csv(Path('data') / 'cross_validation_submission_format.csv')
        #Get issue date encoding
        with open(Path('data') / "issue_date_encoded", "rb") as fp:
            issue_date_encoded = pickle.load(fp)
        #Get month_day and year
        submission['month_day'] = submission.issue_date.str[5:]
        submission['year'] = submission.issue_date.str[:4]
        submission['year'] = submission.year.astype('int')
        #Encode issue dates to get issue_date_no_year from training
        submission['issue_date_no_year'] = submission.month_day.map(issue_date_encoded)
        #Drop volume columns that will be replaced with predictions
        submission.drop(['volume_10', 'volume_50', 'volume_90'],
                        axis = 1, inplace = True)
        #Merge submission format with predictions
        submission = pd.merge(
            submission[['site_id', 'issue_date', 'issue_date_no_year', 'year']],
            preds[['site_id', 'issue_date_no_year', 'volume_10', 'volume_50', 'volume_90', 'year']],
            how = 'left',
            on = ['site_id', 'issue_date_no_year', 'year'])

        #Keep only columns for submission
        submission.drop(['issue_date_no_year', 'year'], axis = 1, inplace = True)
        #Save submission based on type of training
        if RESIDUALS == False and NO_NAT_FLOW_SITES == False:
            submission.to_csv('submission_volume_26_site_ids.csv', index = False)
        elif RESIDUALS == True and NO_NAT_FLOW_SITES == False:
            submission.to_csv('submission_residuals_23_site_ids.csv', index = False)
        elif RESIDUALS == False and NO_NAT_FLOW_SITES == True:
            submission.to_csv('submission_volume_3_site_ids.csv', index = False)

#Hyperparameters tuning
if RUN_HYPERPARAMS_TUNING == True:
    if FINAL_TUNING == True:
        path_suff = 'final'
    else:
        path_suff = 'initial'
    #Maximum number of LightGBM model iterations
    NUM_BOOST_ROUND = 2000
    #All months should take 35-50 hours to run. It is recommended to optimize
    #one month at a time
    for issue_month in tqdm(issue_months):
        if NO_NAT_FLOW_SITES == False:
            #Choose features from given month
            if issue_month in [1, 2]:
                #Jan/Feb number of trials
                if FINAL_TUNING == True:
                    N_TRIALS = 80
                else:
                    N_TRIALS = 150
            elif issue_month in [3, 4]:
                #Mar/Apr number of trials. It's a little smaller, as they
                #should take longer
                if FINAL_TUNING == True:
                    N_TRIALS = 70
                else:
                    N_TRIALS = 130
            else:
                #May-Jul number of trials for 23 site ids with naturalized flow
                #history. May-July months have strong prediction power, so it
                #isn't that important to train them for long. They also take
                #more time for training 1 iteration and there is additionally
                #training for 3 site ids to run
                if FINAL_TUNING == True:
                    N_TRIALS = 50
                else:
                    N_TRIALS = 60
            train_feat = feat_dict[issue_month]
            sampler = TPESampler(seed = 22) #to get the same results all the time
            #Perform hyperparameters tuning. Ranges of values to select from are
            #already chosen in the function
            study = optuna.create_study(direction = 'minimize', sampler = sampler)
            if RESIDUALS == True:
                study.optimize(lambda trial: objective(trial,
                                                       train_23,
                                                       labels_residuals,
                                                       issue_month,
                                                       years_cv,
                                                       YEAR_RANGE,
                                                       train_feat,
                                                       categorical,
                                                       min_max_site_id,
                                                       PATH_DISTR,
                                                       distr_perc_dict,
                                                       NUM_BOOST_ROUND,
                                                       NUM_BOOST_ROUND_START,
                                                       EARLY_STOPPING_ROUNDS,
                                                       EARLY_STOPPING_STEP,
                                                       FINAL_TUNING,
                                                       RESIDUALS,
                                                       labels_23),
                               n_trials = N_TRIALS)
            else:
                study.optimize(lambda trial: objective(trial,
                                                       train,
                                                       labels,
                                                       issue_month,
                                                       years_cv,
                                                       YEAR_RANGE,
                                                       train_feat,
                                                       categorical,
                                                       min_max_site_id,
                                                       PATH_DISTR,
                                                       distr_perc_dict,
                                                       NUM_BOOST_ROUND,
                                                       NUM_BOOST_ROUND_START,
                                                       EARLY_STOPPING_ROUNDS,
                                                       EARLY_STOPPING_STEP,
                                                       FINAL_TUNING,
                                                       RESIDUALS,
                                                       pd.Series([])),
                               n_trials = N_TRIALS)
            #Save study from given month
            joblib.dump(study,
                        f"results/hyperparams_tuning/study_{path_suff}_{pd.to_datetime('today').strftime('%Y_%m_%d_%H_%M_%S')}_month_{issue_month}.pkl")
        else:
            #May-Jul number of trials for 3 site ids. The number of iterations
            #is smaller, as it is only for 3 site ids predictions but still
            #requiring training on all 26 site ids.
            N_TRIALS = 40
            sampler = TPESampler(seed = 22) #to get the same results all the time
            #Perform hyperparameters tuning. Ranges of values to select from are
            #already chosen in the function
            study = optuna.create_study(direction = 'minimize', sampler = sampler)
            study.optimize(lambda trial: objective(trial,
                                                   train,
                                                   labels,
                                                   issue_month,
                                                   years_cv,
                                                   YEAR_RANGE,
                                                   feat_dict_no_nat_flow[issue_month],
                                                   categorical,
                                                   min_max_site_id,
                                                   PATH_DISTR,
                                                   distr_perc_dict_no_nat_flow,
                                                   NUM_BOOST_ROUND,
                                                   NUM_BOOST_ROUND_START,
                                                   EARLY_STOPPING_ROUNDS,
                                                   EARLY_STOPPING_STEP,
                                                   FINAL_TUNING,
                                                   RESIDUALS,
                                                   pd.Series([]),
                                                   NO_NAT_FLOW_SITES),
                           n_trials = N_TRIALS)
            #Save study from given month
            joblib.dump(study,
                        f"results/hyperparams_tuning/study_3_site_ids_{pd.to_datetime('today').strftime('%Y_%m_%d_%H_%M_%S')}_month_{issue_month}.pkl")

#Train models for each month and keep them in different lists according to
#quantiles
if RUN_TRAINING == True:
    for month_idx, month in enumerate(issue_months):
        print(f'Month: {month}')
        #Choose features from given month
        train_feat = feat_dict[month]
        #Choose params from given month
        params = params_dict[month]
        #Train models for different quantiles
        print('0.1 quantile')
        lgb_models_10 = train_pipeline(train,
                                       labels,
                                       month,
                                       params,
                                       train_feat,
                                       num_rounds_months,
                                       categorical,
                                       0.1,
                                       lgb_models_10,
                                       EVAL)
        print('0.5 quantile')
        lgb_models_50 = train_pipeline(train,
                                       labels,
                                       month,
                                       params,
                                       train_feat,
                                       num_rounds_months,
                                       categorical,
                                       0.5,
                                       lgb_models_50,
                                       EVAL)
        print('0.9 quantile')
        lgb_models_90 = train_pipeline(train,
                                       labels,
                                       month,
                                       params,
                                       train_feat,
                                       num_rounds_months,
                                       categorical,
                                       0.9,
                                       lgb_models_90,
                                       EVAL)
    #Save models
    save_models(issue_months,
                lgb_models_10,
                lgb_models_50,
                lgb_models_90)

end = time.time()
elapsed = end - start
print("Seconds elapsed:", elapsed)
