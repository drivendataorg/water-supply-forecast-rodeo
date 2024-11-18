import joblib
from pathlib import Path

#Set general LightGBM hyperparameters the same for each month
BAGGING_FREQ = 50
OBJECTIVE = 'quantile'
METRIC = 'quantile'
VERBOSE = -1
REG_ALPHA = 0
MIN_GAIN_TO_SPLIT = 0.0
MIN_SUM_HESSIAN_IN_LEAF = 0.001
FEATURE_FRACTION_SEED = 22
SEED = 22
#Keep all repetitive hyperparams in one dictionary
joblib.dump([BAGGING_FREQ,
             OBJECTIVE,
             METRIC,
             VERBOSE,
             REG_ALPHA,
             MIN_GAIN_TO_SPLIT,
             MIN_SUM_HESSIAN_IN_LEAF,
             FEATURE_FRACTION_SEED,
             SEED],
            Path('data') / 'general_hyperparams_final.pkl')

#Set different params for each month. These are best params from hyperparameters
#optimization

#Jan
params_1 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.08668237528700982,
            'max_depth': 4,
            'num_leaves': 50,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 7.921707376269314,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.7802262313131088,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.8,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 120,
            'min_data_in_leaf': 21,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}
#Feb
params_2 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.122248501301184,
            'max_depth': 6,
            'num_leaves': 63,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.17616556280119494,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.7005458097707512,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.8571428571428571,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 143,
            'min_data_in_leaf': 30,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}
#Mar
params_3 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.06185209436616753,
            'max_depth': 5,
            'num_leaves': 35,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.022069278840869757,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.7961156775816042,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.8571428571428571,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 200,
            'min_data_in_leaf': 29,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}
#Apr
params_4 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.05711692601228281,
            'max_depth': 6,
            'num_leaves': 91,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.010781401143472777,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.7146513043023093,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.875,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 182,
            'min_data_in_leaf': 17,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}

#May
params_5 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.02876144836229584,
            'max_depth': 7,
            'num_leaves': 63,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 2.733556118022411,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.7513484660835019,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.9,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 190,
            'min_data_in_leaf': 22,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}
#Jun
params_6 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.03281676866609218,
            'max_depth': 7,
            'num_leaves': 59,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.9395397222125083,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.753041555955779,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 0.8888888888888888,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 187,
            'min_data_in_leaf': 22,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}
#Jul
params_7 = {'objective': OBJECTIVE,
            'metric': METRIC,
            'learning_rate': 0.03322427099826354,
            'max_depth': 10,
            'num_leaves': 77,
            'lambda_l1': REG_ALPHA,
            'lambda_l2': 0.3328591152833702,
            'min_gain_to_split': MIN_GAIN_TO_SPLIT,
            'subsample': 0.8720589281122795,
            'bagging_freq': BAGGING_FREQ,
            'bagging_seed': FEATURE_FRACTION_SEED,
            'feature_fraction': 1.0,
            'feature_fraction_seed': FEATURE_FRACTION_SEED,
            'max_bin': 165,
            'min_data_in_leaf': 17,
            'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
            'verbose': VERBOSE,
            'seed': SEED}

#Keep all params in one dictionary
params_dict = {1: params_1,
               2: params_2,
               3: params_3,
               4: params_4,
               5: params_5,
               6: params_6,
               7: params_7}

#Save params to .pkl
joblib.dump(params_dict, Path('data') / 'lgbm_model_params_final.pkl')

#Set different features for each month
#Jan
train_feat_1 = ['site_id',
                'WTEQ_DAILY_prev',
                'issue_date_no_year',
                'pdsi_prev',
                'sd_forecasts']
#Feb
train_feat_2 = ['site_id',
                'WTEQ_DAILY_prev',
                'issue_date_no_year',
                'discharge_cfs_mean_since_Oct_std',
                'longitude',
                'pdsi_prev',
                'sd_forecasts']
#Mar
train_feat_3 = ['site_id',
                'nat_flow_prev',
                'WTEQ_DAILY_prev',
                'issue_date_no_year',
                'discharge_cfs_mean_since_Oct_std',
                'longitude',
                'sd_forecasts_with_jun']
#Apr
train_feat_4 = ['site_id',
                'WTEQ_DAILY_prev',
                'issue_date_no_year',
                'discharge_cfs_mean_since_Oct_std',
                'longitude',
                'pdsi_prev_to_last_month_diff',
                'pdsi_prev',
                'sd_forecasts_with_jun']
#May
train_feat_5 = ['site_id',
                'WTEQ_DAILY_prev',
                'issue_date_no_year',
                'discharge_cfs_mean_since_Oct_std',
                'longitude',
                'WTEQ_DAILY_Apr_mean',
                'pdsi_prev',
                'PREC_DAILY_Apr_mean',
                'PREC_DAILY_Apr_prev_diff',
                'sd_prev']
#Jun
train_feat_6 = ['site_id',
                'WTEQ_DAILY_prev',
                'WTEQ_DAILY_Jun_prev_diff',
                'issue_date_no_year',
                'discharge_cfs_mean_since_Oct_std',
                'longitude',
                'pdsi_prev',
                'PREC_DAILY_Apr_prev_diff',
                'sd_prev']
#Jul
train_feat_7 = ['site_id',
                'nat_flow_prev',
                'issue_date_no_year',
                'longitude',
                'nat_flow_Apr_mean',
                'WTEQ_DAILY_Apr_mean',
                'pdsi_prev',
                'sd_prev',
                'WTEQ_DAILY_Jul_prev_diff']

#Keep all features in one dictionary
train_feat_dict = {1: train_feat_1,
                   2: train_feat_2,
                   3: train_feat_3,
                   4: train_feat_4,
                   5: train_feat_5,
                   6: train_feat_6,
                   7: train_feat_7}
#Save features to .pkl
joblib.dump(train_feat_dict, Path('data') / 'lgbm_model_feats_final.pkl')

#Set different features for site_ids without naturalized flow features
train_feat_no_nat_flow_5 = ['site_id',
                            'issue_date_no_year',
                            'discharge_cfs_mean_since_Oct_std',
                            'longitude',
                            'WTEQ_DAILY_Apr_mean',
                            'pdsi_prev',
                            'PREC_DAILY_Apr_mean',
                            'sd_prev']

train_feat_no_nat_flow_6 = ['site_id',
                            'WTEQ_DAILY_prev',
                            'WTEQ_DAILY_Apr_mean',
                            'WTEQ_DAILY_Jun_prev_diff',
                            'issue_date_no_year',
                            'discharge_cfs_mean_since_Oct_std',
                            'longitude',
                            'pdsi_prev',
                            'sd_prev']

train_feat_no_nat_flow_7 = ['site_id',
                            'issue_date_no_year',
                            'longitude',
                            'WTEQ_DAILY_Apr_mean',
                            'discharge_cfs_mean_Apr_mean',
                            'pdsi_prev',
                            'pdsi_prev_to_last_month_diff']
#Keep all features in one dictionary
train_feat_no_nat_flow_dict = {5: train_feat_no_nat_flow_5,
                               6: train_feat_no_nat_flow_6,
                               7: train_feat_no_nat_flow_7}
#Save features to .pkl
joblib.dump(train_feat_no_nat_flow_dict,
            Path('data') / 'lgbm_model_feats_no_nat_flow_final.pkl')

#May
params_5_no_nat_flow = {'objective': OBJECTIVE,
                        'metric': METRIC,
                        'learning_rate': 0.07055428692224101,
                        'max_depth': 7,
                        'num_leaves': 85,
                        'lambda_l1': REG_ALPHA,
                        'lambda_l2': 4.995708705325918,
                        'min_gain_to_split': MIN_GAIN_TO_SPLIT,
                        'subsample': 0.7230053533134918,
                        'bagging_freq': BAGGING_FREQ,
                        'bagging_seed': FEATURE_FRACTION_SEED,
                        'feature_fraction': 1.0,
                        'feature_fraction_seed': FEATURE_FRACTION_SEED,
                        'max_bin': 137,
                        'min_data_in_leaf': 28,
                        'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
                        'verbose': VERBOSE,
                        'seed': SEED}
#Jun
params_6_no_nat_flow = {'objective': OBJECTIVE,
                        'metric': METRIC,
                        'learning_rate': 0.057517057231425125,
                        'max_depth': 6,
                        'num_leaves': 128,
                        'lambda_l1': REG_ALPHA,
                        'lambda_l2': 3.593916596909529,
                        'min_gain_to_split': MIN_GAIN_TO_SPLIT,
                        'subsample': 0.7486863261543473,
                        'bagging_freq': BAGGING_FREQ,
                        'bagging_seed': FEATURE_FRACTION_SEED,
                        'feature_fraction': 1.0,
                        'feature_fraction_seed': FEATURE_FRACTION_SEED,
                        'max_bin': 135,
                        'min_data_in_leaf': 17,
                        'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
                        'verbose': VERBOSE,
                        'seed': SEED}
#Jul
params_7_no_nat_flow = {'objective': OBJECTIVE,
                        'metric': METRIC,
                        'learning_rate': 0.04736884542644427,
                        'max_depth': 8,
                        'num_leaves': 115,
                        'lambda_l1': REG_ALPHA,
                        'lambda_l2': 2.3433145604798717,
                        'min_gain_to_split': MIN_GAIN_TO_SPLIT,
                        'subsample': 0.840652186517837,
                        'bagging_freq': BAGGING_FREQ,
                        'bagging_seed': FEATURE_FRACTION_SEED,
                        'feature_fraction': 1.0,
                        'feature_fraction_seed': FEATURE_FRACTION_SEED,
                        'max_bin': 160,
                        'min_data_in_leaf': 28,
                        'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
                        'verbose': VERBOSE,
                        'seed': SEED}

#Keep all params in one dictionary
params_dict_no_nat_flow = {5: params_5_no_nat_flow,
                           6: params_6_no_nat_flow,
                           7: params_7_no_nat_flow}
#Save params to .pkl
joblib.dump(params_dict_no_nat_flow, Path('data') / 'lgbm_model_params_no_nat_flow.pkl')
