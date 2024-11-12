#!/usr/bin/env python

'''
conda_on
conda activate /fmi/scratch/project_2002138/streamflow_water_supply/streamflow_env

cd /fmi/scratch/project_2002138/streamflow_water_supply/water-supply-forecast-rodeo-runtime

python -m wsfr_download bulk data_download/hindcast_train_config.yml


'''


# Read modules
import sys, glob, ast, importlib, datetime, itertools, os
import numpy as np
import pandas as pd
import xarray as xr




import optuna


from pandas.tseries.offsets import MonthEnd
from datetime import timedelta 


import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeaveOneOut


from joblib import dump, load


import warnings
warnings.filterwarnings('ignore')












exprm_name = str(sys.argv[1])
site = str(sys.argv[2])
issue_time = str(sys.argv[3])
optimize = ast.literal_eval(sys.argv[4])
use_preopt_results = ast.literal_eval(sys.argv[5])
fit_models = ast.literal_eval(sys.argv[6])
fit_shap = ast.literal_eval(sys.argv[7])

code_dir = str(sys.argv[8])+'/'
data_dir = str(sys.argv[9])+'/'






# Create the output directory if it doesn't exist
os.makedirs(data_dir+'models/', exist_ok=True)


print('Starting', exprm_name, site, issue_time, flush=True)


issue_month, issue_day = issue_time.split('-')


n_estimators = 500

ntrials = 200



target_quantiles = [0.10, 0.50, 0.90]




#code_dir = '/users/kamarain/streamflow_water_supply/'
#data_dir = '/fmi/scratch/project_2002138/streamflow_water_supply/water-supply-forecast-rodeo-runtime/data/'






# Read own functions
sys.path.append(code_dir)
import functions as fcts





df_metadata = pd.read_csv(data_dir+'metadata.csv', dtype={"usgs_id": "string"}, index_col='site_id')

sites = list(df_metadata.index)



# Monthly naturalized flow
#df_monthly = pd.read_csv(data_dir+'train_monthly_naturalized_flow.csv', parse_dates=[['year','month']])
df_monthly_prior = pd.read_csv(data_dir+'prior_historical_monthly_flow.csv', parse_dates=[['year','month']])
df_monthly_crval = pd.read_csv(data_dir+'cross_validation_monthly_flow.csv', parse_dates=[['year','month']])

df_monthly = pd.concat([df_monthly_prior, df_monthly_crval], axis=0)


df_monthly = df_monthly.pivot(index='year_month', columns='site_id', values='volume')
df_monthly.index = df_monthly.index + MonthEnd(); df_monthly.rename_axis('time',inplace=True)

# All months of the data
all_months = pd.date_range(start='1890-04-30', end='2023-07-31', freq='M')
all_months = pd.DataFrame(index=all_months)

# Put in missing months 
df_monthly = pd.concat([all_months, df_monthly], axis=1)




# Read the labels
df_train_hist = pd.read_csv(data_dir+'prior_historical_labels.csv')
df_train_cval = pd.read_csv(data_dir+'cross_validation_labels.csv')

df_train = pd.concat([df_train_hist, df_train_cval], axis=0)
df_train = df_train.pivot(index='year', columns='site_id', values='volume')













if fit_models:
    models = {}
    preprocessing_models = {}


if not fit_models: 
    models = load(f'{data_dir}models/models_{site}_issuetime={issue_time}.joblib') 
    preprocessing_models = load(f'{data_dir}models/preprocessing_models_{site}.joblib') 





if exprm_name=='kfold3':
    cv = KFold(n_splits=3, shuffle=True, random_state=99)
    nfolds = 3
    
if exprm_name=='loocv':
    cv = LeaveOneOut()
    nfolds = 20




metrics_file = data_dir+'metrics_'+exprm_name+'_'+site+'_'+issue_time+'.csv'
crosval_file = data_dir+'crosval_'+exprm_name+'_'+site+'_'+issue_time+'.csv'

metrics = pd.DataFrame(index=[site], columns=['r', 'r2', 'ic', 'mql', 'l10', 'l50', 'l90'])


clock = fcts.stopwatch('start')






# Season parameters
ssn_bgn = df_metadata.loc[site, 'season_start_month']
ssn_end = df_metadata.loc[site, 'season_end_month']




if ssn_end==6: days = '30' 
if ssn_end==7: days = '31' 


# Time series begins from the first water year
bgn_year = str(df_train[site].dropna().index[0] - 1)
end_year = str(df_train[site].dropna().index[-1])

# All train years and a subset for validation
all_train_years = np.arange(1982, 2024)
all_valid_years = np.arange(2004, 2024) 

# End of season dates for the target site
t_axis_target = df_train.index.astype(str) + '-'+str(ssn_end).zfill(2)+'-'+days
t_axis_target = pd.to_datetime(t_axis_target)

# Target data in a dataframe, daily time resolution
df_target = pd.DataFrame(index=t_axis_target, data=df_train[site].values, columns=[site])
Y = df_target.resample('1D').mean()

# Find and match issue times and end-of-season dates
Y[issue_time] = fcts.match_target_and_issue_dates(df_target)[[issue_time]]
Y = Y.rename_axis('time')

# Extend the inspection window to get more data for fitting. Wider window for winter,
# narrowing the window width towards the end of season
issue_month = int(issue_time[0:2]); insp_window = int(round((1/issue_month)*14+4))
Y[issue_time] = fcts.select_window_around_issuetime(Y[issue_time], issue_time, insp_window, fill=True)


# Prediction targets
Y['q0p1'] = np.nan
Y['q0p5'] = np.nan
Y['q0p9'] = np.nan











# Cross-validation over year blocks/groups
for i, (train_years_idx, valid_years_idx) in enumerate(cv.split(all_valid_years)):
    i+=1
    
    # Find training and testing years for the split
    _, valid_years = fcts.define_split_years(all_valid_years, valid_years_idx, train_years_idx)
    train_years = np.setdiff1d(all_train_years, valid_years)
    
    #if valid_years[0] != 2023: continue
    
    print(f'Train {site}: {train_years}\nValidate {site}: {valid_years}\n{issue_time}\n', flush=True)
    print(site,issue_time,exprm_name,flush=True) 
    t1 = fcts.stopwatch('start')
    
    # Define predictors and fit PCA models for the train split
    predictor_data_train, climatology_models, pca_models = fcts.read_process_everything(
        data_dir, data_dir, site, train_years, train_or_test='train', 
        remove_outliers=True); fcts.print_ram_state()
    
    fig, axes = plt.subplots(len(predictor_data_train.keys()),1, figsize=(20,40))#, sharex=True)
    for ax,dataset in zip(axes, predictor_data_train.keys()):
        predictor_data_train[dataset].plot(ax=ax); ax.set_title(dataset); ax.legend(ncols=6, loc='upper left')
    
    plt.tight_layout(); fig.savefig(data_dir+'fig_predictortimeseries_'+site+'_train.png')
    plt.clf(); plt.close('all')
    
    
    # Define predictors and apply previously fitted PCA models for the validation split
    predictor_data_valid, _, _ = fcts.read_process_everything(
        data_dir,data_dir, site, valid_years,
        train_or_test='train', climatology_models=climatology_models, 
        pca_models=pca_models, remove_outliers=True); fcts.print_ram_state()
    
    fig, axes = plt.subplots(len(predictor_data_valid.keys()),1, figsize=(14,40))#, sharex=True)
    for ax,dataset in zip(axes, predictor_data_valid.keys()):
        predictor_data_valid[dataset].plot(ax=ax); ax.set_title(dataset); ax.legend(ncols=3, loc='upper left')
    
    plt.tight_layout(); fig.savefig(data_dir+'fig_predictortimeseries_'+site+'_valid.png')
    plt.clf(); plt.close('all')
    
    
    # Split data for fitting and validation samples
    X_trn, Y_trn, X_val, Y_val, sample_weights = fcts.define_split_details(
            predictor_data_train, predictor_data_valid, 
            train_years, valid_years, Y, issue_time); fcts.print_ram_state()
    
    print(train_years, Y_trn.dropna().index.year.unique())
    print(valid_years,Y_val.dropna().index.year.unique())
    
    
    # Calculate climatological quantiles for visualization
    X_all = pd.concat(predictor_data_train.values(), axis=1)
    dataset_names = list(predictor_data_train.keys())
    observed_PC_climatology = {}
    for q in [0.01, 0.1, 0.5, 0.9, 0.99]:
        observed_PC_climatology[q] = X_all.groupby('dayofyear').quantile(q).loc[1:202]
    
    
    # Collect preprocessing models
    preprocessing_models[f'climatology_models_fold={i}'] = climatology_models.copy()
    preprocessing_models[f'pca_models_fold={i}']  = pca_models.copy()
    preprocessing_models[f'observed_climatology_fold={i}']  = observed_PC_climatology.copy()
    
    
    
    
    if optimize and use_preopt_results: 
        sys.exit('This combination of "optimize" and "use_preopt_results" not allowed, check input params')
    
    if optimize and not use_preopt_results:
        
        # Fine tune parameters 
        best_params = fcts.tune_hyperparams(X_trn, Y_trn[[issue_time]], quantile=target_quantiles, num_trials=ntrials)
        fcts.print_ram_state()
        
        models[f'params_fold={i}'] = best_params
    
        print(f'Optimizing {site} took',fcts.stopwatch('stop',t1), flush=True)
    
    
    if not optimize and not use_preopt_results:
        best_params = fcts.params_qregr()
    
    
    if not optimize and use_preopt_results: 
        param_dicts = []; 
        for k in np.arange(nfolds)+1:
            param_dicts.append(models[f'params_fold={k}'])
        
        # Get median of parameters
        best_params = {key: np.median(values) for key, values in zip(param_dicts[0], zip(*(d.values() for d in param_dicts)))}
        best_params['bootstrap'] = best_params['bootstrap'].astype(bool)
        best_params['bootstrap_features'] = best_params['bootstrap_features'].astype(bool)
     
    
    
    best_params['n_estimators'] = n_estimators # 100 # 200 # 500
    best_params['n_jobs'] = -1
    
    params_qregr = fcts.params_qregr()
    params_qregr.update(best_params)
    
    
    if fit_models:
        # Fit the model to the train part of the cross-validation split using the tuned hyperparameters
        quantile_model = fcts.bagging_multioutput_model(X_trn, Y_trn[[issue_time]], 
                                                    params_qregr, target_quantiles, sample_weights)
        
        # Collect the model 
        models[f'model_fold={i}'] = quantile_model
    
    if not fit_models:
        # Select the proviously fitted model
        quantile_model = models[f'model_fold={i}']
    
    # Predict the validation data
    Y.loc[Y_val.index,['q0p1','q0p5','q0p9']] = quantile_model.predict(X_val) 
    
    # Ensure correct order of quantiles 0.1 < 0.5 < 0.9
    Y.loc[Y_val.index,['q0p1','q0p5','q0p9']] = np.sort(Y.loc[Y_val.index,['q0p1','q0p5','q0p9']].values, axis=1)
    
    # Ensure positive values
    Y.loc[Y_val.index,['q0p1','q0p5','q0p9']] = Y.loc[Y_val.index,['q0p1','q0p5','q0p9']].clip(0, None)
    
    
    # SHAP models. Fold=20 corresponds to forecast year 2023.
    if fit_shap and i==20:
        #_, explainer   = fcts.extract_shap_values(X_trn.sample(int(len(X_trn)/3)), quantile_model)
        _, explainer   = fcts.extract_shap_values(X_trn, quantile_model)
        shap_values, _ = fcts.extract_shap_values(X_val, quantile_model, explainer)
        
        models[f'explainer_fold={i}'] = explainer
    
    
    # Calculate metrics
    common_time = Y['q0p5'].dropna().index
    r_q0p5 = fcts.calc_corr(Y.loc[common_time,issue_time], Y.loc[common_time,'q0p5']); print(site,issue_time,'corr:',r_q0p5)
    r2_q0p5 = fcts.calc_r2ss(Y.loc[common_time,issue_time], Y.loc[common_time,'q0p5'])[0]; print(site,issue_time,'r2:',r2_q0p5, flush=True)
    ic = fcts.interval_coverage(Y.loc[common_time,issue_time], Y.loc[common_time,'q0p1'], Y.loc[common_time,'q0p9'])
    ql10 = fcts.calc_pbloss(Y.loc[common_time,issue_time], Y.loc[common_time,'q0p1'], 0.1)
    ql50 = fcts.calc_pbloss(Y.loc[common_time,issue_time], Y.loc[common_time,'q0p5'], 0.5)
    ql90 = fcts.calc_pbloss(Y.loc[common_time,issue_time], Y.loc[common_time,'q0p9'], 0.9)
    
    
    print(f'Fitting one fold in {exprm_name} for {site} took',fcts.stopwatch('stop',t1), flush=True)




if fit_models:
    # Save models
    dump(models, f'{data_dir}models/models_{site}_issuetime={issue_time}.joblib') 
    
    # Save preprocessing models
    if issue_time == '01-01':
        dump(preprocessing_models, f'{data_dir}models/preprocessing_models_{site}.joblib') 





# Save cross-validation metrics 
metrics.loc[site,'issue_time'] = issue_time
metrics.loc[site,'r'] = r_q0p5
metrics.loc[site,'r2'] = r2_q0p5
metrics.loc[site,'ic'] = ic
metrics.loc[site,'mql'] = np.mean([ql10,ql50,ql90])
metrics.loc[site,'l10'] = ql10
metrics.loc[site,'l50'] = ql50
metrics.loc[site,'l90'] = ql90

metrics.to_csv(metrics_file)



# Save cross-validation predictions and the observed value for reference
Z = Y[['q0p1','q0p5','q0p9',issue_time]]
Z[['q0p1','q0p5','q0p9']] = Z[['q0p1','q0p5','q0p9']]
Z = Z.dropna()

# Pick the exact dates
issue_dates = (Z.index.month == int(issue_month)) & (Z.index.day == int(issue_day))
Z = Z.iloc[issue_dates]

# Ensure quantiles are in correct order
Z[['q0p1','q0p5','q0p9']] = np.sort(Z[['q0p1','q0p5','q0p9']].values, axis=1)

Z['site_id'] = site
Z = Z.rename_axis('issue_date').rename(columns={'q0p1':'volume_10',
                                                'q0p5':'volume_50',
                                                'q0p9':'volume_90',
                                                issue_time:'observed'}).reset_index()

Z[['site_id','issue_date','volume_10','volume_50','volume_90','observed']].to_csv(crosval_file, index=False)






print('Finished', exprm_name, site, issue_time, 'fitting took',fcts.stopwatch('stop', clock), flush=True)



