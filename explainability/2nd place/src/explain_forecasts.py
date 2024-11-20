#!/usr/bin/env python





# Read modules
import sys, glob, ast, importlib, datetime, itertools, os
import numpy as np
import pandas as pd
import xarray as xr

import requests

from pandas.tseries.offsets import MonthEnd
from datetime import timedelta 

import shap

import geopandas as gpd
import rioxarray as rxr


from pathlib import Path

import matplotlib; matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns 


from joblib import dump, load

from wsfr_download.config import DATA_ROOT



#data_dir = Path('/fmi/scratch/project_2002138/streamflow_water_supply/water-supply-forecast-rodeo-runtime/data/')

data_dir = DATA_ROOT 




# Read own functions
import functions as fcts



issue_times = fcts.all_issue_times()

sites = fcts.all_sites()



# FOLD=20 is the CV fold which has 2023 as validation data in the LOOCV experiment
FOLD = 20

'''

python explain_forecasts.py 2023-03-15 owyhee_r_bl_owyhee_dam &
python explain_forecasts.py 2023-05-15 owyhee_r_bl_owyhee_dam &
python explain_forecasts.py 2023-03-15 pueblo_reservoir_inflow &
python explain_forecasts.py 2023-05-15 pueblo_reservoir_inflow &

python explain_forecasts.py 2023-07-15 hungry_horse_reservoir_inflow &
python explain_forecasts.py 2023-07-15 virgin_r_at_virtin &



forecast_year = 2023

issue_time = '05-15'

issue_date = f'{forecast_year}-{issue_time}'


site = 'owyhee_r_bl_owyhee_dam'
site = 'pueblo_reservoir_inflow'



'''


issue_date, site = str(sys.argv[1]), str(sys.argv[2])







forecast_year, month, day = issue_date.split('-')
issue_time = f'{month}-{day}'


forecast_year = int(forecast_year)



valid_years = [forecast_year] 

all_years   = np.arange(1982,2025)
train_years = np.arange(1982,2023)



print(f'Analysis for {site} {issue_date}\n', flush=True)




# Station metadata
df_metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={"usgs_id": "string"}, index_col='site_id')



# Geospatial catchment polygons
gdf_polygons = gpd.read_file(data_dir / 'geospatial.gpkg')


# SNOTEL metadata
snotel_meta     = pd.read_csv(data_dir / 'snotel/station_metadata.csv')
sites_to_snotel = pd.read_csv(data_dir / 'snotel/sites_to_snotel_stations.csv')




# Observed naturalized flow
df_train_hist = pd.read_csv(data_dir / 'prior_historical_labels.csv')
df_train_cval = pd.read_csv(data_dir / 'cross_validation_labels.csv')

df_target = pd.concat([df_train_hist, df_train_cval], axis=0)
df_target = df_target.pivot(index='year', columns='site_id', values='volume')


# Observed quantiles of targets 1981-2023
quantiles = [0.10, 0.50, 0.90]
obs_quantiles = {}
for q in quantiles:
    obs_quantiles[q] = df_target[site].loc['1981':'2023'].quantile(q)








# Read models and preprocessing models. 
models = load(f'{data_dir}/models/models_{site}_issuetime={issue_time}.joblib') 
preprocessing_models = load(f'{data_dir}/models/preprocessing_models_{site}.joblib')

climatology_models      = preprocessing_models[f'climatology_models_fold={FOLD}']
pca_models              = preprocessing_models[f'pca_models_fold={FOLD}']
observed_PC_climatology = preprocessing_models[f'observed_climatology_fold={FOLD}']  
explainer               = models[f'explainer_fold={FOLD}']








# Extract the raw predictor data without diff/lags/mave or PCA 
predictor_data_raw, _, _ = fcts.read_process_everything(
    f'{data_dir}/',f'{data_dir}/', site, all_years, 
    train_or_test='train', return_raw=True); fcts.print_ram_state()


# Define predictors by applying previously fitted PCA models for the train split
predictor_data_train, _, _ = fcts.read_process_everything(
    f'{data_dir}/',f'{data_dir}/', site, train_years,
    train_or_test='train', climatology_models=climatology_models, 
    pca_models=pca_models, remove_outliers=True); fcts.print_ram_state()


# Define predictors by applying previously fitted PCA models for the validation split
predictor_data_valid, _, _ = fcts.read_process_everything(
    f'{data_dir}/',f'{data_dir}/', site, valid_years,
    train_or_test='train', climatology_models=climatology_models, 
    pca_models=pca_models, remove_outliers=True); fcts.print_ram_state()



fig, axes = plt.subplots(np.ceil(len(predictor_data_valid.keys())/2).astype(int),2, figsize=(14,20))#, sharex=True)
for ax,dataset in zip(axes.ravel(), predictor_data_valid.keys()):
    predictor_data_valid[dataset].dropna().plot(ax=ax); ax.set_title(dataset)
    ax.legend(ncols=3, loc='upper left', fontsize='xx-small')

plt.tight_layout(); fig.savefig(str(data_dir)+'/fig_explainability_predictortimeseries_'+site+'_valid.png', bbox_inches='tight', pad_inches=0.035)
#plt.show(); plt.clf(); plt.close('all')


# Join the individual datasets into one predictor matrix
X_trn = pd.concat(predictor_data_train.values(), axis=1)
X_val = pd.concat(predictor_data_valid.values(), axis=1)

# Shift the predictor data of the validation set one day to not break the competition rules
X_trn = X_trn.shift(1)
X_val = X_val.loc[:issue_date].shift(1)



X_trn_sample = X_trn.dropna().sample(300)

shap_values_trn, _ = fcts.extract_shap_values(X_trn_sample,   models[f'model_fold={FOLD}'], explainer)
shap_values_val, _ = fcts.extract_shap_values(X_val.dropna(), models[f'model_fold={FOLD}'], explainer)








shap_data_trn = pd.DataFrame(index=X_trn_sample.index, columns=X_trn_sample.columns, data=shap_values_trn.values)
shap_data_val = pd.DataFrame(index=X_val.dropna().index, columns=X_val.columns, data=shap_values_val.values)


#shap_data = shap_values_val.loc[issue_date].reset_index(drop=True)
#shap_absm = np.abs(shap_data.loc[issue_date]).mean(axis=0).sort_values(ascending=False); most_important = list(shap_absm[0:8].index)
shap_absm_global = np.abs(shap_data_trn).mean(axis=0).sort_values(ascending=False);     shap_features_global    = list(shap_absm_global[0:10].index)
shap_absm_issdat = np.abs(shap_data_val.loc[issue_date]).sort_values(ascending=False);  shap_features_issuedate = list(shap_absm_issdat[0:10].index)


# Identify the six most important features and the six most important datasets
top6_features = shap_features_issuedate[0:6]

'''
top6_datasets = []
for item in shap_features_issuedate:
    if item.split('_PC')[0] not in top6_datasets and len(top6_datasets) < 6:
        top6_datasets.append(item.split('_PC')[0])
'''



# Plot most important features for given issue date
sns.set_style('whitegrid')
fig, axes = plt.subplots(1,1, figsize=(6,5))
b2 = axes.barh(shap_features_issuedate,      shap_absm_issdat.iloc[0:10], height=0.6, color='deepskyblue', label=f'Importance for {issue_date}'); 
b3 = axes.barh(shap_features_issuedate[0:6], shap_absm_issdat.iloc[0:6],  height=0.6, color='deepskyblue', label=f'TOP-6 for {issue_date}'); 
b1 = axes.barh(shap_features_issuedate,      shap_absm_global[shap_features_issuedate], height=0.8, color='m', alpha=0.15, label='Mean importance in 1981–2022'); 
for bar in b3:
    bar.set_hatch('////')


axes.set_ylabel('Feature'); axes.set_xlabel('mean(|SHAP|): average impact on model output')
axes.set_title(f'mean(|SHAP|) of the TOP-10 features for\n{site.upper().replace("_"," ")} at {issue_date}')
axes.invert_yaxis()

plt.tight_layout(); plt.legend()#title='', fontsize='small')
axes.grid(True, which='both', linestyle='--', linewidth=0.5)
#fig.savefig(str(data_dir)+f'/fig_explainability_shap_barplot_{site}_{issue_date}.pdf', bbox_inches='tight', pad_inches=0.035)
fig.savefig(str(data_dir)+f'/fig_explainability_shap_barplot_{site}_{issue_date}.png', bbox_inches='tight', pad_inches=0.035)
plt.clf(); plt.close('all')
#plt.show(); plt.clf(); plt.close('all')








#shap_absm_global[shap_features_global].plot.barh(); plt.tight_layout(); plt.show()


#sns.boxenplot(shap_data_trn[shap_features_global], orient='h', color='lightgray',width_method="linear", showfliers=False, width=2, linewidth=0.3);
fig, axes = plt.subplots(1,1, figsize=(6,6))
idx = shap_data_val.index==issue_date
shap.waterfall_plot(shap_values_val[idx][0], max_display=11, show=False); plt.tight_layout(); 
#plt.savefig(str(data_dir)+f'/fig_explainability_shap_waterfall_{site}_{issue_date}.pdf', bbox_inches='tight', pad_inches=0.035)
plt.savefig(str(data_dir)+f'/fig_explainability_shap_waterfall_{site}_{issue_date}.png', bbox_inches='tight', pad_inches=0.035)
plt.clf(); plt.close('all')
#plt.show()



#shap.summary_plot(shap_values_trn, plot_size=(8,6), max_display=20, plot_type='violin', show=False); plt.tight_layout(); plt.show()
#shap.summary_plot(shap_values_trn, plot_size=(8,6), max_display=20, plot_type='bar', show=False); plt.tight_layout(); plt.show()


#shap.decision_plot(shap_values_val[idx][0].base_values, shap_values_val[idx][0].values, plot_color='BrBG')

#shap.dependence_plot('usgs_neighborflow_PC1', shap_values_val[idx][0].values, )# X_test.iloc[:,features])
#shap.force_plot(explainer_q0p5.expected_value, shap_values_q0p5, feature_names=estimator_columns, link='identity', plot_cmap='RdBu', matplotlib=True)

#shap.bar_plot(shap_values_q0p5, X_test.iloc[:,features], estimator_columns)




# Predict all dates for the forecast year until the issue date
issue_times = fcts.all_issue_times()
issue_times = issue_times[:issue_times.index(issue_time)+1]
previous_forecasts = []
for it in issue_times:
    print(it)
    prvs_seven_days = pd.to_datetime(f'{forecast_year}-{it}') - timedelta(days=6)
    date_range = pd.date_range(prvs_seven_days, f'{forecast_year}-{it}')
    mdls = load(f'{data_dir}/models/models_{site}_issuetime={it}.joblib') 
    
    date = X_val.loc[date_range].dropna().index
    result = pd.DataFrame(columns=['date','issue_date','Q10','Q50','Q90','member'])
    
    for i, (features, estimator) in enumerate(zip(mdls[f'model_fold={FOLD}'].estimators_features_, mdls[f'model_fold={FOLD}'].estimators_)):
        #print(features, estimator, i)
        estim_rslt = estimator.predict(X_val.loc[date_range].iloc[:,features].dropna())
        
        result = pd.DataFrame(columns=['date','Q10','Q50','Q90','member'])
        result[['Q10','Q50','Q90']] = estim_rslt #mdls[f'model_fold={FOLD}'].predict(X_val.loc[date_range].dropna())
        result['date'] = date
        result['issue_date'] = f'{forecast_year}-{it}'
        result['member'] = i
        previous_forecasts.append(result)


previous_forecasts = pd.concat(previous_forecasts, axis=0).reset_index(drop='True')





# Plot the evolution of the forecasts from the beginning of the year
sns.set_style('white')
fig, axes = plt.subplots(1,1, figsize=(12, 4))


plt.axhline(obs_quantiles[0.9], lw=2, ls='--', color='gray',      label='Q90, observed 1981–2023')
plt.axhline(obs_quantiles[0.5], lw=3, ls='--', color='black',     label='Q50, observed 1981–2023')
plt.axhline(obs_quantiles[0.1], lw=2, ls='--', color='gray',      label='Q10, observed 1981–2023')

sns.lineplot(data=previous_forecasts, x='date', y='Q90', estimator='mean', ax=axes,  color='tab:cyan', ci=100,lw=3, label='Q90, model') 
sns.lineplot(data=previous_forecasts, x='date', y='Q50', estimator='mean', ax=axes,  color='tab:blue', ci=100,lw=4, label='Q50, model') 
sns.lineplot(data=previous_forecasts, x='date', y='Q10', estimator='mean', ax=axes,  color='tab:cyan', ci=100,lw=3, label='Q10, model')

issue_date_forecast = previous_forecasts.loc[previous_forecasts['date']==issue_date][['Q10','Q50','Q90']].mean()
boxst = dict(boxstyle='round', fc='w', ec='k', alpha=0.5)
axes.text(pd.to_datetime(issue_date), issue_date_forecast['Q10'], str(issue_date_forecast['Q10'].astype(int)), ha='center', va='center',bbox=boxst)
axes.text(pd.to_datetime(issue_date), issue_date_forecast['Q50'], str(issue_date_forecast['Q50'].astype(int)), ha='center', va='center',bbox=boxst)
axes.text(pd.to_datetime(issue_date), issue_date_forecast['Q90'], str(issue_date_forecast['Q90'].astype(int)), ha='center', va='center',bbox=boxst)

for id in previous_forecasts['issue_date'].unique(): 
    label = ''
    if id==issue_date: label='Issue dates'
    plt.axvline(pd.to_datetime(id), lw=1, ls='--', color='lightgray', zorder=-1)


axes.set_ylabel('Naturalized Flow (KAF)'); axes.set_xlabel('')
axes.set_title(f'Forecasts of the FY{forecast_year} end-of-season naturalized flow for {site.upper().replace("_"," ")}')
#plt.legend(title='Lines: ensemble median\nShading: ensemble spread', fontsize='small')
plt.legend(fontsize='small', loc='upper left', ncols=2)
plt.tight_layout(rect=[0,0,1,1]); 
#fig.savefig(str(data_dir)+f'/fig_explainability_forecasttimeseries_{site}_{issue_date}.pdf', bbox_inches='tight', pad_inches=0.035)
fig.savefig(str(data_dir)+f'/fig_explainability_forecasttimeseries_{site}_{issue_date}.png', bbox_inches='tight', pad_inches=0.035)
#plt.show(); plt.clf(); plt.close('all')





# Calculate the daily climatological values of the raw predictor data 
from sklearn.preprocessing import QuantileTransformer

dataset_names = list(predictor_data_raw.keys())
dataset_names.remove('dayofyear')

quantiles = [1.0, 0.99, 0.90, 0.75, 0.50, 0.25, 0.10, 0.01,0.0]

predictor_data_climatology = {} 
predictor_data_percentiles = {} 

for name in dataset_names:
    raw_data = predictor_data_raw[name].copy(deep=True)
    
    if name=='snotel_temperature':
        raw_data = raw_data.rolling(window=30, min_periods=10).mean()
    #raw_data = fcts.Gauss_filter(raw_data, sigma=(1,0))
    
    
    raw_data['dayofyear']  = raw_data.index.dayofyear
    
    # Delete data outside the water year
    raw_data.loc[(raw_data.index.dayofyear>=203)&(raw_data.index.dayofyear<=274)] = np.nan
    
    # Climatology as smoothed timeseries
    grouped = raw_data.groupby('dayofyear')
    median_timeser = grouped.transform(lambda x: x.quantile(0.5)) + 1e-9
    median_timeser = fcts.Gauss_filter(median_timeser, sigma=(3,0))
    
    '''
    # Transform to percentage of median
    percentage = (raw_data.drop(columns='dayofyear') / median_timeser) * 100
    percentage = percentage.where(~np.isinf(percentage))
    
    predictor_data_percentage[name] = percentage.copy(deep=True)
    '''
    
    # Delete data outside the water year
    #raw_data.loc[(raw_data.index.dayofyear>=203)&(raw_data.index.dayofyear<=266)] = np.nan
    
    # Climatological quantiles
    grouped = raw_data.groupby('dayofyear')
    predictor_data_climatology[name] = {}
    for q in quantiles:
        predictor_data_climatology[name][q] = grouped.quantile(q)#.loc[1:202]


    # Apply QuantileTransformer to each feature for each day of the year
    transformed_df = pd.DataFrame(index=raw_data.index)
    for column in raw_data.columns[:-1]:  # Exclude the 'dayofyear' column
        transformed_data = []
        
        for doy in range(1, 366):  # Loop through each day of the year (1 to 366)
            day_data = raw_data[raw_data['dayofyear'] == doy][[column]]
            
            if not day_data.empty:
                qt = QuantileTransformer(output_distribution='uniform')
                transformed_day_data = qt.fit_transform(day_data)
                transformed_data.append(pd.Series(transformed_day_data.flatten(), index=day_data.index))
        
        # Concatenate the transformed data for the column
        transformed_df[column] = pd.concat(transformed_data).sort_index()

    # Multiply by 100 to get percentiles (0 to 100)
    predictor_data_percentiles[name] = transformed_df * 100






fig, axes = plt.subplots(2,3, figsize=(10,3), sharex=True)
for i, (feat,ax) in enumerate(zip(top6_features,axes.ravel())): #dataset_names:
#for feat in top6_features:
    name = feat.split('_PC')[0]
    doy = pd.to_datetime(issue_date).dayofyear
    doy = 366
    
    quantiles, colors, lws, lss = [0.99, 0.50, 0.01], ['k', 'tab:red', 'k'], [0.5, 1.5, 0.5], ['--','-','--']
    if 'flow' in name:
        quantiles, colors, lws, lss = [0.50], ['tab:red'], [1.5], ['-']
    
    for quantile, color, lw, ls in zip(quantiles, colors, lws, lss):
        ax.plot(observed_PC_climatology[quantile][feat][0:doy].values, color=color, lw=lw, ls=ls, label=f'Q{int(quantile*100)}')
    
    if not 'flow' in name:
        ax.fill_between(x=observed_PC_climatology[0.1].index[0:doy],y2=observed_PC_climatology[0.1][feat][0:doy],  
                    y1=observed_PC_climatology[0.9][feat][0:doy],  color='tab:pink', alpha=0.25, label='80%', zorder=-1)

    #plt.plot(observed_PC_climatology[0.1].index,observed_PC_climatology[0.50][feat],  color='b')

    previous_year = forecast_year - 1
    #observed = pd.DataFrame(predictor_data_raw[name].loc[f'{previous_year}-09-22':f'{forecast_year}-09-21']).mean(axis=1)
    observed = X_val[feat].copy(deep=True)
    #if name=='snotel_temperature': observed = observed.rolling(30, min_periods=10).mean()
    
    observed.loc[issue_date:] = np.nan; observed.index = observed.index.dayofyear; observed = observed.loc[1:]
    ax.plot(observed, color='k',lw=2.5, label=f'{forecast_year}')
    #ax.plot(np.arange(len(observed)), observed.drop(columns='dayofyear'), color='r', lw=1.5, label=f'{forecast_year}')
    #ax.plot(range(len(new_order)), observed.values, color='tab:red', lw=2.5, label=f'FY{forecast_year}')
    
    #if 'flow' in name: # and observed_PC_climatology[0.01][feat].min() >= 0:
    #    ax.set_yscale('symlog')
    
    ax.set_title(feat)
    ax.set_xlabel('Day of Year')
    if i==0: ax.legend(fontsize='x-small'); 
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)




#fig.suptitle(f'TOP-6 features for {issue_date} {site.upper().replace("_"," ")}', fontsize=12)
plt.tight_layout()
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.subplots_adjust(wspace=0.3, hspace=0.5)
#fig.savefig(str(data_dir)+f'/fig_explainability_featuresquantiles_{site}_{issue_date}.pdf', bbox_inches='tight', pad_inches=0.035)
fig.savefig(str(data_dir)+f'/fig_explainability_featuresquantiles_{site}_{issue_date}.png', bbox_inches='tight', pad_inches=0.035)
#plt.show(); plt.clf(); plt.close('all')







basic_datasets = ['snotel_waterequivalent', 'snotel_precipitation', 'snotel_temperature', 'pdsi', 'naturalflow']


#fig, axes = plt.subplots(2,3, figsize=(14,5))
fig, axes = plt.subplots(5,1, figsize=(4,8), sharex=True)
for i, (name,ax) in enumerate(zip(basic_datasets,axes.ravel())): #dataset_names:
#for name in dataset_names:
    
    # Dataset name
    #name = feat.split('_PC')[0]
    
    multidims = False; legend_title = ''
    #if predictor_data_climatology[name][0.0].shape[1] > 1: 
    #    multidims = True; legend_title = 'Values are\nmeans over\nthe catchment'
    
    # Calculate mean across columns for each quantile
    df_means = {}
    quantiles = [0.99, 0.9, 0.50, 0.1, 0.01]
    for quantile in quantiles:
        df_means[quantile] = predictor_data_climatology[name][quantile].mean(axis=1)
    
    # Create a new order for the days and reindex the DataFrame according to the new order
    new_order = list(range(265, 366)) + list(range(1, 265))
    reindexed_means = {quantile: df_means[quantile].reindex(new_order) for quantile in quantiles}
    
    # Plotting
    
    for quantile, color, lw, ls in zip([0.99, 0.50, 0.01], ['k', 'tab:blue', 'k'], [0.5, 1.5, 0.5], ['--','-','--']):
        ax.plot(reindexed_means[quantile].values, color=color, lw=lw, ls=ls, label=f'Q{int(quantile*100)}')

    ax.fill_between(range(len(new_order)), reindexed_means[0.10].values, reindexed_means[0.90].values, 
                        color='tab:blue', alpha=0.2, label='80%', zorder=-1)
    
    previous_year = forecast_year - 1
    observed = pd.DataFrame(predictor_data_raw[name].loc[f'{previous_year}-09-22':f'{forecast_year}-09-21']).mean(axis=1)
    if name=='snotel_temperature': 
        observed = observed.rolling(30, min_periods=10).mean()
    
    observed.iloc[0:10] = np.nan; observed.loc[issue_date:] = np.nan; observed.index = observed.index.dayofyear
    #ax.plot(np.arange(len(observed)), observed.drop(columns='dayofyear'), color='r', lw=1.5, label=f'{forecast_year}')
    ax.plot(range(len(observed)), observed.values, color='tab:red', lw=2.5, label=f'{forecast_year}')
    
    
    #if reindexed_means[0.0].min() >= 0:
    if 'flow' in name:
        ax.set_yscale('log')
        
    
    # Set custom x-axis tick labels
    tick_spacing = 30  # every 30 days
    ticks = list(range(0, len(new_order), tick_spacing))
    tick_labels = [new_order[tick] % 365 for tick in ticks]
    ax.set_xticks(ticks, tick_labels)
    
    ax.set_title(name)
    ax.set_xlabel('Day of Year')
    #ax.set_title(f'Climatological quantiles and raw {name.upper().replace("_"," ")} data for FY{forecast_year} {site.upper().replace("_"," ")}')
    ax.set_title(f'{name.upper().replace("_"," ")}')
    #plt.tight_layout(); plt.legend(title='Lines: ensemble median\nShading: ensemble spread', fontsize='small')
    #ax.set_ylabel('Mean Streamflow')
    if i==0: ax.legend(title=legend_title, fontsize='small'); 
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


#fig.suptitle(f'{issue_date} {site.upper().replace("_"," ")}\nClimatological quantiles and data for FY{forecast_year}\n', fontsize=12)
#fig.suptitle(f'{issue_date} {site.upper().replace("_"," ")}', fontsize=12)
plt.tight_layout()
#fig.savefig(str(data_dir)+f'/fig_explainability_datasetsquantiles_{site}_{issue_date}.pdf', bbox_inches='tight', pad_inches=0.035)
fig.savefig(str(data_dir)+f'/fig_explainability_datasetsquantiles_{site}_{issue_date}.png', bbox_inches='tight', pad_inches=0.035)
#plt.show(); plt.clf(); plt.close('all')





def get_percentile(dataframe, number):
    percentile = 50.0
    try:
        for idx in dataframe.index:
            if idx.split('_')[2] == number:
                percentile = dataframe[idx]
    except:
        pass
    return percentile



# Read PDSI info
pdsi_file = glob.glob(str(data_dir / f'pdsi/FY{forecast_year}')+'/*')[0]

# PDSI Data domain around the catchments
spatial_extents = fcts.catchment_extent(data_dir / 'geospatial.gpkg', 0.5)


latmin, lonmin, latmax, lonmax = spatial_extents[site]
ds_PDSI = xr.open_dataset(pdsi_file)['daily_mean_palmer_drought_severity_index'].rename({'day':'time'}).resample(time='1D').ffill()
ds_PDSI = ds_PDSI.sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax), time=issue_date)
ds_PDSI = ds_PDSI.rio.write_crs('EPSG:4326') 


# Plot catchment map with PDSI background
fig, ax = plt.subplots(1,1, figsize=(7,7)); #ax = ax.ravel()
ds_PDSI.plot(ax=ax, cmap='BrBG', vmin=-5, vmax=5, center=0, alpha=0.75,
            add_colorbar=True, extend='both', cbar_kwargs={'shrink':0.7, 'label':'PDSI'})

# Plot catchent boundaries
polygon = gdf_polygons.loc[gdf_polygons.site_id==site].geometry.to_crs(ds_PDSI.rio.crs)
polygon.plot(ax=ax, facecolor='none', edgecolor='w', linewidth=4)
polygon.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=1.5)

ax.set_title(f'Palmer Drought Severity Index and SNOTEL data\nfor {issue_date} {site.upper().replace("_"," ")}')

# SNOTEL station data
snotel_catchment = sites_to_snotel.loc[sites_to_snotel.site_id==site]
watereq_to_date = predictor_data_percentiles['snotel_waterequivalent'].ffill(limit=5).loc[issue_date]
precipi_to_date = predictor_data_percentiles['snotel_precipitation'].ffill(limit=5).loc[issue_date]
tempera_to_date = predictor_data_percentiles['snotel_temperature'].ffill(limit=5).loc[issue_date]

lons=[]; lats = []; 
percentiles_wateqv = []; percentiles_precip = []; percentiles_temper = []
for sttn in snotel_catchment.stationTriplet:
    number = sttn.split(':')[0]

    percentiles_wateqv.append(get_percentile(watereq_to_date, number))
    percentiles_precip.append(get_percentile(precipi_to_date, number))
    percentiles_temper.append(get_percentile(tempera_to_date, number))

    row = snotel_meta.loc[snotel_meta.stationTriplet == sttn]
    lats.append(row.latitude.values[0])
    lons.append(row.longitude.values[0])


lats = np.array(lats); lons = np.array(lons)

sc1 = ax.scatter(lons,lats-0.07,c=percentiles_temper,edgecolors='k',marker='d',cmap='Spectral',vmin=0,vmax=100,s=150) #,label=str(watereq))
sc2 = ax.scatter(lons-0.07,lats,c=percentiles_precip,edgecolors='k',marker='s',cmap='Spectral',vmin=0,vmax=100,s=120) #,label=str(watereq))
sc3 = ax.scatter(lons+0.07,lats,c=percentiles_wateqv,edgecolors='k',marker='o',cmap='Spectral',vmin=0,vmax=100,s=150) #,label=str(watereq))

#fig.colorbar(sc, ax=ax, location='bottom', shrink=0.7)

from matplotlib.lines import Line2D
legend_elements = fcts.create_legend('Spectral', num_bins=8)
legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Snow w. equiv.', markeredgecolor='k', markerfacecolor=None, markersize=12))
legend_elements.append(Line2D([0], [0], marker='s', color='w', label='Accumul. precip.', markeredgecolor='k', markerfacecolor=None, markersize=10))
legend_elements.append(Line2D([0], [0], marker='d', color='w', label='Temp. 30-day mean', markeredgecolor='k', markerfacecolor=None, markersize=12))
ax.legend(handles=legend_elements, loc='upper right', fontsize='small', title='Climatological\nPercentiles')

#
#legend_elmnts = []

#ax.legend(handles=legend_elmnts, loc='center right', fontsize='medium', title='SNOTEL variables')

#ax.legend(handles=create_legend(), loc='upper right', fontsize='small', title='SNOTEL\n% normal')
plt.tight_layout()
#fig.savefig(str(data_dir)+f'/fig_explainability_map_PDSI_SNOTEL_{site}_{issue_date}.pdf', bbox_inches='tight', pad_inches=0.035)
fig.savefig(str(data_dir)+f'/fig_explainability_map_PDSI_SNOTEL_{site}_{issue_date}.png', bbox_inches='tight', pad_inches=0.035)
#plt.show(); plt.clf(); plt.close('all')









'''
observed_PC_climatology = {}
for q in [0.01, 0.1, 0.5, 0.9, 0.99]:
    observed_PC_climatology[q] = X_all.groupby('dayofyear').quantile(q).loc[1:202]
    #observed_PC_climatology[q][:] = np.roll(observed_PC_climatology[q],15,axis=0)
    #observed_PC_climatology[q].index = np.roll(observed_PC_climatology[q].index,15,axis=0).astype(str)#.shift(3)#np.roll(observed_PC_climatology[q],3,axis=0)


for ds in dataset_names[1:]:
    plt.fill_between(x=observed_PC_climatology[0.1].index,y2=observed_PC_climatology[0.01][f'{ds}_PC1']-200,  y1=observed_PC_climatology[0.99][f'{ds}_PC1']-200,  color='b', alpha=0.15)
    plt.fill_between(x=observed_PC_climatology[0.1].index,y2=observed_PC_climatology[0.01][f'{ds}_PC2']-100,  y1=observed_PC_climatology[0.99][f'{ds}_PC2']-100,  color='r', alpha=0.15)
    plt.fill_between(x=observed_PC_climatology[0.1].index,y2=observed_PC_climatology[0.01][f'{ds}_PC3']-0,    y1=observed_PC_climatology[0.99][f'{ds}_PC3']-0,    color='g', alpha=0.15)
    plt.fill_between(x=observed_PC_climatology[0.1].index,y2=observed_PC_climatology[0.01][f'{ds}_PC4']+100,  y1=observed_PC_climatology[0.99][f'{ds}_PC4']+100,  color='c', alpha=0.15)
    plt.title(ds); plt.show()

'''
