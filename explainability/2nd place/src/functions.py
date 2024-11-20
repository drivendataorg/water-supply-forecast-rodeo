#!/usr/bin/env python



import itertools, glob
import numpy as np
import pandas as pd
import xarray as xr


from datetime import datetime














# --- Metadata ---


def all_sites():

    sites = ['hungry_horse_reservoir_inflow',
             'snake_r_nr_heise',
             'pueblo_reservoir_inflow',
             'sweetwater_r_nr_alcova',
             'missouri_r_at_toston',
             'animas_r_at_durango',
             'yampa_r_nr_maybell',
             'libby_reservoir_inflow',
             'boise_r_nr_boise',
             'green_r_bl_howard_a_hanson_dam',
             'taylor_park_reservoir_inflow',
             'dillon_reservoir_inflow',
             'ruedi_reservoir_inflow',
             'fontenelle_reservoir_inflow',
             'weber_r_nr_oakley',
             'san_joaquin_river_millerton_reservoir',
             'merced_river_yosemite_at_pohono_bridge',
             'american_river_folsom_lake',
             'colville_r_at_kettle_falls',
             'stehekin_r_at_stehekin',
             'detroit_lake_inflow',
             'virgin_r_at_virtin',
             'skagit_ross_reservoir',
             'boysen_reservoir_inflow',
             'pecos_r_nr_pecos',
             'owyhee_r_bl_owyhee_dam']
    
    return sites


def all_issue_times():
    
    issue_times = []
    for month,day in itertools.product( ['01', '02', '03', '04', '05', '06', '07'], 
                                        ['01', '08', '15', '22']):
        
        issue_times.append(month+'-'+day)
    
    return issue_times



def usgs_hucs(site):
    
    
    hucs = {'hungry_horse_reservoir_inflow':            '17010209',
            'snake_r_nr_heise':                         '17040104',
            'pueblo_reservoir_inflow':                  '11020002',
            'sweetwater_r_nr_alcova':                   '10180006',
            'missouri_r_at_toston':                     '10030101',
            'animas_r_at_durango':                      '14080104',
            'yampa_r_nr_maybell':                       '14050002',
            'libby_reservoir_inflow':                   '17010101',
            'boise_r_nr_boise':                         '17050114',
            'green_r_bl_howard_a_hanson_dam':           '17110013',
            'taylor_park_reservoir_inflow':             '14020001',
            'dillon_reservoir_inflow':                  '14010002',
            'ruedi_reservoir_inflow':                   '14010004',
            'fontenelle_reservoir_inflow':              '14040101',
            'weber_r_nr_oakley':                        '16020101',
            'san_joaquin_river_millerton_reservoir':    '18040001',
            'merced_river_yosemite_at_pohono_bridge':   '18040008',
            'american_river_folsom_lake':               '18020111',
            'colville_r_at_kettle_falls':               '17020003',
            'stehekin_r_at_stehekin':                   '17020009',
            'detroit_lake_inflow':                      '17090005',
            'virgin_r_at_virtin':                       '15010008',
            'skagit_ross_reservoir':                    '17110005',
            'boysen_reservoir_inflow':                  '10080005',
            'pecos_r_nr_pecos':                         '13060001',
            'owyhee_r_bl_owyhee_dam':                   '17050110',}
    
    return hucs[site]



def active_cdec_stations():
    # These contain data in 2023    
    
    station_dict = {
        'san_joaquin_river_millerton_reservoir':    ['BLC', 'CAP', 'CXS', 'FRN', 'GKS', 'HYS', 'LOS', 'SIL', 
                                                     'ALP', 'RBB', 'RBP', 'SCN', 'VVL', 'BLD', 'BLK', 'BLS', 
                                                     'BSK', 'DDM', 'EBB', 'EP5', 'FDC', 'FLL', 'GNL', 'HGM', 
                                                     'HHM', 'HOR', 'HRS', 'HVN', 'IDC', 'IDP', 'INN', 'LBD', 
                                                     'LVM', 'LVT', 'MDW', 'MNT', 'MRL', 'MSK', 'PLP', 'PSN', 
                                                     'RCC', 'REL', 'RP2', 'SDW', 'SLM', 'SPS', 'SPT', 'SQV', 
                                                     'SSM', 'TCC', 'TK2', 'WC3', 'BMW', 'CSL', 'GOL'],
        
        'american_river_folsom_lake':               ['CHM', 'DPO', 'GRM', 'GRV', 'HNT', 'KSP', 'PSR', 'TMR', 
                                                     'VLC', 'BCB', 'BGP', 'BIM', 'BSH', 'CFL', 'CMW', 'CRL', 
                                                     'CVM', 'DAN', 'ERY', 'FLV', 'GEM', 'GIN', 'HRS', 'KIB', 
                                                     'KUB', 'KUP', 'MHP', 'MPG', 'MTM', 'PDS', 'RCK', 'SLI', 
                                                     'SLK', 'STL', 'STR', 'SWM', 'TES', 'TNY', 'TUM', 'UBC', 
                                                     'UTY', 'VRG', 'WHW', 'WWC', 'GNF'],
        
        'merced_river_yosemite_at_pohono_bridge':   ['FLV', 'STR', 'TNY', 'CFL', 'CHM', 'CMW', 'CVM', 'DAN', 
                                                     'DDM', 'DPO', 'ERY', 'GEM', 'GIN', 'GNL', 'GRM', 'GRV', 
                                                     'HNT', 'HRS', 'KIB', 'KSP', 'KUP', 'LBD', 'LVM', 'LVT', 
                                                     'MHP', 'MPG', 'PDS', 'PSR', 'RCK', 'REL', 'SDW', 'SLI', 
                                                     'SPS', 'TES', 'TMR', 'TUM', 'UBC', 'VLC', 'VRG', 'WHW']}
    
    return station_dict




# --- Data correction ---


def replace_outliers_with_nans_moving_average(df, window_size=5, threshold=2):
    # Destroy suspicious data. This function identifies large deviations
    # from the recent past values, so it is suitable for real-time identification
    # of outliers.
    
    moving_avg = df.rolling(window=window_size).mean()
    deviation = np.abs(df - moving_avg)
    outliers = deviation > threshold * moving_avg.std()
    df[outliers] = np.nan
    
    return df



def replace_outliers_with_nans_iqr(df):
    # Destroy suspicious data. Use interquartile range to identify deviations:
    # This function is good for handling long historical time series.
    
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (df < lower_bound) | (df > upper_bound)
    df[outliers] = np.nan
    
    return df




def destroy_illegal(df, years):
    # Destroy data extending to potentially illegal time ranges
    
    previous_years = np.array(years).astype(int) - 1
    
    # Only data before July 21 allowed for a calendar year and after 01 October from the previous calendar year
    legal_period_1 = np.isin(df.index.year, years) & (df.index.month.values == 7) & (df.index.day.values <= 21)
    legal_period_2 = np.isin(df.index.year, years) & (df.index.month.values <= 6)
    legal_period_3 = np.isin(df.index.year, previous_years) & (df.index.month.values >= 10) 
    
    legal_period = legal_period_1 | legal_period_2 | legal_period_3
    
    illegal_period = ~ legal_period
    
    df.loc[illegal_period] = np.nan
    
    return df




from sklearn.base import BaseEstimator, TransformerMixin

class ClimatologyImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # Calculate multi-year median daily climatology for all samples
        self.climatology_ = X.groupby([X.index.month, X.index.day]).median()
        self.climatology_ = self.climatology_.interpolate().ffill().bfill().fillna(0)
        return self
    
    def transform(self, X):
        
        for col in X.columns:
            # Climatological column
            clim_col = pd.merge(X[[col]], self.climatology_[[col]], left_on=[X.index.month, X.index.day], 
                                right_index=True, how='left', suffixes=('', '_clim'))
            
            clim_col.index = X.index
            
            # Identify columns with all NaN values
            if X[col].isnull().all():
                X[col] = clim_col[col+'_clim']
            
            # Fill missing values with climatological values, adjust for trends
            else:
                diff_col = (X[col] - clim_col[col+'_clim']).ffill()
                X[col].fillna(clim_col[col+'_clim'] + diff_col, inplace=True)
                
                X[col].fillna(clim_col[col+'_clim'], inplace=True)
                
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)








# --- Reading and preparing input data sources from disk ---


def read_process_everything(data_dir, preprocessed_dir, site, years,
                            train_or_test='test', climatology_models=False, pca_models=False, 
                            remove_outliers=True, return_raw=False):
    
    windows = [1,2,6,10,20,30,]
    lags    = [0,2,6,10,15,20,25,30,35,40,45,50,55,60]
    ncomp   = 4
    
    
    
    data = {}; new_climatology_models = {}; new_pca_models = {}
    def append_data(input_data, clim_data, pca_data):
        
        if input_data is not None:
            data.update(input_data)
        
        if clim_data is not None:
            new_climatology_models.update(clim_data)
        
        if pca_data is not None:
            new_pca_models.update(pca_data)
    
    
    
    # Day-of-year predictor
    index = pd.date_range(str(np.min(years)-1)+'-10-01', str(np.max(years))+'-07-22')
    dayofyear = dayofyear_predictor(index)
    
    append_data(dayofyear, None, None)
    
    
    
    # Principal components of monthly local natural flow 
    pcs_nflow, clim_nflow, pca_nflow = \
        read_data_for_X(f'{data_dir}/', site, '1D', years, 'naturalflow', 
        climatology_models=climatology_models, pca_models=pca_models, windows=windows, lags=lags, 
        ncomp=ncomp+1, train_or_test=train_or_test, return_raw=return_raw) 

    append_data(pcs_nflow, clim_nflow, pca_nflow)
    
    
    
    # Principal components of local USGS daily antecedent streamflow for sites with currently available observations
    if site not in ['pueblo_reservoir_inflow', 'ruedi_reservoir_inflow', 'fontenelle_reservoir_inflow', 
                    'american_river_folsom_lake', 'skagit_ross_reservoir', 'sweetwater_r_nr_alcova',
                    'boise_r_nr_boise', 'boysen_reservoir_inflow', ]:
        
        pcs_usgs_local, clim_usgs_local, pca_usgs_local = read_data_for_X(
            f'{data_dir}usgs_streamflow/', f'*/{site}*', '1D', years, 'usgs_localflow', 
            fy_in_path=True, col_prefixes=['00060_Mean'], climatology_models=climatology_models, pca_models=pca_models,
            windows=windows, lags=lags, ncomp=ncomp+1, train_or_test=train_or_test, return_raw=return_raw) # 8
        
        append_data(pcs_usgs_local, clim_usgs_local, pca_usgs_local)
    
    
    
    # Principal components of USGS daily streamflow and other observations in the neighboring sites 
    pcs_usgs_neigh, clim_usgs_neigh, pca_usgs_neigh = read_data_for_X(
        f'{preprocessed_dir}usgs_neighborhood/', f'*{site}*', '1D', years, 'usgs_neighborflow', 
        climatology_models=climatology_models, pca_models=pca_models, windows=windows, lags=lags, 
        ncomp=ncomp+2, train_or_test=train_or_test, return_raw=return_raw) # 14
    
    append_data(pcs_usgs_neigh, clim_usgs_neigh, pca_usgs_neigh)
    
    
    
    # Principal components of SST-based teleindices
    pcs_teleind, clim_teleind, pca_teleind = \
        read_data_for_X(f'{data_dir}/teleindices/', 'teleindices', '1D', years, 'teleind_sst', 
        climatology_models=climatology_models, pca_models=pca_models, windows=windows, lags=lags, 
        ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) 

    append_data(pcs_teleind, clim_teleind, pca_teleind)
    
    
    
    # Principal components of MJO teleindices
    pcs_mjo, clim_mjo, pca_mjo = \
        read_data_for_X(f'{data_dir}/teleindices/', 'mjo', '1D', years, 'teleind_mjo', 
        climatology_models=climatology_models, pca_models=pca_models, windows=windows, lags=lags, 
        ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) 

    append_data(pcs_mjo, clim_mjo, pca_mjo)
    
    
    
    # Principal components of ECMWF seasonal forecast from CDS
    pcs_ecmwf_t2m, clim_ecmwf_t2m, pca_ecmwf_t2m = read_data_for_X(
        f'{preprocessed_dir}cds/', f'*{site}*', '1D', years, 'ecmwf_temperature', 
        col_prefixes=['t2m'], climatology_models=climatology_models, pca_models=pca_models, windows=[1], lags=[0],  
        ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) 
    
    pcs_ecmwf_tp, clim_ecmwf_tp, pca_ecmwf_tp    = read_data_for_X(
        f'{preprocessed_dir}cds/', f'*{site}*', '1D', years, 'ecmwf_precipitation', 
        col_prefixes=['tp'], climatology_models=climatology_models, pca_models=pca_models, windows=[1], lags=[0], 
        ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) 
    
    pcs_ecmwf_e, clim_ecmwf_e, pca_ecmwf_e       = read_data_for_X(
        f'{preprocessed_dir}cds/', f'*{site}*', '1D', years, 'ecmwf_evaporation', 
        col_prefixes=['e'], climatology_models=climatology_models, pca_models=pca_models, windows=[1], lags=[0], 
        ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) 
    
    pcs_ecmwf_ro, clim_ecmwf_ro, pca_ecmwf_ro       = read_data_for_X(
        f'{preprocessed_dir}cds/', f'*{site}*', '1D', years, 'ecmwf_runoff', 
        col_prefixes=['ro'], climatology_models=climatology_models, pca_models=pca_models, windows=[1], lags=[0], 
        ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) 
    
    pcs_ecmwf_rsn, clim_ecmwf_rsn, pca_ecmwf_rsn    = read_data_for_X(
        f'{preprocessed_dir}cds/', f'*{site}*', '1D', years, 'ecmwf_snowdensity', 
        col_prefixes=['rsn'], climatology_models=climatology_models, pca_models=pca_models, windows=[1], lags=[0], 
        ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) 
    
    pcs_ecmwf_sd, clim_ecmwf_sd, pca_ecmwf_sd    = read_data_for_X(
        f'{preprocessed_dir}cds/', f'*{site}*', '1D', years, 'ecmwf_snowdepth', 
        col_prefixes=['rsn'], climatology_models=climatology_models, pca_models=pca_models, windows=[1], lags=[0], 
        ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) 
    
    append_data(pcs_ecmwf_t2m, clim_ecmwf_t2m, pca_ecmwf_t2m)
    append_data(pcs_ecmwf_tp, clim_ecmwf_tp, pca_ecmwf_tp)
    append_data(pcs_ecmwf_e, clim_ecmwf_e, pca_ecmwf_e)
    append_data(pcs_ecmwf_ro, clim_ecmwf_ro, pca_ecmwf_ro)
    append_data(pcs_ecmwf_rsn, clim_ecmwf_rsn, pca_ecmwf_rsn)
    append_data(pcs_ecmwf_sd, clim_ecmwf_sd, pca_ecmwf_sd)
    
        
    
    # Principal components of Palmer Drought Severity Index 
    pcs_pdsi, clim_pdsi, pca_pdsi = read_data_for_X(
        f'{preprocessed_dir}pdsi/', f'*{site}*', '1D', years, 'pdsi', 
        climatology_models=climatology_models, pca_models=pca_models, windows=windows, lags=lags, 
        ncomp=ncomp+1, train_or_test=train_or_test, return_raw=return_raw) # 8
    
    append_data(pcs_pdsi, clim_pdsi, pca_pdsi)
    
    
    
    # Principal components of SNOTEL variables: precipitation, snow water equivalent, and temperature
    pcs_snotel_prc, clim_snotel_prc, pca_snotel_prc = read_data_for_X(
        f'{preprocessed_dir}snotel/', f'*{site}*', '1D', years, 'snotel_precipitation', 
        col_prefixes=['PREC'], remove_outliers=remove_outliers, climatology_models=climatology_models, pca_models=pca_models, 
        windows=windows, lags=lags, ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) # 8
    
    pcs_snotel_wtq, clim_snotel_wtq, pca_snotel_wtq = read_data_for_X(
        f'{preprocessed_dir}snotel/', f'*{site}*', '1D', years, 'snotel_waterequivalent', 
        col_prefixes=['WTEQ'], remove_outliers=remove_outliers, climatology_models=climatology_models, pca_models=pca_models, 
        windows=windows, lags=lags, ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) # 8
    
    pcs_snotel_tem, clim_snotel_tem, pca_snotel_tem = read_data_for_X(
        f'{preprocessed_dir}snotel/', f'*{site}*', '1D', years, 'snotel_temperature', 
        col_prefixes=['TAVG'], remove_outliers=remove_outliers, climatology_models=climatology_models, pca_models=pca_models, 
        windows=windows, lags=lags, ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) # 8
    
    append_data(pcs_snotel_prc, clim_snotel_prc, pca_snotel_prc)
    append_data(pcs_snotel_wtq, clim_snotel_wtq, pca_snotel_wtq)
    append_data(pcs_snotel_tem, clim_snotel_tem, pca_snotel_tem)
    
    
    
    if site in ['san_joaquin_river_millerton_reservoir', 'american_river_folsom_lake', 'merced_river_yosemite_at_pohono_bridge']:
        # Principal components of CDEC variables: precipitation, snow water equivalent, and temperature
        pcs_cdec_wtq, clim_cdec_wtq, pca_cdec_wtq = read_data_for_X(
            f'{preprocessed_dir}cdec/', f'*{site}*', '1D', years, 'cdec_waterequivalent', 
            col_prefixes=['SNOW WC'], remove_outliers=remove_outliers, climatology_models=climatology_models, pca_models=pca_models, 
            windows=windows, lags=lags, ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) # 8
        
        """
        pcs_cdec_prc, clim_cdec_prc, pca_cdec_prc = read_data_for_X(
            f'{preprocessed_dir}cdec/', f'*{site}*', '1D', years, 'cdec_prc', 
            col_prefixes=['RAIN'], remove_outliers=remove_outliers, climatology_models=climatology_models, pca_models=pca_models, 
            windows=windows, lags=lags, ncomp=3, train_or_test=train_or_test, return_raw=return_raw) # 8
        
        pcs_cdec_tem, clim_cdec_tem, pca_cdec_tem = read_data_for_X(
            f'{preprocessed_dir}cdec/', f'*{site}*', '1D', years, 'cdec_tem', 
            col_prefixes=['TEMP AV'], remove_outliers=remove_outliers, climatology_models=climatology_models, pca_models=pca_models, 
            windows=windows, lags=lags, ncomp=3, train_or_test=train_or_test, return_raw=return_raw) # 8
        """
        
        append_data(pcs_cdec_wtq, clim_cdec_wtq, pca_cdec_wtq)
        #append_data(pcs_cdec_prc, clim_cdec_prc, pca_cdec_prc)
        #append_data(pcs_cdec_tem, clim_cdec_tem, pca_cdec_tem)
    
    
    # Principal components of SWANN variables: accumulated precipitation, instantaneous precipitation, snow water equivalent
    pcs_swann_acc, clim_swann_acc, pca_swann_acc = read_data_for_X(
        f'{preprocessed_dir}swann/', f'*{site}*', '1D', years, 'swann_accumulprecip', 
        col_prefixes=['accumulated'], remove_outliers=False, climatology_models=climatology_models, pca_models=pca_models, 
        windows=windows, lags=lags, ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) # 8
    
    pcs_swann_ins, clim_swann_ins, pca_swann_ins = read_data_for_X(
        f'{preprocessed_dir}swann/', f'*{site}*', '1D', years, 'swann_instantanprecip', 
        col_prefixes=['instantaneous'], remove_outliers=False, climatology_models=climatology_models, pca_models=pca_models, 
        windows=windows, lags=lags, ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) # 8
        
    pcs_swann_swe, clim_swann_swe, pca_swann_swe = read_data_for_X(
        f'{preprocessed_dir}swann/', f'*{site}*', '1D', years, 'swann_waterequivalent', 
        col_prefixes=['average'], remove_outliers=False, climatology_models=climatology_models, pca_models=pca_models, 
        windows=windows, lags=lags, ncomp=ncomp, train_or_test=train_or_test, return_raw=return_raw) # 8
    
    append_data(pcs_swann_acc, clim_swann_acc, pca_swann_acc)
    append_data(pcs_swann_ins, clim_swann_ins, pca_swann_ins)
    append_data(pcs_swann_swe, clim_swann_swe, pca_swann_swe)
    
    
    if pca_models:
        return data, climatology_models, pca_models

    return data, new_climatology_models, new_pca_models










from pandas.tseries.offsets import MonthEnd

def read_data_for_X(data_path, file_pattern, data_freq, years, data_prefix, 
                    fy_in_path=False, col_prefixes=False, 
                    climatology_models=False, pca_models=False, remove_outliers=False,
                    windows=[0,5,10,20,30], lags=[0,10,30], ncomp=5, train_or_test='test', return_raw=False):
    
    '''
    Function for reading the preprocessed input data sources. 
    Reads different data in the CSV format, treats potential outliers, calculates 
    moving averages, lags, and differences, and then extracts the most important principal components from the data.
    
    Also destroys data extending to outside each forecast year / water year, which
    eliminates the risk of using illegal data accidentally. See also function calc_mave_lags_diff.
    '''
    
    #print(f'Reading and deriving predictors for {data_prefix} {file_pattern}')
    
    
    try:
        if data_prefix == 'teleind_sst':
            # Read teleconnection indices and resample to daily resolution
            df_data = pd.read_csv(f'{data_path}teleindices_monthly.csv', parse_dates=True, index_col='time')
            df_data = df_data.resample(data_freq).mean().ffill(limit=31)
        
        elif data_prefix == 'teleind_mjo':
            # Read MJO and resample to daily resolution
            df_data = pd.read_csv(f'{data_path}mjo_index.csv', parse_dates=True, index_col='time')
            df_data = df_data.resample(data_freq).mean().ffill(limit=7)
        
        elif data_prefix == 'naturalflow':
            # Monthly naturalized flow
            df_prior = pd.read_csv(f'{data_path}prior_historical_monthly_flow.csv', parse_dates=[['year','month']])
            df_crsvl = pd.read_csv(f'{data_path}cross_validation_monthly_flow.csv', parse_dates=[['year','month']])
            df_data = pd.concat([df_prior, df_crsvl], axis=0)
            
            df_data = df_data.pivot(index='year_month', columns='site_id', values='volume')[[file_pattern]]
            df_data.index = df_data.index + MonthEnd(); df_data.rename_axis('time',inplace=True)
            df_data = df_data.rename(columns={file_pattern:'naturalflow'})
            
            df_data = df_data.resample(data_freq).mean().ffill(limit=31)
            all_dates = pd.date_range(f'{min(years)-1}-10-01', f'{max(years)}-07-22', freq=data_freq)
            
            df_data = df_data.reindex(all_dates)
        
        else:
            # Read specified years only
            data = []
            for year in years:
                try:
                    if not fy_in_path: file = glob.glob(f'{data_path}{file_pattern}*{year}*.csv')[0]
                    if fy_in_path:     file = glob.glob(f'{data_path}*{year}*{file_pattern}.csv')[0]
                    
                    df = pd.read_csv(file, index_col=0, parse_dates=True).tz_localize(None)
                    
                    # Remove quality code columns
                    df = df.drop(df.filter(regex='_cd$').columns, axis=1)
                    
                    # Identify and select certain columns
                    if col_prefixes: 
                        for prfx, col in itertools.product(col_prefixes, df.columns):
                            if not prfx in col:
                                df = df.drop(columns=col)
                    
                    data.append(df.apply(pd.to_numeric, errors='coerce').rename_axis('time'))
                except:
                    pass
            
            df_data = pd.concat(data, axis=0)
            
            # Ensure correct data frequency and number of samples in the data
            df_data = df_data.resample(data_freq).mean()
        
        
        # Remove outliers
        if remove_outliers:
            if train_or_test=='train':
                df_data_filtered_1 = replace_outliers_with_nans_moving_average(df_data.copy(deep=True))
                df_data_filtered_2 = replace_outliers_with_nans_iqr(df_data_filtered_1)
                
                '''
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2,1, figsize=(10,6))
                ax[0].plot(df_data);            ax[0].set_title(f'Non-filtered data for {data_prefix} at {file_pattern}')  
                ax[1].plot(df_data_filtered_2); ax[1].set_title(f'Cleaned data for {data_prefix} at {file_pattern}')
                plt.tight_layout(); fig.savefig(f'{data_path}fig_{data_prefix}_{file_pattern}.png')
                plt.clf(); plt.close('all')
                '''
                
            # Only moving average filter can be applied for test phase because of less data
            if train_or_test=='test':
                df_data_filtered_2 = replace_outliers_with_nans_moving_average(df_data.copy(deep=True))
            
            
            df_data = df_data_filtered_2.copy(deep=True)
        
        
        
        if train_or_test=='test':
            year_now = datetime.now().year
            year_prv = datetime.now().year - 1
            FY_dates = pd.date_range(f'{year_prv}-10-01', f'{year_now}-07-22', freq='1D')
            
            df_data = df_data.reindex(FY_dates)
        
        if return_raw:
            return {data_prefix: df_data}, None, None
        
        # Impute with climatology
        if climatology_models:
            clim = climatology_models[data_prefix]
        
        if not climatology_models:
            climatology_models = {}
            clim = ClimatologyImputer().fit(df_data)

            climatology_models[data_prefix] = clim
        
        # Impute NaNs with climatology
        df_data = clim.transform(df_data)
        
        # Destroy data extending to potentially illegal time ranges
        df_data = destroy_illegal(df_data, years)
        
        # Calculate moving averages, lags, and differences. This creates new NaNs to the data
        lagged_data = calc_mave_lags_diff(df_data, windows, lags)
        
        
        # Replace values with NaNs where at least one value in the row is NaN
        lagged_data[lagged_data.isnull().any(axis=1)] = np.nan
        
        index = lagged_data.index
        all_nan_steps = np.all(np.isnan(lagged_data), axis=1)
        all_ok_steps = ~all_nan_steps
        
        
        # Apply principal component analysis
        if pca_models:
            components, pca = apply_PCA(lagged_data.loc[all_ok_steps], ncomp, pca_models[data_prefix])
        
        if not pca_models:
            components, pca = apply_PCA(lagged_data.loc[all_ok_steps], ncomp)
            pca_models = {}
            pca_models[data_prefix] = pca
        
        # Arrange the principal components to a dataframe
        df_components = pd.DataFrame(data=components, index=lagged_data.index[all_ok_steps],
                                 columns=[f'{data_prefix}_PC{i}' for i in range(1, ncomp + 1)])
        
        df_components = df_components.reindex(index)
        
         
        df_components = destroy_illegal(df_components, years)
        df_components = df_components.loc[f'{np.min(years)-1}-10-01':f'{np.max(years)}-07-22']
        
        data_out = {data_prefix: df_components}
        print(f'{data_prefix} {file_pattern} OK')
    
    except:
        data_out = None; climatology_models = None; pca_models = None
        print(f'{data_prefix} {file_pattern} FAILED')
    
    return data_out, climatology_models, pca_models





def calc_mave_lags_diff(df, windows, lags):
    """
    This function calculates moving averages and lags and differences from
    time series. To avoid using data outside each water year, only certain windows 
    and lags are allowed. Window widths and lags depend on the issue date: the earlier the issue date, 
    the shorter windows and lags can be used to not overlap with the previous year. 
    
    Examples producing correct/legal data:
    
    Issue date 01 Jan, DAILY data:
    Maximum window width + Maximum lag = 90 steps
    
    Issue date 01 Mar, DAILY data:
    Maximum window width + Maximum lag = 150 steps
    """
    
    # Calculate moving averages
    mave = []
    for col in df:
        for window in windows:
            sma_col = col+'_sma='+str(window)
            ewm_col = col+'_ewm='+str(window)
            
            # Separately: simple moving average and exp. weighted mean
            sma = df[[col]].rolling(window=window, min_periods=window).mean()
            ewm = df[[col]].ewm(span=window, min_periods=window, adjust=False).mean()
            
            mave.append(sma.rename(columns={col:sma_col}))
            mave.append(ewm.rename(columns={col:ewm_col}))
            
    
    mave = pd.concat(mave, axis=1)
    
    # Lag and differentiate 
    lagged = []
    for col in mave.columns:
        for lag in lags:
            
            mave_lagged = mave[[col]].shift(lag)
            mave_differ = mave[[col]].diff(lag+1)
            
            lagged.append(mave_lagged.rename(columns={col:col+'_lag='+str(lag)}) )
            lagged.append(mave_differ.rename(columns={col:col+'_dif='+str(lag+1)}) )
    
    lagged = pd.concat(lagged, axis=1)
    
    return lagged




'''

def calc_mave_lags_diff(df, windows, lags):
    
    # Calculate moving averages
    mave = []
    for col in df:
        for window in windows:
            sma_col = col+'_sma='+str(window)
            
            sma = df[[col]].rolling(window=window, min_periods=window).mean()
            
            mave.append(sma.rename(columns={col:sma_col}))
            
    
    mave = pd.concat(mave, axis=1)
    
    # Lag and differentiate 
    lagged = []
    for col in mave.columns:
        for lag in lags:
            lag_col = col+'_lag='+str(lag)
            dif_col = col+'_dif='+str(lag+1)
            
            mave_lag = mave[[col]].shift(lag)
            mave_dif = mave[[col]].diff(lag+1)
            
            lagged.append(mave_lag.rename(columns={col:lag_col}))
            lagged.append(mave_dif.rename(columns={col:dif_col}))
    
    lagged = pd.concat(lagged, axis=1)
    
    return lagged










from sklearn.linear_model import LinearRegression
def calculate_linear_trend(window_data, time):
    # Filter out NaN values
    valid_indices = ~np.isnan(window_data)
    filtered_data = window_data[valid_indices].reshape(-1, 1)
    filtered_time = time[valid_indices].reshape(-1, 1)
    
    if len(filtered_data) < 2:
        # Not enough data to calculate a trend
        return np.nan
    else:
        # Fit linear model to filtered data
        model = LinearRegression()
        model.fit(filtered_time, filtered_data)
        return model.coef_[0][0]  # Return slope of the linear trend


import pandas as pd
from scipy.stats import skew, kurtosis

def calc_mave_lags_diff(df, windows, lags):
    """
    Enhanced function to calculate moving averages, lags, differences, and additional
    statistical features (volatility, skewness, kurtosis) from time series data.
    
    :param df: DataFrame with time series data.
    :param windows: List of integers for moving average windows.
    :param lags: List of integers for lag features.
    """
    
    # Initialize DataFrame for results
    results = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        for window in windows:
            # Calculate simple moving average (SMA) and exponential weighted mean (EWM)
            sma_col = f'{col}_sma_{window}'
            ewm_col = f'{col}_ewm_{window}'
            results[sma_col] = df[col].rolling(window=window, min_periods=1).mean()
            results[ewm_col] = df[col].ewm(span=window, adjust=False).mean()
            
            # Calculate volatility (rolling standard deviation)
            vol_col = f'{col}_vol_{window}'
            results[vol_col] = df[col].rolling(window=window, min_periods=1).std()
            
            """
            # Calculate skewness
            skew_col = f'{col}_skew_{window}'
            results[skew_col] = df[col].rolling(window=window, min_periods=1).apply(skew, raw=False)
            
            # Calculate kurtosis
            kurt_col = f'{col}_kurt_{window}'
            results[kurt_col] = df[col].rolling(window=window, min_periods=1).apply(kurtosis, raw=False)
            """
            
    # Calculate lagged and differenced features
    for col in results.columns:
        for lag in lags:
            lagged_col = f'{col}_lag_{lag}'
            diffed_col = f'{col}_diff_{lag}'
            results[lagged_col] = results[col].shift(lag)
            results[diffed_col] = results[col].diff(lag+1)
    
    return results
    





def calc_mave_lags_diff(df, windows, lags):
    """
    This function calculates moving averages and lags and differences from
    time series. To avoid using data outside each water year, only certain windows 
    and lags are allowed. Window widths and lags depend on the issue date: the earlier the issue date, 
    the shorter windows and lags can be used to not overlap with the previous year. 
    
    Examples producing correct/legal data:
    
    Issue date 01 Jan, DAILY data:
    Maximum window width + Maximum lag = 90 steps
    
    Issue date 01 Mar, DAILY data:
    Maximum window width + Maximum lag = 150 steps
    """
    
    # Calculate moving averages
    mave = []
    for col in df:
        for window in windows:
            sma_col = col+'_sma='+str(window)
            med_col = col+'_sma='+str(window)
            ewm_col = col+'_ewm='+str(window)
            vol_col = col+'_vol='+str(window)
            
            # Separately: simple moving average and exp. weighted mean
            #sma = df[[col]].rolling(window=window, min_periods=window).mean()
            med = df[[col]].rolling(window=window, min_periods=window).median()
            ewm = df[[col]].ewm(span=window, min_periods=window, adjust=False).mean()
            #vol = df[[col]].rolling(window=window, min_periods=window).std()
            #print(vol.dropna())
            
            #mave.append(sma.rename(columns={col:sma_col}))
            mave.append(ewm.rename(columns={col:ewm_col}))
            mave.append(med.rename(columns={col:med_col}))
            
    
    mave = pd.concat(mave, axis=1)
    
    # Lag and differentiate 
    lagged = []
    for col in mave.columns:
        for lag in lags:
            
            mave_lagged = mave[[col]].shift(lag)
            mave_differ = mave[[col]].diff(lag+1)
            
            lagged.append(mave_lagged.rename(columns={col:col+'_lag='+str(lag)}) )
            lagged.append(mave_differ.rename(columns={col:col+'_dif='+str(lag+1)}) )
    
    lagged = pd.concat(lagged, axis=1)
    
    return lagged


'''



def dayofyear_predictor(index):
    
    index = pd.to_datetime(index)
    df = pd.DataFrame(index=index)
    df['dayofyear'] = index.dayofyear
    
    return {'dayofyear': df}




# --- Data analysis and evaluation ---



"""

def extract_shap_values(X, quantile_model, fitted_explainers=False):
    
    import shap
    
    all_estimators = quantile_model.estimators_
    all_features   = quantile_model.estimators_features_

    if not fitted_explainers: 
        explainers = []
    else:
        explainers = fitted_explainers

    shap_values = [] # pd.DataFrame(columns=['estimator_index']+list(X.columns))
    
    #model_for_shap = ShapQuantileWrapper(quantile_model.estimator_, 1)
    #model_for_shap = ShapQuantileWrapper(quantile_model, 1)
    
    #explainer = shap.Explainer(model_for_shap.predict, X)
    #shap_values = explainer(X)
        
    model_for_shap = ShapBaggingQuantileWrapper(quantile_model, quantile_index=1)  # Example for the second quantile
    explainer = shap.Explainer(model_for_shap.predict, X)
    shap_values = explainer(X)

    # Plot SHAP values
    shap.summary_plot(shap_values, X)
    
    i=0
    for estimator, features in zip(all_estimators, all_features):
        #print(estimator, features)
        
        estimator_columns = list(X.iloc[:,features].columns)
        
        #model_q0p1 = estimator.models[0]
        model_q0p5 = estimator.models[1]
        #model_q0p9 = estimator.models[2]
        
        if not fitted_explainers:
            #explainer_q0p1 = shap.Explainer(model_q0p1, X.iloc[:,features])
            explainer_q0p5 = shap.Explainer(model_q0p5, X.iloc[:,features])
            #explainer_q0p9 = shap.Explainer(model_q0p9, X.iloc[:,features])
            explainers.append(explainer_q0p5)
            
        else:
            explainer_q0p5 = explainers[i]
        
        
        #shap_values_q0p1 = explainer_q0p1.shap_values(X.iloc[:,features])
        shap_values_q0p5 = explainer_q0p5.shap_values(X.iloc[:,features])
        shap_values_q0p5 = explainer_q0p5(X.iloc[:,features])
        #shap_values_q0p9 = explainer_q0p9.shap_values(X.iloc[:,features])
        
        shap_values_q0p5 = pd.DataFrame(index=X.index, columns=estimator_columns, data=shap_values_q0p5)
        #shap_values_q0p5['estimator_index'] = i
        
        shap_values.append(shap_values_q0p5)#.reset_index,drop=True)
        
        i+=1
        # shap.summary_plot(shap_values_q0p1, feature_names=estimator_columns, max_display=7, plot_type='violin')
        # shap.decision_plot(explainer_q0p5.expected_value, shap_values_q0p5[0], feature_names=estimator_columns, plot_color='BrBG')
        # shap.dependence_plot('usgs_neighborflow_PC1', shap_values_q0p5, X_test.iloc[:,features])
        # shap.force_plot(explainer_q0p5.expected_value, shap_values_q0p5, feature_names=estimator_columns, link='identity', plot_cmap='RdBu', matplotlib=True)
        # shap.monitoring_plot(1, shap_values_q0p5, X_test.iloc[:,features], show=True)
        # shap.bar_plot(shap_values_q0p5, X_test.iloc[:,features], estimator_columns)
    
    shap_values = pd.concat(shap_values, axis=0).fillna(0)
    #shap_values = shap_values.reset_index(drop=True)
    
    '''
    if len(np.array(shap_values).shape) == 3:
        shap_values = np.array(shap_values).mean(axis=0)
    
    data = pd.DataFrame(np.abs(shap_values).mean(axis=0), index=list(X_test.columns))
    names, importance = list(data.index), data.values.squeeze()
    
    
    shap_values = pd.DataFrame(columns=X_test.columns, index=X_test.index, data=shap_values)
    '''
    
    #return sorted_names, sorted_importance, shap_values
    #return names, importance, shap_values
    return shap_values, explainers
"""



import shap
def extract_shap_values(X, quantile_model, fitted_explainer=False):
    
    
    if not fitted_explainer: 
        # quantile_index=0 corresponds to Q10
        # quantile_index=1 corresponds to Q50 
        # quantile_index=2 corresponds to Q90 
        model_for_shap = ShapBaggingQuantileWrapper(quantile_model, quantile_index=1)  
        explainer = shap.Explainer(model_for_shap.predict, X)
    else:
        explainer = fitted_explainer
    
    
    shap_values = explainer(X)
    
    # Plot SHAP values
    #shap.summary_plot(shap_values, X, max_display=20, show=False); plt.tight_layout(); plt.show()
    #shap.plots.bar(shap_values ,max_display=40, show=False); plt.tight_layout(); plt.show()
    #shap.plots.waterfall(shap_values[100], max_display=20, show=False); plt.tight_layout(); plt.show()
    
    
    return shap_values, explainer



from sklearn.utils.validation import check_is_fitted
class ShapBaggingQuantileWrapper:
    def __init__(self, bagging_model, quantile_index):
        self.bagging_model = bagging_model
        self.quantile_index = quantile_index
    
    def predict(self, X):
        # Ensure the BaggingRegressor and its base models are fitted
        check_is_fitted(self.bagging_model, ['estimators_', 'estimators_features_'])
        
        # Aggregate predictions from all estimators for the specific quantile
        predictions = []
        for estimator, features in zip(self.bagging_model.estimators_, self.bagging_model.estimators_features_):
            X_subset = X.iloc[:, features]  # Select the same subset of features used during training
            pred = estimator.predict(X_subset)[:, self.quantile_index]  # Predict for the specific quantile
            predictions.append(pred)
        
        return np.mean(predictions, axis=0)





def interval_coverage(true_values, lower_quantile_preds, upper_quantile_preds):
    
    # Check if the lengths of true values and predictions match
    assert len(true_values) == len(lower_quantile_preds) == len(upper_quantile_preds), "Lengths do not match"

    # Count the number of true values within the predicted interval
    correct_predictions = np.sum((true_values >= lower_quantile_preds) & (true_values <= upper_quantile_preds))

    # Calculate interval coverage
    coverage = correct_predictions / len(true_values)

    return coverage



from sklearn.metrics import mean_pinball_loss
def calc_pbloss(obs,mod,q=0.5): 
    
    return mean_pinball_loss(obs, mod, alpha=q)*2




def calc_pbloss_all(obs,mod,quantiles): 
    
    losses = []
    for i, q in enumerate(quantiles):
        loss = mean_pinball_loss(obs, mod[:,i], alpha=q)*2
        losses.append(loss)
    
    return np.mean(losses)



from sklearn.metrics import r2_score
def calc_r2ss(obs,mod,multioutput='raw_values'): 
    
    
    mask_o = np.isnan(obs) | np.isinf(obs)
    mask_m = np.isnan(mod) | np.isinf(mod)
    mask = mask_o + mask_m
    
    if len(obs[~mask]) == 0 or len(mod[~mask]) == 0: 
        result = np.nan
    else:
        result = r2_score(obs[~mask], mod[~mask], multioutput=multioutput)
    
    return result




def calc_rmse(a,b,axis=0): 
    return np.sqrt(np.nanmean((a-b)**2, axis=axis))

def calc_mse(a,b,axis=0):  
    return np.nanmean((a-b)**2, axis=axis)
    

def calc_sprc(a, b):
    from scipy import stats
    return stats.spearmanr(a, b, nan_policy='omit')


def calc_nse(obs,mod):
    return 1-(np.nansum((obs-mod)**2)/np.nansum((obs-np.nanmean(mod))**2))


def calc_corr(a, b, axis=0):
    mask_a = np.isnan(a) | np.isinf(a)
    mask_b = np.isnan(b) | np.isinf(b)
    mask = mask_a + mask_b
    _a = a.copy()
    _b = b.copy()
    try:
        _a[mask] = np.nan
        _b[mask] = np.nan
    except:
        pass
    _a = _a - np.nanmean(_a, axis=axis, keepdims=True)
    _b = _b - np.nanmean(_b, axis=axis, keepdims=True)
    std_a = np.sqrt(np.nanmean(_a**2, axis=axis)) 
    std_b = np.sqrt(np.nanmean(_b**2, axis=axis)) 
    return np.nanmean(_a * _b, axis=axis)/(std_a*std_b)







# --- Manipulators ---




def Gauss_filter(data, sigma=(0,1,1), mode='wrap'):
    """ Smooth data (spatially in 3D as default) using Gaussian filter """   
    import scipy.ndimage.filters as flt
    
    try: data_vals=data.values
    except: data_vals=data
    
    U=data_vals.copy()
    V=data_vals.copy()
    V[np.isnan(U)]=0
    VV=flt.gaussian_filter(V,sigma=sigma, mode=mode)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=flt.gaussian_filter(W,sigma=sigma, mode=mode)

    Z=VV/WW
    Z[np.isnan(U)] = np.nan
    
    data[:] = Z
    
    return data #Z



def regrid_dataset(ds, lat_new, lon_new, method='linear'):
    # Interpolate using xarray's interp method
    return ds.interp(lat=lat_new, lon=lon_new, method=method)





    
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
def apply_PCA(data, ncomp, pca_model_in=False, svd_solver='randomized'): 
    """ Decomposition of data with principal component analysis. 
        Assumes input data to be of shape (time, predictor). 
        Assumes input does not contain missing (NaN) data.
    """
    
    
    if data.shape[1] < ncomp:
        ncomp = data.shape[1]
        print('Reducing number of components to',ncomp)
    
    
    if pca_model_in:
        pca_model = pca_model_in
        cps = pca_model.transform(data)
    
    if not pca_model_in:
        pca_model = PCA(n_components=ncomp, whiten=False, random_state=99, svd_solver=svd_solver).fit(data)
        cps = pca_model.transform(data)
    
    return cps, pca_model








# --- Indexing and data selection/extraction ---


def select_window_around_issuetime(df, issue_time, ndays, fill=False):
    # Selecting target data for fitting
    
    years = np.unique(df.index.year).astype(str)
    
    out = []
    
    for year in years:
        bgn = pd.to_datetime(f'{year}-{issue_time}') - pd.DateOffset(days=ndays)
        end = pd.to_datetime(f'{year}-{issue_time}') + pd.DateOffset(days=ndays)
        
        df_window = df.loc[bgn:end]
        
        if fill:
            df_window = df_window.ffill().bfill().dropna()
        
        out.append(df_window.copy())
        

    return pd.concat(out, axis=0)







def match_target_and_issue_dates(df_target):
    """ Match the end-of-season target values with the issue dates.
    """
    from datetime import timedelta
    
    y = df_target.copy(deep=True)
    y = y.resample('1D').mean()
    
    site = list(y.columns)[0]
    
    for year,month,day in itertools.product(y.index.year.unique().astype(str), 
                                            ['01', '02', '03', '04', '05', '06', '07'], 
                                            ['01', '08', '15', '22']): 
        col = f'{month}-{day}'
        if not col in y.columns:
            y[col] = np.nan
        
        issue_date = pd.Timestamp(f'{year}-{month}-{day}')
        
        if site=='detroit_lake_inflow':
            end_of_season = pd.Timestamp(f'{year}-06-30')
        else:
            end_of_season = pd.Timestamp(f'{year}-07-31')
        
        y.loc[issue_date, col] = y.loc[end_of_season, site]
    
    return pd.DataFrame(y)



def define_split_years(all_years, valid_years_idx, train_years_idx):
    
    train_years = all_years[train_years_idx]
    valid_years = all_years[valid_years_idx]
    
    return train_years, valid_years





def find_forecast_year(list_of_dates):
    import pandas as pd
    
    return [pd.to_datetime(item).year - 1 if   pd.to_datetime(item).month >= 10 and pd.to_datetime(item).month <= 12 \
                                          else pd.to_datetime(item).year for item in list_of_dates]




def define_split_details(predictor_data_train, predictor_data_valid, 
                         train_years, valid_years, y, issue_time): #, drop=False):
    
    
    # Join the individual datasets into one predictor matrix
    x_val = pd.concat(predictor_data_valid.values(), axis=1)
    
    # Shift the predictor data of the validation set one day to not break the competition rules
    x_val = x_val.shift(1)
    
    # Find the end-of-season target data 
    y_val = y.loc[np.isin(y.index.year, np.unique(list(valid_years)+list(valid_years-1))), [issue_time]]
    
    # Ensure x and y are synchronized
    merged_val = pd.concat([x_val, y_val], axis=1).loc[y_val.index].rename_axis('time')#.dropna()
    
    # Separate X and Y again
    X_val = merged_val.loc[:,list(x_val.columns)] 
    Y_val = merged_val.loc[:,list(y_val.columns)]

    
    
    try:
        # Join the individual datasets into one predictor matrix
        x_trn = pd.concat(predictor_data_train.values(), axis=1)
        
        # Shift the predictor data of the training set 
        x_trn = x_trn.shift(1)
        
        # Find the end-of-season target data 
        y_trn = y.loc[np.isin(y.index.year, np.unique(list(train_years)+list(train_years-1))), [issue_time]]
        
        # Ensure x and y are synchronized
        merged_trn = pd.concat([x_trn, y_trn], axis=1).loc[y_trn.index].rename_axis('time').dropna()
        
        
        # Separate X and Y again
        X_trn = merged_trn.loc[:,list(x_trn.columns)] 
        Y_trn = merged_trn.loc[:,list(y_trn.columns)] 
        
        
        # Add more weight to samples containing recent data
        sample_weights = define_sample_weights(X_trn)
        
    except:
        print('No training data')
        X_trn, Y_trn = None, None
        sample_weights = None
    
    
    return X_trn, Y_trn, X_val, Y_val, sample_weights








def define_sample_weights(X_trn):
    
    sample_weights = np.array([1.0] * len(X_trn))
    
    '''
    # Add more weight to samples representing the most recent years
    idxs_1990 = Y_trn.index.year >= 1990
    idxs_2004 = Y_trn.index.year >= 2004
    #idxs_2015 = Y_trn.index.year >= 2015
    sample_weights[idxs_1990] += 3
    sample_weights[idxs_2004] += 3
    #sample_weights[idxs_2015] += 2
    '''
    
    return sample_weights








def last_day_of_month(year, month):
    # Find the last day of the month using calendar.monthrange
    import calendar
    
    _, last_day = calendar.monthrange(year, month)
    return last_day



def bool_index_to_int_index(bool_index):
    return np.where(bool_index)[0]


def adjust_lats_lons(ds):
    coord_names =   [['longitude', 'latitude'],
                    ['X', 'Y'],]
    
    for nmes in coord_names:
        try:
            ds = ds.rename({nmes[0]: 'lon', nmes[1]: 'lat'}) 
        except: 
            pass  
    
    
    if(ds.lon.values.max() > 180):
        print('Transforming longitudes from [0,360] to [-180,180]', flush=True)
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        
    return ds.sortby(['lon','lat'])


def nearest_gridpoints(ds, points):
    import xarray as xr
    
    grid = xr.Dataset({"lat": (["lat"], ds.lat.values),"lon": (["lon"], ds.lon.values),})
    
    points_grid = grid.sel(lat=xr.DataArray(np.array(points)[:,0], dims='gcl'), \
                           lon=xr.DataArray(np.array(points)[:,1], dims='gcl'), method='nearest')
    
    nearest_points = [tuple(row) for row in np.array([points_grid.gcl.lat.values, 
                                                      points_grid.gcl.lon.values]).T]
    
    return nearest_points



def extract_timestamp(filename):
    import re
    
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2})', filename)
    if match:
        return match.group(1)
    
    return None











def catchment_extent(geospat_file_path, dDeg):
    import geopandas as gpd
    
    # Read the spatial extent of catchment areas
    gdf = gpd.read_file(geospat_file_path)
    
    spatial_extents = {}
    for index, row in gdf.iterrows():
        site_id = row['site_id']
        bbox = row['geometry'].bounds
        
        latmin, lonmin, latmax, lonmax = bbox[1], bbox[0], bbox[3], bbox[2]
        spatial_extents[site_id] = [latmin-dDeg, lonmin-dDeg, latmax+dDeg, lonmax+dDeg]
    
    return spatial_extents




























# --- Fitting and optimizing ---





from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LassoLars, LassoLarsCV, QuantileRegressor
from sklearn.model_selection import KFold

from sklearn.exceptions import ConvergenceWarning
import warnings; warnings.filterwarnings('ignore', category=ConvergenceWarning)


def bagging_multioutput_model(X, Y, params, quantiles, sample_weights):
    
    X = X.ffill().fillna(0)
    Y = Y.ffill().fillna(0)
    
    # Duplicate Y for each quantile
    Y_multi = np.tile(Y, (1, len(quantiles)))
    
    # Initialize the custom MultiQuantileRegressor with the specified quantiles
    base_estimator = MultiQuantileRegressor(alpha=params['alpha'], 
                                            solver='highs', 
                                            quantiles=quantiles)
    
    # Use BaggingRegressor with the custom multi-output estimator
    bagging_ensemble = BaggingRegressor(
                        estimator=base_estimator,
                        n_estimators=params['n_estimators'],
                        max_samples=params['p_smpl'], 
                        max_features=params['p_feat'],
                        bootstrap=params['bootstrap'],   
                        bootstrap_features=params['bootstrap_features'],
                        oob_score=False,
                        n_jobs=params['n_jobs'], 
                        random_state=99,
                        verbose=False).fit(X, Y_multi, sample_weight=sample_weights)
    
    return bagging_ensemble







from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MultiQuantileRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, quantiles, alpha=1.0, solver='highs'):
        self.quantiles = quantiles
        self.alpha = alpha
        self.solver = solver
        # Initialize models here without setting quantile, alpha, or solver
        self.models = []

    def fit(self, X, Y, sample_weight=None):
        X, Y = check_X_y(X, Y, multi_output=True, y_numeric=True)
        self.models = []  # Clear previous models
        for i, q in enumerate(self.quantiles):
            model = QuantileRegressor(quantile=q, alpha=self.alpha, solver=self.solver)
            model.fit(X, Y[:, i], sample_weight=sample_weight)
            self.models.append(model)
        
        return self
    
    def predict(self, X):
        # Check if fit had been called
        check_is_fitted(self, 'models')
        # Input validation
        X = check_array(X)
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return predictions

    def get_params(self, deep=True):
        # Return parameters to enable sklearn's clone operation
        return {"quantiles": self.quantiles, "alpha": self.alpha, "solver": self.solver}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self













import optuna

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.ensemble import HistGradientBoostingRegressor

def optuna_objective(trial, X, Y, quantile):
    
    
    param = {
        'n_jobs': -1,
        'n_estimators': 200,
        'alpha':  1.0, #trial.suggest_float('alpha', 1e-3, 1e1, log=True),
        'bootstrap': False, #trial.suggest_categorical('bootstrap', [True, False]),
        'bootstrap_features': False, #trial.suggest_categorical('bootstrap_features', [True, False]),
        'p_smpl': trial.suggest_float('p_smpl', 0.3, 1.0), 
        'p_feat': trial.suggest_float('p_feat', 0.1, 1.0)}
    
    
    
    years = np.unique(X.index.year)
    
    kf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=99)
    
    pb_scores = []
    for train_year_idx, valid_year_idx in kf.split(years):
        
        train_idx = np.isin(X.index.year, years[train_year_idx])
        valid_idx = np.isin(X.index.year, years[valid_year_idx])
        
        X_trn, Y_trn = X.iloc[train_idx], Y.iloc[train_idx]
        X_val, Y_val = X.iloc[valid_idx], Y.iloc[valid_idx]
        
        sample_weights = define_sample_weights(X_trn)
        
        model = bagging_multioutput_model(X_trn, Y_trn, param, quantile, sample_weights)
        pred_labels = model.predict(X_val.ffill().fillna(0))
        pb_score = calc_pbloss_all(Y_val, pred_labels, quantile)
        
        pb_scores.append(pb_score)
        
    
    return np.nanmean(pb_scores)







def tune_hyperparams(X_train, Y_train, quantile=0.5, num_trials=100, params_in=False):
    """
    Hyperparameter tuning using Optuna.
    
    Parameters:
    - X_train (pd.DataFrame): The training features.
    - Y_train (pd.Series): The training labels.
    - quantile (float, optional): The quantile level for parameter tuning. Defaults to 0.5.
    - num_trials (int, optional): The total number of optimization trials. Defaults to 100.
    
    Returns:
    - dict: The optimized hyperparameters.
    """
    
    import optuna
    
    
    
    
    # Set up Optuna sampler with a third of trials dedicated to exploration
    num_startup_trials = int(num_trials / 3)
    sampler = optuna.samplers.TPESampler(n_startup_trials=num_startup_trials)
    
    
    # Create an Optuna study 
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    
    # Initialize parameters for the first trial
    if not params_in:
        params = params_qregr()
    else:
        params = params_in.copy()
    
    # Enqueue the first trial
    study.enqueue_trial(params)
    
    # Optimize hyperparameters 
    study.optimize(lambda trial: optuna_objective(trial, X_train, Y_train, quantile), n_trials=num_trials)
    
    # Update the initial parameters with the best hyperparameters found
    best_params = study.best_params
    params.update(best_params)
    
    print('Best trial:', best_params, flush=True)
    
    return params







def params_qregr():
    params = dict(
        n_jobs=-1,
        n_estimators=100, # 200,
        alpha=1.0,
        bootstrap=False,   
        bootstrap_features=False,
        p_smpl=0.8, 
        p_feat=0.5,)
    return params







# --- Misc / Utils / Plotting ---

def print_ram_state():
    import resource
    
    # Get memory usage in kilobytes
    ram_state_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    def mb(kbytes):
        return kbytes / 1024
    
    # Convert kilobytes to megabytes
    ram_state_mb = mb(ram_state_kb)
    
    print(f"RAM: {ram_state_mb:.2f} MB", flush=True)




def stopwatch(start_stop, t=-99):
    import time
    
    if start_stop=='start':
        t = time.time()
        return t
    
    if start_stop=='stop':
        elapsed = time.time() - t
        return time.strftime("%H hours, %M minutes, %S seconds",time.gmtime(elapsed))







from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
def create_legend(cmap, num_bins=9):
    
    vmin=0; vmax=100
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    # Generate legend elements based on the number of bins
    value_bins = np.linspace(norm.vmin, norm.vmax, num_bins + 1)
    legend_elmnts = []

    for i in range(num_bins):
        color_value = (value_bins[i] + value_bins[i + 1]) / 2
        color = colormap(norm(color_value))
        label = f'{value_bins[i]:.1f} – {value_bins[i + 1]:.1f} %'

        legend_elmnts.append(Line2D([0], [0], marker='s', color='w', label=label,
                                    markeredgecolor=None, markerfacecolor=color, markersize=12))

    return legend_elmnts




def classify_percentages(df,):# column_name):
    # Define the bins and labels
    bins = [-float('inf'), 0, 25, 50, 75, 100, 125, 150, 175, 200, 1000, float('inf')]
    #bins = [0, 25, 50, 75, 100, 125, 150, 175, 200,99999999]
    #labels = ['≤ 0%', '0-25%', '25-50%', '50-75%', '75-100%', '100-125%', '125-150%', '150-175%', '175-200%', '≥ 200%']
    #colors = ['#e50000', '#ff8d00', '#ffc500', '#ffff00', '#b0dd00', '#54c500', '#00aa5b', '#008b94', '#005bbb', '#2b00d7']
    #colors = ['#E15A55', '#EC8B50', '#F8BC52', '#F7DAA3', '#FFFEFD', '#B1F3AB', '#6BE368', '#62A0AE', '#565DED']
    
    
    # Create a mapping from labels to colors
    color_mapping = dict(zip(bins[:-1], colors))

    # Process each column
    for col in df.columns:
        categories = pd.cut(df[col], bins=bins, labels=bins[:-1], right=False)
        df[col] = categories

    # Create a new DataFrame for colors
    df_colors = pd.DataFrame(index=df.index)

    for col in df.columns:
        df_colors[col] = df[col].map(color_mapping)

    return df.astype(float), df_colors



