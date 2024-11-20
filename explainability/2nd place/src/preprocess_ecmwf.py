#!/usr/bin/env python



import os, sys, ast, datetime, importlib, glob, itertools
import numpy as np
import xarray as xr
import pandas as pd


from pathlib import Path

from wsfr_download.config import DATA_ROOT



forecast_year = str(sys.argv[1])





trgt_dir = DATA_ROOT / 'cds'
data_dir = DATA_ROOT 



# Create the output directory if it doesn't exist
os.makedirs(str(trgt_dir), exist_ok=True)



# Read own functions
import functions as fcts



skip_existing = True


year_now = datetime.datetime.now().year




# Station metadata
df_metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={"usgs_id": "string"}, index_col='site_id')
sites = df_metadata.index.values


# Data domain around the catchments
spatial_extents = fcts.catchment_extent(data_dir / 'geospatial.gpkg', 3.1)




# Check for existing target files
processed_files = []
for site in sites: 
    target_file = os.path.join(trgt_dir, f'ecmwf_{site}_FY{forecast_year}.csv')
    file_exists = os.path.exists(target_file)

    processed_files.append(file_exists)


if all(processed_files) and skip_existing and int(forecast_year) != year_now:
    sys.exit(f'Skipping {forecast_year} for ECMWF - Files already processed.')


variables = ['t2m', 'tp', 'e', 'ro', 'rsn', 'sd']
accum_variables = ['tp', 'e', 'ro']


columns =  []
for v,l in itertools.product(variables, np.arange(7).astype(str)):
    columns.append(f'{v}_lead={l}')

all_dates = pd.date_range(f'{int(forecast_year)-1}-12-01', f'{forecast_year}-07-21', freq='1D')

results = {}
for site in sites: 
    results[site] = pd.DataFrame(index=all_dates, columns=columns)



for month in range(1,8):
    
    
    mnth_this_file = str(month).zfill(2)
    mnth_prev_file = '12'
    year_this_file = forecast_year
    year_prev_file = forecast_year 
    
    if month == 1:
        year_prev_file = str(int(forecast_year)-1)
    
    if month > 1: 
        mnth_prev_file = str(month-1).zfill(2)
    


    print('Processing ECMWF',mnth_this_file, mnth_prev_file, year_this_file, year_prev_file)
    
    try:
        ds_this = xr.open_dataset(trgt_dir / f'seasonal_ecmwf_{year_this_file}-{mnth_this_file}.nc')
        ds_this = fcts.adjust_lats_lons(ds_this.median('number'))
        
        # Transform accumulated variables to instantaneous
        for v in accum_variables:
            ds_this[v] = ds_this[v].diff('time')
        
        ds_this = ds_this.resample(time='1M').median()
    except:
        ds_this = None
    
    try:
        ds_prev = xr.open_dataset(trgt_dir / f'seasonal_ecmwf_{year_prev_file}-{mnth_prev_file}.nc')    
        ds_prev = fcts.adjust_lats_lons(ds_prev.median('number'))
        
        # Transform accumulated variables to instantaneous
        for v in accum_variables:
            ds_prev[v] = ds_prev[v].diff('time')
        
        ds_prev = ds_prev.resample(time='1M').median()
    except:
        ds_prev = None
    
    # No data found at all
    if ds_this is None and ds_prev is None: continue 
    
    
    for site in sites:
        latmin, lonmin, latmax, lonmax = spatial_extents[site]
        
        df_this = None
        if ds_this is not None:
            ds_catchment_this = ds_this.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))
            ds_catchment_this = ds_catchment_this.mean(['lat','lon'])
            df_this = ds_catchment_this.to_dataframe()
            df_this.index = np.arange(len(df_this))

        df_prev = None
        if ds_prev is not None:
            ds_catchment_prev = ds_prev.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))
            ds_catchment_prev = ds_catchment_prev.mean(['lat','lon'])
            df_prev = ds_catchment_prev.to_dataframe()
            #df_prev = df_prev.shift(-1)
            df_prev.index = np.arange(len(df_prev))-1
        
        '''
        for v in variables:
            df_this[v].plot()
            df_prev[v].plot(); plt.show()
        '''
        
        # Real-time forecasts are released once per month on the 6th at 12UTC for ECMWF
        # Prior to that, the forecast from previous month must be used
        row_index_this = (results[site].index.month == int(mnth_this_file)) & (results[site].index.day > 6)
        row_index_prev = ((results[site].index.month == int(mnth_prev_file)) & (results[site].index.day > 6)) | \
                         ((results[site].index.month == int(mnth_this_file)) & (results[site].index.day <= 6))
        
        for column in results[site].columns:
            vrbl =     column.split('_')[0]
            lead = int(column.split('=')[1])
            
            if df_prev is not None:
                if lead in df_prev.index:
                    results[site].loc[row_index_prev, column] = df_prev.loc[lead, vrbl]
            
            if df_this is not None: 
                if lead in df_this.index and lead in df_prev.index:
                    results[site].loc[row_index_this, column] = df_this.loc[lead, vrbl]
        
        '''
        #results[site].loc[row_index_prev].plot(); plt.show()
        
        # Save gridcell time series
        output_path = os.path.join(trgt_dir, f'{output_base}_{site}_FY{forecast_year}.csv')
        df.to_csv(output_path)
        '''



for site in sites: 
    # Save results
    output_path = os.path.join(trgt_dir, f'ecmwf_{site}_FY{forecast_year}.csv')
    results[site].to_csv(output_path)
    print(f'ecmwf_{site}_FY{forecast_year}.csv PROCESSED')







