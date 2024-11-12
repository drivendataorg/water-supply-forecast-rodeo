#!/usr/bin/env python



import os, sys, ast, datetime, requests, importlib, glob
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

from pathlib import Path

from io import StringIO

from wsfr_download.config import DATA_ROOT



forecast_year       = int(sys.argv[1])
previous_year       = forecast_year - 1



trgt_dir = DATA_ROOT / 'swann'
data_dir = DATA_ROOT 

# Create the output directory if it doesn't exist
os.makedirs(str(trgt_dir), exist_ok=True)



# Read own functions
import functions as fcts



skip_existing = True


year_now = datetime.datetime.now().year





# All days of the forecast year
FY_dates = pd.date_range(f'{forecast_year-1}-10-01', f'{forecast_year}-07-21', freq='1D')






# Station metadata
df_metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={"usgs_id": "string"}, index_col='site_id')
sites = df_metadata.index.values






# URL base 
base_url = 'https://snowview.arizona.edu/csv/Download/Watersheds'


# Download and process files
for site in sites:

    target_file = trgt_dir / f'swann_{site}_{forecast_year}.csv'
    file_exists = os.path.exists(target_file)
    if skip_existing and file_exists and int(forecast_year) != year_now:
        print(f'Skipping {forecast_year} for SWANN {site} - File already processed.')
        continue

    if int(forecast_year) <= 1981:
        print(f'SWANN not defined for {forecast_year} {site}')
        continue
    
    #try:
    digit_8_basin = fcts.usgs_hucs(site)
    digit_6_basin = digit_8_basin[:-2]
    digit_4_basin = digit_8_basin[:-4]
    
    print('Processing SWANN', site, digit_8_basin, digit_6_basin, digit_4_basin)
    
    list_basins = []
    for basin in [digit_8_basin, digit_6_basin, digit_4_basin]:
        
        url = f'{base_url}/{basin}.csv'
        df = pd.read_csv(StringIO(requests.get(url, verify=False, timeout=30).text), index_col='Date', parse_dates=True).rename_axis('time')
        
        df = df.rename(columns={'Average Accumulated Water Year PPT (in)': f'accumulated_ppt_{basin}', 
                                'Average SWE (in)': f'average_swe_{basin}'})
        df[f'instantaneous_ppt_{basin}'] = df[f'accumulated_ppt_{basin}'].diff()
        df[f'instantaneous_ppt_{basin}'] = df[f'instantaneous_ppt_{basin}'].where(df[f'instantaneous_ppt_{basin}'] >= 0, other=0)
        
        df = df.loc[FY_dates]
        
        list_basins.append(df)
    
    df_basins = pd.concat(list_basins, axis=1)
    
    #df_basins.loc['2021-01-01':'2024-02-28'].plot(); plt.show()
    
    # Save data
    df_basins.to_csv(target_file)
        
    #except:
    #    print(f'SWANN {site} {forecast_year} FAILED')


