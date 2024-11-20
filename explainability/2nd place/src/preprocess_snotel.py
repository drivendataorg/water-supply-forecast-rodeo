#!/usr/bin/env python



# Read modules
import sys, os, glob, ast, importlib, datetime, itertools, requests
import numpy as np

import pandas as pd
import geopandas as gpd


from pathlib import Path

from wsfr_download.config import DATA_ROOT



forecast_year = int(sys.argv[1])
previous_year = forecast_year - 1

trgt_dir = DATA_ROOT / 'snotel'
data_dir = DATA_ROOT 

# Create the output directory if it doesn't exist
os.makedirs(str(trgt_dir), exist_ok=True)



# Read own functions
import functions as fcts



skip_existing = True


year_now = datetime.datetime.now().year




# Station metadata
df_metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={"usgs_id": "string"}, index_col='site_id')

sites = list(df_metadata.index.values)

#
sites_to_snotel = pd.read_csv(data_dir / 'snotel/sites_to_snotel_stations.csv')

# Geospatial catchment polygons
gdf_polygons = gpd.read_file(data_dir / 'geospatial.gpkg')



# List Snotel files for the forecast year
snotel_files = glob.glob(str(data_dir / f'snotel/FY{forecast_year}/*'))




fy_dates = pd.date_range(f'{previous_year}-10-01', f'{forecast_year}-07-21', freq='1D')


# Read snotel
for site in sites:
    
    target_file = trgt_dir / f'snotel_{site}_{forecast_year}.csv' 
    file_exists = os.path.exists(target_file)
    if skip_existing and file_exists and int(forecast_year) != year_now:
        print(f'Skipping {forecast_year} for SNOTEL {site} - File already processed.')
        continue
    
    if int(forecast_year) <= 1970:
        print(f'SNOTEL not defined for {forecast_year}')
        continue
    
    try:
        triplet_ids = list(sites_to_snotel.loc[sites_to_snotel['site_id']==site]['stationTriplet'].values)
        
        data_out = []
        for triplet_id in triplet_ids:
            file_id = triplet_id.replace(':','_')
            
            triplet_file = glob.glob(str(data_dir / f'snotel/FY{forecast_year}/*{file_id}*'))
            
            #if len(triplet_file) > 0:
            columns = ['PREC_DAILY', 'SNWD_DAILY', 'TAVG_DAILY', 'TMAX_DAILY', 'TMIN_DAILY','WTEQ_DAILY']
            triplet_data = pd.DataFrame(index=fy_dates, columns=columns).rename_axis('date')
            
            if len(triplet_file) > 0:
                df = pd.read_csv(triplet_file[0], index_col='date', parse_dates=True)
                common_cols = list(np.intersect1d(columns, list(df.columns)))
            
                triplet_data.loc[df.index, common_cols] = df
            
            for col in triplet_data.columns:
                new_col = col+'_'+file_id
                triplet_data = triplet_data.rename(columns={col:new_col})
            
            data_out.append(triplet_data)
        
        data_out = pd.concat(data_out, axis=1).astype(float)
        
        data_out.to_csv(target_file)
        print('Preprocessing SNOTEL',site,forecast_year,'OK')
        
        fcts.print_ram_state()
        
    except:
        print('Preprocessing SNOTEL',site,forecast_year,'FAILED')




