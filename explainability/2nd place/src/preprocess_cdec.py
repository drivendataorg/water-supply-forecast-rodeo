#!/usr/bin/env python



# Read modules
import sys, os, glob, ast, importlib, datetime, itertools, requests
import numpy as np

import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point


from pathlib import Path

from wsfr_download.config import DATA_ROOT






forecast_year = int(sys.argv[1])
previous_year = forecast_year - 1

trgt_dir = DATA_ROOT / 'cdec'
data_dir = DATA_ROOT 

# Create the output directory if it doesn't exist
os.makedirs(str(trgt_dir), exist_ok=True)



# Read own functions
import functions as fcts



skip_existing = True


year_now = datetime.datetime.now().year




# Station metadata
df_metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={"usgs_id": "string"}, index_col='site_id')



#
#sites_to_cdec_stations = pd.read_csv(data_dir / 'cdec/sites_to_cdec_stations.csv')
#sites = list(sites_to_cdec_stations['site_id'].unique())

sites_to_cdec_stations = fcts.active_cdec_stations()
sites = list(sites_to_cdec_stations.keys())



# Geospatial catchment polygons
gdf_polygons = gpd.read_file(data_dir / 'geospatial.gpkg')






sensor_types = ['SNOW WC',
                'SNO ADJ',
                'RAIN',
                'TEMP AV',
                'TEMP MX',
                'TEMP MN',
                'SNOW DP',
                'SNWCMIN',
                'SNWCMAX']

'''
files = glob.glob(str(trgt_dir / '*/*'))
for file in files:
    print(file)
    df = pd.read_csv(file)
    
    if 'sensorType' in df.columns:
        types = list(df['sensorType'].unique())
        for item in types:
            if item not in sensor_types: sensor_types.append(item)

'''


fy_dates = pd.date_range(f'{previous_year}-10-01', f'{forecast_year}-07-21', freq='1D')


# Read cdec
for site in sites:
    
    active_stations = []
    
    target_file = trgt_dir / f'cdec_{site}_{forecast_year}.csv' 
    file_exists = os.path.exists(target_file)
    if skip_existing and file_exists and int(forecast_year) != year_now:
        print(f'Skipping {forecast_year} for CDEC {site} - File already processed.')
        continue
    
    if int(forecast_year) <= 1977:
        print(f'CDEC not defined for {forecast_year}')
        continue
    
    try:
        #station_ids = list(sites_to_cdec_stations.loc[sites_to_cdec_stations['site_id']==site]['station_id'].values)
        station_ids = sites_to_cdec_stations[site]
        
        print('Preprocessing CDEC for',site,forecast_year)
        
        data_out = []
        for station_id in station_ids:
            
            station_file = glob.glob(str(data_dir / f'cdec/FY{forecast_year}/*{station_id}.csv'))
            #print(station_id, station_file)
            
            #if len(station_file) > 0:
            columns = sensor_types
            station_data = pd.DataFrame(index=fy_dates, columns=columns).rename_axis('date')
            
            if len(station_file) > 0:
                df = pd.read_csv(station_file[0], index_col='date', parse_dates=True)
                df = df.pivot(columns='sensorType', values='value')
                df = df.where(df > -9999)
                
                not_nans = ~np.isnan(df.values)
                
                common_cols = list(np.intersect1d(columns, list(df.columns)))
            
                station_data.loc[df.index, common_cols] = df
                
                if np.sum(not_nans) > 0:
                    active_stations.append(station_id)
                
            for col in station_data.columns:
                new_col = col+'_'+station_id
                station_data = station_data.rename(columns={col:new_col})
            
            data_out.append(station_data)
        
        print('CDEC', site, active_stations)
        data_out = pd.concat(data_out, axis=1).astype(float)
        
        data_out.to_csv(target_file)
        
        fcts.print_ram_state()
        
    except:
        print('Preprocessing CDEC',site,forecast_year,'FAILED')




