#!/usr/bin/env python



# Read modules
import sys, glob, ast, importlib, datetime, itertools, requests, os
import numpy as np

# Set the XARRAY environment variable to disable bottleneck
#os.environ['XARRAY'] = 'DISABLE_BOTTLENECK=1'

import pandas as pd
import xarray as xr

import rioxarray as rxr



import geopandas as gpd



from pathlib import Path

from wsfr_download.config import DATA_ROOT


forecast_year = int(sys.argv[1])
previous_year = forecast_year - 1

trgt_dir = DATA_ROOT / 'pdsi'
data_dir = DATA_ROOT 





# Create the output directory if it doesn't exist
os.makedirs(str(trgt_dir), exist_ok=True)



# Read own functions
import functions as fcts



skip_existing = True


year_now = datetime.datetime.now().year





FY_dates = pd.date_range(f'{forecast_year-1}-10-01', f'{forecast_year}-07-21', freq='1D')






# Station metadata
df_metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={"usgs_id": "string"}, index_col='site_id')
sites = df_metadata.index.values


# Geospatial catchment polygons
gdf_polygons = gpd.read_file(data_dir / 'geospatial.gpkg')


# Data domain around the catchments
spatial_extents = fcts.catchment_extent(data_dir / 'geospatial.gpkg', 0.5)




# Read pdsi
for site in sites:
    #print(f'Processing PDSI forecast year {forecast_year} for {site}')
    
    target_file = trgt_dir / f'pdsi_{site}_{forecast_year}.csv'
    file_exists = os.path.exists(target_file)
    if skip_existing and file_exists and int(forecast_year) != year_now:
        print(f'Skipping {forecast_year} for PDSI - File already processed.')
        continue
    
    pdsi_file = glob.glob(str(data_dir / f'pdsi/FY{forecast_year}')+'/*')
    if len(pdsi_file) == 0:
        print(f'No PDSI data for {forecast_year}')
        continue
    
    try:
        
        
        latmin, lonmin, latmax, lonmax = spatial_extents[site]
        catchment_ds_FY = xr.open_dataset(pdsi_file[0])['daily_mean_palmer_drought_severity_index'].rename({'day':'time'})
        catchment_ds_FY = catchment_ds_FY.sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax))
        
        drain_area = 800
        if not np.isnan(df_metadata.loc[site, 'drainage_area']):
            drain_area = df_metadata.loc[site, 'drainage_area']
        
        length_scale = np.sqrt(drain_area)
        dx, dy = np.round(length_scale/200, 2), np.round(length_scale/200, 2)
        sigma = np.round(length_scale/20, 2)
        
        
        # Blur the data spatially (not temporally) prior to coarsening
        catchment_ds_FY[:] = fcts.Gauss_filter(catchment_ds_FY.copy(deep=True), sigma=(0,sigma,sigma), mode='nearest')
        
        
        
        # Interpolate to a coarser resolution
        lats_new = np.arange(np.round(latmin,2), np.round(latmax,2), dy)
        lons_new = np.arange(np.round(lonmin,2), np.round(lonmax,2), dx)
        catchment_ds_FY = fcts.regrid_dataset(catchment_ds_FY, lats_new, lons_new, method='linear')
        
        
        # Include missing days
        catchment_ds_FY = catchment_ds_FY.reindex(time=FY_dates) 
        
        # Include the CRS information
        catchment_ds_FY = catchment_ds_FY.rio.write_crs('EPSG:4326') 
        
        
        # Select data from the catchment only
        polygon = gdf_polygons.loc[gdf_polygons.site_id==site].geometry.to_crs(catchment_ds_FY.rio.crs)
        catchment_ds_FY = catchment_ds_FY.rio.clip(polygon.values, polygon.crs, invert=False, drop=False, all_touched=True)
        
        
        # From 3D xarray dataset to 2D pandas dataframe
        catchment_ds_FY = catchment_ds_FY.stack(gridcell=('lat','lon')).dropna('gridcell', how='all')
        
        n_cells = catchment_ds_FY.shape[1]
        df = pd.DataFrame(index=catchment_ds_FY.time, 
                          columns=['PDSI_gcl={}'.format(i) for i in range(n_cells)], 
                          data=catchment_ds_FY.values)
        
        
        
        df.to_csv(target_file)
        print(f'PDSI forecast year {forecast_year} for {site} PROCESSED')
        
        
    except:
        print(f'PDSI forecast year {forecast_year} for {site} FAILED')









