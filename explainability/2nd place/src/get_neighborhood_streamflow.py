#!/usr/bin/env python







import requests, sys, os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from io import StringIO

from pathlib import Path




from wsfr_download.config import DATA_ROOT




forecast_year = 2024
forecast_year = int(sys.argv[1])
previous_year = forecast_year - 1





trgt_dir = DATA_ROOT / 'usgs_neighborhood'
data_dir = DATA_ROOT 


# Create the output directory if it doesn't exist
os.makedirs(str(trgt_dir), exist_ok=True)



# Read own functions
#sys.path.append(str(src_dir))
import functions as fcts



skip_existing = True

year_now = datetime.now().year




df_metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={"usgs_id": "string"}, index_col='site_id')

sites = list(df_metadata.index.values)




# Read other observations around target sites
spatial_extents = fcts.catchment_extent(data_dir / 'geospatial.gpkg',0.3)




for site in sites:
    
    target_file = trgt_dir / f'usgs_neighbors_{site}_{forecast_year}.csv'
    file_exists = os.path.exists(target_file)
    if skip_existing and file_exists:
        print(f'Skipping {forecast_year} {site} for neighborhood streamflow - file already downloaded')
        continue
    
    latmin, lonmin, latmax, lonmax = spatial_extents[site]
    bbox = lonmin,latmin,lonmax,latmax
    bbox = ','.join(np.array(bbox).round(4).astype(str))
    
    site_neighbors = pd.read_csv(f'usgs_neighborhood_sites_{site}.csv', dtype='string')
    
    locations = site_neighbors['site'].values
    locations = ','.join(locations)
    
    variables = np.unique(site_neighbors['variable'])
    variables = ','.join(variables)
    
    statcodes = np.unique(site_neighbors['statCd'])
    statcodes = ','.join(statcodes)
    
    dates = pd.date_range(str(previous_year)+"-10-01", str(forecast_year)+"-07-21", freq='1D')
    
    neighbor_columns = []
    for i,row in site_neighbors.iterrows():
        neighbor_columns.append('_'.join(row.values))
    
    df_neighbors = pd.DataFrame(index=dates, columns=neighbor_columns)
    
    observation_url = \
        "https://waterservices.usgs.gov/nwis/dv/?format=rdb"+\
        "&bBox="+bbox+\
        "&variable="+variables+\
        "&startDT="+str(previous_year)+"-10-01" +\
        "&endDT="+str(forecast_year)+"-07-21"
    
    
    try:
        
        response = requests.get(observation_url)
        
        if response.status_code == 200:
            print('Request OK for USGS neighborhood streamflow',site,str(forecast_year))
            data = response.text
            
            # Split the data into blocks separated by comments
            data_blocks = data.split("Data provided for site")
            
            # Initialize an empty list to collect dataframes
            dataframes = []
            
            # Process each data block for a specific site
            for block in data_blocks[1:]:  # Skip the first empty block
                
                # Extract data and column names from the block
                site_data = block.split('agency_cd')[1].split('\n')
                columns = site_data[0].split('\t')
                
                # Create a temporary dataframe for the site
                df = pd.read_csv(
                    StringIO('\n'.join(site_data)),  # Convert list of strings to a single string
                    delimiter='\t', names=columns, low_memory=False,
                    skiprows=2, parse_dates=['datetime'], index_col='datetime').drop(columns='')
                
                df.sort_index(inplace=True)
                print(df)
                # Append the site's dataframe to the list
                dataframes.append(df)
            
            # Concatenate all dataframes into one
            final_df = pd.concat(dataframes,axis=1).rename_axis('time')
            final_df = final_df.loc[final_df.index.dropna()]
            
            # Convert strings to NaN and convert the DataFrame to float
            final_df = final_df.apply(pd.to_numeric, errors='coerce')
            
            common_cols = np.intersect1d(neighbor_columns, final_df.columns)
            df_neighbors.loc[final_df.index, common_cols] = final_df[common_cols]
            
            df_neighbors.to_csv(trgt_dir / target_file)
            
            
            print('Data downloaded and processed USGS neighborhood streamflow for',site,str(forecast_year))
            #df_neighbors.plot(); plt.show()
            fcts.print_ram_state()
    
    except: print('Failed',site,str(forecast_year))





