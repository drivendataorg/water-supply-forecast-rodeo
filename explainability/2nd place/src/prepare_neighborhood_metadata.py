#!/usr/bin/env python






import requests, sys, os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from io import StringIO

from pathlib import Path




from wsfr_download.config import DATA_ROOT





trgt_dir = DATA_ROOT / 'usgs_neighborhood'
data_dir = DATA_ROOT


# Create the output directory if it doesn't exist
os.makedirs(str(trgt_dir), exist_ok=True)



# Read own functions
import functions as fcts






today = str(datetime.now().date())




df_metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={"usgs_id": "string"}, index_col='site_id')
sites = list(df_metadata.index.values)



# Read other observations around target sites
spatial_extents = fcts.catchment_extent(data_dir / 'geospatial.gpkg',0.3)





for site in sites:

    latmin, lonmin, latmax, lonmax = spatial_extents[site]
    bbox = lonmin,latmin,lonmax,latmax
    bbox = ','.join(np.array(bbox).round(4).astype(str))

    product = (lonmax-lonmin)*(latmax-latmin)

    # Request all even remotely relevant variables from active stations/sites
    observation_url = \
        "https://waterservices.usgs.gov/nwis/dv/?format=rdb"+\
        "&bBox="+bbox+\
        "&variable=00060,00061,00062,00065,30208,30207,30210,62601,72008,72019,72021,72016,72025,72137,99019,00054,00055,00056,00058,00059"+\
        "&startDT=1950-01-01" +\
        "&endDT="+today+"&siteStatus=active"

    print(np.round(product,2), bbox, site)
    print(observation_url+'\n')

    try:

        response = requests.get(observation_url)

        if response.status_code == 200:
            print('Request OK for',site)
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

                drops = ['site_no']
                for col in df.columns:

                    # Only numerical columns allowed
                    not_numerical = '_cd' in col

                    # More than 20 active years needed for fitting and no more than 5% of missing observations allowed
                    start_date = '1995-01-01'
                    data_length = df[(df.index >= start_date) & (df.index <= today)][col].shape[0]
                    data_missng = df[(df.index >= start_date) & (df.index <= today)][col].dropna().shape[0]
                    miss_percent = (data_length - data_missng)/(data_length+0.1) * 100

                    not_long_or_not_complete = (data_length < 7300) | (miss_percent > 5)

                    # Reject useless columns
                    if not_long_or_not_complete or not_numerical: drops.append(col)

                df = df.drop(columns=drops)

                # Append the site's dataframe to the list
                dataframes.append(df)

            # Concatenate all dataframes into one
            final_df = pd.concat(dataframes,axis=1).rename_axis('time')

            # Convert strings to NaN and convert the DataFrame to float
            final_df = final_df.apply(pd.to_numeric, errors='coerce')

            site_ids = []; vrbs = []; stats = []
            for col in final_df.columns:
                ste, vrb, sts = col.split('_')

                site_ids.append(ste)
                vrbs.append(vrb)
                stats.append(sts)

            df_neighbors = pd.DataFrame(data=np.array([site_ids, vrbs, stats]).T, columns=['site','variable','statCd'])
            df_neighbors.to_csv('usgs_neighborhood_sites_'+site+'.csv', index=False)
            print(df_neighbors)
            print(site,'processed',final_df.shape)

    except: print('Failed',site)
