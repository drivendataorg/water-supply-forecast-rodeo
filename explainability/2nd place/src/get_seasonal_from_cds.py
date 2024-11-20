#!/usr/bin/env python



import os, glob, sys
import numpy as np

from datetime import datetime

#import concurrent.futures

import cdsapi


from pathlib import Path

import warnings
warnings.filterwarnings("ignore")



from wsfr_download.config import DATA_ROOT







year = str(sys.argv[1])
mnth = str(sys.argv[2])





trgt_dir = DATA_ROOT / 'cds'
data_dir = DATA_ROOT 


# Create the output directory if it doesn't exist
os.makedirs(str(trgt_dir), exist_ok=True)




skip_existing = True

now = datetime.now()
year_now = str(now.year)



target_file = trgt_dir / f'seasonal_ecmwf_{year}-{mnth}.nc'
file_exists = os.path.exists(str(target_file))

if skip_existing and file_exists and year is not year_now:
    sys.exit(f'Skipping {target_file} for ECMWF seasonal forecast - file already downloaded')


if int(year) >= now.year and int(mnth) >= now.month:
    sys.exit(f'ECMWF data not defined for {year}-{mnth}')


else:
    print(f'Retrieving {target_file} from CDS')
    
    c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api/v2',
                  key='10112:80b20536-a88b-4360-bbdf-daa87d7a6745')
    
    c.retrieve(
        'seasonal-original-single-levels',
        {   'originating_centre': 'ecmwf',
            'system': '51',
            'variable': ['2m_temperature', 'total_precipitation', 'evaporation',
                         'runoff', 'snow_density', 'snow_depth'],
            'area': [54, -132, 22, -89,],
            'grid': [1.0, 1.0],
            'year': [year],
            'month': [mnth],
            'day': ['01'],
            'leadtime_hour': list(np.arange(24, 5161, 24).astype(str)),
            'format': 'netcdf',}, target_file)


    print(f'{target_file} SUCCESSFULLY RETRIEVED from CDS')


