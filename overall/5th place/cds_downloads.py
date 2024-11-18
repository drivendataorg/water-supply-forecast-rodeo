from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import cdsapi
import xarray as xr
import os
from loguru import logger


def convert_nc_to_dateframe(path_input: str,
                            path_output: str,
                            all_touched: bool):
    """
    Convert .nc gridded file to a DataFrame. Iterate over different site ids,
    choose grids associated to the site id, average results over different grids
    and save results.

    Args:
        path_input (str): An input path to .nc file to process
        path_output (str): An output .pkl path
        all_touched (bool): Take into account gridded coordinates if any part of
            the coordinates is within site id polygon's coordinates (True). If
            coordinates' pixels should be in a center of polygon's coordinates,
            assign value to False. True value is used if coordinate step is by
            1. False is for 0.1 step. When step is bigger (by 1), any value that
            touches the polygons should be assigned, otherwise there will be no match.
            More about all_touched parameter could be found in the link below:
            https://corteva.github.io/rioxarray/html/rioxarray.html#rioxarray.raster_array.RasterArray
    """
    # Initialize a list where results from different dates are stored
    stats_cds = []
    # Read data from a specific year (forecast year)
    cds_one_month = xr.open_dataset(path_input)
    # Change encoding of coordinates
    cds_one_month.rio.write_crs("epsg:4326", inplace=True)
    # Get variable names
    cds_vars = list(cds_one_month.keys())
    print('Processing different site ids:\n')
    # Iterate over rows from geospatial.gpkg
    for _, geo in geospatial.iterrows():
        site_id = geo.site_id
        print(site_id)
        # Keep only information from specific site_id
        if all_touched:
            # Use all_touched parameter
            cds_data = cds_one_month.rio.clip([geo.geometry], all_touched=True)
        else:
            cds_data = cds_one_month.rio.clip([geo.geometry], all_touched=False)
            # Get number of different dates to iterate over
        num_days = cds_data.dims['time']
        # Iterate over different variables
        for var in cds_vars:
            date_values = []
            # Iterate over different dates
            for num_day in range(num_days):
                # Get date
                date = cds_data.time[num_day].values
                # Get average value over different grids
                mean_val = np.nanmean(cds_data[var][num_day])
                date_values.append([site_id, date, mean_val])
                # Append results from a specific forecast year - site_id combination
                stats_cds.append([var, site_id, date, mean_val])
    print('Creating a DataFrame')
    # Crate a DataFrame from results
    stats_cds = pd.DataFrame(stats_cds)
    stats_cds.columns = ['cds_var', 'site_id', 'date', 'mean_value']
    # Keep only site_id, date and CDS variables as columns
    stats_cds = pd.pivot_table(stats_cds,
                               values='mean_value',
                               index=['site_id', 'date'],
                               columns='cds_var').reset_index()
    # Add date columns
    stats_cds['year'] = stats_cds.date.dt.year
    stats_cds['month'] = stats_cds.date.dt.month
    stats_cds['day'] = stats_cds.date.dt.day
    stats_cds['hour'] = stats_cds.date.dt.hour
    # Add forecast year
    stats_cds['year_forecast'] = stats_cds.year
    stats_cds.loc[stats_cds['month'].astype(int).between(10, 12), 'year_forecast'] = \
        stats_cds.year_forecast + 1
    # Save data
    print(f'\nSaving data to {path_output}')
    stats_cds.to_pickle(path_output)


# Get site_id locations combinations. Locations are provided as polygons
geospatial = gpd.read_file('data/geospatial.gpkg')
# Initialize cdsapi client
c = cdsapi.Client()

# Monthly CDS data
# Monthly CDS data uses coordinates step by 0.1, so all_touched parameter isn't used.
grib_path = Path("data") / "cds_raw" / 'cds_monthly_snow.grib'
grib_path.parent.mkdir(parents=True, exist_ok=True)
if not grib_path.exists():
    c.retrieve(
        'reanalysis-era5-land-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': 'snow_depth_water_equivalent',
            'year': [
                '1960', '1961', '1962', '1963', '1964', '1965', '1966',
                '1967', '1968', '1969', '1970', '1971', '1972', '1973',
                '1974', '1975', '1976', '1977', '1978', '1979', '1980',
                '1981', '1982', '1983', '1984', '1985', '1986', '1987',
                '1988', '1989', '1990', '1991', '1992', '1993', '1994',
                '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                '2002', '2003', '2004', '2005', '2006', '2007', '2008',
                '2009', '2010', '2011', '2012', '2013', '2014', '2015',
                '2016', '2017', '2018', '2019', '2020', '2021', '2022',
                '2023',
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '10', '11', '12',
            ],
            'time': '00:00',
            'area': [
                51, -125, 32,
                -100,
            ],
            'format': 'grib',
        },
        str(grib_path)
    )
else:
    logger.info(f"Skipping {grib_path.name}, exists.")
# Convert .grib data to .nc format, save it to data/cds
nc_path = Path("data") / "cds" / "cds_monthly_snow.nc"
nc_path.parent.mkdir(parents=True, exist_ok=True)
os.system(f"grib_to_netcdf -o {nc_path} {grib_path}")
# Convert data to a DataFrame
df_path = Path("data") / "cds" / "cds_monthly_snow.pkl"
convert_nc_to_dateframe(str(nc_path),
                        str(df_path),
                        False)

# Seasonal forecasts data
# Coordinates for this data are with step 1, so all_touched parameter is used.
# Seasonal forecasts start from 1981. Forecasts for the end of March, April,
# May and June are downloaded (approximately from the last day of the month,
# 30 days step is used).

# Dec
grib_path = Path("data") / "cds_raw" / 'seasonal_dec.grib'
if not grib_path.exists():
    c.retrieve(
        'seasonal-original-single-levels',
        {
            'format': 'grib',
            'originating_centre': 'ecmwf',
            'system': '51',
            'variable':'snow_depth',
            'year': [
                '1981', '1982', '1983',
                '1984', '1985', '1986',
                '1987', '1988', '1989',
                '1990', '1991', '1992',
                '1993', '1994', '1995',
                '1996', '1997', '1998',
                '1999', '2000', '2001',
                '2002', '2003', '2004',
                '2005', '2006', '2007',
                '2008', '2009', '2010',
                '2011', '2012', '2013',
                '2014', '2015', '2016',
                '2017', '2018', '2019',
                '2020', '2021', '2022',
                '2023',
            ],
            'month': '12',
            'day': '01',
            'leadtime_hour': [
                '2880', '3600', '4320', '5040',
            ],
            'area': [
                51, -125, 32,
                -100,
            ],
        },
        str(grib_path)
    )
else:
    logger.info(f"Skipping {grib_path.name}, exists.")
# Convert .grib data to .nc format, save it to data/cds
nc_path = Path("data") / "cds" / "seasonal_dec.nc"
os.system(f"grib_to_netcdf -o {nc_path} {grib_path}")
# Convert data to a DataFrame
df_path = Path("data") / "cds" / "seasonal_dec.pkl"
convert_nc_to_dateframe(str(nc_path),
                        str(df_path),
                        True)

# Jan
grib_path = Path("data") / "cds_raw" / 'seasonal_jan.grib'
if not grib_path.exists():
    c.retrieve(
        'seasonal-original-single-levels',
        {
            'format': 'grib',
            'originating_centre': 'ecmwf',
            'system': '51',
            'variable':'snow_depth',
            'year': [
                '1981', '1982', '1983',
                '1984', '1985', '1986',
                '1987', '1988', '1989',
                '1990', '1991', '1992',
                '1993', '1994', '1995',
                '1996', '1997', '1998',
                '1999', '2000', '2001',
                '2002', '2003', '2004',
                '2005', '2006', '2007',
                '2008', '2009', '2010',
                '2011', '2012', '2013',
                '2014', '2015', '2016',
                '2017', '2018', '2019',
                '2020', '2021', '2022',
                '2023',
            ],
            'month': '01',
            'day': '01',
            'leadtime_hour': [
                '2160', '2880', '3600', '4320',
            ],
            'area': [
                51, -125, 32,
                -100,
            ],
        },
        str(grib_path)
    )
else:
    logger.info(f"Skipping {grib_path.name}, exists.")
# Convert .grib data to .nc format, save it to data/cds
nc_path = Path("data") / "cds" / "seasonal_jan.nc"
os.system(f"grib_to_netcdf -o {nc_path} {grib_path}")
# Convert data to a DataFrame
df_path = Path("data") / "cds" / "seasonal_jan.pkl"
convert_nc_to_dateframe(str(nc_path),
                        str(df_path),
                        True)

# Feb
grib_path = Path("data") / "cds_raw" / 'seasonal_feb.grib'
if not grib_path.exists():
    c.retrieve(
        'seasonal-original-single-levels',
        {
            'format': 'grib',
            'originating_centre': 'ecmwf',
            'system': '51',
            'variable':'snow_depth',
            'year': [
                '1981', '1982', '1983',
                '1984', '1985', '1986',
                '1987', '1988', '1989',
                '1990', '1991', '1992',
                '1993', '1994', '1995',
                '1996', '1997', '1998',
                '1999', '2000', '2001',
                '2002', '2003', '2004',
                '2005', '2006', '2007',
                '2008', '2009', '2010',
                '2011', '2012', '2013',
                '2014', '2015', '2016',
                '2017', '2018', '2019',
                '2020', '2021', '2022',
                '2023',
            ],
            'month': '02',
            'day': '01',
            'leadtime_hour': [
                '1440', '2160', '2880', '3600',
            ],
            'area': [
                51, -125, 32,
                -100,
            ],
        },
        str(grib_path)
    )
else:
    logger.info(f"Skipping {grib_path.name}, exists.")
# Convert .grib data to .nc format, save it to data/cds
nc_path = Path("data") / "cds" / "seasonal_feb.nc"
os.system(f"grib_to_netcdf -o {nc_path} {grib_path}")
# Convert data to a DataFrame
df_path = Path("data") / "cds" / "seasonal_feb.pkl"
convert_nc_to_dateframe(str(nc_path),
                        str(df_path),
                        True)

# Mar
grib_path = Path("data") / "cds_raw" / 'seasonal_mar.grib'
if not grib_path.exists():
    c.retrieve(
        'seasonal-original-single-levels',
        {
            'format': 'grib',
            'originating_centre': 'ecmwf',
            'system': '51',
            'variable':'snow_depth',
            'year': [
                '1981', '1982', '1983',
                '1984', '1985', '1986',
                '1987', '1988', '1989',
                '1990', '1991', '1992',
                '1993', '1994', '1995',
                '1996', '1997', '1998',
                '1999', '2000', '2001',
                '2002', '2003', '2004',
                '2005', '2006', '2007',
                '2008', '2009', '2010',
                '2011', '2012', '2013',
                '2014', '2015', '2016',
                '2017', '2018', '2019',
                '2020', '2021', '2022',
                '2023',
            ],
            'month': '03',
            'day': '01',
            'leadtime_hour': [
                '720', '1440', '2160', '2880',
            ],
            'area': [
                51, -125, 32,
                -100,
            ],
        },
        str(grib_path)
    )
else:
    logger.info(f"Skipping {grib_path.name}, exists.")
# Convert .grib data to .nc format, save it to data/cds
nc_path = Path("data") / "cds" / "seasonal_mar.nc"
os.system(f"grib_to_netcdf -o {nc_path} {grib_path}")
# Convert data to a DataFrame
df_path = Path("data") / "cds" / "seasonal_mar.pkl"
convert_nc_to_dateframe(str(nc_path),
                        str(df_path),
                        True)

# Apr
grib_path = Path("data") / "cds_raw" / 'seasonal_apr.grib'
if not grib_path.exists():
    c.retrieve(
        'seasonal-original-single-levels',
        {
            'format': 'grib',
            'originating_centre': 'ecmwf',
            'system': '51',
            'variable':'snow_depth',
            'year': [
                '1981', '1982', '1983',
                '1984', '1985', '1986',
                '1987', '1988', '1989',
                '1990', '1991', '1992',
                '1993', '1994', '1995',
                '1996', '1997', '1998',
                '1999', '2000', '2001',
                '2002', '2003', '2004',
                '2005', '2006', '2007',
                '2008', '2009', '2010',
                '2011', '2012', '2013',
                '2014', '2015', '2016',
                '2017', '2018', '2019',
                '2020', '2021', '2022',
                '2023',
            ],
            'month': '04',
            'day': '01',
            'leadtime_hour': [
                '720', '1440', '2160',
            ],
            'area': [
                51, -125, 32,
                -100,
            ],
        },
        str(grib_path)
    )
else:
    logger.info(f"Skipping {grib_path.name}, exists.")
# Convert .grib data to .nc format, save it to data/cds
nc_path = Path("data") / "cds" / "seasonal_apr.nc"
os.system(f"grib_to_netcdf -o {nc_path} {grib_path}")
# Convert data to a DataFrame
df_path = Path("data") / "cds" / "seasonal_apr.pkl"
convert_nc_to_dateframe(str(nc_path),
                        str(df_path),
                        True)
