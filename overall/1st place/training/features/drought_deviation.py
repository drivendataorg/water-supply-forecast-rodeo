import geopandas
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from shapely.geometry import Point

from features.feature_utils import compute_deviation


def generate_drought_deviation(data_dir: Path,
                               preprocessed_dir: Path,
                               df_features: pd.DataFrame,
                               cv_year: int,
                               test_features: pd.DataFrame,
                               all_lat_lon: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    logger.info('Generating PDSI deviation features.')
    pdsi_file = preprocessed_dir / 'train_lat_lon_pdsi.csv'
    if all_lat_lon is not None:
        pass
    elif pdsi_file.is_file():
        logger.info('PDSI file already exists, pulling existing.')
        all_lat_lon = pd.read_csv(preprocessed_dir / 'train_lat_lon_pdsi.csv')
    else:
        geo_sites = geopandas.read_file(data_dir / 'geospatial.gpkg')

        all_min_lon = 9999
        all_min_lat = 9999
        all_max_lon = -9999
        all_max_lat = -9999
        for idx, rw in geo_sites.iterrows():
            min_lon, min_lat, max_lon, max_lat = rw['geometry'].bounds
            if min_lon < all_min_lon:
                all_min_lon = np.floor(min_lon)
            if min_lat < all_min_lat:
                all_min_lat = np.floor(min_lat)
            if max_lon > all_max_lon:
                all_max_lon = np.ceil(max_lon)
            if max_lat > all_max_lat:
                all_max_lat = np.ceil(max_lat)

        ds = xr.open_dataset(data_dir / f'pdsi/FY2023/pdsi_2022-10-01_2023-07-21.nc')
        pdsi_df = ds.to_dataframe()
        pdsi_df.reset_index(inplace=True)
        sub_pdsi_df = pdsi_df[(pdsi_df['lat'] >= all_min_lat) &
                              (pdsi_df['lon'] >= all_min_lon) &
                              (pdsi_df['lat'] <= all_max_lat) &
                              (pdsi_df['lon'] <= all_max_lon)].reset_index(drop=True)

        masks_df = sub_pdsi_df.groupby(['lat', 'lon']).count().reset_index()[['lat', 'lon']]
        for idx, rw in geo_sites.iterrows():
            site_id = rw['site_id']
            geometry = rw['geometry']
            min_lon, min_lat, max_lon, max_lat = geometry.bounds
            sub_masks_df = masks_df[(masks_df['lat'] >= min_lat) &
                                    (masks_df['lon'] >= min_lon) &
                                    (masks_df['lat'] <= max_lat) &
                                    (masks_df['lon'] <= max_lon)]

            masks_site = sub_masks_df.apply(lambda rw: geometry.contains(Point(rw['lon'], rw['lat'])), axis=1)
            masks_df = masks_df.join(masks_site.to_frame(name=site_id), how='left').fillna(False)

        lat_lon_dfs = []
        for year in df_features['year'].unique():
            try:
                ds = xr.open_dataset(data_dir / f'pdsi/FY{year}/pdsi_{year - 1}-10-01_{year}-07-21.nc')
                pdsi_df = ds.to_dataframe()
                pdsi_df.reset_index(inplace=True)
                sub_pdsi_df = pdsi_df[(pdsi_df['lat'] >= all_min_lat) &
                                      (pdsi_df['lon'] >= all_min_lon) &
                                      (pdsi_df['lat'] <= all_max_lat) &
                                      (pdsi_df['lon'] <= all_max_lon)].reset_index(drop=True)

                for site_id in df_features['site_id'].unique():
                    site_pdsi = pd.merge(sub_pdsi_df,
                                         masks_df[masks_df[site_id] == True][['lat', 'lon']],
                                         on=['lat', 'lon'], how='inner')
                    site_pdsi['site_id'] = site_id
                    lat_lon_dfs.append(site_pdsi)

            except FileNotFoundError:
                print(f'No PDSI data for {year}')

        all_lat_lon = pd.concat(lat_lon_dfs)
        all_lat_lon['lat'] = np.round(all_lat_lon['lat'], 1)
        all_lat_lon['lon'] = np.round(all_lat_lon['lon'], 1)
        all_lat_lon = all_lat_lon.groupby(by=['day', 'lat', 'lon'])[
            'daily_mean_palmer_drought_severity_index'].mean().reset_index()
        all_lat_lon.drop_duplicates(subset=['day', 'lat', 'lon'], inplace=True)
        all_lat_lon['date'] = pd.to_datetime(all_lat_lon['day'])
        all_lat_lon['year'] = all_lat_lon['date'].dt.year
        all_lat_lon['month'] = all_lat_lon['date'].dt.month
        all_lat_lon['day_of_year'] = all_lat_lon['date'].dt.day_of_year
        all_lat_lon['forecast_year'] = all_lat_lon.apply(lambda rw: rw['year'] if rw['month'] < 8 else rw['year'] + 1,
                                                         axis=1)
        all_lat_lon['month_day'] = all_lat_lon['date'].dt.strftime('%m%d')

        all_lat_lon = all_lat_lon.rename(columns={'daily_mean_palmer_drought_severity_index': 'pdsi'})
        all_lat_lon['lat_lon'] = all_lat_lon['lat'].astype(str) + '_' + all_lat_lon['lon'].astype(str)

        all_lat_lon.to_csv(preprocessed_dir / 'train_lat_lon_pdsi.csv', index=False)

    applicable_sites = df_features['site_id'].unique()

    df_features, test_features = compute_deviation(
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        cv_year=cv_year,
        input_col='pdsi',
        output_col='pdsi_deviation',
        pivot_col='lat_lon',
        base_data=all_lat_lon,
        df_features=df_features,
        applicable_sites=applicable_sites,
        rolling_windows=[2, 5],
        min_num_locations=7,
        thresh=15,
        test_features=test_features,
        agg_months=[1, 2, 3, 4, 5, 6, 7, 12],
        sort_ascending=False,
        ffill=True
    )

    return df_features, test_features
