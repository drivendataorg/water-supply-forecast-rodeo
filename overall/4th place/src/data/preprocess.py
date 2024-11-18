from loguru import logger
from pathlib import Path
from typing import Hashable, Any
import pandas as pd
import time
import os
from data.swann import download_swann, read_swann

def preprocess(
    src_dir: Path, data_dir: Path, preprocessed_dir: Path
) -> dict[Hashable, Any]:
    """An optional function that performs setup or processing.

    Args:
        src_dir (Path): path to the directory that your submission ZIP archive
            contents are unzipped to.
        data_dir (Path): path to the mounted data drive.
        preprocessed_dir (Path): path to a directory where you can save any
            intermediate outputs for later use.

    Returns:
        (dict[Hashable, Any]): a dictionary containing any assets you want to
            hold in memory that will be passed to to your 'predict' function as
            the keyword argument 'assets'.
    """
    # so that the runtime has enough time to load everything
    time.sleep(5)
    logger.info('Available Data')
    logger.info(os.listdir(data_dir))
    
    # ---------------------------------------------------------------------
    # SWANN data
    # ---------------------------------------------------------------------
    # Original source of the HUC mapping: https://www.sciencebase.gov/catalog/item/631405c4d34e36012efa315f

    logger.info('Loading SWANN data')
    huc_to_site = pd.read_csv(data_dir / 'huc-to-site.csv', dtype={'huc8': str})
    logger.info('.. downloading SWANN data')
    download_swann(huc_to_site['huc8'].unique(), preprocessed_dir / 'swann')
    logger.info('.. creating SWANN data set')
    swann = read_swann(preprocessed_dir / 'swann')
    swann = huc_to_site.merge(swann, on='huc8')
    swann = swann.groupby(['site_id', 'Date'])[['Average Accumulated Water Year PPT (in)', 'Average SWE (in)']].mean().reset_index()
    swann = swann.dropna(axis=0)
    # For faster feature computation during prediction time, better to have SWANN data separated by site_id
    swann = {site_id: group for site_id, group in swann.groupby('site_id')}
    
    # ---------------------------------------------------------------------
    # Monthly flow data
    # ---------------------------------------------------------------------
    logger.info('Loading monthly flow data')
    monthly_flow = pd.read_csv(data_dir / 'cross_validation_monthly_flow.csv')
    monthly_flow_train = pd.read_csv(data_dir / 'prior_historical_monthly_flow.csv')
    monthly_flow = pd.concat([monthly_flow_train, monthly_flow])
    monthly_flow = monthly_flow.dropna()

    # ---------------------------------------------------------------------
    # SNOTEL data
    # ---------------------------------------------------------------------
    logger.info('Loading SNOTEL data')
    snotel_path = data_dir / 'snotel'
    snotel_meta = pd.read_csv(snotel_path.joinpath('station_metadata.csv'))
    snotel_mapping = pd.read_csv(snotel_path.joinpath('sites_to_snotel_stations.csv'))

    # Read in all the SNOTEL years and stations
    csv_files = snotel_path.glob('**/*.csv')
    ignored_files = {'sites_to_snotel_stations.csv', 'station_metadata.csv'}
    snotel_files = [
        pd.read_csv(file).assign(station=file.stem.replace('_', ':'))
        for file in csv_files if file.name not in ignored_files
    ]

    # Create the stations data
    snotel = pd.concat(snotel_files, ignore_index=True)
    snotel = snotel.merge(snotel_mapping, left_on='station', right_on='stationTriplet')
    snotel['date'] = pd.to_datetime(snotel['date'])
    
    # Averaging stations per site
    snotel = snotel.groupby(['site_id', 'date']).agg({
        'WTEQ_DAILY': 'mean',
    }).reset_index()
    snotel['wateryear'] = snotel['date'].apply(lambda d: d.year + 1 if d.month >= 10 else d.year)
    # For faster feature computation during prediction time, better to have SNOTEL data separated by site_id
    snotel_dict = {site_id: group for site_id, group in snotel.groupby('site_id')}
    
    return {
        'monthly_flow': monthly_flow,
        'snotel': snotel_dict,
        'swann': swann
    }


def create_issue_dates(year: int) -> pd.DataFrame:
    """Creates the issue dates for the first seven months of a given year.

    This function generates a pandas DataFrame with issue dates for each week (1st, 8th, 15th, 22nd) of the months January through July.

    Args:
        year (int): The year for which to create the dates DataFrame. The year should be a four-digit number.

    Returns:
        pd.DataFrame: A DataFrame with two columns:
                      - `forecast_year`: The year of the forecast.
                      - `issue_date`: The issue date in datetime format (YYYY-MM-DD).
    """
    issue_dates = [1, 8, 15, 22]
    months = range(1, 8)  # From January (1) to July (7)
    data = [{'forecast_year': year, 'issue_date': f"{year}-{month:02d}-{day:02d}"}
            for month in months for day in issue_dates]
    df = pd.DataFrame(data)
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    return df
