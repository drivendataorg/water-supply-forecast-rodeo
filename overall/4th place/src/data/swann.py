from pathlib import Path
import pandas as pd
import numpy as np
import os
import urllib3
from loguru import logger
import requests
from data.snotel import compute_conditional_swe_date

def download_swann(huc_list: list, directory: Path) -> None:
    """
    Download SWANN SWE data for specified HUCs (Hydrologic Unit Codes) from a given URL 
    and save them as CSV files in the specified directory.

    Parameters:
    huc_list (list): A list of HUCs for which the SWE data is to be downloaded.
    directory (Path): A Path object pointing to the directory where the CSV files will be saved.

    Returns:
    None: This function does not return anything. It saves the downloaded files in the specified directory.
    """
    urllib3.disable_warnings()
    base_url = 'https://climate.arizona.edu/snowview/csv/Download/Watersheds/'
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        logger.info('creating directory for swann data')
        os.makedirs(directory)
        
    for huc in huc_list:
        file_name = str(huc) + '.csv'
        file_path = os.path.join(directory, file_name)

        # Check if the file already exists
        if os.path.isfile(file_path):
            logger.info(f"'{file_name}' already downloaded. Skipping download.")
            continue
            
        file_url = base_url + file_name
        response = requests.get(file_url, verify=False)
        
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)

            logger.info(f"Downloaded '{file_url}' to '{file_path}'")
        else:
            logger.error(f"Failed to download '{file_url}' (Status code: {response.status_code})")

def read_swann(directory: Path) -> pd.DataFrame:
    """
    Read and combine SWANN SWE data from multiple CSV files in a specified directory into a single DataFrame.

    Parameters:
    directory (Path): A Path object pointing to the directory containing the CSV files.

    Returns:
    pd.DataFrame: A DataFrame containing combined data from all CSV files in the directory.
    """
    swann_data = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            df['huc8'] = file.replace('.csv', '')
            swann_data.append(df)
    concatenated_df = pd.concat(swann_data, ignore_index=True)
    concatenated_df['Date'] = pd.to_datetime(concatenated_df['Date'])
    return concatenated_df

def compute_swann_features(swann: pd.DataFrame, site_id: str, issue_date: pd.Timestamp) -> pd.Series:
    """
    Compute specific features from the SWANN dataset for a given site and date.

    This function filters the SWANN dataset for a specified site ID and dates prior to the given issue date. 
    It then calculates and returns the last recorded snow water equivalent (SWE) and the accumulated water year 
    precipitation up to that date.

    Parameters:
    swann (pd.DataFrame): The SWANN dataset containing snow water equivalent and precipitation data.
    site_id (str): The site ID for which the features are to be computed.
    issue_date (pd.Timestamp): The cutoff date up to which data should be considered.

    Returns:
    pd.Series: A pandas Series containing the last SWE and accumulated water year precipitation values.
               If no data is found for the specified site and date, NaN values are returned.

    """
    water_year_start = pd.Timestamp(f'{issue_date.year - 1}-10-01')
    # Filter data for the specified site_id and date range
    filtered_data = swann[(swann['site_id'] == site_id) &
                           (swann['Date'] < issue_date) &
                           (swann['Date'] >= water_year_start)]
    in_season = {
        "detroit_lake_inflow": [4, 5, 6],
        "pecos_r_nr_pecos": [3, 4, 5, 6, 7]
    }.get(site_id, [4, 5, 6, 7])  # Default in-season months
    
    # Return NaNs if the filtered dataset is empty
    if filtered_data.empty:
        if not os.environ.get('ENV') == 'this-is-my-beloved-macbook':  # For training the models on macbook
            logger.info(f'No SWANN data found for site {site_id} and date {issue_date}')
        ppt_unaccounted = np.nan
        ppt_unaccounted = np.nan
        ppt_latest = np.nan
        swe_latest = np.nan
        ppt_conditional = np.nan
        swe_conditional = np.nan
    else:
        swe_latest = filtered_data['Average SWE (in)'].iloc[-1]
        ppt_latest = filtered_data['Average Accumulated Water Year PPT (in)'].iloc[-1]
        conditional_swe_date = compute_conditional_swe_date(site_id, issue_date)
        conditional_data = filtered_data[filtered_data['Date'] <= conditional_swe_date]
        conditional_data = conditional_data.sort_values('Date', ascending=True)
        swe_conditional = conditional_data['Average SWE (in)'].iloc[-1]
        ppt_conditional = conditional_data['Average Accumulated Water Year PPT (in)'].iloc[-1]
        ppt_unaccounted = ppt_latest - ppt_conditional


    return pd.Series({
                      'swann_swe_conditional': swe_conditional,
                      'swann_ppt_unaccounted': ppt_unaccounted,
                      'swann_ppt_conditional': ppt_conditional,
                      'swann_ppt_latest': ppt_latest,
                      'swann_swe_latest': swe_latest,
                     })
