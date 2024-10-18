import os
import pandas as pd
import numpy as np
from loguru import logger

def compute_conditional_swe_date(site_id: str, issue_date: pd.Timestamp) -> pd.Timestamp:
    """
    Computes the most relevant data for the SWE measurement for a site

    This function takes the site and issue date and computes the date at which teh swe should be captured.
    """

    in_season = {
        "detroit_lake_inflow": [4, 5, 6],
        "pecos_r_nr_pecos": [3, 4, 5, 6, 7]
    }.get(site_id, [4, 5, 6, 7])

    no_flow_sites = [
       'san_joaquin_river_millerton_reservoir',
       'merced_river_yosemite_at_pohono_bridge',
       'american_river_folsom_lake',
    ]

    if issue_date.month in in_season:
        if site_id in no_flow_sites:
            return issue_date.replace(month=3, day=31)
        else:
            return issue_date - pd.offsets.MonthEnd(1)
    else:
        return issue_date - pd.offsets.Day(1)

def compute_snotel_features(
    snotel: pd.DataFrame, site_id: str, issue_date: pd.Timestamp
) -> pd.Series:
    """
    Computes SNOTEL features based on the last week's data for a given site and issue date.

    This function filters the SNOTEL data for the given site_id and the 7 days preceding the issue_date.
    It calculates the last water equivalent, maximum water equivalent, total precipitation,
    and average maximum daily temperature within this period.

    Args:
        snotel (pd.DataFrame): DataFrame containing SNOTEL data.
        site_id (str): The site ID for which features are to be computed.
        issue_date (pd.Timestamp): The issue date for computing features.

    Returns:
        pd.Series: A Series containing the computed SNOTEL features:
                   - snotel_swe_conditional: Last water equivalent value.
                   Returns NaN for all features if no data is available for the specified period.
    """
    # Filter data for the specified site_id and date range
    filtered_data = snotel[(snotel['site_id'] == site_id) &
                           (snotel['date'] < issue_date) &
                           (snotel['wateryear'] == issue_date.year)]

    filtered_data = filtered_data.dropna(subset=['WTEQ_DAILY'])

    # Return NaNs if the filtered dataset is empty
    if filtered_data.empty:
        if not os.environ.get('ENV') == 'this-is-my-beloved-macbook':  # For training the models on macbook
            logger.info(f'No SNOTEL data found for site {site_id} and date {issue_date}')
        wteq_latest = np.nan
        wteq_conditional = np.nan
    else:
        filtered_data = filtered_data.sort_values('date', ascending=True)
        wteq_latest = filtered_data['WTEQ_DAILY'].iloc[-1]
        conditional_swe_date = compute_conditional_swe_date(site_id, issue_date)
        conditional_data = filtered_data[filtered_data['date'] <= conditional_swe_date]
        wteq_conditional = conditional_data['WTEQ_DAILY'].iloc[-1]



    return pd.Series({
        'snotel_swe_latest': wteq_latest,
        'snotel_swe_conditional': wteq_conditional,
        })
