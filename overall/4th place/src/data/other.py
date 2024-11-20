import pandas as pd
import numpy as np
import os
from loguru import logger

def compute_antecedent_flow(flow: pd.DataFrame, site_id: str, issue_date: pd.Timestamp) -> pd.Series:
    """
    Computes the antecedent flow volume for the month preceding the specified issue date for a given site.

    This function filters the flow data for the given site_id and retrieves the flow volume for the month before the issue_date.

    Args:
        flow (pd.DataFrame): DataFrame containing flow data.
        site_id (str): The site ID for which antecedent flow is to be computed.
        issue_date (pd.Timestamp): The issue date for computing antecedent flow.

    Returns:
        float: The antecedent flow volume for the month before the issue date.
               Returns NaN if data is unavailable.
    """
    # Extract the previous month
    antecedent_month = (issue_date - pd.DateOffset(months=1)).month

    # Filter data for the specified site_id and date range
    flow_filtered = flow[(flow['site_id'] == site_id) &
                         (flow['forecast_year'] == issue_date.year) &
                         (flow['month'] == antecedent_month)]

    antecedent_flow = flow_filtered['volume'].iloc[0] if not flow_filtered.empty else np.nan

    if flow_filtered.empty and not os.environ.get('ENV') == 'this-is-my-beloved-macbook':
        logger.info(f'No antecedent flow data found for site {site_id} and date {issue_date}')

    return pd.Series({'antecedent_flow': antecedent_flow})


def compute_offset(flow: pd.DataFrame, site_id: str, issue_date: pd.Timestamp) -> pd.Series:
    """
    Computes the total water flow volume for the months that are already in season
    up to a given issue date for a specified site.

    Args:
        flow (pd.DataFrame): DataFrame containing flow data.
        site_id (str): The site ID for which the offset volume is to be computed.
        issue_date (pd.Timestamp): The issue date for computing the offset volume.

    Returns:
        pd.Series: A Series containing the offset volume:
                   - offset_volume: Total water flow volume for in-season months up to the issue date.
                   Returns 0 for sites with no flow or if no data is available.

    """
    # Filter the flow data for the specified site and year, and months before the issue date
    filtered_flow = flow[(flow['site_id'] == site_id) &
                         (flow['forecast_year'] == issue_date.year) &
                         (flow['month'] < issue_date.month)]

    # In theory, could remove the last month here, because last month is never in filtered_flow pd.DataFrame due to filtering.
    # Makes no difference, though. Leaving for readability.
    in_season = {
        "detroit_lake_inflow": [4, 5, 6],
        "pecos_r_nr_pecos": [3, 4, 5, 6, 7]
    }.get(site_id, [4, 5, 6, 7])  # Default in-season months

    in_season_flow = filtered_flow[filtered_flow['month'].isin(in_season)]
    offset_volume = in_season_flow['volume'].sum() if not in_season_flow.empty else 0

    return pd.Series({'offset_volume': offset_volume})
