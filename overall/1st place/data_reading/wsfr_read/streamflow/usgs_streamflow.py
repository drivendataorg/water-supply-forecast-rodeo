import datetime
from pathlib import Path

import pandas as pd

from wsfr_read.config import DATA_ROOT
from wsfr_read.sites import read_metadata

# About: https://help.waterdata.usgs.gov/codes-and-parameters/parameters
# Look up: https://help.waterdata.usgs.gov/parameter_cd_nm
MEAN_DISCHARGE_RAW_COL = "00060_Mean"
MEAN_DISCHARGE_READABLE_COL = "discharge_cfs_mean"


def read_usgs_streamflow_data(
    site_id: str, issue_date: str | datetime.date | pd.Timestamp
) -> pd.DataFrame:
    """Read USGS daily mean streamflow data for a given forecast site as of a given forecast issue
    date.

    Args:
        site_id (str): Identifier for forecast site
        issue_date (str | datetime.date | pd.Timestamp): Date that forecast is being issued for

    Returns:
        pd.DateFrame: dateframe with columns ["datetime", "discharge_cfs_mean"]
    """

    issue_date = pd.to_datetime(issue_date)
    path = get_path_to_file(site_id, issue_date)
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df[df["datetime"].dt.date < issue_date.date()][["datetime", MEAN_DISCHARGE_RAW_COL]]
    df = df.rename(columns={MEAN_DISCHARGE_RAW_COL: MEAN_DISCHARGE_READABLE_COL})
    return df.copy()


def get_path_to_file(site_id: str, issue_date: str | datetime.date | pd.Timestamp) -> Path:
    """Get path to data file given site_id and an issue_date (for the forecast year of that issue
    date).

    Args:
        site_id (str): Identifier for forecast site
        issue_date (str | datetime.date | pd.Timestamp): Date that forecast is being issued for

    Returns:
        Path: path to CSV file
    """
    issue_date = pd.to_datetime(issue_date)
    if site_id not in read_metadata().index:
        raise ValueError(f"Invalid site_id: {site_id}")
    forecast_year = issue_date.year
    return DATA_ROOT / "usgs_streamflow" / f"FY{forecast_year}" / f"{site_id}.csv"
