import datetime
from pathlib import Path

import pandas as pd

from wsfr_read.config import DATA_ROOT

FILE_PATH_PARTS = ("teleconnections", "oni.txt")


# Maps each SEAS string to month when data becomes available
# e.g., DJF is Dec-Jan-Feb so data is available in Mar
SEAS_TO_AVAILABLE_MONTH = {
    "DJF": 3,
    "JFM": 4,
    "FMA": 5,
    "MAM": 6,
    "AMJ": 7,
    "MJJ": 8,
    "JJA": 9,
    "JAS": 10,
    "ASO": 11,
    "SON": 12,
    "OND": 1,
    "NDJ": 2,
}

# Maps each SEAS string to a delta year when data becomes available.
# Generally this is same year
# For OND/NDJ, the isn't available until Jan/Feb following year
SEAS_TO_AVAILABLE_YEAR_DELTA = {
    "DJF": 0,
    "JFM": 0,
    "FMA": 0,
    "MAM": 0,
    "AMJ": 0,
    "MJJ": 0,
    "JJA": 0,
    "JAS": 0,
    "ASO": 0,
    "SON": 0,
    "OND": 1,
    "NDJ": 1,
}


def read_oni_data(
    issue_date: str | datetime.date | pd.Timestamp,
    path: Path | None = None,
) -> pd.DataFrame:
    """Loads the Oceanic Nino Index (ONI) data appropriately subset for issue_date.

    Args:
        issue_date (str | datetime.date | pd.Timestamp): Issue date of forecast. This is used to
            subset the data returned so that the forecast does not use future data. Must be
            something pandas.to_datetime can parse, e.g., the string "2021-03-15".
        path (Path | None, optional): Path to data file. Default of None will use a default
            path relative to the data root directory.

    Returns:
        pd.DataFrame: Dataframe with the columns "SEAS", "YR", "TOTAL", and "ANOM"
    """
    issue_date = pd.to_datetime(issue_date)

    df = _read_full_oni_data(path=path)
    yr_avail = df["YR"] + df["SEAS"].map(SEAS_TO_AVAILABLE_YEAR_DELTA)
    mo_avail = df["SEAS"].map(SEAS_TO_AVAILABLE_MONTH)
    df = df[
        (yr_avail < issue_date.year)
        | ((yr_avail == issue_date.year) & (mo_avail <= issue_date.month))
    ]
    return df.copy()


def _read_full_oni_data(path: Path | None = None) -> pd.DataFrame:
    """Loads the full ONI dataframe. You should use the `read_oni_data` function instead to
    properly subset by time."""
    path = path or DATA_ROOT.joinpath(*FILE_PATH_PARTS)
    return pd.read_csv(path, sep=r"\s+")
