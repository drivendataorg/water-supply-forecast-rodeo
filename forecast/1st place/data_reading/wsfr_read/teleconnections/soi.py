import datetime
from pathlib import Path

import pandas as pd

from wsfr_read.config import DATA_ROOT

FILE_PATH_PARTS = ("teleconnections", "soi.txt")


def read_soi_data(
    issue_date: str | datetime.date | pd.Timestamp,
    path: Path | None = None,
) -> pd.DataFrame:
    """Loads the Southern Oscillation Index (SOI) data  appropriately subset for issue_date.

    Args:
        issue_date (str | datetime.date | pd.Timestamp): Issue date of forecast. This is used to
            subset the data returned so that the forecast does not use future data. Must be
            something pandas.to_datetime can parse, e.g., the string "2021-03-15".
        path (Path | None, optional): Path to data file. Default of None will use a default
            path relative to the data root directory.

    Returns:
        pd.DataFrame: Dataframe with the columns "year", "month", "soi"
    """
    issue_date = pd.to_datetime(issue_date)

    # Melt to get long format
    df = pd.melt(
        frame=_read_full_soi_data(path=path),
        id_vars=("YEAR",),
        var_name="month",
        value_name="soi",
    )
    df = df.rename(columns={"YEAR": "year"})
    # Convert month abbrevation to number value
    df["month"] = df["month"].apply(datetime.datetime.strptime, args=(r"%b",)).dt.month
    df = df.sort_values(["year", "month"], ignore_index=True)
    df = df[
        (df["year"] < issue_date.year)
        | ((df["year"] == issue_date.year) & (df["month"] < issue_date.month))
    ]
    return df


def _read_full_soi_data(path: Path | None = None) -> pd.DataFrame:
    """Loads full SOI dataframe. You should use the `read_soi_data` function instead to properly
    subset by time."""
    # Raw data file contains two fixed-width files: first is not standardized, and second is
    # standardized. The standardized values are the most common representation of SOI.
    path = path or DATA_ROOT.joinpath(*FILE_PATH_PARTS)
    with path.open("r") as fp:
        # Get line number that contains "STANDARDIZED DATA"
        line_no = 0
        for line_no, line in enumerate(fp):
            if "STANDARDIZEDDATA" in line.replace(" ", ""):
                break
    # Skip that line and the next one
    skiprows = line_no + 2
    return pd.read_fwf(path, widths=(4,) + (6,) * 12, skiprows=skiprows, na_values=("-999.9",))
