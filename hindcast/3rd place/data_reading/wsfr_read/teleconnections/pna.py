import datetime
from pathlib import Path

import pandas as pd

from wsfr_read.config import DATA_ROOT

FILE_PATH_PARTS = ("teleconnections", "pna.txt")


def read_pna_data(
    issue_date: str | datetime.date | pd.Timestamp,
    path: Path | None = None,
) -> pd.DataFrame:
    """Loads the Pacific-North American (PNA) Index data  appropriately subset for issue_date.

    Args:
        issue_date (str | datetime.date | pd.Timestamp): Issue date of forecast. This is used to
            subset the data returned so that the forecast does not use future data. Must be
            something pandas.to_datetime can parse, e.g., the string "2021-03-15".
        path (Path | None, optional): Path to data file. Default of None will use a default
            path relative to the data root directory.

    Returns:
        pd.DataFrame: Dataframe with the columns "year", "month", "pna_index"
    """
    issue_date = pd.to_datetime(issue_date)

    # Melt to get long format
    df = pd.melt(
        frame=_read_full_pna_data(path=path),
        id_vars=("year",),
        var_name="month",
        value_name="pna_index",
    )
    # Convert month abbrevation to number value
    df["month"] = df["month"].apply(datetime.datetime.strptime, args=(r"%b",)).dt.month
    df = df.sort_values(["year", "month"], ignore_index=True)
    # Subset by issue_date
    df = df[
        (df["year"] < issue_date.year)
        | ((df["year"] == issue_date.year) & (df["month"] < issue_date.month))
    ]
    return df


def _read_full_pna_data(path: Path | None = None) -> pd.DataFrame:
    """Loads the full PNA dataframe. You should use the `read_pna_data` function instead to
    properly subset by time."""
    path = path or DATA_ROOT.joinpath(*FILE_PATH_PARTS)
    return pd.read_fwf(path, widths=(4,) + (7,) * 12).rename(columns={"Unnamed: 0": "year"})
