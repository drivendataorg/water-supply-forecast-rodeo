import datetime
from pathlib import Path

import pandas as pd

from wsfr_read.config import DATA_ROOT

FILE_PATH_PARTS = ("teleconnections", "mjo.txt")
DATE_COL_FORMAT = r"%Y%m%d"


def read_mjo_data(
    issue_date: str | datetime.date | pd.Timestamp,
    path: Path | None = None,
) -> pd.DataFrame:
    """Loads the Madden-Julian Oscillation index data appropriately subset for issue_date.

    Args:
        issue_date (str | datetime.date | pd.Timestamp): Issue date of forecast. This is used to
            subset the data returned so that the forecast does not use future data. Must be
            something pandas.to_datetime can parse, e.g., the string "2021-03-15".
        path (Path | None, optional): Path to data file. Default of None will use a default
            path relative to the data root directory.

    Returns:
        pd.DataFrame: Dataframe with the columns ["DATE", "INDEX_9", "INDEX_10", "INDEX_1",
            "INDEX_2", "INDEX_3", "INDEX_4", "INDEX_5", "INDEX_6", "INDEX_7", "INDEX_8"]
    """
    issue_date = pd.to_datetime(issue_date)

    df = _read_full_mjo_data(path=path)

    df["DATE"] = pd.to_datetime(df["DATE"], format=DATE_COL_FORMAT)
    df = df[df["DATE"] <= issue_date]
    return df.copy()


def _read_full_mjo_data(path: Path | None = None) -> pd.DataFrame:
    """Loads the full MJO index dataframe. You should use the `read_mjo_data` function instead to
    properly subset by time."""

    data_file = path or DATA_ROOT.joinpath(*FILE_PATH_PARTS)
    with data_file.open("r") as fp:
        # Read first line to get column headers
        cols = next(fp).split()
        # Skip next row which has the longitude of the index pattern centers
        next(fp)
        df = pd.read_csv(fp, sep=r"\s+", header=None, names=["DATE"] + cols, na_values=("*****",))
    return df
