import datetime
from pathlib import Path

import pandas as pd

from wsfr_read.config import DATA_ROOT

FILE_PATH_PARTS = ("teleconnections", "nino_regions_sst.txt")


def read_nino_regions_sst_data(
    issue_date: str | datetime.date | pd.Timestamp,
    path: Path | None = None,
) -> pd.DataFrame:
    """Loads the Nino Regions Sea Surface Temperature (SST) data  appropriately subset for
    issue_date.

    Args:
        issue_date (str | datetime.date | pd.Timestamp): Issue date of forecast. This is used to
            subset the data returned so that the forecast does not use future data. Must be
            something pandas.to_datetime can parse, e.g., the string "2021-03-15".
        path (Path | None, optional): Path to data file. Default of None will use a default
            path relative to the data root directory.

    Returns:
        pd.DataFrame: Dataframe with the columns ["YR", "MON", "NINO1+2", "NINO1+2 ANOM", "NINO3",
            "NINO3 ANOM", "NINO4", "NINO4 ANOM", "NINO3.4", "NINO3.4 ANOM"]
    """
    issue_date = pd.to_datetime(issue_date)

    df = _read_full_nino_regions_sst_data(path=path)
    df = df[
        (df["YR"] < issue_date.year)
        | ((df["YR"] == issue_date.year) & (df["MON"] < issue_date.month))
    ]
    return df.copy()


def _read_full_nino_regions_sst_data(path: Path | None = None) -> pd.DataFrame:
    """Loads the full Nino Regions SST dataframe. You should use the `read_nino_regions_sst_data`
    function instead to properly subset by time."""
    data_file = path or DATA_ROOT.joinpath(*FILE_PATH_PARTS)
    df = pd.read_csv(data_file, sep=r"\s+")
    # Rename "ANOM" columns to include region labels
    return df.rename(
        columns={
            "ANOM": "NINO1+2 ANOM",
            "ANOM.1": "NINO3 ANOM",
            "ANOM.2": "NINO4 ANOM",
            "ANOM.3": "NINO3.4 ANOM",
        }
    )
