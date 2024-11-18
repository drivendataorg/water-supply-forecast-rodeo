import datetime
from functools import cache

import pandas as pd

from wsfr_read.config import DATA_ROOT

FILE = DATA_ROOT / "test_monthly_naturalized_flow.csv"


def read_test_monthly_naturalized_flow(
    site_id: str, issue_date: str | datetime.date | pd.Timestamp
):
    """Read monthly antecedent naturalized flow for test set years. Will subset to months before
    issue_date.

    Args:
        site_id (str): Identifier for forecast site
        issue_date (str | datetime.date | pd.Timestamp): Date that forecast is being issued for

    Returns:
        pd.DateFrame: dateframe index ("year", "month") and column "volume"
    """

    issue_date = pd.to_datetime(issue_date)
    df = _read_full_test_monthly_naturalized_flow()
    df = df.loc[site_id].loc[issue_date.year]
    # Subset by issue date
    df = df[
        (df["year"] < issue_date.year)
        | ((df["year"] == issue_date.year) & (df["month"] < issue_date.month))
    ]
    return df.set_index(["year", "month"])


@cache
def _read_full_test_monthly_naturalized_flow():
    """Reads the full 'test_monthly_naturalized_flow.csv' file to a dataframe. You should use
    'read_test_monthly_naturalized_flow' instead to get subsetting to a particular site and by
    time for an issue date."""
    return pd.read_csv(FILE, index_col=["site_id", "forecast_year"])
