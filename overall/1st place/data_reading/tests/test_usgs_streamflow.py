import datetime

from wsfr_read.sites import read_metadata
from wsfr_read.streamflow import read_usgs_streamflow_data


def test_read_usgs_streamflow_data():
    df = read_usgs_streamflow_data("animas_r_at_durango", "2021-03-15")
    assert list(df.columns.values) == ["datetime", "discharge_cfs_mean"]
    assert df.shape[0] > 0
    assert max(df["datetime"]).date() == datetime.date(2021, 3, 14)
