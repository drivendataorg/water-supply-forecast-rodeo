import pandas as pd

from wsfr_read import teleconnections


def test_mjo():
    df = teleconnections.read_mjo_data("2021-03-15")
    assert list(df.columns) == [
        "DATE",
        "INDEX_9",
        "INDEX_10",
        "INDEX_1",
        "INDEX_2",
        "INDEX_3",
        "INDEX_4",
        "INDEX_5",
        "INDEX_6",
        "INDEX_7",
        "INDEX_8",
    ]
    assert df.shape[0] > 0
    assert df.equals(df.sort_values("DATE"))
    assert df["DATE"].max() < pd.to_datetime("2021-03-15")


def test_nino_regions_sst():
    df = teleconnections.read_nino_regions_sst_data("2021-03-15")
    assert list(df.columns) == [
        "YR",
        "MON",
        "NINO1+2",
        "NINO1+2 ANOM",
        "NINO3",
        "NINO3 ANOM",
        "NINO4",
        "NINO4 ANOM",
        "NINO3.4",
        "NINO3.4 ANOM",
    ]
    assert df.shape[0] > 0
    assert df[df["YR"] == 2020].shape[0] == 12
    assert df.equals(df.sort_values(["YR", "MON"]))
    assert (df["YR"].values[-1], df["MON"].values[-1]) < (2021, 3)


def test_oni():
    df = teleconnections.read_oni_data("2021-03-15")
    assert list(df.columns) == ["SEAS", "YR", "TOTAL", "ANOM"]
    assert df.shape[0] > 0
    assert df[df["YR"] == 2020].shape[0] == 12
    assert df["SEAS"].values[-1] == "DJF"
    assert df["YR"].values[-1] == 2021


def test_pdo():
    df = teleconnections.read_pdo_data("2021-03-15")
    assert list(df.columns) == ["year", "month", "pdo_index"]
    assert df.shape[0] > 0
    assert df[df["year"] == 2020].shape[0] == 12
    assert df.equals(df.sort_values(["year", "month"]))
    assert (df["year"].values[-1], df["month"].values[-1]) < (2021, 3)


def test_pna():
    df = teleconnections.read_pna_data("2021-03-15")
    assert list(df.columns) == ["year", "month", "pna_index"]
    assert df.shape[0] > 0
    assert df[df["year"] == 2020].shape[0] == 12
    assert df.equals(df.sort_values(["year", "month"]))
    assert (df["year"].values[-1], df["month"].values[-1]) < (2021, 3)


def test_soi():
    df = teleconnections.read_soi_data("2021-03-15")
    assert list(df.columns) == ["year", "month", "soi"]
    assert df.shape[0] > 0
    assert df[df["year"] == 2020].shape[0] == 12
    assert df.equals(df.sort_values(["year", "month"]))
    assert (df["year"].values[-1], df["month"].values[-1]) < (2021, 3)
