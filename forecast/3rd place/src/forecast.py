"""
Additional code for inference on Forecast stage
"""
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthEnd


def as_date(issue_date):
    issue_date = pd.to_datetime(issue_date)
    return issue_date


def get_year(issue_date):
    issue_year = as_date(issue_date).year
    if get_month(issue_date) in [10, 11, 12]:
        issue_year = issue_year + 1
    return issue_year


def get_month(issue_date):
    issue_month = as_date(issue_date).month
    return issue_month


def get_date_from_df(df, use_eom=False):
    df = df.copy()
    df = df.assign(
        date=lambda x: pd.to_datetime(
            x["year"].astype(str) + "-" + x["month"].astype(str) + "-" + x["day"].astype(str)
        ).astype(str)
    )
    if use_eom:
        df = df.assign(date=lambda x: (pd.to_datetime(x["date"]) + MonthEnd(0)).astype(str))

    return df


def get_lag_date(date_str, days=1):
    date_str = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=days)
    date_str = date_str.strftime("%Y-%m-%d")

    return date_str


def filter_issue_date(df, issue_date):
    df = df.copy()
    df = df[df["wyear"] == get_year(issue_date)]
    df = df[df["date"] < issue_date]

    return df


def get_snotel_sites_chosen(src_dir, RUN_DIR, EXP_NAME, years, k=9):
    df_snotel_sites = pd.read_feather(src_dir / "data/meta/snotel_sites.feather")
    df_snotel_sites_chosen = []
    for val_year in years:
        _df_snotel_sites_chosen = (
            pd.read_csv(f"{RUN_DIR}/{EXP_NAME}/snotel_sites_basins/y{val_year}.csv", dtype={"snotel_id": "object"})
            .groupby("site_id")
            .head(k)
        )
        _df_snotel_sites_chosen = pd.merge(
            _df_snotel_sites_chosen,
            df_snotel_sites[["snotel_id", "snotel_triplet", "snotel_start_date"]],
        ).assign(val_year=val_year)
        df_snotel_sites_chosen.append(_df_snotel_sites_chosen)
    df_snotel_sites_chosen = pd.concat(df_snotel_sites_chosen)
    logger.info("There are {} SNOTEL sites from the metadata".format(df_snotel_sites_chosen.snotel_id.nunique()))

    return df_snotel_sites_chosen


def get_swe_features(df_swe_filtered, df_snotel_sites_chosen, val_year):
    df_swe_filtered = df_swe_filtered.copy()
    df_swe_features = (
        pd.merge(
            df_swe_filtered,
            df_snotel_sites_chosen[["snotel_id", "site_id"]][df_snotel_sites_chosen["val_year"] == val_year],
        )
        .drop(columns=["date", "snotel_id", "swe", "prec_cml"])
        .groupby(["site_id", "year", "month", "day", "yday"], as_index=False)
        .mean()
    )

    return df_swe_features
