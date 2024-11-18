"""
Additional code for inference on Forecast stage
"""

import numpy as np
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


def get_snotel_sites_chosen(src_dir, RUN_DIR, EXP_NAME, years, fname="data/meta/snotel_sites.feather", k=9):
    df_snotel_sites = pd.read_feather(src_dir / fname)
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
            how="left"
        ).assign(val_year=val_year)
        df_snotel_sites_chosen.append(_df_snotel_sites_chosen)
    df_snotel_sites_chosen = pd.concat(df_snotel_sites_chosen)
    logger.info("There are {} SNOTEL sites from the metadata".format(df_snotel_sites_chosen.snotel_id.nunique()))

    return df_snotel_sites_chosen


def get_swe_stats(RUN_DIR, EXP_NAME, years):
    df_swe_stats = []
    for val_year in years:
        df_swe_stats.append(
            pd.read_csv(f"{RUN_DIR}/{EXP_NAME}/swe_stats_{val_year}.csv", dtype={"snotel_id": "object"})
        )
    df_swe_stats = pd.concat(df_swe_stats)

    return df_swe_stats


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


def get_swe_features_mm(
    df_swe_filtered, df_snotel_sites_chosen, df_swe_stats, swe_lag_features, val_year, comp_thr=0.65
):
    df_swe_filtered = df_swe_filtered.copy()
    df_swe_filtered_mm = pd.merge(
        df_swe_filtered,
        df_snotel_sites_chosen[["snotel_id", "site_id"]][df_snotel_sites_chosen["val_year"] == val_year].assign(
            n_sites=lambda x: x.groupby("site_id")["site_id"].transform("size")
        ),
    )
    df_swe_filtered_mm = pd.merge(df_swe_filtered_mm, df_swe_stats[df_swe_stats["val_year"] == val_year])
    for feature in swe_lag_features:
        if "swe" in feature:
            df_swe_filtered_mm[feature] = df_swe_filtered_mm[feature] / df_swe_filtered_mm["swe_max"]
        elif "prec_cml" in feature:
            df_swe_filtered_mm[feature] = df_swe_filtered_mm[feature] / df_swe_filtered_mm["prec_cml_max"]
    df_swe_filtered_mm = df_swe_filtered_mm.drop(columns=["swe_max", "prec_cml_max"])

    df_swe_features = (
        df_swe_filtered_mm.drop(columns=["date", "snotel_id", "swe", "prec_cml"])
        .groupby(["site_id", "year", "month", "day", "yday", "n_sites"], as_index=False)
        .agg({k: ["count", "mean"] for k in swe_lag_features})
    )
    df_swe_features.columns = [
        "__".join(col).strip() if "count" in col else col[0] for col in df_swe_features.columns.values
    ]
    for col in swe_lag_features:
        df_swe_features[col] = np.where(
            df_swe_features[col + "__count"] / df_swe_features["n_sites"] < comp_thr, np.NaN, df_swe_features[col]
        )
    df_swe_features = df_swe_features.drop(
        columns=["n_sites"] + [x for x in df_swe_features.columns if "__count" in x]
    )

    return df_swe_features
