"""
QA and logger

- Check if there's missing date
- Check if there's null value
- Check if there's outlier
- etc.
"""

import pandas as pd
from loguru import logger
from src.features.base import get_day_diff
from src.forecast import get_lag_date
from datetime import datetime, timedelta


def log_data_summary(df, df_name, df_desc, cols=None, show_col=False):
    df = df.copy()
    if show_col:
        logger.info(list(df))
    if cols:
        df = df[cols]
    logger.info(f"{df_desc} overview\n{df_name} shape: {df.shape}\n{df_name} summary:\n{df.describe().to_string()}")


def log_data(df, df_name, df_desc, cols=None, show_col=False):
    df = df.copy()
    df = shorten_id(df)
    if show_col:
        logger.info(list(df))
    if cols:
        df = df[cols]
    logger.info(f"{df_desc} data\n{df_name} shape: {df.shape}\n{df.to_string()}")


def write_feather(df, filename, is_cache=False, message=None):
    if is_cache:
        df.to_feather(filename)
        if message:
            logger.info(message)


def write_csv(df, filename, is_cache=False, message=None):
    if is_cache:
        df.to_csv(filename, index=False)
        if message:
            logger.info(message)


def check_missing_dates(
    df, cols=["snotel_id", "date", "swe", "prec_cml", "day_diff"], groupby_cols=["snotel_id", "wyear"], return_df=False
):
    df = df.copy()
    _res = get_day_diff(df, groupby_cols=groupby_cols).query("day_diff > 1")
    _res_sites = _res[groupby_cols[0]].nunique()
    if 0 < len(_res) <= 20:
        logger.warning(f"There are {len(_res)} missing dates\n{_res[cols]}")
    elif len(_res) > 20:
        logger.error(f"There are more than {len(_res)} missing dates")
    if return_df:
        return _res


def check_duplicates(df, cols=["site_id", "date"]):
    df = df.copy()
    df_dup = df[df.duplicated(subset=cols)]
    if len(df_dup) > 0:
        logger.warning(f"There are {len(df_dup)} duplicated rows")


def check_max_dates(df, issue_date, col="swe", id_col="snotel_id"):
    df = df.copy()
    issue_date_d1 = get_lag_date(issue_date, 1)
    _res = df[~df[col].isna()]
    _res_smr_all = df[~df[col].isna()].groupby(id_col).agg(max_date=("date", "max")).sort_values("max_date")
    _res_smr_all = _res_smr_all.assign(
        gap=lambda x: (pd.to_datetime(issue_date) - pd.to_datetime(x["max_date"])).dt.days
    )
    _res_smr = _res_smr_all.query("max_date < @issue_date_d1")
    if len(_res_smr) > 0:
        logger.warning(f"For {col}, there are {len(_res_smr)} sites with dates less than {issue_date} \n{_res_smr}")
    if df[id_col].nunique() != _res[id_col].nunique():
        _res_missing = set(df[id_col]) - set(_res[id_col])
        logger.critical(f"For {col}, there are {len(_res_missing)} missing sites: {_res_missing}")
    return _res_smr_all


def check_missing_sites(df, df_snotel_sites_chosen):
    df = df.copy()
    df_snotel_sites_chosen = df_snotel_sites_chosen.copy()
    missing_sites = set(df_snotel_sites_chosen.snotel_id) - set(df.snotel_id)

    if len(missing_sites) > 0:
        logger.warning(f"There are {len(missing_sites)} missing SNOTEL sites!\n{missing_sites}")


def check_negative_values(df, col):
    df = df.copy()
    df = df[df[col] < 0]
    if len(df) > 0:
        logger.warning(f"For {col}, there are {len(df)} records with negative value!")


def shorten_id(df):
    df = df.copy()
    df["site_id"] = df["site_id"].str.split("_").str[0]
    return df


if __name__ == "__main__":
    df_sample = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    x = 1
    log_data_summary(df_sample, "df_sample", "Sample data")
