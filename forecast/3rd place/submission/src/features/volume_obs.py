import numpy as np
import pandas as pd
from src.data.base import read_usbr_all, read_usgs_all, read_discharge
from src.features.base import (
    convert_cfs_to_kaf,
    parse_time_features,
    expand_date,
    forward_fill,
    filter_season_month,
    get_lag_features,
)
from src.config import td_usbr_sites, td_usgs_sites, USGS_DIR

_sites_with_missing_monthly_flow = [
    "american_river_folsom_lake",
    # "san_joaquin_river_millerton_reservoir", # no USGS and USBR site
    "merced_river_yosemite_at_pohono_bridge",
]


def prepare_usgs_data(df_meta, usgs_dir=USGS_DIR, usgs_format="rdb"):
    if usgs_format == "rdb":
        df_usgs = read_discharge("data/external/usgs_streamflow/raw_1900-01-01_2023-10-21_daily.txt")
    elif usgs_format == "csv":
        df_usgs = read_usgs_all(usgs_dir)
    df_usgs = convert_cfs_to_kaf(df_usgs, col="discharge")
    df_usgs = df_usgs.rename(columns={"discharge": "volume_obs"})
    df_usgs = pd.merge(df_usgs, df_meta[["usgs_id", "site_id"]])
    df_usgs = df_usgs[df_usgs["site_id"].isin(td_usgs_sites)]

    return df_usgs


def prepare_usbr_data(usbr_dir="data/external/usbr"):
    df_usbr = read_usbr_all(usbr_dir)
    df_usbr = df_usbr.rename(columns={"value": "volume_obs"})
    df_usbr = df_usbr[df_usbr["site_id"].isin(td_usbr_sites)]
    df_usbr = df_usbr.groupby(["site_id", "date"], as_index=False).agg(volume_obs=("volume_obs", "sum"))

    return df_usbr


def prepare_volume_obs(df_meta, usbr_dir="data/external/usbr", usgs_dir=USGS_DIR, usgs_format="rdb"):
    df_usbr = prepare_usbr_data(usbr_dir)
    df_usgs = prepare_usgs_data(df_meta, usgs_dir, usgs_format)

    df_volume_obs = pd.concat(
        [
            df_usbr[["date", "site_id", "volume_obs"]].assign(source="usbr"),
            df_usgs[["date", "site_id", "volume_obs"]].assign(source="usgs"),
        ]
    )
    del df_usbr, df_usgs
    df_volume_obs = parse_time_features(df_volume_obs)
    df_volume_obs = expand_date(df_volume_obs, groupby_cols=["site_id", "wyear"])
    df_volume_obs = parse_time_features(df_volume_obs)
    df_volume_obs = forward_fill(df_volume_obs, groupby_cols=["site_id", "wyear"])

    return df_volume_obs


def prepare_volume_obs_diff(df_volume_obs, df_meta):
    df_volume_obs_filtered = filter_season_month(df_volume_obs, df_meta)
    df_volume_obs_filtered = df_volume_obs_filtered.assign(
        volume_obs_cml=lambda x: x.groupby(["site_id", "wyear"])["volume_obs"].transform("cumsum"),
        volume_obs_month_cml=lambda x: x.groupby(["site_id", "wyear", "month"])["volume_obs"].transform("cumsum"),
    )
    df_volume_obs_filtered, _ = get_lag_features(
        df_volume_obs_filtered[
            [
                "site_id",
                "date",
                "day",
                "yday",
                "month",
                "year",
                "wyear",
                "volume_obs_cml",
                "volume_obs_month_cml",
            ]
        ],
        groupby_cols=["site_id", "wyear"],
        features=["volume_obs_cml", "volume_obs_month_cml"],
        lag_list=[1],
    )
    df_volume_obs_filtered = df_volume_obs_filtered.drop(columns=["wyear"])
    df_volume_obs_filtered.loc[df_volume_obs_filtered["day"] == 1, "volume_obs_month_cml_lag1"] = np.NaN

    return df_volume_obs_filtered


def get_volume_target_diff_extra(
    df_train_base,
    df_volume_obs_filtered,
    remove_cols=["diff_prev"],
):
    df_train_base = pd.merge(
        df_train_base,
        df_volume_obs_filtered.drop(columns=["date", "yday", "volume_obs_cml", "volume_obs_month_cml"]),
        how="left",
    )
    df_train_base["volume_obs_cml_lag1"] = df_train_base["volume_obs_cml_lag1"].fillna(0)
    df_train_base["volume_obs_month_cml_lag1"] = df_train_base["volume_obs_month_cml_lag1"].fillna(0)
    df_train_base = df_train_base.assign(
        is_use_diff_extra=lambda x: (x["volume_obs_month_cml_lag1"] > 0).astype(int),
        diff_prev=lambda x: x["diff"],
        diff=lambda x: np.where(
            x["site_id"].isin(_sites_with_missing_monthly_flow),
            x["volume_obs_cml_lag1"],
            x["diff"] + x["volume_obs_month_cml_lag1"],
        ),
        volume=lambda x: np.where(
            x["site_id"].isin(_sites_with_missing_monthly_flow),
            x["volume"] - x["diff"],
            x["volume"] - x["volume_obs_month_cml_lag1"],
        ),
        is_use_diff=lambda x: np.where(
            x["site_id"].isin(_sites_with_missing_monthly_flow) & x["diff"] > 0,
            1,
            x["is_use_diff"],
        ),
    )
    df_train_base = df_train_base.drop(columns=["volume_obs_cml_lag1", "volume_obs_month_cml_lag1"] + remove_cols)

    return df_train_base
