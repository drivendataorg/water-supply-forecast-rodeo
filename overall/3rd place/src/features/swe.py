"""
SWE features data processing
"""

import pandas as pd
from src.utils import groupby_cor


def get_snotel_sites_basins(path, df_sites):
    df = pd.read_csv(path, dtype={"snotel_id": "object"})
    df = (
        df.query("rnk <= 5 or rnk != rnk")
        .groupby(["site_id", "snotel_id"], as_index=False)
        .agg(method=("method", ",".join), dist=("dist", "min"), rnk=("rnk", "min"))
    )
    df = pd.merge(df, df_sites)
    return df


def get_basin_swe(
    df_volume,
    df_swe,
    df_pair,
    site_id="libby_reservoir_inflow",
    filter_condition="month in [1,2,3,4,5,6,7]",
):
    _res = pd.merge(
        pd.merge(df_volume[df_volume["site_id"] == site_id], df_pair),
        df_swe.query(filter_condition),
    )
    return _res


def get_basin_sites(df_pair, site_id="libby_reservoir_inflow"):
    _res = df_pair[df_pair["site_id"] == site_id]
    return _res


def get_basin_sites_r2(
    df_volume,
    df_swe,
    df_pair,
    site_id="libby_reservoir_inflow",
    groupby=["snotel_id", "yday"],
    cols=["prec_cml", "volume"],
):
    _res = groupby_cor(
        get_basin_swe(df_volume, df_swe, df_pair, site_id=site_id),
        groupby=groupby,
        cols=cols,
        is_r2=True,
        is_count=True,
    )
    if len(groupby) > 1:
        _res = _res.groupby("snotel_id", as_index=False).agg(
            size=("size", "sum"),
            cor=("cor", "mean"),
            r2=("r2", "mean"),
        )
    _res = (
        _res.sort_values("r2", ascending=False)
        .set_index("snotel_id")
        .join(
            get_basin_sites(df_pair, site_id=site_id).set_index("snotel_id")[
                ["method", "dist", "snotel_elevation"]
            ]
        )
    )

    return _res
