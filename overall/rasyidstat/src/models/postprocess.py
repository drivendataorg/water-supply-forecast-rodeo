import numpy as np
import pandas as pd
from src.features.base import forward_fill
from src.config import cfg, td_usbr_sites, td_usgs_sites


def scale_prediction(df, factor=[0.95, 1, 1.05]):
    df = df.copy()
    df = df.assign(
        pred_volume_10=lambda x: x["pred_volume_10"] * factor[0],
        pred_volume_50=lambda x: x["pred_volume_50"] * factor[1],
        pred_volume_90=lambda x: x["pred_volume_90"] * factor[2],
    )

    return df


def rearrange_prediction(df):
    df = df.copy()
    df = df.assign(
        pred=lambda x: x[["pred_volume_10", "pred_volume_50", "pred_volume_90"]].apply(
            lambda x: np.sort(np.array(x)), axis=1
        )
    ).assign(
        pred_volume_10=lambda x: x["pred"].apply(lambda x: x[0]),
        pred_volume_50=lambda x: x["pred"].apply(lambda x: x[1]),
        pred_volume_90=lambda x: x["pred"].apply(lambda x: x[2]),
    )

    return df


def clip_prediction(df, site_summary=None):
    df = df.copy()
    df[["pred_volume_10", "pred_volume_50", "pred_volume_90"]] = df[
        ["pred_volume_10", "pred_volume_50", "pred_volume_90"]
    ].clip(lower=0)
    if site_summary:
        pass

    return df


def round_prediction(df, rounding=1):
    df = df.copy()
    df[["pred_volume_10", "pred_volume_50", "pred_volume_90"]] = df[
        ["pred_volume_10", "pred_volume_50", "pred_volume_90"]
    ].round(rounding)

    return df


def add_diff(df_pred):
    df_pred = df_pred.copy()
    pred_cols = list(df_pred.filter(like="pred"))

    for col in ["volume"] + pred_cols:
        df_pred[col] = df_pred[col] + df_pred["diff"]

    return df_pred


def rescale(df_pred, method="max"):
    df_pred = df_pred.copy()
    pred_cols = list(df_pred.filter(like="pred"))

    if method == "max":
        for col in ["volume"] + pred_cols:
            df_pred[col] = df_pred[col] * df_pred["volume_max"]

    return df_pred


def use_previous_forecast(
    df,
    months=[5, 6, 7],
    days=[8, 15, 22],
    cols=cfg["pred_cols"],
):
    """
    Use previous issue date forecast
    """
    df = df.copy()
    for col in cols:
        df[col] = np.where((df["month"].isin(months)) & (df["day"].isin(days)), np.NaN, df[col])
    df = df.ffill()

    return df


def use_previous_forecast_sites(
    df,
    months=[5, 6, 7],
    days=[8, 15, 22],
    cols=cfg["pred_cols"],
    excluded_sites=td_usgs_sites + td_usbr_sites,
):
    df = df.copy()
    df.loc[
        (~df["site_id"].isin(excluded_sites)) & (df["month"].isin(months)) & (df["day"].isin(days)),
        cols,
    ] = np.NaN
    df = forward_fill(df, groupby_cols=["site_id", "year"])

    return df


def get_correction_factor(cf=0.005, start=1):
    end = (1 - (start - 1)) if start > 1 else start
    df_cf = pd.DataFrame(
        {
            "md_id": list(range(27, -1, -1)),
            "correction_factor_lower": [start - x * cf for x in range(28)],
            "correction_factor_upper": [end + x * cf for x in range(28)],
        }
    )
    return df_cf


def calibrate_interval(df_pred, cf=0.005, start=1):
    df_pred = df_pred.copy()
    df_cf = get_correction_factor(cf=cf, start=start)
    df_pred = pd.merge(df_pred, df_cf)
    df_pred["pred_volume_10"] = df_pred["pred_volume_10"] * df_pred["correction_factor_lower"]
    df_pred["pred_volume_90"] = df_pred["pred_volume_90"] * df_pred["correction_factor_upper"]

    return df_pred
