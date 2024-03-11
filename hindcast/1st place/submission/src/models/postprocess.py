import numpy as np
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


def add_diff(df_pred):
    df_pred = df_pred.copy()
    pred_cols = list(df_pred.filter(like="pred"))

    for col in ["volume"] + pred_cols:
        df_pred[col] = df_pred[col] + df_pred["diff"]

    return df_pred


def use_previous_issue_date_forecast(
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


def use_previous_issue_date_forecast_sites(
    df,
    months=[5, 6, 7],
    days=[8, 15, 22],
    cols=cfg["pred_cols"],
    excluded_sites=td_usgs_sites + td_usbr_sites,
):
    """
    Use previous issue date forecast with additional params of list of excluded sites
    """
    df = df.copy()
    df.loc[
        (~df["site_id"].isin(excluded_sites)) & (df["month"].isin(months)) & (df["day"].isin(days)),
        cols,
    ] = np.NaN
    df = forward_fill(df, groupby_cols=["site_id", "year"])

    return df
