import numpy as np
from src.config import cfg


def mean_ensemble(df_pred, cols=["volume"] + cfg["pred_cols"], groupby_cols=["site_id", "year", "month", "day"]):
    df_pred = df_pred.copy()
    df_pred["site_id"] = df_pred["site_id"].astype(str)
    df_pred = df_pred.groupby(groupby_cols, as_index=False)[cols].mean()

    return df_pred


def custom_ensemble(
    df_pred,
    groupby_cols=["site_id", "year", "month", "day"],
    agg_func=[
        lambda x: np.quantile(x, 0.1),
        "mean",
        lambda x: np.quantile(x, 0.9),
    ],
):
    df_pred = df_pred.copy()
    df_pred["site_id"] = df_pred["site_id"].astype(str)
    df_pred = df_pred.groupby(groupby_cols, as_index=False).agg(
        volume=("volume", agg_func[1]),
        pred_volume_10=("pred_volume_10", agg_func[0]),
        pred_volume_50=("pred_volume_50", agg_func[1]),
        pred_volume_90=("pred_volume_90", agg_func[2]),
    )

    return df_pred
