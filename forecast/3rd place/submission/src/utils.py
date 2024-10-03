import numpy as np
import pandas as pd
import random

# from IPython.display import display
from sklearn.metrics import (
    mean_pinball_loss,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)
from src.data.base import read_sub


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def agg_error_metrics(df, y_actual="volume", y_pred=["pred_volume_10", "pred_volume_50", "pred_volume_90"]):
    y_pred_10 = y_pred[0]
    y_pred_90 = y_pred[2]
    y_pred = y_pred[1]
    n = len(df[y_actual])
    mpl10 = 2 * mean_pinball_loss(df[y_actual], df[y_pred_10], alpha=0.1)
    mpl50 = 2 * mean_pinball_loss(df[y_actual], df[y_pred], alpha=0.5)
    mpl90 = 2 * mean_pinball_loss(df[y_actual], df[y_pred_90], alpha=0.9)
    int_cvr = np.sum(np.where((df[y_actual] > df[y_pred_10]) & (df[y_actual] < df[y_pred_90]), 1, 0)) / n
    rmse = np.sqrt(mean_squared_error(df[y_actual], df[y_pred]))
    r2 = r2_score(df[y_actual], df[y_pred])
    mae = mean_absolute_error(df[y_actual], df[y_pred])
    mape = mean_absolute_percentage_error(df[y_actual], df[y_pred])
    bias = np.mean(df[y_actual] - df[y_pred])
    actual_mean = np.mean(df[y_actual])
    pred_mean = np.mean(df[y_pred])

    return pd.Series(
        dict(
            n=n,
            mpl=(mpl10 + mpl50 + mpl90) / 3,
            mpl10=mpl10,
            mpl50=mpl50,
            mpl90=mpl90,
            int_cvr=int_cvr,
            rmse=rmse,
            r2=r2,
            mape=mape,
            bias=bias,
            actual_mean=actual_mean,
            pred_mean=pred_mean,
        )
    )


def eval_agg(pred_df, grouper=["year"], is_include_mean_std=False):
    eval_agg_df = pred_df.groupby(grouper).apply(agg_error_metrics)
    if "site_id" in grouper:
        eval_agg_df = eval_agg_df.sort_values("actual_mean", ascending=False)
    if (is_include_mean_std == True) & (len(eval_agg_df) > 1):
        eval_agg_df = pd.concat(
            [
                eval_agg_df,
                eval_agg_df.mean().to_frame().T,
                eval_agg_df.std().to_frame().T,
            ],
            axis=0,
        )

    return eval_agg_df


def eval_all(pred_df, grouper_list=[["year"]], html=True, is_include_mean_std=True):
    for grouper in grouper_list:
        if html:
            print(grouper)
            display(eval_agg(pred_df, grouper=grouper, is_include_mean_std=is_include_mean_std))
            print("\n")
        else:
            print(grouper)
            print(eval_agg(pred_df, grouper=grouper, is_include_mean_std=is_include_mean_std))
            print("\n")


def groupby_cor(df, groupby, cols, is_count=False, is_r2=False):
    df = df.copy()
    smr_cor = df.groupby(groupby)[cols].corr().iloc[0::2, -1].reset_index().rename(columns={cols[-1]: "cor"})
    if is_r2:
        smr_cor["r2"] = np.round(smr_cor["cor"] ** 2, 6)
    if is_count:
        smr_cor = pd.merge(df.groupby(groupby, as_index=False).size(), smr_cor)

    return smr_cor


# def generate_hindcast_submission(df_pred, df_sub=read_sub(), fname=None, dirname="data/sub/hindcast"):
#     df_sub = pd.merge(
#         df_sub.drop(columns=["volume_10", "volume_50", "volume_90"]),
#         df_pred.assign(
#             issue_date=lambda x: pd.to_datetime(
#                 x["year"].astype(str) + "-" + x["month"].astype(str) + "-" + x["day"].astype(str)
#             ),
#             volume_10=lambda x: x["pred_volume_10"],
#             volume_50=lambda x: x["pred_volume_50"],
#             volume_90=lambda x: x["pred_volume_90"],
#         ).assign(issue_date=lambda x: x["issue_date"].astype(str))[
#             ["site_id", "issue_date", "volume_10", "volume_50", "volume_90"]
#         ],
#     )
#     if dirname:
#         df_sub.to_csv(f"{dirname}/{fname}.csv", index=False)

#     return df_sub
