import numpy as np
from src.data.base import read_train

df_train = read_train(is_forecast=True)
df_pred_median_base = (
    df_train.query("year > 1990 & year < 2021")
    .groupby("site_id", as_index=False)
    .agg(
        volume_mean=("volume", "mean"),
        pred_volume_10=("volume", lambda x: np.nanquantile(x, 0.1)),
        pred_volume_50=("volume", lambda x: np.nanquantile(x, 0.5)),
        pred_volume_90=("volume", lambda x: np.nanquantile(x, 0.9)),
    )
)
df_pred_median_base.set_index("site_id").round(2).to_csv("data/meta/1991_2020_train_stats.csv")
