import pandas as pd

from src.utils import *
from src.features.base import *
from src.features.swe import *
from src.features.volume_obs import *
from src.data.base import *
from src.config import *
from src.models.postprocess import *
from src.models.lgb import *

import warnings

warnings.filterwarnings("ignore")

df_train = read_train(is_forecast=True)
df_train = get_year_cat(df_train, val_years=[2004] + cfg["val_years"][0], test_years=cfg["test_years"])
df_meta = read_meta(f"data/meta/metadata_proc.csv")
df_monthly = read_monthly_naturalized_flow(is_forecast=True)

df_train_base = generate_base_train(df_train, use_new_format=True)
df_target_diff = generate_target_diff(df_monthly, df_meta)
df_train_base = get_volume_target_diff(
    df_train_base, df_target_diff, 0
)
df_volume_obs = prepare_volume_obs(df_meta)
df_volume_obs = prepare_volume_obs_diff(df_volume_obs, df_meta)
df_train_base = get_volume_target_diff_extra(df_train_base, df_volume_obs)

df_train_base_median = df_train_base.query('year > 1980 & year < 2004').groupby(['site_id','month'], as_index=False).agg(
    volume_mean = ('volume','mean'),
    pred_volume_10 = ('volume', lambda x: np.nanquantile(x, 0.1)),
    pred_volume_50 = ('volume', lambda x: np.nanquantile(x, 0.5)),
    pred_volume_90 = ('volume', lambda x: np.nanquantile(x, 0.9)),
)

df_pred_median_diff = generate_base_train(
    generate_base_year(read_meta(), years=cfg["test_years"]+cfg["val_years"][0]+[2004]
), use_new_format=True)
df_pred_median_diff = pd.merge(
    df_pred_median_diff, df_train_base
)
df_pred_median_diff = pd.merge(
    df_pred_median_diff,
    df_train_base_median
)

df_pred_median_diff.reset_index(drop=True).to_feather("data/sub/pred_median_dp.feather")