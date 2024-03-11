import os
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
import argparse
import importlib

from loguru import logger

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description="Training script for Hindcast stage", argument_default=argparse.SUPPRESS
)

parser.add_argument("-c", "--config")
parser.add_argument("-s", "--seed", type=int, default=2024)
parser.add_argument("-y", "--year_min", type=int, default=1980)
parser.add_argument("-v", "--val_years", nargs="+", type=int)

args = parser.parse_args()

EXP_NAME = args.config
output_dir = f"runs/hindcast_stage/{EXP_NAME}"
if args.seed != 2024:
    output_dir = output_dir + f"_s{args.seed}"
if args.year_min != 1980:
    output_dir = output_dir + f"_y{args.year_min}"
model_dir = f"{output_dir}/models"
pred_dir = f"{output_dir}/preds"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

cfg = importlib.import_module(f"configs.lgb.{EXP_NAME}").cfg
cfg.update(vars(args))
year_min = cfg.get("year_min")

logger.add(f"{output_dir}/log.txt")

logger.info("Prepare train and metadata")
df_meta = read_meta("data/meta/metadata_proc.csv")
df_train = read_train(meta=df_meta, test_years=cfg["test_years"])
df_monthly = read_monthly_naturalized_flow()

if cfg.get("swe"):
    logger.info("Prepare SNOTEL SWE data")
    df_snotel_sites_chosen = pd.read_feather(
        "data/meta/snotel_sites_basins/snotel_sites_basins_chosen.feather"
    )
    if cfg["filter_snotel_sites"]:
        from src.meta.get_available_sites import filter_available_snotel_sites

        logger.info("Filter available SNOTEL sites based on runtime")
        df_snotel_sites_chosen = filter_available_snotel_sites(df_snotel_sites_chosen)

    df_swe = read_snotel_swe(SNOTEL_SWE_DIR, df_snotel_sites_chosen.snotel_id.tolist())
    df_swe = parse_time_features(
        df_swe, cols=["date"], level=["day", "yday", "month", "year", "wyear", "wyday"]
    )
    df_swe = forward_fill(df_swe, groupby_cols=["snotel_id", "wyear"])
    df_swe_filtered, swe_lag_features = get_lag_features(
        df_swe[
            ["snotel_id", "date", "day", "yday", "month", "year", "wyear"]
            + cfg["swe"]["features"]
        ].query("swe==swe"),
        groupby_cols=["snotel_id", "wyear"],
        features=cfg["swe"]["features"],
        start_lag=cfg["swe"]["start_lag"],
        lags=cfg["swe"]["lags"],
        interval=cfg["swe"]["interval"],
    )
    df_swe_filtered = df_swe_filtered.drop(columns=["wyear"])

df_train_base = generate_base_train(df_train, use_new_format=True)
if cfg["target_prep"]["diff"]:
    logger.info("Prepare target diff")
    df_target_diff = generate_target_diff(df_monthly, df_meta)
    df_train_base = get_volume_target_diff(
        df_train_base, df_target_diff, cfg["diff_gap"]
    )
    cfg["remove_features"] = cfg["remove_features"] + [
        "volume_actual",
        "diff",
        "month_tf",
    ]
if cfg["target_prep"].get("diff_extra"):
    logger.info("Prepare target diff extra from USGS and USBR data")
    df_volume_obs = prepare_volume_obs(df_meta)
    df_volume_obs = prepare_volume_obs_diff(df_volume_obs, df_meta)
    df_train_base = get_volume_target_diff_extra(df_train_base, df_volume_obs)

df_pred_all = []
for year in cfg["val_years"]:
    logger.info(f"Prepare training data with {year} as validation year")
    df_train = get_year_cat(df_train, val_years=year, test_years=cfg["test_years"])
    df_train_base = get_year_cat(
        df_train_base, val_years=year, test_years=cfg["test_years"]
    )
    df_train_all = df_train_base.copy()

    features = []
    if cfg.get("swe"):
        df_swe_features = []
        df_basin_sites_r2 = []
        for site_id in df_meta.site_id:
            _df_basin_sites_r2 = get_basin_sites_r2(
                df_train.query('cat=="train"'),
                df_swe.query("swe==swe & month in [1,2,3,4,5,6,7]"),
                df_snotel_sites_chosen,
                site_id=site_id,
                cols=[cfg["swe"].get("rank_feature", "prec_cml"), "volume"],
            ).sort_values("r2", ascending=False)
            _snotel_id_selected = _df_basin_sites_r2.head(
                cfg["swe"]["top_k"]
            ).index.tolist()
            df_basin_sites_r2.append(_df_basin_sites_r2.assign(site_id=site_id))

            _res = (
                get_basin_swe(
                    df_train[["site_id", "year"]],
                    df_swe_filtered.query("swe==swe & month in [1,2,3,4,5,6,7]"),
                    df_snotel_sites_chosen[["snotel_id", "site_id"]],
                    site_id=site_id,
                )
                .query("snotel_id in @_snotel_id_selected")
                .drop(columns=["snotel_id"] + cfg["swe"]["features"])
                .groupby(["site_id", "date", "year", "month", "day"], as_index=False)
                .mean()
                .drop(columns=["date"])
            )
            df_swe_features.append(_res)
        df_swe_features = pd.concat(df_swe_features)
        df_basin_sites_r2 = pd.concat(df_basin_sites_r2)
        bss_dir = f"{output_dir}/snotel_sites_basins"
        os.makedirs(bss_dir, exist_ok=True)
        df_basin_sites_r2.to_csv(f"{bss_dir}/y{year}.csv")

        df_train_all = pd.merge(df_train_all, df_swe_features, how="left")
        df_train_all = df_train_all.query("year >= @year_min")
        df_train_all = df_train_all.query("yday == yday").reset_index(drop=True)

        features = features + swe_lag_features

    df_train_all = get_meta_features(df_train_all, df_meta)

    if cfg["synthetic"]["n"] > 0:
        _df_synthetic, scale_factor = generate_synthetic_data(
            df_train_all,
            cols=["volume"] + features,
            n_synthetic=cfg["synthetic"]["n"],
            scale_factor=cfg["synthetic"]["scale_factor"],
            seed=cfg["seed"],
        )
        sf_dir = f"{output_dir}/scale_factor"
        os.makedirs(sf_dir, exist_ok=True)
        scale_factor.to_csv(f"{sf_dir}/y{year}.csv", index=False)
        df_train_all = pd.concat([df_train_all, _df_synthetic]).reset_index(drop=True)
        del _df_synthetic

    logger.info("Train LGBM models with quantile loss")
    mdl, df_pred = train_lgb(
        df_train_all,
        params=cfg["lgb_params"],
        cfg=cfg,
        use_log=cfg["target_prep"]["log"],
        model_output=f"{output_dir}/models/{EXP_NAME}_y{year}",
        pred_output=f"{output_dir}/preds/{EXP_NAME}_y{year}",
        pred_iters=[300, 400, 500],
    )

    if cfg["mode"]["reg"]:
        logger.info("Train LGBM models with Tweedie loss")
        mdl_reg, df_pred_reg = train_lgb(
            df_train_all.assign(volume=lambda x: x.volume.clip(0)),
            params=cfg["lgb_reg_params"],
            cfg=cfg,
            use_log=cfg["target_prep"]["log"],
            model_output=f"{output_dir}/models/{EXP_NAME}_reg_y{year}",
            pred_output=f"{output_dir}/preds/{EXP_NAME}_reg_y{year}",
            pred_iters=[800, 1000, 1400, 1200],
            quantiles=[50],
        )
        df_pred["pred_volume_reg"] = df_pred_reg["pred_volume_50"]

    df_pred_all.append(df_pred)
    print("\n")

df_pred_all = pd.concat(df_pred_all)

if cfg["target_prep"]["diff"]:
    df_pred_all = add_diff(df_pred_all)

df_pred_all.to_csv(f"{output_dir}/pred.csv", index=False)
