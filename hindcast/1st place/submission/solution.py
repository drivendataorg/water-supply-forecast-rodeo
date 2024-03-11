"""This is a template for the expected code submission format. Your solution must
implement the 'predict' function. The 'preprocess' function is optional."""

import os
import sys
import importlib
import glob
from loguru import logger
from pathlib import Path

sys.path.append("/code_execution/src/")
logger.info(f"Current working directory is {os.getcwd()}")

import numpy as np
import pandas as pd

from src.data.base import read_meta, read_snotel_swe, read_monthly_naturalized_flow
from src.data.usbr import download_usbr_hindcast
from src.features.base import generate_base_year, generate_base_train
from src.features.base import generate_target_diff, get_volume_target_diff
from src.features.base import (
    parse_time_features,
    forward_fill,
    expand_date,
    get_lag_features,
    get_meta_features,
)
from src.features.volume_obs import prepare_volume_obs, prepare_volume_obs_diff, get_volume_target_diff_extra
from src.models.lgb import predict_lgb
from src.models.ensemble import custom_ensemble
from src.models.postprocess import add_diff, use_previous_issue_date_forecast_sites
from src.qa import log_data_summary


EXP_NAME = "_test_2020_lgb_sweK9L2S1_diff_S4_m3_ff"
EXP_NAME = "lgb_sweK9L2S1_diffp_S4_m3_ff"
cfg = importlib.import_module(f"configs.lgb.{EXP_NAME}").cfg

cfg["use_ensemble_seeds"] = True
cfg["model_seeds"] = [cfg["seed"]]
if cfg["use_ensemble_seeds"]:
    cfg["model_seeds"] = cfg["model_seeds"] + [int(f"{s}024") for s in range(1, 10) if s != 2]


def prepare_swe_data(src_dir, SNOTEL_SWE_DIR, RUN_DIR):
    df_snotel_sites = pd.read_feather(src_dir / "data/meta/snotel_sites.feather")
    df_snotel_sites_chosen = []
    for val_year in cfg["val_years"]:
        _df_snotel_sites_chosen = (
            pd.read_csv(f"{RUN_DIR}/{EXP_NAME}/snotel_sites_basins/y{val_year}.csv", dtype={"snotel_id": "object"})
            .groupby("site_id")
            .head(cfg["swe"]["top_k"])
        )
        _df_snotel_sites_chosen = pd.merge(
            _df_snotel_sites_chosen,
            df_snotel_sites[["snotel_id", "snotel_triplet", "snotel_start_date"]],
        ).assign(val_year=val_year)
        df_snotel_sites_chosen.append(_df_snotel_sites_chosen)
    df_snotel_sites_chosen = pd.concat(df_snotel_sites_chosen)
    logger.info("There are {} SNOTEL sites from the metadata".format(df_snotel_sites_chosen.snotel_id.nunique()))

    df_swe = read_snotel_swe(SNOTEL_SWE_DIR, df_snotel_sites_chosen.snotel_id.tolist())
    df_swe = parse_time_features(df_swe)
    df_swe = df_swe[df_swe["wyear"].isin(cfg["test_years"])]
    df_swe = forward_fill(df_swe, groupby_cols=["snotel_id", "wyear"])
    logger.info("There are {} SNOTEL sites from the directory".format(df_swe.snotel_id.nunique()))

    if df_snotel_sites_chosen.snotel_id.nunique() != df_swe.snotel_id.nunique():
        logger.warnings("Missing SNOTEL sites from directory!")
    log_data_summary(df_swe, "df_swe", "SNOTEL SWE", cols=["swe", "prec_cml"])

    df_swe = expand_date(df_swe, groupby_cols=["snotel_id", "wyear"], max_mm_dd="07-22")
    df_swe = parse_time_features(df_swe)

    df_swe_filtered, swe_lag_features = get_lag_features(
        df_swe[["snotel_id", "date", "day", "yday", "month", "year", "wyear"] + cfg["swe"]["features"]],
        groupby_cols=["snotel_id", "wyear"],
        features=cfg["swe"]["features"],
        start_lag=cfg["swe"]["start_lag"],
        lags=cfg["swe"]["lags"],
        interval=cfg["swe"]["interval"],
    )
    df_swe_filtered = df_swe_filtered.drop(columns=["wyear"])

    return df_swe_filtered, df_snotel_sites_chosen, swe_lag_features


def get_swe_features(df_swe_filtered, df_snotel_sites_chosen, val_year):
    df_swe_filtered = df_swe_filtered.copy()
    df_swe_features = (
        pd.merge(
            df_swe_filtered,
            df_snotel_sites_chosen[["snotel_id", "site_id"]][df_snotel_sites_chosen["val_year"] == val_year],
        )
        .drop(columns=["date", "snotel_id"] + cfg["swe"]["features"])
        .groupby(["site_id", "year", "month", "day"], as_index=False)
        .mean()
    )

    return df_swe_features


def preprocess(src_dir, data_dir, preprocessed_dir):
    logger.info(f"Current working directory is {os.getcwd()}")

    RAW_DATA_DIR = data_dir
    RUN_DIR = src_dir / "runs"
    SNOTEL_SWE_DIR = data_dir / "snotel"
    USGS_DIR = data_dir / "usgs_streamflow"
    USBR_DIR = preprocessed_dir / "usbr"

    if not USBR_DIR.exists():
        USBR_DIR.mkdir(exist_ok=True)
        download_usbr_hindcast(USBR_DIR, wyears=cfg["test_years"])

    logger.info(EXP_NAME)
    logger.info(cfg["model_seeds"])
    logger.info(f"RAW_DATA_DIR is {RAW_DATA_DIR}")
    logger.info(f"RUN_DIR is {RUN_DIR}")
    logger.info(f"SNOTEL_SWE_DIR is {SNOTEL_SWE_DIR}")
    logger.info(f"USGS_DIR is {USGS_DIR}")

    logger.info("Prepare monthly naturalized flow and metadata...")
    df_meta = read_meta(src_dir / f"data/meta/metadata_proc.csv")
    df_test = generate_base_year(df_meta, years=cfg["test_years"])
    df_test["volume"] = np.NaN
    df_monthly = read_monthly_naturalized_flow(["test"], dirname=RAW_DATA_DIR)

    logger.info("Prepare SNOTEL SWE data...")
    df_swe_filtered, df_snotel_sites_chosen, swe_lag_features = prepare_swe_data(src_dir, SNOTEL_SWE_DIR, RUN_DIR)

    df_test_base = generate_base_train(df_test, use_new_format=True)
    if cfg["target_prep"]["diff"]:
        logger.info("Prepare target diff based on lagged monthly naturalized flow...")
        df_target_diff = generate_target_diff(df_monthly, df_meta)
        df_test_base = get_volume_target_diff(df_test_base, df_target_diff, cfg["diff_gap"])
        cfg["remove_features"] = cfg["remove_features"] + ["volume_actual", "diff", "month_tf"]

    if cfg["target_prep"].get("diff_extra"):
        logger.info("Prepare target diff extra based on lagged daily of observed flow from USGS and USBR data...")
        logger.info(glob.glob(f"{USBR_DIR}/**/*.csv", recursive=True))
        df_volume_obs = prepare_volume_obs(df_meta, USBR_DIR, USGS_DIR, usgs_format="csv")
        df_volume_obs = expand_date(df_volume_obs, groupby_cols=["site_id", "wyear"], max_mm_dd="07-22")
        df_volume_obs = parse_time_features(df_volume_obs)
        df_volume_obs = prepare_volume_obs_diff(df_volume_obs, df_meta)
        df_test_base = get_volume_target_diff_extra(df_test_base, df_volume_obs)

    df_pred_test_all = []
    logger.info("Generate prediction...")
    for val_year in cfg["val_years"]:
        df_swe_features = get_swe_features(df_swe_filtered, df_snotel_sites_chosen, val_year=val_year)
        df_test_all = df_test_base.copy()
        df_test_all = pd.merge(df_test_all, df_swe_features, how="left")
        df_test_all = get_meta_features(df_test_all, df_meta)
        log_data_summary(
            df_test_all,
            "df_test_all: swe_features",
            f"Model input of SWE features from fold={val_year}",
            cols=swe_lag_features,
        )

        for seed in cfg["model_seeds"]:
            EXP_NAME_SEED = EXP_NAME if seed == 2024 else f"{EXP_NAME}_s{seed}"
            df_pred_test_reg = predict_lgb(
                df_test_all,
                cfg=cfg,
                model_output=f"{RUN_DIR}/{EXP_NAME_SEED}/models/{EXP_NAME}_reg_y{val_year}",
                quantiles=[50],
            )
            df_pred_test = predict_lgb(
                df_test_all,
                cfg=cfg,
                model_output=f"{RUN_DIR}/{EXP_NAME_SEED}/models/{EXP_NAME}_y{val_year}",
                quantiles=[10, 90],
            )
            df_pred_test["pred_volume_50"] = df_pred_test_reg["pred_volume_50"]
            df_pred_test["seed"] = seed
            df_pred_test["val_year"] = val_year
            if cfg["target_prep"]["diff"]:
                df_pred_test = add_diff(df_pred_test)
            df_pred_test_all.append(df_pred_test)

    df_pred_test_all = pd.concat(df_pred_test_all)
    log_data_summary(
        df_pred_test_all, "df_pred_test_all", "All prediction from all models", cols=["diff"] + cfg["pred_cols"]
    )

    logger.info("Generate prediction ensemble...")
    df_pred_test_final = custom_ensemble(df_pred_test_all)

    logger.info("Postprocess - use forecast of day 1 of the month for sites with no daily observed flow")
    df_pred_test_final = use_previous_issue_date_forecast_sites(
        df_pred_test_final, months=[5, 6, 7], cols=["pred_volume_10", "pred_volume_50"]
    )

    logger.success("Prediction data frame is ready")
    log_data_summary(df_pred_test_final, "df_pred_test_final", "Prediction ensemble")

    df_pred_test_final = df_pred_test_final.assign(
        issue_date=lambda x: pd.to_datetime(
            x["year"].astype(str) + "-" + x["month"].astype(str) + "-" + x["day"].astype(str)
        ).astype(str)
    )
    # df_pred_test_final.to_csv("data/runs/pred.csv", index=False)

    return {"preds": df_pred_test_final.set_index(["site_id", "issue_date"]).T.to_dict()}


def predict(site_id, issue_date, assets, src_dir, data_dir, preprocessed_dir):
    p10 = assets["preds"][(site_id, issue_date)]["pred_volume_10"]
    p50 = assets["preds"][(site_id, issue_date)]["pred_volume_50"]
    p90 = assets["preds"][(site_id, issue_date)]["pred_volume_90"]

    return p10, p50, p90
