import os
import sys
from loguru import logger
from pathlib import Path

sys.path.append("/code_execution/src/")
logger.info(f"Current working directory is {os.getcwd()}")

import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path
import json
import zipfile

from src.utils import *
from src.features.base import *
from src.features.volume_obs import *
from src.features.pdsi import *
from src.data.base import *
from src.data.usbr import *
from src.forecast import *

from src.models.postprocess import *
from src.models.lgb import *
from src.models.ensemble import *
from src.qa import *

import warnings

warnings.filterwarnings("ignore")


def main(issue_date, src_dir, data_dir, preprocessed_dir):
    issue_year = get_year(issue_date)
    issue_month = get_month(issue_date)

    if not preprocessed_dir:
        preprocessed_dir = Path(f"data/processed/exp_base/predict_{issue_date}")
        preprocessed_dir.mkdir(exist_ok=True, parents=True)
    logger.remove()
    logger.add(sys.stderr)
    logger.add(preprocessed_dir / "log.txt")

    RAW_DATA_DIR = data_dir
    RUN_DIR = src_dir / "runs/forecast_v2_prd"
    SNOTEL_SWE_DIR = data_dir / "snotel"
    USGS_DIR = data_dir / "usgs_streamflow"
    USBR_DIR = preprocessed_dir / "usbr"
    PDSI_DIR = preprocessed_dir / "pdsi"

    if not RUN_DIR.exists():
        model_zip = src_dir / "runs.zip"
        logger.info("Unzipping models...")
        with zipfile.ZipFile(model_zip, "r") as zip_ref:
            zip_ref.extractall(src_dir)
        os.remove(model_zip)

    logger.info(f"Issue date is {issue_date}")
    logger.info(f"RAW_DATA_DIR is {RAW_DATA_DIR}")
    logger.info(f"RUN_DIR is {RUN_DIR}")
    logger.info(f"SNOTEL_SWE_DIR is {SNOTEL_SWE_DIR}")
    logger.info(f"USGS_DIR is {USGS_DIR}")
    logger.info(f"USBR_DIR is {USBR_DIR}")
    logger.info(f"PDSI_DIR is {PDSI_DIR}")

    # Models
    model_list = [
        "pdsi_swe_k9_s1_dp",
        "pdsi_swe_k5_s3_dp",
        "swe_k9_s1_dp",
        "swe_k5_s3_dp",
    ]
    BASE_EXP_NAME = "pdsi_swe_k9_s1_dp"
    cfg_base = json.load(open(f"{RUN_DIR}/{BASE_EXP_NAME}/config.json", "r"))
    cfg_base["is_cache"] = True
    cfg_base["use_ensemble_seeds"] = True
    cfg_base["model_seeds"] = [cfg_base["seed"]]
    if cfg_base["use_ensemble_seeds"]:
        cfg_base["model_seeds"] = cfg_base["model_seeds"] + [int(f"{s}024") for s in range(1, 4) if s != 2]

    # Params for simulation
    cfg_sim = {}
    cfg_sim["gap_natflow"] = 0
    cfg_sim["gap_obsflow"] = 0
    cfg_sim["gap_swe"] = 0

    logger.info("Prepare base and metadata...")
    df_meta = read_meta(src_dir / f"data/meta/metadata_proc.csv")
    df_test_base_all = generate_base_train(generate_base_year(df_meta, [issue_year]), use_new_format=True)
    df_test_base_all["volume"] = np.NaN
    df_test_base_all = get_date_from_df(df_test_base_all)
    df_test_base = df_test_base_all[df_test_base_all["date"] <= issue_date]

    logger.info("Prepare monthly naturalized flow...")
    df_monthly = read_monthly_naturalized_flow(cats=["test"], dirname=data_dir, is_dropna=False)
    df_monthly = df_monthly[df_monthly["forecast_year"] == issue_year]
    df_monthly = get_date_from_df(df_monthly.assign(day=1), use_eom=True)
    df_monthly = df_monthly.drop(columns=["forecast_year"])
    df_monthly = parse_time_features(df_monthly)
    df_monthly = filter_issue_date(df_monthly, issue_date=issue_date)
    check_negative_values(df_monthly, col="volume")

    if cfg_sim["gap_natflow"] > 0:
        np.random.seed(cfg_base["seed"])
        diff_gap = np.random.randint(cfg_sim["gap_natflow"], size=23)
        df_monthly_filter = (
            df_monthly.groupby(["site_id"], as_index=False)
            .agg(max_month=("month", "max"))
            .assign(diff_gap=diff_gap + 1, max_month_after=lambda x: x["max_month"] - x["diff_gap"])
        )
        df_monthly = pd.merge(df_monthly, df_monthly_filter).query("month <= max_month_after")
    smr_natflow_qa = check_max_dates(df_monthly, col="volume", id_col="site_id", issue_date=issue_date[:7] + "-01")

    df_monthly = df_monthly.assign(date=lambda x: pd.to_datetime(x["date"]).dt.strftime("%Y-%m-01"))
    df_monthly = parse_time_features(df_monthly)
    df_monthly = expand_date(df_monthly, groupby_cols=["site_id"], max_ymd=issue_date, freq="MS")
    df_monthly = parse_time_features(df_monthly)
    df_monthly["volume"] = df_monthly["volume"].fillna(-999999)

    logger.info("Prepare target diff from monthly naturalized flow...")
    df_target_diff = generate_target_diff(df_monthly, df_meta)
    df_test_base = get_volume_target_diff(df_test_base, df_target_diff, 0)
    log_data(
        df_test_base.query("diff>=0").groupby("site_id", as_index=False).last(),
        "df_test_base",
        "Target diff latest date",
        cols=["site_id", "date", "diff", "is_use_diff"],
    )

    logger.info("Prepare daily observed flow from USGS data...")
    df_usgs = prepare_usgs_data(df_meta, USGS_DIR, usgs_format="csv")
    log_data(
        df_usgs.groupby("site_id").tail(7).assign(),
        "df_usgs",
        "USGS raw data (last 7 records)",
        cols=["site_id", "date", "volume_obs", "discharge_qa"],
    )
    check_negative_values(df_usgs, col="volume_obs")
    df_usgs["volume_obs"] = df_usgs["volume_obs"].mask(df_usgs["volume_obs"] < 0, np.NaN)
    df_usgs = df_usgs.dropna(subset=["volume_obs"])
    df_usgs = parse_time_features(df_usgs)
    df_usgs = filter_issue_date(df_usgs, issue_date=issue_date)
    check_missing_dates(df_usgs, cols=["site_id", "date", "volume_obs", "day_diff"], groupby_cols=["site_id", "wyear"])
    check_duplicates(df_usgs, cols=["site_id", "date"])
    usgs_missing_sites = set(td_usgs_sites) - set(df_usgs.site_id)
    if len(usgs_missing_sites) > 0:
        logger.critical(f"There are {len(usgs_missing_sites)} missing USGS sites: {usgs_missing_sites}")

    if issue_month >= 1:
        if not USBR_DIR.exists():
            logger.info("Download USBR data...")
            USBR_DIR.mkdir()
            download_usbr_forecast(USBR_DIR, issue_date=issue_date)
        logger.info("Prepare daily observed flow from USBR data...")
        df_usbr = prepare_usbr_data(USBR_DIR)
        log_data(df_usbr.groupby("site_id").tail(7), "df_usbr", "USBR raw data (last 7 records)")
        check_negative_values(df_usbr, col="volume_obs")
        df_usbr["volume_obs"] = df_usbr["volume_obs"].mask(df_usbr["volume_obs"] < 0, np.NaN)
        df_usbr = df_usbr.dropna(subset=["volume_obs"])
        df_usbr = parse_time_features(df_usbr)
        df_usbr = filter_issue_date(df_usbr, issue_date=issue_date)
        check_missing_dates(
            df_usbr, cols=["site_id", "date", "volume_obs", "day_diff"], groupby_cols=["site_id", "wyear"]
        )
        check_duplicates(df_usbr, cols=["site_id", "date"])
        usbr_missing_sites = set(td_usbr_sites) - set(df_usbr.site_id)
        if len(usbr_missing_sites) > 0:
            logger.critical(f"There are {len(usbr_missing_sites)} missing USBR sites: {usbr_missing_sites}")
            for site_id in usbr_missing_sites:
                df_usbr = pd.concat([df_usbr, df_usbr.iloc[0:1].assign(site_id=site_id, volume_obs=np.NaN)])
                logger.info(f"Set NA value for {site_id}")
                log_data(df_usbr.tail(1), "df_usbr", "USBR missing sites")
    else:
        df_usbr = pd.DataFrame({}, columns=["date", "site_id", "volume_obs"])

    logger.info("Combine USGS and USBR data...")
    df_volume_obs = pd.concat(
        [
            df_usbr[["date", "site_id", "volume_obs"]].assign(source="usbr"),
            df_usgs[["date", "site_id", "volume_obs"]].assign(source="usgs"),
        ]
    ).reset_index(drop=True)

    if cfg_sim["gap_obsflow"] > 0:
        np.random.seed(cfg_base["seed"])
        diff_gap = np.random.randint(cfg_sim["gap_obsflow"], size=14)
        df_volume_obs_filter = (
            df_volume_obs.groupby(["site_id"], as_index=False)
            .agg(max_date=("date", "max"))
            .assign(
                diff_gap=diff_gap + 1,
                max_date_after=lambda x: pd.to_datetime(x["max_date"]) - pd.to_timedelta(x["diff_gap"], unit="d"),
            )
        )
        df_volume_obs = pd.merge(df_volume_obs, df_volume_obs_filter).query("date <= max_date_after")
    smr_obsflow_qa = check_max_dates(df_volume_obs, col="volume_obs", id_col="site_id", issue_date=issue_date)
    df_volume_obs = parse_time_features(df_volume_obs)
    df_volume_obs = expand_date(df_volume_obs, groupby_cols=["site_id", "wyear"])
    df_volume_obs = parse_time_features(df_volume_obs)
    write_csv(
        df_volume_obs, preprocessed_dir / "volume_obs.csv", message="Cache volume obs", is_cache=cfg_base["is_cache"]
    )
    df_volume_obs = fill_missing_value(df_volume_obs, cols=["volume_obs"], groupby_cols=["site_id", "wyear"], limit=5)
    df_volume_obs = expand_date(df_volume_obs, groupby_cols=["site_id", "wyear"], max_ymd=issue_date)
    df_volume_obs = parse_time_features(df_volume_obs)
    df_volume_obs["volume_obs"] = df_volume_obs["volume_obs"].fillna(-999999)

    logger.info("Prepare target diff from daily observed flow...")
    df_volume_obs = prepare_volume_obs_diff(df_volume_obs, df_meta)
    df_test_base = get_volume_target_diff_extra(df_test_base, df_volume_obs)
    log_data(
        df_test_base.query("diff>=0").groupby("site_id", as_index=False).last(),
        "df_test_base (after obsflow)",
        "Target diff extra latest date",
        cols=["site_id", "date", "diff", "is_use_diff", "is_use_diff_extra"],
    )

    logger.info("Prepare PDSI data...")
    if not PDSI_DIR.exists():
        logger.info("Preprocess PDSI data...")
        PDSI_DIR.mkdir()
        df_meta_poly, _ = read_meta_geo(path=data_dir / "geospatial.gpkg")
        preprocess_pdsi(dirname=data_dir / "pdsi", df_meta_poly=df_meta_poly, out_file=PDSI_DIR / "pdsi_v2.csv")

    df_pdsi = pd.read_csv(PDSI_DIR / "pdsi_v2.csv")
    df_pdsi = df_pdsi.drop_duplicates()
    log_data(df_pdsi.groupby("site_id").tail(2), "df_pdsi", "PDSI (last 3 records)")
    check_duplicates(df_pdsi, cols=["site_id", "date"])
    df_pdsi = parse_time_features(df_pdsi, cols=["date"])
    df_pdsi = df_pdsi.sort_values(["site_id", "date"]).reset_index(drop=True)
    df_pdsi = filter_issue_date(df_pdsi, issue_date=get_lag_date(issue_date, 5))

    df_pdsi_expand = expand_date(df_pdsi, groupby_cols=["site_id", "wyear"], max_ymd=issue_date)
    df_pdsi_expand = parse_time_features(df_pdsi_expand, cols=["date"])
    df_pdsi_expand = forward_fill(df_pdsi_expand, groupby_cols=["site_id", "wyear"], limit=5)
    df_pdsi_features, pdsi_lag_features = get_lag_features(
        df_pdsi_expand[["site_id", "date", "day", "yday", "month", "year", "wyear"] + cfg_base["pdsi"]["features"]],
        groupby_cols=["site_id", "wyear"],
        features=cfg_base["pdsi"]["features"],
        lag_list=cfg_base["pdsi"]["lag_list"],
    )
    df_pdsi_features = df_pdsi_features.query("month in [1,2,3,4,5,6,7]")
    df_pdsi_features[pdsi_lag_features] = df_pdsi_features[pdsi_lag_features].fillna(-999999)
    _nrows_before = len(df_pdsi_features)
    df_pdsi_features = df_pdsi_features.dropna(subset=pdsi_lag_features)
    _nrows_after = len(df_pdsi_features)
    logger.info(f"df_pdsi_features: {_nrows_after-_nrows_before} rows removed")
    smr_pdsi_qa = check_max_dates(
        df_pdsi_features, col=pdsi_lag_features[0], id_col="site_id", issue_date=get_lag_date(issue_date, 5)
    )
    df_pdsi_features = df_pdsi_features.drop(columns=["pdsi", "date", "wyear"])

    logger.info("Prepare SNOTEL SWE data...")
    df_snotel_sites_chosen = get_snotel_sites_chosen(
        src_dir=src_dir, RUN_DIR=RUN_DIR, EXP_NAME=BASE_EXP_NAME, years=[2020, 2021, 2022], k=cfg_base["swe"]["top_k"]
    )
    df_swe_stats = get_swe_stats(RUN_DIR=RUN_DIR, EXP_NAME=BASE_EXP_NAME, years=[2020, 2021, 2022])
    df_swe = read_snotel_swe(SNOTEL_SWE_DIR, df_snotel_sites_chosen.snotel_id.tolist())
    df_swe = parse_time_features(df_swe)
    df_swe = filter_issue_date(df_swe, issue_date=issue_date)
    check_missing_dates(df_swe, cols=["snotel_id", "date", "swe", "day_diff"], groupby_cols=["snotel_id", "wyear"])
    df_swe = expand_date(df_swe, groupby_cols=["snotel_id", "wyear"])
    df_swe = parse_time_features(df_swe)
    df_swe = fill_missing_value(
        df_swe, cols=["swe", "prec_cml"], groupby_cols=["snotel_id", "wyear"], limit=7, rounding=1
    )

    if cfg_sim["gap_swe"] > 0:
        np.random.seed(cfg_base["seed"])
        diff_gap = np.random.randint(cfg_sim["gap_swe"], size=df_swe.snotel_id.nunique())
        df_swe_filter = (
            df_swe.groupby(["snotel_id"], as_index=False)
            .agg(max_date=("date", "max"))
            .assign(
                diff_gap=diff_gap + 1,
                max_date_after=lambda x: pd.to_datetime(x["max_date"]) - pd.to_timedelta(x["diff_gap"], unit="d"),
            )
        )
        df_swe = pd.merge(df_swe, df_swe_filter).query("date <= max_date_after")
    smr_swe_qa = check_max_dates(df_swe, col="swe", id_col="snotel_id", issue_date=issue_date)
    log_data(
        shorten_id(df_snotel_sites_chosen)[
            df_snotel_sites_chosen["snotel_id"].isin(smr_swe_qa.query("gap > 1").index)
        ][["snotel_id", "site_id"]]
        .drop_duplicates()
        .groupby(["snotel_id"])["site_id"]
        .apply(lambda x: ",".join(x))
        .reset_index(),
        "df_snotel_sites_chosen",
        "SNOTEL x stations with huge gap",
    )

    if df_snotel_sites_chosen.snotel_id.nunique() != df_swe.snotel_id.nunique():
        logger.warnings("Missing SNOTEL sites from directory!")
    log_data_summary(df_swe, "df_swe", "SNOTEL SWE", cols=["swe", "prec_cml"])

    df_swe = expand_date(df_swe, groupby_cols=["snotel_id", "wyear"], max_ymd=issue_date)
    df_swe = parse_time_features(df_swe)
    # df_swe[["swe", "prec_cml"]] = df_swe[["swe", "prec_cml"]].fillna(-999999)

    logger.info("Generate prediction...")
    df_pred_test_all = []
    for model in model_list:
        EXP_NAME = model
        cfg_mdl = json.load(open(f"{RUN_DIR}/{EXP_NAME}/config.json", "r"))

        df_swe_filtered, swe_lag_features = get_lag_features(
            df_swe[["snotel_id", "date", "day", "yday", "month", "year", "wyear"] + cfg_mdl["swe"]["features"]],
            groupby_cols=["snotel_id", "wyear"],
            features=cfg_mdl["swe"]["features"],
            lag_list=cfg_mdl["swe"]["lag_list"],
        )
        df_swe_filtered = df_swe_filtered.drop(columns=["wyear"])
        df_swe_filtered = df_swe_filtered[df_swe_filtered["month"].isin(range(8))]

        if cfg_mdl["target_prep"]["diff"]:
            cfg_mdl["remove_features"] = cfg_mdl["remove_features"] + [
                "volume_actual",
                "diff",
                "month_tf",
            ]
        logger.info(f"model_id='{EXP_NAME}', K={cfg_mdl['swe']['top_k']}, swe_features={swe_lag_features}")

        for val_year in [2020, 2021, 2022]:
            df_swe_features = get_swe_features_mm(
                df_swe_filtered,
                df_snotel_sites_chosen.groupby(["site_id", "val_year"]).head(cfg_mdl["swe"]["top_k"]),
                df_swe_stats,
                swe_lag_features,
                val_year,
                comp_thr=0.65,
            )
            # df_swe_features[df_swe_features[swe_lag_features] < 0] = np.NaN
            _nrows_before = len(df_swe_features)
            df_swe_features = df_swe_features.dropna(subset=swe_lag_features)
            _nrows_after = len(df_swe_features)
            logger.info(f"df_swe_features: {_nrows_after-_nrows_before} rows removed")

            df_test_all = df_test_base.copy()
            df_test_all = df_test_all[df_test_all["diff"] >= 0]
            df_test_all = pd.merge(df_test_all, df_swe_features, how="inner")
            df_test_all = df_test_all.drop(columns=["date"])

            features = get_features(df_test_all, cfg_mdl["remove_features"])
            logger.info(f"features: {features}")

            if cfg_mdl.get("pdsi"):
                df_test_all = pd.merge(df_test_all, df_pdsi_features, how="inner")
                features = features + pdsi_lag_features

            df_test_all = get_meta_features(df_test_all, df_meta)
            log_data_summary(
                df_test_all,
                "df_test_all - features",
                f"Features from model_id='{EXP_NAME}', fold={val_year}",
                cols=[x for x in features if "lag" in x],
            )

            for seed in cfg_base["model_seeds"]:
                EXP_NAME_SEED = EXP_NAME if seed == 2024 else f"{EXP_NAME}_s{seed}"
                df_pred_test_reg = predict_lgb(
                    df_test_all,
                    cfg=cfg_mdl,
                    model_output=f"{RUN_DIR}/{EXP_NAME_SEED}/models/{EXP_NAME}_reg_y{val_year}",
                    quantiles=[50],
                )
                df_pred_test = predict_lgb(
                    df_test_all,
                    cfg=cfg_mdl,
                    model_output=f"{RUN_DIR}/{EXP_NAME_SEED}/models/{EXP_NAME}_y{val_year}",
                    quantiles=[10, 90],
                )
                df_pred_test["pred_volume_50"] = df_pred_test_reg["pred_volume_50"]
                df_pred_test["seed"] = seed
                df_pred_test["val_year"] = val_year
                df_pred_test["model_id"] = model
                if cfg_mdl["target_prep"]["diff"]:
                    df_pred_test = add_diff(df_pred_test)
                df_pred_test_all.append(df_pred_test)

    df_pred_test_all = pd.concat(df_pred_test_all)
    write_feather(
        df_pred_test_all.reset_index(drop=True),
        preprocessed_dir / "pred_all.feather",
        is_cache=cfg_base["is_cache"],
        message="Caching all raw predictions",
    )
    df_pred_test_all = get_date_from_df(df_pred_test_all)

    logger.info("Generate prediction ensemble...")
    df_pred_test = mean_ensemble(
        df_pred_test_all, groupby_cols=["site_id", "year", "month", "day", "model_id"], is_count=True
    )
    df_pred_test = clip_prediction(rearrange_prediction(df_pred_test))
    df_pred_test_ens = custom_ensemble(
        df_pred_test,
        groupby_cols=["site_id", "year", "month", "day"],
        is_count=True,
    )

    logger.info("Generate final prediction...")
    df_median = pd.read_csv(src_dir / "data/meta/1991_2020_train_stats.csv")
    df_pred_test_best = df_pred_test_ens.groupby(["site_id"]).last()
    df_pred_test_best = get_date_from_df(df_pred_test_best)
    df_pred_test_best = df_pred_test_best[["date", "size"] + cfg_base["pred_cols"]]
    df_pred_test_best = df_pred_test_best.rename(
        columns={
            "size": "n_ens",
            "pred_volume_10": "volume_10",
            "pred_volume_50": "volume_50",
            "pred_volume_90": "volume_90",
        }
    )
    df_pred_test_best = pd.merge(
        df_pred_test_best.reset_index(),
        df_median[["site_id", "pred_volume_50"]].rename(columns={"pred_volume_50": "median"}),
    )
    df_pred_test_best["pct_median"] = df_pred_test_best["volume_50"] / df_pred_test_best["median"]
    df_pred_test_best = df_pred_test_best.sort_values("median", ascending=False)
    df_pred_test_best.to_csv(preprocessed_dir / "pred.csv", index=False)

    smr_pred = df_pred_test_best.copy()
    smr_pred["site_id"] = smr_pred["site_id"].str.split("_").str[0]
    smr_pred["date"] = smr_pred["date"].str[5:]
    smr_pred[["volume_10", "volume_50", "volume_90", "median"]] = smr_pred[
        ["volume_10", "volume_50", "volume_90", "median"]
    ].astype(int)
    smr_pred["pct_median"] = (smr_pred["pct_median"] * 100).astype(int)
    smr_pred = smr_pred.rename(
        columns={
            "date": "mm-dd",
            "n_ens": "ns",
            "volume_10": "vol10",
            "volume_50": "vol50",
            "volume_90": "vol90",
            "median": "med",
            "pct_median": "%med",
        }
    )
    logger.info(f"Prediction summary\n{smr_pred.set_index('site_id').to_string()}")
    logger.info(
        f'Prediction summary (overall): \n{smr_pred.drop(columns=["site_id","mm-dd","ns"]).mean().astype(int)}'
    )


def predict(site_id, issue_date, assets, src_dir, data_dir, preprocessed_dir):
    if not (preprocessed_dir / "pred.csv").exists():
        logger.info(f"Excecute submission for {issue_date}")
        main(issue_date=issue_date, src_dir=src_dir, data_dir=data_dir, preprocessed_dir=preprocessed_dir)
    df_pred = pd.read_csv(preprocessed_dir / "pred.csv")
    preds = df_pred.set_index("site_id").T.to_dict()
    p10 = preds[site_id]["volume_10"]
    p50 = preds[site_id]["volume_50"]
    p90 = preds[site_id]["volume_90"]
    pmedian = preds[site_id]["median"]
    pct_median = p50 / pmedian
    logger.info(
        f"site_id: {site_id} | p50: {p50:.0f} | p90: {p90:.0f} | %median: {pct_median:.0%} | median: {pmedian:.0f}"
    )

    return p10, p50, p90
