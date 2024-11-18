import os
import json
import pandas as pd
from src.utils import *
from src.features.base import *
from src.features.swe import *
from src.features.volume_obs import *
from src.features.pdsi import *
from src.features.uaswe import *
from src.features.era5 import *
from src.features.seas51 import *
from src.data.base import *
from src.config import *
from src.models.postprocess import *
from src.models.lgb import *
from src.qa import *

import warnings
import argparse
import importlib
import datetime

from loguru import logger

warnings.filterwarnings("ignore")


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


parser = argparse.ArgumentParser(description="Training script for Final stage", argument_default=argparse.SUPPRESS)

parser.add_argument("-c", "--config")
parser.add_argument("-s", "--seed", type=int, default=2024)
parser.add_argument("-y", "--year_min", type=int)
parser.add_argument("-v", "--val_years", nargs="+", type=int)
parser.add_argument("-t", "--test_years", nargs="+", type=int)
parser.add_argument("-f", "--is_forecast", type=boolean_string, default=True)
parser.add_argument("-e", "--exclude_years", nargs="+", type=int, default=[])
parser.add_argument("-d", "--dirname_suffix", type=str, default="")
parser.add_argument("-es", "--early_stopping", type=int, default=None)
parser.add_argument("-ch", "--is_cache", type=boolean_string, default=False)
parser.add_argument("-td", "--is_target_diff", type=boolean_string, default=None)
parser.add_argument("-dm", "--dirname_main", type=str, default="final")
parser.add_argument("-sy", "--n_synthetic", type=int, default=None)

args = parser.parse_args()

EXP_NAME = args.config
cfg = importlib.import_module(f"configs.{args.dirname_main}.{EXP_NAME}").cfg

if args.is_forecast:
    output_dir = f"runs/{args.dirname_main}{args.dirname_suffix}/{EXP_NAME}"
    if len(np.ravel(cfg["val_years"]).tolist()) > 1:
        cfg["val_years"] = list(range(2004, 2024))
    cfg["test_years"] = []
else:
    output_dir = f"runs/{args.dirname_main}{args.dirname_suffix}/{EXP_NAME}"

cfg.update(vars(args))
year_min = cfg.get("year_min", 1980)
logger.info(f"Minimum training year: {year_min}")

if args.seed != 2024:
    output_dir = output_dir + f"_s{args.seed}"
if hasattr(args, "year_min"):
    output_dir = output_dir + f"_y{args.year_min}"
if args.early_stopping:
    cfg["lgb_params"]["early_stopping_rounds"] = args.es
    cfg["lgb_reg_params"]["early_stopping_rounds"] = args.es
    logger.info("Enable early stopping")
if args.is_target_diff is not None:
    cfg["target_prep"]["diff"] = args.is_target_diff
    cfg["target_prep"]["diff_extra"] = args.is_target_diff
    cfg["remove_features"] = [x for x in cfg["remove_features"] if x not in ["is_use_diff", "is_use_diff_extra"]]
    logger.info(f"is_target_diff={args.is_target_diff}")
if args.n_synthetic is not None:
    cfg["synthetic"]["n"] = args.n_synthetic
model_dir = f"{output_dir}/models"
pred_dir = f"{output_dir}/preds"
eval_dir = f"{output_dir}/evals"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

json.dump(cfg, open(f"{output_dir}/config.json", "w"), indent=4)
logger.add(f"{output_dir}/log.txt")

logger.info(f"Validation years: {cfg['val_years']}, test years: {cfg['test_years']}")
logger.info("Prepare train and metadata")
df_meta = read_meta(f"data/meta/metadata_proc.csv")
df_train = read_train(meta=df_meta, test_years=cfg["test_years"], is_forecast=cfg["is_forecast"])
df_monthly = read_monthly_naturalized_flow(is_forecast=cfg["is_forecast"])

if cfg.get("swe"):
    logger.info("Prepare SNOTEL SWE data")
    df_snotel_sites_chosen = pd.read_feather("data/meta/snotel_sites_basins/snotel_sites_basins_chosen.feather")
    if cfg["filter_snotel_sites"]:
        from src.meta.get_available_sites import filter_available_snotel_sites

        logger.info("Filter available SNOTEL sites based on runtime")
        df_snotel_sites_chosen = filter_available_snotel_sites(df_snotel_sites_chosen)

    df_swe = read_snotel_swe(SNOTEL_SWE_DIR, df_snotel_sites_chosen.snotel_id.tolist())
    df_swe = parse_time_features(df_swe, cols=["date"], level=["day", "yday", "month", "year", "wyear", "wyday"])
    if cfg["swe"].get("ffill_limit"):
        logger.info("Use SWE forward fill limit")
        df_swe = forward_fill(df_swe, groupby_cols=["snotel_id", "wyear"], limit=cfg["swe"]["ffill_limit"])
    else:
        df_swe = forward_fill(df_swe, groupby_cols=["snotel_id", "wyear"])

    if cfg["swe"].get("use_cdec"):
        logger.info("Prepare CDEC SWE data")
        df_cdec_sites_chosen = pd.read_feather("data/meta/cdec_sites_chosen.feather")
        df_swe_cdec = read_cdec_swe(CDEC_SWE_DIR, is_preprocess=True)
        df_swe_cdec = parse_time_features(
            df_swe_cdec, cols=["date"], level=["day", "yday", "month", "year", "wyear", "wyday"]
        )
        if cfg["swe"].get("ffill_limit"):
            df_swe_cdec = forward_fill(df_swe_cdec, groupby_cols=["cdec_id", "wyear"], limit=cfg["swe"]["ffill_limit"])
        else:
            df_swe_cdec = forward_fill(df_swe_cdec, groupby_cols=["cdec_id", "wyear"], limit=7)
        df_swe = pd.concat([df_swe, df_swe_cdec.rename(columns={"cdec_id": "snotel_id"})])
        df_snotel_sites_chosen = pd.concat(
            [df_snotel_sites_chosen, df_cdec_sites_chosen.rename(columns={"station_id": "snotel_id"})]
        )

    if cfg["swe"].get("lag_list"):
        df_swe_filtered, swe_lag_features = get_lag_features(
            df_swe[["snotel_id", "date", "day", "yday", "month", "year", "wyear"] + cfg["swe"]["features"]].query(
                "swe==swe"
            ),
            groupby_cols=["snotel_id", "wyear"],
            features=cfg["swe"]["features"],
            lag_list=cfg["swe"]["lag_list"],
        )
    else:
        df_swe_filtered, swe_lag_features = get_lag_features(
            df_swe[["snotel_id", "date", "day", "yday", "month", "year", "wyear"] + cfg["swe"]["features"]].query(
                "swe==swe"
            ),
            groupby_cols=["snotel_id", "wyear"],
            features=cfg["swe"]["features"],
            start_lag=cfg["swe"]["start_lag"],
            lags=cfg["swe"]["lags"],
            interval=cfg["swe"]["interval"],
        )
    df_swe_filtered = df_swe_filtered.drop(columns=["wyear"])

if cfg.get("uaswe"):
    logger.info("Prepare UA-SWANN SWE data")
    if not os.path.isfile(f"{UASWANN_DIR}/uaswe_huc6.feather"):
        df_usgs_sites = pd.read_feather("data/meta/nwis_usgs_meta.feather")
        df_usgs_sites["huc"] = df_usgs_sites["huc_cd"].astype(str).str.zfill(8)
        df_usgs_sites["huc8"] = df_usgs_sites["huc"].str[0:8]
        df_usgs_sites["huc6"] = df_usgs_sites["huc"].str[0:6]
        df_uaswe_huc6 = preprocess_uaswe_huc(
            UASWANN_DIR, df_meta, df_usgs_sites, huc_level=6, out_file=f"{UASWANN_DIR}/uaswe_huc6.feather"
        )
        df_uaswe_huc8 = preprocess_uaswe_huc(
            UASWANN_DIR, df_meta, df_usgs_sites, huc_level=8, out_file=f"{UASWANN_DIR}/uaswe_huc8.feather"
        )
    else:
        df_uaswe_huc6 = pd.read_feather(f"{UASWANN_DIR}/uaswe_huc6.feather")
        df_uaswe_huc8 = pd.read_feather(f"{UASWANN_DIR}/uaswe_huc8.feather")
    df_uaswe_huc6 = df_uaswe_huc6.rename(columns={"swe": "uaswe_huc6", "prec_cml": "uaprec_cml_huc6"})
    df_uaswe_huc8 = df_uaswe_huc8.rename(columns={"swe": "uaswe_huc8", "prec_cml": "uaprec_cml_huc8"})

    if not os.path.isfile(f"{UASWANN_DIR}/uaswe_basin.csv"):
        df_uaswe_basin = preprocess_uaswe_basin_all(
            UASWANN_DIR, read_meta_geo()[0], out_file=f"{UASWANN_DIR}/uaswe_basin.csv"
        )
    else:
        df_uaswe_basin = pd.read_csv(f"{UASWANN_DIR}/uaswe_basin.csv")
    df_uaswe_basin = df_uaswe_basin.groupby(["site_id", "time"], as_index=False).last()
    df_uaswe_basin = df_uaswe_basin.rename(columns={"time": "date", "SWE": "uaswe", "DEPTH": "uasdepth"})
    df_uaswe_basin = parse_time_features(df_uaswe_basin, level=[])

    df_uaswe = pd.merge(df_uaswe_huc6, df_uaswe_huc8)
    df_uaswe = pd.merge(df_uaswe, df_uaswe_basin)
    del df_uaswe_huc6, df_uaswe_huc8, df_uaswe_basin

    df_uaswe = parse_time_features(df_uaswe)
    df_uaswe = expand_date(df_uaswe, groupby_cols=["site_id", "wyear"], max_mm_dd="07-22")
    df_uaswe = parse_time_features(df_uaswe)
    df_uaswe_features, uaswe_lag_features = get_lag_features(
        df_uaswe[["site_id", "date", "day", "yday", "month", "year", "wyear"] + cfg["uaswe"]["features"]],
        groupby_cols=["site_id", "wyear"],
        features=cfg["uaswe"]["features"],
        lag_list=cfg["uaswe"]["lag_list"],
    )
    df_uaswe_features = df_uaswe_features.drop(columns=cfg["uaswe"]["features"] + ["date", "wyear"])

if cfg.get("pdsi"):
    logger.info("Prepare PDSI")
    if not os.path.isfile(f"{PDSI_DIR}/pdsi_v2.csv"):
        df_pdsi = preprocess_pdsi(PDSI_DIR, read_meta_geo()[0], out_file=f"{PDSI_DIR}/pdsi_v2.csv")
    df_pdsi = pd.read_csv(f"{PDSI_DIR}/pdsi_v2.csv")
    df_pdsi = df_pdsi.drop_duplicates()
    df_pdsi = df_pdsi.rename(columns={"day": "date", "daily_mean_palmer_drought_severity_index": "pdsi"})
    df_pdsi = parse_time_features(df_pdsi, cols=["date"])
    df_pdsi = df_pdsi.sort_values(["site_id", "date"]).reset_index(drop=True)

    df_pdsi_expand = expand_date(df_pdsi, groupby_cols=["site_id", "wyear"], max_mm_dd="07-22")
    df_pdsi_expand = parse_time_features(df_pdsi_expand, cols=["date"])
    df_pdsi_expand = df_pdsi_expand.assign(pdsi=lambda x: x.groupby(["site_id", "wyear"])["pdsi"].transform("ffill"))
    df_pdsi_features, pdsi_lag_features = get_lag_features(
        df_pdsi_expand[["site_id", "date", "day", "yday", "month", "year", "wyear"] + cfg["pdsi"]["features"]],
        groupby_cols=["site_id", "wyear"],
        features=cfg["pdsi"]["features"],
        lag_list=cfg["pdsi"]["lag_list"],
    )
    df_pdsi_features = df_pdsi_features.query("month in [1,2,3,4,5,6,7]")
    df_pdsi_features = df_pdsi_features.drop(columns=["pdsi", "date", "wyear"])

if cfg.get("cds"):
    if cfg["cds"].get("s51"):
        if not os.path.isfile(f"{SEAS51_DIR}/seas51_forecast_m1_1981_2023_conus.feather"):
            df_seas51 = preprocess_seas51(
                SEAS51_DIR, read_meta_geo()[0], out_file=f"{SEAS51_DIR}/seas51_forecast_m1_1981_2023_conus.feather"
            )
        else:
            df_seas51 = pd.read_feather(f"{SEAS51_DIR}/seas51_forecast_m1_1981_2023_conus.feather")
        df_seas51["t2m"] = df_seas51["t2m"] - 273.15
        df_seas51 = df_seas51.dropna()
        if cfg["cds"].get("use_same_ens"):
            df_seas51 = df_seas51.query("number < 25")
        df_seas51_agg, seas51_agg_features = get_agg_features(
            df_seas51,
            groupby_cols=["site_id", "date"],
            features=cfg["cds"]["s51"]["features"],
            agg_cols=cfg["cds"]["s51"]["agg_cols"],
        )
        df_seas51_agg = parse_time_features(df_seas51_agg, level=["wyear", "month"])
        df_seas51_agg_expand = get_month_features(df_seas51_agg, ["site_id", "wyear"], seas51_agg_features)
        df_seas51_agg_expand = (
            df_seas51_agg_expand.rename(columns={"wyear": "year"}).drop(columns=["date"]).query("month != 12")
        )
        df_seas51_agg_expand = df_seas51_agg_expand.dropna()

    if cfg["cds"].get("era5"):
        if not os.path.isfile(f"{ERA5_DIR}/era5_1980_2023_conus.feather"):
            df_era5 = preprocess_era5_all(
                ERA5_DIR, read_meta_geo()[0], out_file=f"{ERA5_DIR}/era5_1980_2023_conus.feather"
            )
        else:
            df_era5 = pd.read_feather(f"{ERA5_DIR}/era5_1980_2023_conus.feather")
        df_era5 = df_era5.assign(
            time=lambda x: pd.to_datetime(x["time"]),
            date=lambda x: x["time"].dt.date,
            hour=lambda x: x["time"].dt.hour,
        )
        df_era5["t2m"] = df_era5["t2m"] - 273.15
        df_era5 = df_era5.assign(date=lambda x: x["date"] + datetime.timedelta(days=1))
        df_era5 = parse_time_features(df_era5, level=["wyear", "month", "day"])
        df_era5 = df_era5[df_era5["hour"] == cfg["cds"]["era5"]["hour"]]
        era5_features = cfg["cds"]["era5"]["features"]
        df_era5 = df_era5.rename(columns={"wyear": "year"})
        df_era5 = df_era5[["site_id", "year", "month", "day"] + era5_features]

df_train_base = generate_base_train(df_train, use_new_format=True)
if cfg["target_prep"]["diff"]:
    logger.info("Prepare target diff")
    df_target_diff = generate_target_diff(df_monthly, df_meta)
    df_train_base = get_volume_target_diff(df_train_base, df_target_diff, cfg["diff_gap"])
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
    df_train_base = get_year_cat(df_train_base, val_years=year, test_years=cfg["test_years"])
    df_train_all = df_train_base.copy()

    if cfg["target_prep"].get("scale"):
        logger.info("Use target scaling")
        df_train_stats, train_stats_cols = get_agg_features(
            df_train.query('cat=="train"'),
            groupby_cols=["site_id"],
            features=["volume"],
            agg_cols=["max", "mean", "std"],
        )
        os.makedirs(f"{output_dir}/train_stats", exist_ok=True)
        df_train_stats.assign(val_year=year).to_csv(f"{output_dir}/train_stats/train_stats_{year}.csv", index=False)
        if cfg["target_prep"]["scale"] == "max":
            df_train_all = pd.merge(df_train_all, df_train_stats).assign(
                volume=lambda x: x["volume"] / x["volume_max"]
            )
        if len(set(train_stats_cols) & set(cfg["remove_features"])) == 0:
            cfg["remove_features"] = cfg["remove_features"] + train_stats_cols

    features = []
    if cfg.get("swe"):
        if cfg["swe"].get("scale"):
            logger.info("Use SWE scaling")
            df_swe_stats, swe_stats_cols = get_agg_features(
                get_year_cat(
                    df_swe.drop(columns=["year"]).rename(columns={"wyear": "year"}),
                    val_years=year,
                    test_years=cfg["test_years"],
                ).query("cat=='train'"),
                groupby_cols=["snotel_id"],
                features=cfg["swe"]["features"],
                agg_cols=["max", "mean", "std"],
            )
            os.makedirs(f"{output_dir}/swe_stats", exist_ok=True)
            df_swe_stats.assign(val_year=year).to_csv(f"{output_dir}/swe_stats/swe_stats_{year}.csv", index=False)
            df_swe_filtered_mm = pd.merge(df_swe_filtered, df_swe_stats)
            for feature in cfg["swe"]["features"]:
                for lag_feature in swe_lag_features:
                    if lag_feature.startswith(feature):
                        if cfg["swe"]["scale"] == "minmax":
                            df_swe_filtered_mm[lag_feature] = (
                                df_swe_filtered_mm[lag_feature] / df_swe_filtered_mm[feature + "_max"]
                            )
                        elif cfg["swe"]["scale"] == "norm":
                            df_swe_filtered_mm[lag_feature] = (
                                df_swe_filtered_mm[lag_feature] - df_swe_filtered_mm[feature + "_mean"]
                            ) / df_swe_filtered_mm[feature + "_std"]
            df_swe_filtered_mm = df_swe_filtered_mm.drop(columns=swe_stats_cols)
        else:
            df_swe_filtered_mm = df_swe_filtered.copy()
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
            _snotel_id_selected = _df_basin_sites_r2.head(cfg["swe"]["top_k"]).index.tolist()
            df_basin_sites_r2.append(_df_basin_sites_r2.assign(site_id=site_id))

            _res = (
                get_basin_swe(
                    df_train[["site_id", "year"]],
                    df_swe_filtered_mm.query("swe==swe & month in [1,2,3,4,5,6,7]"),
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
        if cfg["swe"].get("extra_features"):
            swe_col = [x for x in swe_lag_features if "swe" in x][0]
            prec_col = [x for x in swe_lag_features if "prec_cml" in x][0]
            df_swe_features = df_swe_features.assign(
                swe_max=lambda x: x.groupby(["site_id", "year"])[swe_col].transform("cummax"),
                swe_prec_ratio=lambda x: x[swe_col] / x[prec_col],
                swe_diff=lambda x: x["swe_max"] - x[swe_col],
                swe_melt_ratio=lambda x: x["swe_diff"] / x["swe_max"],
            )
            drop_swe_features = [
                x
                for x in cfg["swe"]["extra_features"]
                if x not in ["swe_max", "swe_prec_ratio", "swe_diff", "swe_melt_ratio"]
            ]
            df_swe_features = df_swe_features.drop(columns=drop_swe_features)
            features = features + [x for x in cfg["swe"]["extra_features"] if "_ratio" not in x]
            logger.info('SWE extra features: {cfg["swe"]["extra_features"]}')
        df_basin_sites_r2 = pd.concat(df_basin_sites_r2)
        bss_dir = f"{output_dir}/snotel_sites_basins"
        os.makedirs(bss_dir, exist_ok=True)
        df_basin_sites_r2.to_csv(f"{bss_dir}/y{year}.csv")

        df_train_all = pd.merge(df_train_all, df_swe_features, how="left")
        df_train_all = df_train_all.query("year >= @year_min")
        df_train_all = df_train_all.query("yday == yday").reset_index(drop=True)
        logger.info(f"Train years: {df_train_all[df_train_all['cat']=='train'].year.unique().tolist()}")
        logger.info(f"Train size (SWE features): {df_train_all.shape}")

        features = features + swe_lag_features

    if cfg.get("uaswe"):
        df_train_all = pd.merge(df_train_all, df_uaswe_features, how="inner")
        features = features + uaswe_lag_features
        logger.info(f"Train size (UASWE features): {df_train_all.shape}")

    if cfg.get("pdsi"):
        df_train_all = pd.merge(df_train_all, df_pdsi_features, how="inner")
        features = features + pdsi_lag_features
        logger.info(f"Train size (PDSI features): {df_train_all.shape}")

    if cfg.get("cds"):
        if cfg["cds"].get("s51"):
            df_train_all = pd.merge(df_train_all, df_seas51_agg_expand, how="inner")
            features = features + seas51_agg_features
            logger.info(f"Train size (seasonal forecast features): {df_train_all.shape}")
        if cfg["cds"].get("era5"):
            df_train_all = pd.merge(df_train_all, df_era5, how="inner")
            features = features + era5_features
            logger.info(f"Train size (ERA5 features): {df_train_all.shape}")

    df_train_all = get_meta_features(df_train_all, df_meta)
    if len(cfg["exclude_years"]) > 0:
        logger.info(f"Exclude years: {cfg['exclude_years']}")
        logger.info(f"Train original size (before): {df_train_all.shape}")
        df_train_all = df_train_all[
            ((df_train_all["cat"] == "train") & (df_train_all["year"].isin(cfg["exclude_years"]))) == False
        ]
        logger.info(f"Train original size (after): {df_train_all.shape}")

    if cfg["is_cache"]:
        features_dir = f"{output_dir}/features"
        os.makedirs(features_dir, exist_ok=True)
        write_feather(
            df_train_all,
            f"{features_dir}/features_{year}.feather",
            is_cache=True,
            message=f"Cache features input for year={year}",
        )

    if cfg["synthetic"]["n"] > 0:
        logger.info(f"Features for synthetic generation {features}")
        if cfg["synthetic"].get("exclude_features"):
            features = [x for x in features if x not in cfg["synthetic"]["exclude_features"]]
            exclude_features = [x for x in cfg["synthetic"]["exclude_features"] if x in list(df_train_all)]
            if len(exclude_features) > 0:
                logger.info(f"Synthetic excluded features: {exclude_features}")
        neg_features = []
        if cfg["synthetic"].get("cols_neg"):
            for col_neg in cfg["synthetic"]["cols_neg"]:
                for feature in features:
                    if feature.startswith(col_neg):
                        neg_features.append(feature)
        _df_synthetic, scale_factor = generate_synthetic_data(
            df_train_all,
            cols=["volume"] + features,
            n_synthetic=cfg["synthetic"]["n"],
            scale_factor=cfg["synthetic"]["scale_factor"],
            seed=cfg["seed"],
            cols_neg=neg_features,
        )
        sf_dir = f"{output_dir}/scale_factor"
        os.makedirs(sf_dir, exist_ok=True)
        scale_factor.to_csv(f"{sf_dir}/y{year}.csv", index=False)
        df_train_all = pd.concat([df_train_all, _df_synthetic]).reset_index(drop=True)
        del _df_synthetic

        if cfg["is_cache"]:
            features_dir = f"{output_dir}/features_synth"
            os.makedirs(features_dir, exist_ok=True)
            write_feather(
                df_train_all,
                f"{features_dir}/features_{year}.feather",
                is_cache=True,
                message=f"Cache features input for year={year}",
            )

    logger.info("Train LGBM models with quantile loss")
    mdl, df_pred = train_lgb(
        df_train_all,
        params=cfg["lgb_params"],
        cfg=cfg,
        use_log=cfg["target_prep"]["log"],
        model_output=f"{output_dir}/models/{EXP_NAME}_y{year}",
        pred_output=f"{output_dir}/preds/{EXP_NAME}_y{year}",
        pred_iters=[300, 400, 500],
        # eval_output=f"{output_dir}/evals/{EXP_NAME}_y{year}",
        feature_output=f"{output_dir}/feature_names",
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
            pred_iters=[800, 1000, 1200],
            # eval_output=f"{output_dir}/evals/{EXP_NAME}_reg_y{year}",
            quantiles=[50],
        )
        df_pred["pred_volume_reg"] = df_pred_reg["pred_volume_50"]

    df_pred_all.append(df_pred)
    print("\n")

df_pred_all = pd.concat(df_pred_all)

if cfg["target_prep"].get("scale"):
    df_pred_all = rescale(df_pred_all, method=cfg["target_prep"]["scale"])

if cfg["target_prep"]["diff"]:
    df_pred_all = add_diff(df_pred_all)

df_pred_all.to_csv(f"{output_dir}/pred.csv", index=False)
