import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR
import geopandas as gpd
from src.features.base import generate_base_year, convert_cfs_to_kaf
from loguru import logger

_usbr_cols_rename = {
    "Location": "site_id",
    "Parameter": "parameter",
    "Result": "value",
    "Datetime (UTC)": "date",
}
_usbr_site_mapping = {
    "Boysen Reservoir Dam and Powerplant": "boysen_reservoir_inflow",
    "Folsom Lake Dam and Powerplant": "american_river_folsom_lake",
    "Taylor Park Reservoir and Dam": "taylor_park_reservoir_inflow",
    "Fontenelle Reservoir Dam and Powerplant": "fontenelle_reservoir_inflow",
    "Ruedi Reservoir and Dam ": "ruedi_reservoir_inflow",
    "Pueblo Reservoir and Dam": "pueblo_reservoir_inflow",
}

_snotel_cols_rename = {
    "date": "date",
    "WTEQ_DAILY": "swe",
    "SNWD_DAILY": "sdepth",
    "PREC_DAILY": "prec_cml",
    "TMAX_DAILY": "tmax",
    "TMIN_DAILY": "tmin",
    "TAVG_DAILY": "tavg",
    "snotel_id": "snotel_id",
}

_cdec_cols_rename = {
    "stationId": "cdec_id",
    "date": "date",
    "sensorType": "parameter",
    "value": "value",
    # "dataFlag": "value_qa",
}

_cdec_params = {
    "SNO ADJ": "swe",
    # "SNOW DP": "sdepth",
    # "SNOW WC": "swe_raw",
    "RAIN": "prec_cml",
    # "TEMP MX": "tmax",
    # "TEMP MN": "tmin",
    # "TEMP AV": "tavg",
}

_usgs_cols_rename = {
    "site_no": "usgs_id",
    "datetime": "date",
    "00060_Mean": "discharge",
    "00060_Mean_cd": "discharge_qa",
}


def read_train(dirname=RAW_DATA_DIR, meta=None, test_years=None, is_forecast=False):
    if is_forecast:
        df = pd.concat([
            pd.read_csv(f"{dirname}/prior_historical_labels.csv"),
            pd.read_csv(f"{dirname}/cross_validation_labels.csv"),
        ])
        df = df[~df["volume"].isna()].reset_index(drop=True)
    else:
        df = pd.read_csv(f"{dirname}/train.csv")
        df = df[~df["volume"].isna()].reset_index(drop=True)
        if (test_years is not None) & (meta is not None):
            df = pd.concat([df, generate_base_year(meta, years=test_years)]).reset_index(drop=True)

    return df


def read_monthly_naturalized_flow(cats=["train", "test"], dirname=RAW_DATA_DIR, is_forecast=False, is_dropna=True):
    if is_forecast:
        df = pd.concat([
            pd.read_csv(f"{dirname}/prior_historical_monthly_flow.csv"),
            pd.read_csv(f"{dirname}/cross_validation_monthly_flow.csv")
        ])
    else:
        df = []
        for cat in cats:
            df.append(pd.read_csv(f"{dirname}/{cat}_monthly_naturalized_flow.csv"))
        df = pd.concat(df)
    if is_dropna:
        df = df[~df["volume"].isna()].reset_index(drop=True)

    return df


def read_discharge(path):
    df = pd.read_csv(path, sep="\t", comment="#")
    df = df.query('agency_cd=="USGS"').reset_index(drop=True)
    df.columns = ["source", "usgs_id", "date", "discharge", "discharge_qa"]
    df = df.drop(columns=["source"])
    df["discharge"] = pd.to_numeric(df["discharge"])

    return df


def read_usgs_all(dirname):
    usgs_files = glob.glob(f"{dirname}/FY**/*.csv", recursive=True)
    df = []
    for file in usgs_files:
        df.append(pd.read_csv(file, dtype={"site_no": "object"}))
    df = pd.concat(df)
    logger.info("df_usgs columns: {}".format(list(df)))

    df = df.rename(columns=_usgs_cols_rename)
    df = df[_usgs_cols_rename.values()]
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def read_usbr(path, skiprows=7, is_print=False, unit="cfs"):
    df_raw = pd.read_csv(path, skiprows=skiprows)
    df = df_raw.copy()
    if is_print:
        print(df.Units.value_counts().index.tolist())
    df = df[df["Units"] == unit]
    df = df.rename(columns=_usbr_cols_rename)
    df = df[_usbr_cols_rename.values()]
    df["value"] = df["value"].astype(float)
    df["site_id"] = df["site_id"].replace(_usbr_site_mapping)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if unit == "cfs":
        df = convert_cfs_to_kaf(df, col="value")
    if unit == "af":
        df["value"] = df["value"] / 1000

    return df, df_raw


def read_usbr_all(dirname="data/external/usbr", parameter="Lake/Reservoir Inflow", unit="cfs"):
    path_list = glob.glob(f"{dirname}/**/*.csv", recursive=True)
    df = []
    for path in path_list:
        df.append(read_usbr(path, unit=unit)[0])
    df = pd.concat(df).reset_index(drop=True)
    df = df[df["parameter"] == parameter]
    df = df.drop(columns=["parameter"])
    df = df.reset_index(drop=True)

    return df


def read_meta(path=f"{RAW_DATA_DIR}/metadata.csv"):
    df_meta = pd.read_csv(path, dtype={"usgs_id": "object"})
    df_meta["usgs_id"] = df_meta["usgs_id"].str.zfill(8)

    return df_meta


def read_sub(path=f"{RAW_DATA_DIR}/submission_format_d66NoWb.csv"):
    df = pd.read_csv(path)

    return df


def read_meta_geo(path=f"{RAW_DATA_DIR}/geospatial.gpkg"):
    df_meta_poly = gpd.read_file(path, layer="basins")
    df_meta_point = gpd.read_file(path, layer="sites")

    return df_meta_poly, df_meta_point


def read_snotel_swe(dirname, site_list=None):
    swe_files = Path(dirname).rglob("*.csv")
    df_swe = []
    if site_list is None:
        site_list = [os.path.basename(x).replace(".csv", "").split("_")[0] for x in swe_files]
    for file in swe_files:
        snotel_id = os.path.basename(file).replace(".csv", "").split("_")[0]
        if snotel_id in site_list:
            _df = pd.read_csv(file)
            _df["snotel_id"] = snotel_id
            df_swe.append(_df)
    df_swe = pd.concat(df_swe)
    logger.info("df_swe columns: {}".format(list(df_swe)))

    df_swe = df_swe.rename(columns=_snotel_cols_rename)
    df_swe = df_swe[_snotel_cols_rename.values()]

    return df_swe


def read_cdec_swe(dirname, site_list=None, is_preprocess=False):
    swe_files = glob.glob(f"{dirname}/FY**/*.csv", recursive=True)
    df_swe = []
    if site_list is None:
        site_list = [os.path.basename(x).replace(".csv", "").split("_")[0] for x in swe_files]
    for file in swe_files:
        cdec_id = os.path.basename(file).replace(".csv", "").split("_")[0]
        if cdec_id in site_list:
            _df = pd.read_csv(file)
            df_swe.append(_df)
    df_swe = pd.concat(df_swe)
    logger.info("df_swe columns: {}".format(list(df_swe)))

    df_swe = df_swe.rename(columns=_cdec_cols_rename)
    df_swe = df_swe[_cdec_cols_rename.values()]

    if is_preprocess:
        df_swe["parameter"] = df_swe["parameter"].map(_cdec_params)
        df_swe = df_swe[df_swe["parameter"].isin(_cdec_params.values())]
        df_swe["value"] = df_swe["value"].mask(df_swe["value"] < -99, np.NaN)
        df_swe["value"] = df_swe["value"].clip(lower=0)
        df_swe = df_swe.pivot(index=["cdec_id", "date"], columns="parameter", values="value").reset_index()
        df_swe = df_swe.rename_axis(None, axis=1)

    return df_swe
