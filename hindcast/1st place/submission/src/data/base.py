import os
import glob
import re
import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR
import fiona
import geopandas as gpd
import pickle
from src.features.base import generate_base_year, convert_cfs_to_kaf
from loguru import logger

_cols_select = [
    "station_id",
    "station_name",
    "longitude",
    "latitude",
    "elevation",
    "huc6",
    "huc6_name",
    "huc12",
    "huc12_name",
    "start_date",
    "end_date",
]

_cols_rename = [
    "station_id",
    "station_name",
    "longitude",
    "latitude",
    "elevation",
    "start_date",
    "end_date",
]

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
    # "temp",
    "TMAX_DAILY": "tmax",
    "TMIN_DAILY": "tmin",
    "TAVG_DAILY": "tavg",
    "snotel_id": "snotel_id"
    # "prec",
    # "swe_prec",
    # "sdensity",
}

_usgs_cols_rename = {
    "site_no": "usgs_id",
    "datetime": "date",
    "00060_Mean": "discharge",
    "00060_Mean_cd": "discharge_qa",
}


def _rename_cols(
    df,
    network,
    cols=_cols_rename,
):
    _col_names = [network + "_" + x.replace("station_id", "id") for x in cols]
    df = df.rename(columns=dict(zip(cols, _col_names)))

    return df


def read_train(path=f"{RAW_DATA_DIR}/train.csv", meta=None, test_years=None):
    df = pd.read_csv(path)
    df = df[~df["volume"].isna()].reset_index(drop=True)
    if (test_years is not None) & (meta is not None):
        df = pd.concat([df, generate_base_year(meta, years=test_years)]).reset_index(drop=True)

    return df


def read_monthly_naturalized_flow(cats=["train", "test"], dirname=RAW_DATA_DIR):
    df = []
    for cat in cats:
        df.append(pd.read_csv(f"{dirname}/{cat}_monthly_naturalized_flow.csv"))
    df = pd.concat(df)
    df = df[~df["volume"].isna()].reset_index(drop=True)

    return df


def read_sites(network="usgs", path=None, read_mode="rdb", cols=_cols_select, is_rename=True):
    if path is None:
        path = f"data/meta/{network}_sites.txt"

    if read_mode == "rdb":
        df = pd.read_csv(path, sep=",", comment="#", dtype=object)
    elif read_mode == "csv":
        df = pd.read_csv(path, comment="#", dtype=object)

    _col_names = [re.sub(r" ", "_", x.lower()) for x in df.columns]
    _col_names = [re.sub(r"_\(.*\)", "", x) for x in _col_names]
    df.columns = _col_names
    _col_numeric = ["longitude", "latitude", "elevation"]
    df[_col_numeric] = df[_col_numeric].astype(float)
    df["is_active"] = (df["end_date"] == "2100-01-01").astype(int)
    df = df[cols]
    if is_rename:
        df = _rename_cols(df, network=network)

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
    for layername in fiona.listlayers(path):
        with fiona.open(path, layer=layername) as src:
            print(layername, len(src))

    df_meta_poly = gpd.read_file(path, layer="basins")
    df_meta_point = gpd.read_file(path, layer="sites")

    return df_meta_poly, df_meta_point


def read_swe(dirname, site_list=None):
    swe_files = glob.glob(f"{dirname}/*.txt")
    df_swe = []
    if site_list is None:
        site_list = [os.path.basename(x).replace(".txt", "") for x in swe_files]
    for file in swe_files:
        snotel_id = os.path.basename(file).replace(".txt", "")
        if snotel_id in site_list:
            _df = pd.read_csv(file, comment="#")
            _df.columns = [
                "date",
                "swe",
                "sdepth",
                "prec_cml",
                "temp",
                "tmax",
                "tmin",
                "tavg",
                "prec",
                "swe_prec",
                "sdensity",
            ]
            _df["snotel_id"] = snotel_id
            df_swe.append(_df)
    df_swe = pd.concat(df_swe)

    return df_swe


def read_snotel_swe(dirname, site_list=None):
    swe_files = glob.glob(f"{dirname}/FY**/*.csv", recursive=True)
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


def read_monthly_prism(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    df = list()
    for site_id in data.keys():
        print(site_id)
        _res = dict(date=list(), maxt=list(), mint=list(), avgt=list(), pcpn=list())

        for d in data[site_id]["data"]:
            _res["date"].append(d[0])
            _res["maxt"].append(np.array(d[1][0]))
            _res["mint"].append(np.array(d[2][0]))
            _res["avgt"].append(np.array(d[3][0]))
            _res["pcpn"].append(np.array(d[4][0]))
        _df = (
            pd.DataFrame(_res)
            .assign(
                maxt_avg=lambda x: x["maxt"].map(lambda x: np.mean(x)),
                mint_avg=lambda x: x["mint"].map(lambda x: np.mean(x)),
                avgt_avg=lambda x: x["avgt"].map(lambda x: np.mean(x)),
                pcpn_avg=lambda x: x["pcpn"].map(lambda x: np.mean(x)),
            )
            .drop(columns=["maxt", "mint", "avgt", "pcpn"])
            .query("maxt_avg >= 0")
        )
        _df["date"] = pd.to_datetime(_df["date"])
        _df["year"] = _df["date"].dt.year
        _df["month"] = _df["date"].dt.month
        _df["site_id"] = site_id
        df.append(_df)
    df = pd.concat(df)

    return df


if __name__ == "__main__":
    df = read_sites(network="snotel")
    snotel_ids = df.snotel_id[0:20].tolist()
    SNOTEL_DIR = "D:/Projects/data/wsf/snotel_swe_daily"
    df_swe = read_swe(SNOTEL_DIR, snotel_ids)
    RCC_ACIS_DIR = "D:/Projects/data/wsf/rcc_acis"
    df_prism = read_monthly_prism(f"{RCC_ACIS_DIR}/prism_monthly.pkl")
    df_prism = df_prism.reset_index(drop=True)
    df_prism.to_feather("data/processed/prism_monthly.feather")
    print(df.columns)
    print(df)
