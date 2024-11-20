import os
import glob

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray


def preprocess_uaswe_basin(dirname, df_meta_poly, level="daily", is_print=False):
    dirname = f"{dirname}/raw/{level}/*.nc"
    df_swe = []
    for file in glob.glob(dirname):
        try:
            if is_print:
                print(file)
            ds = xr.open_dataset(file)
            ds = ds.drop_dims("time_str_len")
            ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            ds.rio.write_crs("epsg:4326", inplace=True)

            for _, row in df_meta_poly.iterrows():
                _res = ds.rio.clip([row["geometry"]]).mean(("lon", "lat"))
                _res = _res.to_dataframe()[["SWE", "DEPTH"]].reset_index()
                _res["site_id"] = row["site_id"]
                df_swe.append(_res)
        except Exception as e: 
            print(file)
            print(e)
    df_swe = pd.concat(df_swe)

    return df_swe


def preprocess_uaswe_basin_all(dirname, df_meta_poly, use_yearly_data=True, out_file=None, is_print=False):
    df_swe = preprocess_uaswe_basin(dirname, df_meta_poly, level="daily", is_print=is_print)
    if use_yearly_data:
        df_swe_yearly = preprocess_uaswe_basin(dirname, df_meta_poly, level="yearly", is_print=is_print)
        df_swe = pd.concat([df_swe_yearly, df_swe]).reset_index(drop=True)
    if out_file:
        df_swe.to_csv(out_file, index=False)

    return df_swe


def preprocess_uaswe_huc(dirname, df_meta, df_usgs_sites, huc_level, date="2023-07-22", out_file=None):
    dirname = f"{dirname}/raw/huc{huc_level}/*.csv"
    df_swe = []
    for file in glob.glob(dirname):
        _df_swe = pd.read_csv(file)
        _df_swe[f"huc{huc_level}"] = os.path.basename(file).replace(".csv", "")
        df_swe.append(_df_swe)
    df_swe = pd.concat(df_swe)
    df_swe["date"] = pd.to_datetime(df_swe["Date"])
    df_swe["swe"] = df_swe["Average SWE (in)"]
    df_swe["prec_cml"] = df_swe["Average Accumulated Water Year PPT (in)"]
    df_swe = df_swe[[f"huc{huc_level}", "date", "swe", "prec_cml"]]
    df_swe = df_swe[df_swe["date"] < date].reset_index(drop=True)
    df_swe = pd.merge(
        df_swe,
        pd.merge(
            df_usgs_sites[["usgs_id", f"huc{huc_level}"]],
            df_meta.assign(usgs_id=lambda x: np.where(x["usgs_id"].isna(), "12175500", x["usgs_id"])),
        )[[f"huc{huc_level}", "site_id"]],
    )
    df_swe = df_swe.drop(columns=[f"huc{huc_level}"])
    if out_file:
        df_swe.to_feather(out_file)

    return df_swe
