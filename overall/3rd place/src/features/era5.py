import os
import glob

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray


def preprocess_era5(filename, df_meta_poly, is_new_cds=False):
    ds = xr.open_dataset(filename)
    ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    ds.rio.write_crs("epsg:4326", inplace=True)

    df_res = []
    for _, row in df_meta_poly.iterrows():
        _res = ds.rio.clip([row["geometry"]], all_touched=True)
        _res_cols = list(_res.keys())
        _res = _res.mean(("longitude", "latitude"))
        _res = _res.to_dataframe()[_res_cols].reset_index()
        _res["site_id"] = row["site_id"]
        df_res.append(_res)
    df_res = pd.concat(df_res)
    if is_new_cds:
        df_res = df_res.rename(columns={'valid_time':'time'})
    df_res = df_res.set_index(["site_id", "time"])

    return df_res


def preprocess_era5_all(dirname, df_meta_poly, out_file=None, is_new_cds=False):
    era5_files = glob.glob(f"{dirname}/raw/**/*.nc")
    df_res_all = []
    for file in era5_files:
        df_res = preprocess_era5(file, df_meta_poly, is_new_cds)
        df_res_all.append(df_res)
    df_res_all = pd.concat(df_res_all).reset_index()
    if out_file:
        df_res_all.to_feather(out_file)

    return df_res_all
