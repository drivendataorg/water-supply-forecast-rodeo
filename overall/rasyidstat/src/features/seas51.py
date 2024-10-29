import numpy as np
import pandas as pd
import xarray as xr
import rioxarray


_seas51_cols = [
    "2m_temperature",
    "evaporation",
    "runoff",
    "snow_density",
    "snow_depth",
    "snowfall",
    "total_precipitation",
]


def preprocess_seas51(dirname, df_meta_poly, pname="1_forecast_m1_1981_2023_conus", cols=_seas51_cols, out_file=None):
    df_res_all = []
    for col in cols:
        ds = xr.open_dataset(f"{dirname}/raw/seas51_{col}_{pname}.nc")
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
        df_res = df_res.rename(columns={"time": "date"})
        df_res = df_res.rename(columns={"forecast_reference_time": "date"}) # new CDS system
        if 'forecastMonth' in df_res.columns:
            df_res = df_res.drop(columns=['forecastMonth'])
        df_res = df_res.set_index(["site_id", "date", "number"])
        df_res_all.append(df_res)
    df_res_all = pd.concat(df_res_all, axis=1)
    df_res_all = df_res_all.reset_index()
    if out_file:
        df_res_all.to_feather(out_file)

    return df_res_all
