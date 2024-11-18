import glob

import pandas as pd
import xarray as xr
import rioxarray


def preprocess_pdsi(dirname, df_meta_poly, out_file=None):
    pdsi_files = glob.glob(f"{dirname}/**/*.nc", recursive=True)

    df_pdsi = []
    for file in pdsi_files:
        ds = xr.open_dataset(file)
        ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        ds.rio.write_crs("epsg:4326", inplace=True)

        for _, row in df_meta_poly.iterrows():
            _res = ds.rio.clip([row["geometry"]]).mean(("lon", "lat"))
            _res = _res.to_dataframe()[
                ["daily_mean_palmer_drought_severity_index"]
            ].reset_index()
            _res["site_id"] = row["site_id"]
            df_pdsi.append(_res)
    df_pdsi = pd.concat(df_pdsi)
    df_pdsi = df_pdsi.drop_duplicates()
    df_pdsi = df_pdsi.rename(
        columns={"day": "date", "daily_mean_palmer_drought_severity_index": "pdsi"}
    )
    df_pdsi = df_pdsi.sort_values(["site_id", "date"]).reset_index(drop=True)
    if out_file:
        df_pdsi.to_csv(out_file, index=False)

    return df_pdsi
