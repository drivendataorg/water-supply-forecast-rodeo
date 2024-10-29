import os
import cdsapi
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

_CDS_URL = "https://cds.climate.copernicus.eu/api"
# "https://cds.climate.copernicus.eu/api/v2"  old API

_era5_cols = [
    "2m_temperature",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
    "snow_albedo",
    "snow_cover",
    "snow_depth_water_equivalent",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]
_seas51_cols = [
    "2m_temperature",
    "evaporation",
    "runoff",
    "snow_density",
    "snow_depth",
    "snowfall",
    "total_precipitation",
]
_conus_bbox = [
    54.78,
    -127.61,
    29.25,
    -94.39,
]


def download_era5(client, year, month, days, dirname, cols=_era5_cols, area=_conus_bbox, is_new_cds=False):
    dirname = f"{dirname}/raw"
    Path(dirname).mkdir(parents=True, exist_ok=True)
    download_params =  {
        "variable": cols,
        "time": ["00:00", "23:00"],
        "month": month,
        "year": year,
        "day": days,
        "area": area,
    }
    if is_new_cds:
        # download_params.update({"data_format": "netcdf", "download_format": "zip"}) # newest version of CDS, format is different
        download_params.update({"data_format": "netcdf_legacy", "download_format": "zip"})
    else:
        download_params.update({"format": "netcdf.zip"})
    out_path = f"{dirname}/{year}_{month}_era5land_base.netcdf.zip"
    if Path(out_path).exists():
        logger.info(f"Skipping existing {out_path} ...")
    else:
        client.retrieve(
            "reanalysis-era5-land",
            download_params,
            out_path,
        )


def download_seas51_monthly(client, years, leadtime, dirname,
                            cols=_seas51_cols, area=_conus_bbox, pname="forecast_m1_1981_2023_conus",
                            months=["01","02","03","04","05","06","07","12"]):
    dirname = f"{dirname}/raw"
    Path(dirname).mkdir(parents=True, exist_ok=True)
    years = [str(x) for x in years]
    for col in cols:
        out_path = f"{dirname}/seas51_{col}_{leadtime}_{pname}.nc"
        if Path(out_path).exists():
            logger.info(f"Skipping existing {out_path} ...")
        else:
            client.retrieve(
                "seasonal-monthly-single-levels",
                {
                    "format": "netcdf",
                    "originating_centre": "ecmwf",
                    "system": "51",
                    "variable": [col],
                    "product_type": "monthly_mean",
                    "month": months,
                    "leadtime_month": [leadtime],
                    "year": years,
                    "area": area,
                },
                out_path,
            )


if __name__ == "__main__":
    from src.features.base import parse_time_features
    from src.forecast import get_date_from_df
    from src.config import *

    client = cdsapi.Client(url=_CDS_URL, key=os.getenv("CDS_KEY"))

    """
    Generate  dataframe of year, month and list of days with lag t-1 day from issue date
    Data is retrieved for each year and month to speed up the process
    """
    years = list(range(1980, 2024))
    LAG = 1

    df_base = pd.DataFrame(
        index=pd.MultiIndex.from_product([years, list(range(1, 8)), [1, 8, 15, 22]], names=["year", "month", "day"])
    ).reset_index()
    df_base = get_date_from_df(df_base)
    df_base = df_base.assign(date=lambda x: pd.to_datetime(x["date"]))
    df_base_t1 = df_base.assign(date=lambda x: x["date"] - pd.to_timedelta(LAG, unit="d"))
    df_base_t1 = parse_time_features(df_base_t1, level=["year", "month", "day"])
    df_base_t1_ym = df_base_t1.groupby(["year", "month"]).agg({"day": lambda x: list(x)}).reset_index()

    for idx, row in df_base_t1_ym.iterrows():
        year = row["year"]
        month = row["month"]
        days = row["day"]
        download_era5(client, year, month, days, dirname=ERA5_DIR, is_new_cds=True)

    download_seas51_monthly(client, years=list(range(1981, 2024)), leadtime="1", dirname=SEAS51_DIR)
