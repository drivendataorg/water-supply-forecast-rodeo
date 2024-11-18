import pandas as pd
import requests
from pathlib import Path


def _get_water_year(date):
    return date.year if date.month < 10 else date.year + 1


def _parse_uaswann_daily_url(date):
    wy = _get_water_year(date)
    url = f'https://climate.arizona.edu/data/UA_SWE/DailyData_4km/WY{wy}/UA_SWE_Depth_4km_v1_{date.strftime("%Y%m%d")}_stable.nc'

    return url


def download_uaswann_daily(dates, dirname):
    dirname = f"{dirname}/raw/daily"
    Path(dirname).mkdir(parents=True, exist_ok=True)
    for date in dates:
        out_file = f'{dirname}/{date.strftime("%Y%m%d")}.nc'
        try:
            response = requests.get(_parse_uaswann_daily_url(date), verify=False)
            response.raise_for_status()
            with open(out_file, "wb") as file:
                file.write(response.content)
        except Exception as e:
            print(f"{out_file} is not available: {e}")


def _parse_uaswann_yearly_url(wyear):
    url = f"https://climate.arizona.edu/data/UA_SWE/WYData_4km/UA_SWE_Depth_WY{wyear}.nc"

    return url


def download_uaswann_yearly(wyears, dirname):
    dirname = f"{dirname}/raw/yearly"
    Path(dirname).mkdir(parents=True, exist_ok=True)
    for year in wyears:
        out_file = f"{dirname}/UA_SWE_Depth_WY{year}.nc"
        try:
            response = requests.get(_parse_uaswann_yearly_url(year), verify=False)
            response.raise_for_status()
            with open(out_file, "wb") as file:
                file.write(response.content)
        except Exception as e:
            print(f"{out_file} is not available: {e}")


def download_uaswann_huc(huc_list, dirname, huc_level):
    _url = "https://climate.arizona.edu/snowview/csv/Download/Watersheds"
    dirname = f"{dirname}/raw/huc{huc_level}"
    Path(dirname).mkdir(exist_ok=True)
    for huc in huc_list:
        url_all = f"{_url}/{huc}.csv"
        filename = f"{dirname}/{huc}.csv"
        print(url_all)
        response = requests.get(url_all, verify=False)
        with open(filename, "wb") as file:
            file.write(response.content)


if __name__ == "__main__":
    from src.config import UASWANN_DIR
    from src.data.base import read_meta

    dates = pd.date_range("2019-10-01", "2023-12-31")
    download_uaswann_daily(dates=dates, dirname=UASWANN_DIR)

    wyears = range(1982, 2023)
    download_uaswann_yearly(wyears=wyears, dirname=UASWANN_DIR)

    df_meta = read_meta()
    df_usgs_sites = pd.read_feather("data/meta/nwis_usgs_meta.feather")
    df_usgs_sites["huc"] = df_usgs_sites["huc_cd"].astype(str).str.zfill(8)
    df_usgs_sites["huc8"] = df_usgs_sites["huc"].str[0:8]
    df_usgs_sites["huc6"] = df_usgs_sites["huc"].str[0:6]
    df_usgs_sites_filtered = df_usgs_sites[
        df_usgs_sites["usgs_id"].isin(
            df_meta.usgs_id.tolist()
            + [
                "12175500"  # skagit_ross_reservoir proxy https://waterdata.usgs.gov/monitoring-location/12175000/#parameterCode=00065&period=P7D&showMedian=false&timeSeriesId=285479
            ]
        )
    ]
    download_uaswann_huc(huc_list=df_usgs_sites_filtered.huc6.unique().tolist(), dirname=UASWANN_DIR, huc_level=6)
    download_uaswann_huc(huc_list=df_usgs_sites_filtered.huc8.unique().tolist(), dirname=UASWANN_DIR, huc_level=8)
