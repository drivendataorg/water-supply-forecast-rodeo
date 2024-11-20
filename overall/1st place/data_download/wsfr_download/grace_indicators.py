"""Code for downloading GRACE-based soil moisture and groundwater drought indicators. Use the CLI
to download, for example:

    python -m wsfr_download grace_indicators 2021

or use the `bulk` command to download many sources at once based on a config file.

    python -m wsfr_download bulk data_download/hindcast_test_config.yml

You can also import this module and use it as a library.

See the challenge website for more about this approved data source:
https://www.drivendata.org/competitions/254/reclamation-water-supply-forecast-dev/page/801/#grace-based-soil-moisture-and-groundwater-drought-indicators
"""

from datetime import datetime
import functools
from pathlib import Path
import threading
from typing import Annotated

from loguru import logger
import netCDF4 as nc
import pandas as pd
import requests
from tqdm.contrib.concurrent import thread_map
import typer

from wsfr_download.config import DATA_ROOT
from wsfr_download.utils import DownloadResult, log_download_results

lock = threading.Lock()

# Get all possible dates
# Data is available for every 7 days starting Apr 1, 2002
POSSIBLE_DATES = pd.Series(pd.date_range("2002-04-01", "2023-07-21", freq="7D"))


def generate_source_urls(date: str, try_extensions: list[str] = ["030", "040"]) -> list[str]:
    """Get possible source URLs based on a date in the format YYYYMMDD"""
    return [
        f"https://nasagrace.unl.edu/data/{date}/GRACEDADM_CLSM0125US_7D.A{date}.{ext}.nc4"
        for ext in try_extensions
    ]


def dates_in_range(start_date: datetime, end_date: datetime) -> list[str]:
    """Get all of the possible dates within a given time range. Data is
    available for dates starting with 2002-04-01, and every 7 days after
    that.

    Args:
        start_date (datetime): Start of date range
        end_date (datetime): End of date range
    """
    return list(
        POSSIBLE_DATES[POSSIBLE_DATES.between(start_date, end_date)].dt.strftime("%Y%m%d").values
    )


def download_for_date(date: str, out_dir: Path, skip_existing: bool) -> DownloadResult:
    """Download the data for a given date

    Args:
        date (str): Date in the format '%Y%m%d'
        out_dir (Path): Directory to save file
        skip_existing (bool): Whether or not to skip if file already exists
    """
    source_urls = generate_source_urls(date)

    for url in source_urls:
        filename = url.split("/")[-1]
        out_file = out_dir / filename

        # If out file exists, validates file and deletes if bad file
        if skip_existing and out_file.exists():
            if validate_netcdf4(out_file):
                return DownloadResult.SKIPPED_EXISTING

        # Download from source URL and validate
        response = requests.get(url)
        if response.status_code == 200:
            with out_file.open("wb") as fp:
                fp.write(response.content)

            if validate_netcdf4(out_file):
                return DownloadResult.SUCCESS

    logger.warning(
        f"Could not download file for {date}. "
        f"{response.status_code}: {response.reason} error for {url}"
    )

    return DownloadResult.SKIPPED_NO_DATA


def validate_netcdf4(file_path: Path):
    """Validate that the netCDF4 file at file_path can be opened"""
    # Validate that netcdf4 file can be opened
    try:
        with lock, nc.Dataset(file_path) as ds:
            if (ds_keys := set(ds.variables.keys())) != {
                "lat",
                "lon",
                "rtzsm_inst",
                "sfsm_inst",
                "gws_inst",
                "time",
            }:
                logger.warning(f"Data at {file_path} has unexpected variable keys: {ds_keys}")

        return True

    except OSError:
        logger.warning(f"Deleting file that was downloaded but could not be opened: {file_path}")
        file_path.unlink()

        return False


def download_grace_indicators(
    forecast_years: Annotated[list[int], typer.Argument(help="Forecast years to download for.")],
    fy_start_month: Annotated[int, typer.Option(help="Forecast year start month.")] = 10,
    fy_start_day: Annotated[int, typer.Option(help="Forecast year start day.")] = 1,
    fy_end_month: Annotated[int, typer.Option(help="Forecast year end month.")] = 7,
    fy_end_day: Annotated[int, typer.Option(help="Forecast year end day.")] = 21,
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = True,
):
    """Download GRACE indicator data from the NASA GRACE CONUS data archive:
    https://nasagrace.unl.edu/ConusData.aspx
    Each forecast year begins on the specified date of the previous calendar
    year, and ends on the specified date of the current calendar year. By
    default, each forecast year starts on October 1 and ends July 21; e.g.,
    by default, FY2021 starts on 2020-10-01 and ends on 2021-07-21.
    """
    logger.info(f"Downloading GRACE indicator data for forecast years: {forecast_years}")

    date_count = 0
    all_download_results = []
    for fy in forecast_years:
        fy_start = datetime(fy - 1, fy_start_month, fy_start_day)
        fy_end = datetime(fy, fy_end_month, fy_end_day)

        logger.info(
            f"Downloading for FY {fy} ({fy_start.strftime('%Y-%m-%d')} "
            f"to {fy_end.strftime('%Y-%m-%d')})"
        )
        out_dir = DATA_ROOT / f"grace_indicators/FY{fy}"
        out_dir.mkdir(exist_ok=True, parents=True)

        dates = dates_in_range(fy_start, fy_end)
        date_count += len(dates)
        download_results = thread_map(
            functools.partial(download_for_date, out_dir=out_dir, skip_existing=skip_existing),
            dates,
            total=len(dates),
            chunksize=1,
        )
        all_download_results += download_results

    log_download_results(all_download_results, f"[{date_count:,} dates checked.]")
    logger.success("GRACE indicator download complete.")
