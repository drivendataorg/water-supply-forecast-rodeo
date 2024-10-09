"""Code for downloading the Palmer Drought Severity Index (PDSI) from gridMET data. Use the CLI
to download, for example:

    python -m wsfr_download pdsi 2005 2007 2009

or use the `bulk` command to download many sources at once based on a config file.

    python -m wsfr_download bulk data_download/hindcast_test_config.yml

You can also import this module and use it as a library.

See the challenge website for more about this approved data source:
https://www.drivendata.org/competitions/254/reclamation-water-supply-forecast-dev/page/801/#palmer-drought-severity-index-pdsi-from-gridmet
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

from loguru import logger
import netCDF4 as nc
import requests
import typer

from wsfr_download.config import DATA_ROOT
from wsfr_download.utils import DownloadResult, log_download_results


def build_url(start_date: str, end_date: str) -> str:
    """Generate the URL to request PDSI data for a specific time range

    Args:
        start_date (str): Start date in the format %Y-%m-%d
        end_date (str): End date in the format %Y-%m-%d
    """
    url = "https://thredds.northwestknowledge.net/thredds/ncss/agg_met_pdsi_"
    url += "1979_CurrentYear_CONUS.nc?var=daily_mean_palmer_drought_severity_"
    url += "index&disableLLSubset=on&disableProjSubset=on&horizStride=1&"
    url += f"time_start={start_date}T00%3A00%3A00Z&time_end={end_date}"
    url += "T00%3A00%3A00Z&timeStride=1&accept=netcdf"

    return url


def download_time_range(start: str, end: str, out_file: Path) -> DownloadResult:
    """Download PDSI data for the continental US in the specified time
    frame in NetCDF4 format.

    Args:
        start (str): Start date in the format %Y-%m-%d
        end_date (str): End date in the format %Y-%m-%d
        out_file (Path): Path to save the downloaded file
    """
    url = build_url(start, end)
    response = requests.get(url)
    if response.status_code != 200:
        logger.warning(
            f"Could not download file. "
            f"{response.status_code}: {response.reason} error for {url}"
        )
        return DownloadResult.SKIPPED_NO_DATA

    with out_file.open("wb") as fp:
        fp.write(response.content)

    return DownloadResult.SUCCESS


def validate_netcdf4(file_path: Path, fy_start_month: int, fy_end_month: int) -> bool:
    """Validate a NetCDF4 file for PDSI. Returns True is the file is valid and
    False if not.

    Args:
        file_path (Path): Path to file
        fy_start_month (int): Forecast year start month
        fy_end_month (int): Forecast year end month
    """
    try:
        ds = nc.Dataset(file_path)
    except OSError:
        logger.warning(f"Deleting file that is not a valid NetCDF4: {file_path}")
        file_path.unlink()
        return False

    ds_keys = set(ds.variables.keys())
    if ds_keys != {"daily_mean_palmer_drought_severity_index", "day", "lat", "lon"}:
        logger.warning(f"Data at {file_path} has unexpected variable keys: {ds_keys}")

    file_fy = int(file_path.parent.name[-4:])
    calendar_start = datetime(1900, 1, 1)
    min_date = calendar_start + timedelta(days=min(ds.variables["day"][:]))
    if (min_date.year != file_fy - 1) or (min_date.month != fy_start_month):
        logger.warning(f"Unexpected minimum date for {file_path}: {min_date}")

    max_date = calendar_start + timedelta(days=max(ds.variables["day"][:]))
    if (max_date.year != file_fy) or (max_date.month != fy_end_month):
        logger.warning(f"Unexpected maximum date for {file_path}: {min_date}")

    ds.close()

    return True


def download_pdsi(
    forecast_years: Annotated[list[int], typer.Argument(help="Forecast years to download for.")],
    fy_start_month: Annotated[int, typer.Option(help="Forecast year start month.")] = 10,
    fy_start_day: Annotated[int, typer.Option(help="Forecast year start day.")] = 1,
    fy_end_month: Annotated[int, typer.Option(help="Forecast year end month.")] = 7,
    fy_end_day: Annotated[int, typer.Option(help="Forecast year end day.")] = 21,
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = True,
):
    """Download Palmer Drought Severity Index data from the
    THREDDS server (NetcdfSubset):
    https://www.drought.gov/data-maps-tools/us-gridded-palmer-drought-severity-index-pdsi-gridmet
    Each forecast year begins on the specified date of the previous calendar
    year, and ends on the specified date of the current calendar year. By
    default, each forecast year starts on October 1 and ends July 21; e.g.,
    by default, FY2021 starts on 2020-10-01 and ends on 2021-07-21.
    """
    logger.info("Downloading Palmer Drought Severity Index data...")

    download_results = []
    for fy in forecast_years:
        start = datetime(fy - 1, fy_start_month, fy_start_day).strftime("%Y-%m-%d")
        end = datetime(fy, fy_end_month, fy_end_day).strftime("%Y-%m-%d")
        logger.info(f"Downloading PDSI for forecast year {fy} ({start} to {end})")

        out_file = DATA_ROOT / f"pdsi/FY{str(fy)}/pdsi_{start}_{end}.nc"
        if skip_existing and out_file.exists():
            if validate_netcdf4(out_file, fy_start_month, fy_end_month):
                download_results.append(DownloadResult.SKIPPED_EXISTING)
                continue

        out_file.parent.mkdir(exist_ok=True, parents=True)
        result = download_time_range(start, end, out_file)
        if result == DownloadResult.SUCCESS:
            # Validate downloaded files if new file was downloaded
            validate_netcdf4(out_file, fy_start_month, fy_end_month)
            download_results.append(result)

    log_download_results(download_results)
    logger.success("PDSI download complete.")
