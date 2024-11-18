"""Code for downloading the SNODAS snowpack property estimates from data assimilation. Use the CLI
to download, for example:

    python -m wsfr_download snodas 2005 2007 2009

or use the `bulk` command to download many sources at once based on a config file.

    python -m wsfr_download bulk data_download/hindcast_test_config.yml

You can also import this module and use it as a library.

See the challenge website for more about this approved data source:
https://www.drivendata.org/competitions/254/reclamation-water-supply-forecast-dev/page/801/#snodas

See additional references:

Reference for downloading data from online directory
https://nsidc.org/data/user-resources/help-center/how-access-and-download-noaansidc-data

Reference for converting .dat files to netcdf
https://nsidc.org/data/user-resources/help-center/how-do-i-convert-snodas-binary-files-geotiff-or-netcdf

User guide:
https://nsidc.org/sites/default/files/g02158-v001-userguide_2_1.pdf
"""


import calendar
from datetime import datetime
import functools
from pathlib import Path
from typing import Annotated

from loguru import logger
import pandas as pd
import requests
from tqdm.contrib.concurrent import thread_map
import typer

from wsfr_download.config import DATA_ROOT
from wsfr_download.utils import DownloadResult, log_download_results


def urls_for_date_range(start_date: datetime, end_date: datetime) -> list[str]:
    """Get all source urls for files within a given date range (inclusive).

    Args:
        start_date (datetime): Start date
        end_date (datetime): End date
    """
    dates_in_range = pd.date_range(start_date, end_date, freq="1D")

    return [date_url(date) for date in dates_in_range]


def date_url(date: datetime) -> str:
    """Get the source url for the SNODAS file on a given date"""
    url = "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/"
    url += f"{date.year}/{date.month:02}_{calendar.month_abbr[date.month]}"
    url += f"/SNODAS_{date.strftime('%Y%m%d')}.tar"

    return url


def download_from_url(source_url: str, out_dir: Path, skip_existing: bool) -> DownloadResult:
    """Download a SNODAS file based on its source URL. The file will be saved to
    out_dir with the same filename as the URL.
    """
    filename = source_url.split("/")[-1]
    out_file = out_dir / filename
    if skip_existing and out_file.exists():
        return DownloadResult.SKIPPED_EXISTING

    response = requests.get(source_url)
    with open(out_file, "wb") as f:
        f.write(response.content)

    return DownloadResult.SUCCESS


def download_snodas(
    forecast_years: Annotated[list[int], typer.Argument(help="Forecast years to download for.")],
    fy_start_month: Annotated[int, typer.Option(help="Forecast year start month.")] = 10,
    fy_start_day: Annotated[int, typer.Option(help="Forecast year start day.")] = 1,
    fy_end_month: Annotated[int, typer.Option(help="Forecast year end month.")] = 7,
    fy_end_day: Annotated[int, typer.Option(help="Forecast year end day.")] = 21,
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = True,
):
    """Download SNODAS data from NOAA NSIDC:
    https://noaadata.apps.nsidc.org/NOAA/G02158/masked/
    Each forecast year begins on the specified date of the previous calendar
    year, and ends on the specified date of the current calendar year. By
    default, each forecast year starts on October 1 and ends July 21; e.g.,
    by default, FY2021 starts on 2020-10-01 and ends on 2021-07-21.
    """
    logger.info("Downloading SNODAS data...")

    all_download_results = []
    for fy in forecast_years:
        # Get all URLs to download in given forecast year
        fy_start = datetime(fy - 1, fy_start_month, fy_start_day)
        fy_end = datetime(fy, fy_end_month, fy_end_day)
        fy_urls = urls_for_date_range(fy_start, fy_end)

        # Download forecast year files
        out_dir = DATA_ROOT / f"snodas/FY{str(fy)}"
        out_dir.mkdir(exist_ok=True, parents=True)

        logger.info(
            f"Downloading {len(fy_urls)} files for forecast year {fy}"
            f" ({fy_start.strftime('%Y-%m-%d')} to {fy_end.strftime('%Y-%m-%d')})"
        )
        fy_download_results = thread_map(
            functools.partial(download_from_url, out_dir=out_dir, skip_existing=skip_existing),
            fy_urls,
            total=len(fy_urls),
            chunksize=1,
        )
        all_download_results += fy_download_results

    log_download_results(all_download_results)
    logger.success("SNODAS download complete.")


if __name__ == "__main__":
    typer.run(download_snodas)
