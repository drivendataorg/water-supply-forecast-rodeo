"""Code for downloading daily observed streamflow data from USGS streamgages at the forecast sites
in the competition. Use the CLI to download, for example:

    python -m wsfr_download usgs_streamflow 2021

or use the `bulk` command to download many sources at once based on a config file.

    python -m wsfr_download bulk data_download/hindcast_test_config.yml

You can also import this module and use it as a library.

See the challenge website for more about this approved data source:
https://www.drivendata.org/competitions/254/reclamation-water-supply-forecast-dev/page/801/#usgs-streamflow
"""

from pathlib import Path

import pandas as pd
from dataretrieval import nwis
from loguru import logger
from tqdm import tqdm

from wsfr_download.utils import DownloadResult, log_download_results

MEAN_DISCHARGE_RAW_COL = "00060_Mean"
ALTERNATIVE_USGS_IDS = {
    'ruedi_reservoir_inflow': '09080400',
    'fontenelle_reservoir_inflow': '09211200',
    'american_river_folsom_lake': '11446500',
    'skagit_ross_reservoir': '12181000',
    'boysen_reservoir_inflow': '06279500',
    'pueblo_reservoir_inflow': '07109500',
    'boise_r_nr_boise': '13185000',
    'sweetwater_r_nr_alcova': '06235500',
}


def get_daily_values_for_usgs_site(
    usgs_id: str,
    forecast_year: int,
    fy_start_month: int,
    fy_start_day: int,
    fy_end_month: int,
    fy_end_day: int,
) -> pd.DataFrame:
    """Retrieves data from the USGS Daily Values service for a USGS site. This uses the
    dataretrieval Python package published by USGS.
    https://github.com/DOI-USGS/dataretrieval-python

    Args:
        usgs_id (str): USGS 8-digit identifier
        forecast_year (int): Year whose forecast season this data is for.
        fy_start_month (int): Month component of start date. In preceding calendar year.
        fy_start_day (int): Day component of start date. In preceding calendar year.
        fy_end_month (int): Month component of start date.
        fy_end_day (int): Day component of end date.

    Returns:
        pd.DataFrame: dataframe of retrieved data
    """
    start_date = f"{forecast_year - 1}-{fy_start_month:02}-{fy_start_day:02}"
    end_date = f"{forecast_year}-{fy_end_month:02}-{fy_end_day:02}"
    df, _ = nwis.get_dv(
        sites=[usgs_id],
        start=start_date,
        end=end_date,
        statCd="00003",  # statCd 00003 is statistics code for "mean"
    )
    return df


def has_discharge_col(df: pd.DataFrame) -> bool:
    """Checks if USGS DV dataframe contains column for discharge."""
    return MEAN_DISCHARGE_RAW_COL in df.columns


def download_alternative_usgs_streamflow(
    forecast_years: list[int],
    preprocessed_dir: Path,
    skip_existing: bool = True,
):
    """Download daily mean streamflow data from USGS streamgages located at the forecast sites in
    the challenge. The data is downloaded from the USGS Daily Values Service. Each forecast year's
    data begins on the specified date of the previous calendar year, and ends on the specified date
    of the same calendar year. By default, each forecast year starts on October 1 and ends July 21;
    e.g., by default, FY2021 data starts on 2020-10-01 and ends on 2021-07-21.

    To download equivalent data for other locations, see the function
    wsfr_download.usgs_streamflow.get_daily_values_for_usgs_site
    """
    logger.info(f"Downloading USGS streamflow data for forecast years {forecast_years}")
    fy_start_month = 10
    fy_start_day = 1
    fy_end_month = 7
    fy_end_day = 21
    sites_with_usgs = ALTERNATIVE_USGS_IDS
    all_download_results = []
    for forecast_year in forecast_years:
        logger.info(f"Downloading for FY {forecast_year}...")
        fy_dir = preprocessed_dir / "usgs_streamflow" / "alternative" / f"FY{forecast_year}"
        fy_dir.mkdir(exist_ok=True, parents=True)

        for site_id in tqdm(sites_with_usgs.keys()):
            out_file = fy_dir / f"{site_id}.csv"
            if skip_existing and out_file.exists():
                all_download_results.append(DownloadResult.SKIPPED_EXISTING)
                continue
            usgs_id = sites_with_usgs[site_id]
            df = get_daily_values_for_usgs_site(
                usgs_id=usgs_id,
                forecast_year=forecast_year,
                fy_start_month=fy_start_month,
                fy_start_day=fy_start_day,
                fy_end_month=fy_end_month,
                fy_end_day=fy_end_day,
            )
            # Sometimes there is no data, e.g., american_river_folsom_lake 11446220 for 2009
            if not df.empty and has_discharge_col(df):
                df.to_csv(out_file)
                all_download_results.append(DownloadResult.SUCCESS)
            else:
                all_download_results.append(DownloadResult.SKIPPED_NO_DATA)

    log_download_results(all_download_results)
    logger.success("USGS streamflow download complete.")
