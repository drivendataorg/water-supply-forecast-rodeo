"""Code for downloading the SNOTEL daily monitoring data. Use the CLI to download, for example:

    python -m wsfr_download snotel 2005 2007 2009

or use the `bulk` command to download many sources at once based on a config file.

    python -m wsfr_download bulk data_download/hindcast_test_config.yml

You can also import this module and use it as a library.

See the challenge website for more about this approved data source:
https://www.drivendata.org/competitions/254/reclamation-water-supply-forecast-dev/page/801/#nrcs-snotel
"""

import datetime
import functools
from pathlib import Path
import threading
from typing import Annotated

import geopandas as gpd
from loguru import logger
import pandas as pd
import requests
from shapely.geometry import Point
import stamina
from tqdm.contrib.concurrent import thread_map
import typer
import zeep

from wsfr_download.config import DATA_ROOT
from wsfr_download.utils import (
    DownloadResult,
    log_download_results,
    site_geospatial,
    site_geospatial_buffered,
)

SNOTEL_DIR = DATA_ROOT / "snotel"

NRCS_AWDB_SOAP_WSDL_URL = "https://wcc.sc.egov.usda.gov/awdbWebService/services?WSDL"
NRCS_AWDB_REST_DATA_ENDPOINT = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"

ELEMENT_CODES = (
    "WTEQ",  # Snow Water Equivalent (in)
    "SNWD",  # Snow Depth (in)
    "PREC",  # Precipitation Accmulation (in)
    "TMAX",  # Air Temperature Maximum (°F)
    "TMIN",  # Air Temperature Minimum (°F)
    "TAVG",  # Air Temperature Average (°F)
)
DURATION = "DAILY"


def get_snotel_station_metadata(client: zeep.Client):
    """Retrieves dataframe of SNOTEL station metadata."""
    station_triplets = client.service.getStations(networkCds="SNTL", logicalAnd=False)
    data = client.service.getStationMetadataMultiple(stationTriplets=station_triplets)
    # Need to serialize from zeep models to Python dictionaries before passing to pandas
    snotel_df = pd.DataFrame.from_records(zeep.helpers.serialize_object(data))
    return snotel_df.set_index("stationTriplet")


def build_awdb_data_query_string(
    station_triplet: str,
    begin_date: str,
    end_date: str,
    elements: tuple[str, ...],
    duration: str,
):
    """Build querystring for the AWDB REST /data endpoint to get station data."""
    return "&".join(
        [
            f"stationTriplets={station_triplet}",
            f"beginDate={begin_date}",
            f"endDate={end_date}",
            "elements=" + "%2C".join(elements),
            f"duration={duration}",
        ]
    )


thread_local = threading.local()


def _get_session() -> requests.Session:
    """Utility for creating one requests session per thread."""
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


@stamina.retry(
    on=(requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError),
    attempts=5,
    wait_initial=10.0,
    wait_max=60.0,
    wait_exp_base=1.5,
    timeout=None,
)
def get_data_for_station(
    station_triplet: str, begin_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    """Returns data from NRCS AWDB for a SNOTEL station over given date range."""
    url = (
        NRCS_AWDB_REST_DATA_ENDPOINT
        + "?"
        + build_awdb_data_query_string(
            station_triplet=station_triplet,
            begin_date=begin_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            elements=ELEMENT_CODES,
            duration=DURATION,
        )
    )
    session = _get_session()
    response = session.get(url)
    data = {
        entry["stationElement"]["elementCode"]
        + "_"
        + entry["stationElement"]["durationName"]: _series_from_date_value_dicts(entry["values"])
        for entry in response.json()[0]["data"]
    }
    df = pd.DataFrame(data)
    df.index.name = "date"
    return df


def _series_from_date_value_dicts(arr: list[dict]):
    """Utility to build a pandas Series from the {"date": ..., "value": ...} JSON records returned
    by the AWDB REST Service data/ endpoint.
    """
    dates, values = zip(*[(entry["date"], entry["value"]) for entry in arr])
    return pd.Series(values, index=dates, dtype="float")


def download_station_data(
    station_triplet: str,
    begin_date: datetime.date,
    end_date: datetime.date,
    out_dir: Path,
    skip_existing: bool,
) -> DownloadResult:
    """Downloads SNOTEL station daily data to disk."""
    out_file = out_dir / f"{station_triplet.replace(':', '_')}.csv"
    if skip_existing and out_file.exists():
        return DownloadResult.SKIPPED_EXISTING
    try:
        data_df = get_data_for_station(station_triplet, begin_date, end_date)
        data_df.to_csv(out_file)
        return DownloadResult.SUCCESS
    except IndexError:
        logger.warning(f"No data for {station_triplet}, {begin_date} to {end_date}")
        return DownloadResult.SKIPPED_NO_DATA


def download_snotel(
    forecast_years: Annotated[list[int], typer.Argument(help="Forecast years to download for.")],
    fy_start_month: Annotated[int, typer.Option(help="Forecast year start month.")] = 10,
    fy_start_day: Annotated[int, typer.Option(help="Forecast year start day.")] = 1,
    fy_end_month: Annotated[int, typer.Option(help="Forecast year end month.")] = 7,
    fy_end_day: Annotated[int, typer.Option(help="Forecast year end day.")] = 21,
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = True,
):
    """Download SNOTEL station daily station measurements from the NRCS AWDB:
    https://www.nrcs.usda.gov/wps/portal/wcc/home/dataAccessHelp/webService

    This command downloads data for any SNOTEL station within approximately 40 miles of the
    forecast site drainage basins. The following are downloaded:

    \b
    - Metadata for all SNOTEL stations to snotel/station_metadata.csv
    - Result of spatial join between forecast sites and SNOTEL stations in
      snotel/sites_to_snotel_stations.csv
    - Daily measurements partitioned by forecast year and SNOTEL station triplet ID

    Each forecast year begins on the specified date of the previous calendar
    year, and ends on the specified date of the current calendar year. By
    default, each forecast year starts on October 1 and ends July 21; e.g.,
    by default, FY2021 starts on 2020-10-01 and ends on 2021-07-21.
    """
    logger.info("Downloading SNOTEL data...")
    SNOTEL_DIR.mkdir(exist_ok=True, parents=True)

    snotel_metadata_out_file = SNOTEL_DIR / "station_metadata.csv"
    if skip_existing and snotel_metadata_out_file.exists():
        logger.info(f"SNOTEL station metadata exists at {snotel_metadata_out_file}. Skipping.")
        snotel_df = pd.read_csv(snotel_metadata_out_file, index_col="stationTriplet")
    else:
        logger.info("Getting SNOTEL station metadata...")
        with zeep.Client(NRCS_AWDB_SOAP_WSDL_URL) as client:
            snotel_df = get_snotel_station_metadata(client)

        # Save snowtel metadata to disk
        snotel_df.to_csv(snotel_metadata_out_file)
        logger.info(f"SNOTEL station metadata saved to: {snotel_metadata_out_file}")

    # Get snotel station geodataframe
    snotel_gdf = gpd.GeoDataFrame(
        snotel_df.reset_index()[["stationTriplet"]],
        geometry=[
            Point(lon, lat) for lon, lat in zip(snotel_df["longitude"], snotel_df["latitude"])
        ],
        crs="WGS84",
    )

    basins_gdf = site_geospatial(layer="basins")
    buffered_basins_gdf = site_geospatial_buffered()

    logger.info("Performing spatial join from SNOTEL sites to buffered site basin polygons.")
    # Do a spatial join with original basins
    in_basin_gdf = (snotel_gdf.sjoin(basins_gdf[["site_id", "geometry"]], how="inner"))[
        ["site_id", "stationTriplet"]
    ]
    in_basin_gdf["in_basin"] = True

    # Do another spatial join with the buffer basins
    in_buffer_gdf = (snotel_gdf.sjoin(buffered_basins_gdf[["site_id", "geometry"]], how="inner"))[
        ["site_id", "stationTriplet"]
    ]
    in_buffer_gdf["in_basin"] = False

    # Concantenate and drop duplicates
    site_to_snotel_df = pd.concat(
        [in_basin_gdf, in_buffer_gdf], ignore_index=True
    ).drop_duplicates(subset=["site_id", "stationTriplet"], keep="first")
    site_to_snotel_out_file = SNOTEL_DIR / "sites_to_snotel_stations.csv"
    site_to_snotel_df.to_csv(site_to_snotel_out_file, index=False)
    logger.info(f"Site to SNOTEL station mapping saved to: {site_to_snotel_out_file}")

    nearby_snotel_triplets = site_to_snotel_df["stationTriplet"].unique()
    logger.info(f"{len(nearby_snotel_triplets)} nearby SNOTEL stations identified")

    # Availability dates
    avail_df = snotel_df.loc[nearby_snotel_triplets, ["beginDate", "endDate"]].copy()
    avail_df["beginDate"] = pd.to_datetime(avail_df["beginDate"])
    avail_df["endDate"] = pd.to_datetime(avail_df["endDate"])

    all_download_results = []
    for fy in forecast_years:
        fy_start = datetime.date(fy - 1, fy_start_month, fy_start_day)
        fy_end = datetime.date(fy, fy_end_month, fy_end_day)
        logger.info(
            f"Downloading forecast year {fy} "
            f"({fy_start.strftime('%Y-%m-%d')} to {fy_end.strftime('%Y-%m-%d')})"
        )
        out_dir = SNOTEL_DIR / f"FY{fy}"
        out_dir.mkdir(exist_ok=True, parents=True)
        # Get SNOTEL stations with data for this time period
        with_data_snotel_triplets = avail_df[
            # https://stackoverflow.com/a/325964
            (avail_df["beginDate"].dt.date <= fy_end)
            & (avail_df["endDate"].dt.date > fy_start)  # if overlap on xxxx-10-01, no data
        ].index.values

        download_results = thread_map(
            functools.partial(
                download_station_data,
                begin_date=fy_start,
                end_date=fy_end,
                out_dir=out_dir,
                skip_existing=skip_existing,
            ),
            with_data_snotel_triplets,
            total=len(with_data_snotel_triplets),
            chunksize=1,
        )
        all_download_results += download_results

    log_download_results(all_download_results)
    logger.success("SNOTEL download complete.")
