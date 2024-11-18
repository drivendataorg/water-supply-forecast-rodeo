"""Code for downloading CDEC monitoring data. Use the CLI to download, for example:

    python -m wsfr_download cdec 2005 2007 2009

or use the `bulk` command to download many source at once based on a config file.

    python -m wsfr_download bulk data_download/hindcast_test_config.yml

You can also import this module and use it as a library.

See the challenge website for more about this approved data source:
https://www.drivendata.org/competitions/254/reclamation-water-supply-forecast-dev/page/801/#cdec-snow-sensor-network
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Sequence

import geopandas as gpd
from loguru import logger
import pandas as pd
import requests
from shapely.geometry import Point
import stamina
from tqdm import tqdm
import typer

from wsfr_download.config import DATA_ROOT
from wsfr_download.utils import (
    DownloadResult,
    batched,
    log_download_results,
    site_geospatial,
    site_geospatial_buffered,
)

CDEC_DIR = DATA_ROOT / "cdec"

# Only available sensors will be downloaded for each station
SENSOR_NUMBERS = (
    2,  # RAIN | PRECIPITATION, ACCUMULATED | INCHES
    3,  # SNOW WC | SNOW, WATER CONTENT | INCHES
    18,  # SNOW DP | SNOW DEPTH | INCHES
    30,  # TEMP | TEMPERATURE, AIR AVERAGE | DEG F
    31,  # TEMP MX | TEMPERATURE, AIR MAXIMUM | DEG F
    32,  # TEMP MN | TEMPERATURE, AIR MINIMUM | DEG F
    82,  # SNO ADJ | SNOW, WATER CONTENT(REVISED) | INCHES
    237,  # SNWCMIN | SNOW WATER CONTENT, MIN | INCHES
    238,  # SNWCMAX | SNOW WATER CONTENT, MAX | INCHES
)


def process_cdec_station_metadata() -> gpd.GeoDataFrame:
    """
    Load and process CDEC stations metadata into a GeoDataFrame.

    Metadata for all CDEC stations that collect snow data is available on the
    data download page as `cdec_snow_stations.csv`, and should be saved to
    `data/cdec_snow_stations.csv`.

    The metadata includes the union of stations that have a sensor for snow water equivalent
    (sensor number 3) and for snow depth (sensor number 18), downloaded using the CDEC Station
    Search web application.
    https://cdec.water.ca.gov/dynamicapp/staSearch?sta=&sensor_chk=on&sensor=18&collect=NONE+SPECIFIED&dur=&active=&lon1=&lon2=&lat1=&lat2=&elev1=-5&elev2=99000&nearby=&basin=NONE+SPECIFIED&hydro=NONE+SPECIFIED&county=NONE+SPECIFIED&agency_num=160&display=sta
    https://cdec.water.ca.gov/dynamicapp/staSearch?sta=&sensor_chk=on&sensor=3&collect=NONE+SPECIFIED&dur=&active=&lon1=&lon2=&lat1=&lat2=&elev1=-5&elev2=99000&nearby=&basin=NONE+SPECIFIED&hydro=NONE+SPECIFIED&county=NONE+SPECIFIED&agency_num=160&display=sta
    """
    # Load CDEC station metadata
    cdec_stations = pd.read_csv(DATA_ROOT / "cdec_snow_stations.csv")
    cdec_stations.columns = [c.lower().replace(" ", "_") for c in cdec_stations.columns]
    cdec_stations = cdec_stations.rename(columns={"id": "station_id"})

    # Get geodataframe
    cdec_gdf = gpd.GeoDataFrame(
        cdec_stations[["station_id", "station_name"]],
        geometry=[
            Point(lon, lat)
            for lon, lat in zip(cdec_stations["longitude"], cdec_stations["latitude"])
        ],
        crs="WGS84",
    )

    return cdec_gdf


@stamina.retry(
    on=(requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError),
    attempts=5,
    wait_initial=10.0,
    wait_max=60.0,
    wait_exp_base=1.5,
    timeout=None,
)
def download_sensor_data(
    station_ids: Sequence[str],
    start_date: datetime.date,
    end_date: datetime.date,
    out_dir: Path,
    skip_existing: bool,
    sensor_numbers: Sequence[int] = SENSOR_NUMBERS,
    session: requests.Session | None = None,
) -> list[DownloadResult]:
    """Download CDEC data for a list of stations, specified time range, and list
    of sensors. For an explanation of the available sensor numbers, see:
    https://cdec.water.ca.gov/misc/senslist.html

    Args:
        station_ids (Sequence[str]): Three-character station IDs to download
            data for
        start_date (datetime.date): Start date
        end_date (datetime.date): End date
        skip_existing (bool): Whether to skip existing files
        sensor_numbers (Sequence[int], optional): List of sensor numbers to include.
            Defaults to SENSOR_NUMBERS.
        out_dir (Path): Where to save the dataframe. Data will be saved to
            out_dir / {station_id}.csv
    """
    results = []
    if skip_existing:
        existing_station_ids = {p.stem for p in out_dir.glob("*.csv")}
        results += [
            DownloadResult.SKIPPED_EXISTING for _ in set(station_ids) & existing_station_ids
        ]
        station_ids = set(station_ids) - existing_station_ids
        if len(station_ids) == 0:
            return results
    station_ids = sorted(station_ids)

    stations = "%2C".join(sid for sid in station_ids)
    start = start_date.strftime("%Y-%m-%d")
    end = end_date.strftime("%Y-%m-%d")
    sensors = "%2C".join(str(sensor) for sensor in sensor_numbers)

    url = f"https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet?Stations={stations}"
    url += f"&dur_code=d&SensorNums={sensors}&Start={start}&End={end}"

    session = session or requests.Session()
    resp = session.get(url)
    resp.raise_for_status()
    all_data_df = pd.json_normalize(resp.json())
    if all_data_df.shape[0] == 0:
        results += [DownloadResult.SKIPPED_NO_DATA for _ in station_ids]
        return results
    else:
        all_data_df = all_data_df.set_index("stationId")

    for station_id in station_ids:
        try:
            this_station_df = all_data_df.loc[station_id].copy()
            out_file = out_dir / f"{station_id}.csv"
            this_station_df.to_csv(out_file)
            results.append(DownloadResult.SUCCESS)

            # Check for anomalies in station data
            if this_station_df[["SENSOR_NUM", "date", "value"]].isna().any().any():
                logger.warning(f"There are missing values for {station_id}")
            this_station_df["date"] = pd.to_datetime(this_station_df.date)
            if (this_station_df.date.min() < start_date) or (
                this_station_df.date.max() > end_date
            ):
                logger.warning(f"There are unexpected dates for {station_id}")
            if this_station_df[["stationId", "date", "SENSOR_NUM"]].duplicated().any():
                logger.warning(f"There is duplicate data (date + sensor) for {station_id}")

        except KeyError:
            results.append(DownloadResult.SKIPPED_NO_DATA)

    return results


def find_nearby_cdec_stations() -> list[str]:
    """Determine which CDEC stations are within approximately 40 miles of a
    forecast site drainage basin. Returns a list of station IDs.
    """
    cdec_stations = process_cdec_station_metadata()

    basins_gdf = site_geospatial(layer="basins")
    buffered_basins_gdf = site_geospatial_buffered()

    logger.info("Performing spatial join from CDEC sites to buffered site basin polygons.")
    # Do a spatial join with original basins
    in_basin_gdf = (cdec_stations.sjoin(basins_gdf[["site_id", "geometry"]], how="inner"))[
        ["site_id", "station_id"]
    ]
    in_basin_gdf["in_basin"] = True

    # Do another spatial join with the buffered basins
    in_buffer_gdf = (
        cdec_stations.sjoin(buffered_basins_gdf[["site_id", "geometry"]], how="inner")
    )[["site_id", "station_id"]]
    in_buffer_gdf["in_basin"] = False

    # Concantenate and drop duplicates
    site_to_cdec_df = pd.concat([in_basin_gdf, in_buffer_gdf], ignore_index=True).drop_duplicates(
        subset=["site_id", "station_id"], keep="first"
    )
    site_to_cdec_out_file = CDEC_DIR / "sites_to_cdec_stations.csv"
    site_to_cdec_out_file.parent.mkdir(exist_ok=True, parents=True)
    site_to_cdec_df.to_csv(site_to_cdec_out_file, index=False)
    logger.info(f"Site to CDEC station mapping saved to: {site_to_cdec_out_file}")

    nearby_cdec_stations = site_to_cdec_df["station_id"].unique()
    logger.info(f"{len(nearby_cdec_stations)} nearby CDEC stations identified")

    return nearby_cdec_stations


def download_cdec(
    forecast_years: Annotated[list[int], typer.Argument(help="Forecast years to download for.")],
    fy_start_month: Annotated[int, typer.Option(help="Forecast year start month.")] = 10,
    fy_start_day: Annotated[int, typer.Option(help="Forecast year start day.")] = 1,
    fy_end_month: Annotated[int, typer.Option(help="Forecast year end month.")] = 7,
    fy_end_day: Annotated[int, typer.Option(help="Forecast year end day.")] = 21,
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = True,
):
    """Download CDEC monitoring data from the California Data Exchange Center:
    https://cdec.water.ca.gov/dynamicapp/wsSensorData

    This command downloads data for any CDEC station within approximately 40 miles
    of the forecast site drainage basins. The following are downloaded:

    \b
    - Metadata for all CDEC stations to cdec/station_metadata.csv
    - Result of spatial join between forecast sites and CDEC stations to
      cdec/sites_to_cdec_stations.csv
    - Daily measurements partitioned by forecast year. Within each forecast year,
      a separate CSV file is saved for each CDEC station.

    Each forecast year begins on the specified date of the previous calendar
    year, and ends on the specified date of the current calendar year. By
    default, each forecast year starts on October 1 and ends July 21; e.g.,
    by default, FY2021 starts on 2020-10-01 and ends on 2021-07-21.
    """
    nearby_cdec_stations = find_nearby_cdec_stations()

    all_download_results = []
    with requests.Session() as session:
        for fy in forecast_years:
            fy_start = datetime(fy - 1, fy_start_month, fy_start_day)
            fy_end = datetime(fy, fy_end_month, fy_end_day)
            fy_dir = CDEC_DIR / f"FY{fy}"
            fy_dir.mkdir(exist_ok=True, parents=True)

            logger.info(
                f"Downloading forecast year {fy} "
                f"({fy_start.strftime('%Y-%m-%d')} to {fy_end.strftime('%Y-%m-%d')})"
            )

            batches = list(batched(nearby_cdec_stations, 100))
            for batch in tqdm(batches):
                download_results = download_sensor_data(
                    station_ids=batch,
                    start_date=fy_start,
                    end_date=fy_end,
                    out_dir=fy_dir,
                    skip_existing=skip_existing,
                    session=session,
                )
                all_download_results += download_results

    log_download_results(all_download_results)
    logger.success("CDEC download complete.")
