"""Code for downloading the MODIS vegetation indices from the Planetary Computer. Use the CLI
to download, for example:

    python -m wsfr_download modis_vegetation 2005 2007 2009

or use the `bulk` command to download many sources at once based on a config file.

    python -m wsfr_download bulk data_download/hindcast_test_config.yml

You can also import this module and use it as a library.

See the challenge website for more about this approved data source:
https://www.drivendata.org/competitions/254/reclamation-water-supply-forecast-dev/page/801/#modis-vegetation-indices
"""
from datetime import datetime
from functools import partial
import json
from pathlib import Path
import shutil
import threading
from typing import Annotated

from loguru import logger
import odc.stac
import pandas as pd
import planetary_computer as pc
from pystac import Item
import pystac_client
from tqdm.contrib.concurrent import thread_map
import typer

from wsfr_download.config import DATA_ROOT
from wsfr_download.utils import DownloadResult, log_download_results, site_geospatial

MODIS_DIR = DATA_ROOT / "modis_vegetation"


lock = threading.Lock()


def format_date_range(start: datetime, end: datetime) -> str:
    """Format a date range correctly to search the planetary computer"""
    datetime_format = "%Y-%m-%d"
    date_range_str = f"{start.strftime(datetime_format)}/{end.strftime(datetime_format)}"

    return date_range_str


def check_item_exists(item_dir: Path):
    """Check if an item has already been downloaded.

    Args:
        item_dir (Path): Item directory
    """
    if (item_dir / "metadata.json").exists() and (item_dir / "vegetation.nc").exists():
        return True

    return False


def download_item(item: Item, out_dir: Path, skip_existing: bool):
    """Download a PySTAC Item. Two files will be saved in out_dir / item.id:
    - vegetation.nc: netCDF4 file with NDVI (500m_16_days_NDVI) and
        EVI (500m_16_days_EVI) bands
    - metadata.json: Item metadata

    Args:
        item (Item): PySTAC item
        out_dir (Path): Directory to save the item
        skip_existing (bool): Whether to skip items that have already been
            downloaded
    """
    item_dir = out_dir / item.id
    if skip_existing and check_item_exists(item_dir):
        return DownloadResult.SKIPPED_EXISTING
    elif item_dir.exists():
        # Clean up partially downloaded item
        shutil.rmtree(item_dir)

    item_dir.mkdir(exist_ok=True, parents=True)
    # Download the NDVI and EVI bands
    vegetation_path = item_dir / "vegetation.nc"
    vegetation_data = odc.stac.load(
        [pc.sign(item)],
        bands=["500m_16_days_NDVI", "500m_16_days_EVI"],
    )
    # NetCDF is not thread-safe, so need lock
    with lock:
        vegetation_data.to_netcdf(vegetation_path)

    # Download the item metadata
    item_meta_path = item_dir / "metadata.json"
    with item_meta_path.open("w") as fp:
        fp.write(json.dumps(item.to_dict()))

    return DownloadResult.SUCCESS


def identify_items_for_basin(
    bbox: list[float],
    basin_id: str,
    start: datetime,
    end: datetime,
    catalog: pystac_client.Client | None = None,
):
    """Identify all items to download for a specific basin in a given date range.
    Returns a tuple of (basin_id, list of relevant PySTAC items)

    Args:
        bbox (list[float]): Bounding box in the format [min_longitude, min_latitude,
            max_longitude, max_latitude]
        basin_id (str): Basin site id
        start (datetime): Date range start
        end (datetime): Date range end
        catalog (pystac_client.Client | None, optional): PySTAC catalog to search.
            Defaults to None.
    """
    # Connect to the planetary computer, and don't sign in place
    catalog = catalog or pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )

    date_range = format_date_range(start, end)
    res = catalog.search(collections=["modis-13A1-061"], bbox=bbox, datetime=date_range)

    return basin_id, [item for item in res.item_collection()]


def download_modis_vegetation(
    forecast_years: Annotated[list[int], typer.Argument(help="Forecast years to download for.")],
    fy_start_month: Annotated[int, typer.Option(help="Forecast year start month.")] = 10,
    fy_start_day: Annotated[int, typer.Option(help="Forecast year start day.")] = 1,
    fy_end_month: Annotated[int, typer.Option(help="Forecast year end month.")] = 7,
    fy_end_day: Annotated[int, typer.Option(help="Forecast year end day.")] = 21,
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = True,
):
    """Download MODIS vegetation indices from the Planetary Computer:
    https://planetarycomputer.microsoft.com/dataset/modis-13A1-061
    The following are downloaded:

    \b
    - Metadata for all MODIS items for each forecast year downloaded to
      modis_vegetation/FY20XX/sites_to_items.csv. There is one row for each
      combination of MODIS item and forecast site.
    - MODIS items partitioned by forecast year. Within each forecast year,
      there is a folder for each MODIS item indexed by the item ID. The NDVI
      and EVI bands are downloaded as a netCDF file.

    Each forecast year begins in the specified month of the previous calendar
    year, and ends in the specified month of the current calendar year. By
    default, each forecast year starts in October and ends in July; e.g.,
    by default, FY2021 starts in October 2020 and ends in July 2021.
    """
    logger.info("Downloading MODIS Vegetation Indices...")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )

    all_download_results = []
    basins_gdf = site_geospatial(layer="basins")
    basin_bboxes = [basin_geo.bounds for basin_geo in basins_gdf.geometry]
    basin_ids = basins_gdf.site_id

    for fy in forecast_years:
        fy_dir = MODIS_DIR / f"FY{fy}"
        fy_dir.mkdir(exist_ok=True, parents=True)

        # Identify items to download
        fy_start = datetime(fy - 1, fy_start_month, fy_start_day)
        fy_end = datetime(fy, fy_end_month, fy_end_day)
        logger.info(
            f"Identifying items for FY {fy} ({fy_start.strftime('%Y-%m-%d')} to"
            f" {fy_end.strftime('%Y-%m-%d')}) across {len(basins_gdf):,} basins"
        )

        # Get a list of items for each basin
        basin_ids_to_items = thread_map(
            partial(identify_items_for_basin, start=fy_start, end=fy_end, catalog=catalog),
            basin_bboxes,
            basin_ids,
            total=len(basin_ids),
            chunksize=1,
        )

        # Deduplicate all relevant items
        fy_selected_items_dict = {}
        item_metadata = []
        for basin_id, basin_items in basin_ids_to_items:
            for item in basin_items:
                item_metadata.append(
                    {
                        "site_id": basin_id,
                        "item_id": item.id,
                        "start_datetime": item.properties["start_datetime"],
                        "end_datetime": item.properties["end_datetime"],
                        "bounding_box": item.bbox,
                    }
                )
                if item.id not in fy_selected_items_dict:
                    fy_selected_items_dict[item.id] = item

        # Save out item metadata
        item_meta_out_file = fy_dir / "sites_to_items.csv"
        item_metadata_df = pd.DataFrame(item_metadata)
        item_metadata_df.to_csv(item_meta_out_file, index=False)

        # Download items for forecast year
        logger.info(
            f"Downloading {len(fy_selected_items_dict):,} items for FY {fy}."
            f" Sites-to-items mapping saved to {item_meta_out_file}"
        )
        fy_download_results = thread_map(
            partial(download_item, out_dir=fy_dir, skip_existing=skip_existing),
            fy_selected_items_dict.values(),
            total=len(fy_selected_items_dict),
            chunksize=1,
        )
        all_download_results += fy_download_results

    log_download_results(all_download_results)
    logger.success("MODIS Vegetation Indices download complete.")
