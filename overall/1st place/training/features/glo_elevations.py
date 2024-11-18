from itertools import permutations
from pathlib import Path

import geopandas
import numpy as np
import pandas as pd
import planetary_computer
import rasterio
from loguru import logger
from pystac_client import Client
from rasterio import mask


def generate_elevations(data_dir: Path,
                        preprocessed_dir: Path) -> pd.DataFrame:
    elevation_file = preprocessed_dir / 'site_elevations.csv'
    if elevation_file.is_file():
        logger.info("Elevation features file already exists.")
        site_elevations = pd.read_csv(elevation_file)
        return site_elevations

    logger.info("Generating elevation features.")

    geo_sites = geopandas.read_file(data_dir / 'geospatial.gpkg')

    client = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        ignore_conformance=True,
    )

    sites = []
    elevation_means = []
    elevation_stds = []
    southern_gradient_rates = []
    eastern_gradient_rates = []

    for idx, rw in geo_sites.iterrows():
        bbox = [[x, y] for x, y in permutations(rw['geometry'].bounds, 2) if x < 0 and y > 0]
        bbox.append(bbox[0])
        aoi = {'type': 'Polygon', 'coordinates': [bbox]}
        # Get all relevant items within the lat/lon bounds of the df
        search = client.search(
            collections=["cop-dem-glo-30"],
            intersects=aoi,
        )

        items = list(search.get_items())
        elev_list = []
        southern_list = []
        eastern_list = []
        for item in items:
            signed_asset = planetary_computer.sign(item.assets["data"])
            with rasterio.open(signed_asset.href) as src:
                shapes = [rw['geometry']]
                out_image, transformed = mask.mask(src, shapes, crop=True, filled=True)

                # Convert from meters to feet
                elev_matrix = np.squeeze(out_image) * 3.28084
                elev_nonzero = elev_matrix[np.nonzero(elev_matrix)]
                elev_list.append(elev_nonzero)
                southern_gradient = -1 * np.diff(elev_matrix, axis=0)
                southern_nonzero = southern_gradient[np.nonzero(southern_gradient)]
                southern_list.append(southern_nonzero)
                eastern_gradient = np.diff(elev_matrix, axis=1)
                eastern_nonzero = eastern_gradient[np.nonzero(eastern_gradient)]
                eastern_list.append(eastern_nonzero)

        sites.append(rw['site_id'])
        elevation_means.append(np.concatenate(elev_list).mean())
        elevation_stds.append(np.concatenate(elev_list).std())
        southern_gradient_rates.append(
            len(np.nonzero(np.concatenate(southern_list) > 0)[0]) / len(np.concatenate(southern_list)))
        eastern_gradient_rates.append(
            len(np.nonzero(np.concatenate(eastern_list) > 0)[0]) / len(np.concatenate(eastern_list)))

    site_elevations = pd.DataFrame({
        "site_id": sites,
        "elevation_means": elevation_means,
        "elevation_stds": elevation_stds,
        "southern_gradient_rates": southern_gradient_rates,
        "eastern_gradient_rates": eastern_gradient_rates
    })

    site_elevations.to_csv(elevation_file, index=False)

    logger.info("Finished generating elevation features.")

    return site_elevations
