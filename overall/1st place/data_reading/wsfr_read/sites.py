import enum
from functools import cache
from typing import Literal

import geopandas as gpd
import pandas as pd

from wsfr_read.config import GEOSPATIAL_FILE, METADATA_FILE


class Layer(str, enum.Enum):
    BASINS = "basins"
    SITES = "sites"


@cache
def read_metadata() -> pd.DataFrame:
    """Load competition metadata.csv file with site metadata."""
    metadata_df = pd.read_csv(METADATA_FILE, index_col="site_id", dtype={"usgs_id": "string"})
    return metadata_df


def read_geospatial(layer: Literal["basins", "sites"] | Layer):
    """Load competition geospatial.gpkg file with site polygons. Valid layers are "basins" for
    drainage basin delineations or "sites" for forecast site point locations.
    """
    layer = Layer(layer)
    return gpd.read_file(GEOSPATIAL_FILE, layer=layer, index_col="site_id")
