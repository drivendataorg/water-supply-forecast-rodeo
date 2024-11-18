import os
from pathlib import Path

from loguru import logger

DATA_ROOT = Path(os.getenv("WSFR_DATA_ROOT", Path.cwd() / "data"))
METADATA_FILE = DATA_ROOT / "metadata.csv"
GEOSPATIAL_FILE = DATA_ROOT / "geospatial.gpkg"

logger.info(f"DATA_ROOT is {DATA_ROOT}")
