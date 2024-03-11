import os
from pathlib import Path

from loguru import logger
import stamina
from tqdm import tqdm

# Configure loguru logger with tqdm.write
# https://github.com/Delgan/loguru/issues/135
logger.remove(0)
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


DATA_ROOT = Path(os.getenv("WSFR_DATA_ROOT", Path.cwd() / "data"))
METADATA_FILE = DATA_ROOT / "metadata.csv"
GEOSPATIAL_FILE = DATA_ROOT / "geospatial.gpkg"


def log_stamina_retries(details: stamina.instrumentation.RetryDetails):
    """Stamina retry hook to log scheduled retry with loguru logger."""
    logger.warning(
        f"stamina retry scheduled ({details.retry_num}) "
        f"caused by {details.caused_by},"
        f"args={details.args}, kwargs={details.kwargs}"
    )


# Register logging hook
stamina.instrumentation.set_on_retry_hooks([log_stamina_retries])
