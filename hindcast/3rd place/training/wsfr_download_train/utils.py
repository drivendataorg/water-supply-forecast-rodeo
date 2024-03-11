import enum
import itertools

from loguru import logger

# A data collection station must be within this many meters of a drainage basin to be considered
# Default is 40 miles ~= 64K meters
DEFAULT_DRAINAGE_BASIN_BUFFER = 64_373.8


class DownloadResult(str, enum.Enum):
    SUCCESS = "success"
    SKIPPED_EXISTING = "skipped_existing"
    SKIPPED_NO_DATA = "skipped_no_data"


def log_download_results(results: list[DownloadResult], *args: str):
    """Results should be a list of DownloadResult enum values. Any additional str value args will
    be appended to the log statement"""
    n_success = sum(1 for res in results if res == DownloadResult.SUCCESS)
    n_skipped_existing = sum(1 for res in results if res == DownloadResult.SKIPPED_EXISTING)
    n_skipped_no_data = sum(1 for res in results if res == DownloadResult.SKIPPED_NO_DATA)
    msg = " ".join(
        (
            f"Downloaded {n_success:,} new files.",
            f"Skipped {n_skipped_existing:,} existing files.",
            f"Skipped {n_skipped_no_data:,} downloads with no data.",
        )
        + args
    )
    logger.info(msg)


def batched(iterable, n: int) -> tuple:
    """Based on suggested equivalent code for itertools.batched from Python 3.12
    https://docs.python.org/3/library/itertools.html#itertools.batched
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch
