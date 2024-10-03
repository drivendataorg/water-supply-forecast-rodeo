from contextlib import contextmanager
import pandas as pd
import xarray as xr
import rioxarray
import glob
from pathlib import Path
import wsfr_download.pdsi
from wsfr_download.pdsi import download_pdsi


@contextmanager
def _set_data_root(data_root):
    original_data_root = wsfr_download.pdsi.DATA_ROOT
    wsfr_download.pdsi.DATA_ROOT = data_root
    yield
    wsfr_download.pdsi.DATA_ROOT = original_data_root

if __name__ == "__main__":
    from src.config import PDSI_DIR

    Path(PDSI_DIR).mkdir(exist_ok=True)
    years = range(1980, 2024)

    with _set_data_root(Path(PDSI_DIR)):
        download_pdsi(
            forecast_years=years,
            fy_start_month=12,
            fy_start_day=1,
            fy_end_month=7,
            fy_end_day=22,
            skip_existing=False,
        )
