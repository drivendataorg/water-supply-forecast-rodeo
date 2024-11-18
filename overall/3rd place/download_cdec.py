from pathlib import Path

import wsfr_download.cdec

wsfr_download.cdec.CDEC_DIR = Path() / "data" / "external" / "cdec"
wsfr_download.cdec.download_cdec(forecast_years=list(range(1980, 2024)))
