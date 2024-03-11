import requests
from loguru import logger
from src.config import cfg

sites_items_mapping = {
    "10773": "american_river_folsom_lake",
    "4281": "fontenelle_reservoir_inflow",
    "197": "boysen_reservoir_inflow",
    "794": "taylor_park_reservoir_inflow",
}


def _build_url_usbr(start_date, end_date, item_id):
    filename = f"{start_date}_{end_date}_{item_id}"
    url = "https://data.usbr.gov/rise/api/result"
    url = url + f"/download?type=csv&itemId={item_id}"
    url = url + f"&after={start_date}&before={end_date}"
    url = url + f"&filename={filename}&order=ASC"

    return url


def download_usbr(start_date, end_date, item_id, dirname):
    filename = f"{start_date}_{end_date}__{sites_items_mapping[item_id]}"
    out_file = dirname / f"{filename}.csv"
    if out_file.exists():
        logger.info(f"{filename} exists")
    else:
        url = _build_url_usbr(start_date, end_date, item_id)
        response = requests.get(url)
        out_file.parent.mkdir(exist_ok=True, parents=True)
        with out_file.open("w") as fp:
            fp.write(response.text)
        logger.info(f"{filename} is downloaded")


def download_usbr_hindcast(dirname, wyears=cfg["test_years"]):
    for year in wyears:
        for item_id in sites_items_mapping.keys():
            download_usbr(start_date=f"{year}-04-01", end_date=f"{year}-07-21", item_id=item_id, dirname=dirname)


if __name__ == "__main__":
    from pathlib import Path

    dirname = Path("data/usbr")
    download_usbr("2023-04-01", "2023-07-22", "10773", dirname)
