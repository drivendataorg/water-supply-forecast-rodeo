import os
import sys
import requests
from datetime import datetime
from pathlib import Path


def download_usgs(
    site_list,
    start_date="1900-01-01",
    end_date=datetime.now().strftime("%Y-%m-%d"),
    path="",
    output_name="raw",
    data_type="daily",
):
    site_list = ",".join(site_list)

    if data_type == "daily":
        filename = f"{path}{output_name}_{start_date}_{end_date}_{data_type}.txt"
        url = (
            "https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites="
            + site_list
            + "&startDT="
            + start_date
            + "&endDT="
            + end_date
            + "&parameterCd=00060&siteStatus=all"
        )
    elif data_type == "monthly":
        filename = f"{path}{output_name}_{end_date}_{data_type}.txt"
        url = (
            "https://waterservices.usgs.gov/nwis/stat/?format=rdb&sites="
            + site_list
            + "&statReportType=monthly&statTypeCd=all&parameterCd=00010,00060"
        )
    elif data_type == "yearly":
        filename = f"{path}{output_name}_{end_date}_{data_type}.txt"
        url = (
            "https://waterservices.usgs.gov/nwis/stat/?format=rdb&sites="
            + site_list
            + "&statReportType=annual&statTypeCd=all&statYearType=water&missingData=on&parameterCd=00010,00060"
        )

    out_path = Path("data/external/usgs_streamflow") / filename
    out_path.parent.mkdir(exist_ok=True, parents=True)

    if os.path.isfile(out_path):
        sys.stdout.write("{} is already exist\n".format(out_path))
    else:
        print(url)
        with open(out_path, "wb") as f:
            response = requests.get(url, stream=True)
            dl = 0
            for data in response.iter_content(chunk_size=65536):
                dl += len(data)
                f.write(data)
                sys.stdout.write("\r  Dowloaded so far: %s bytes" % (dl))
                sys.stdout.flush()
        sys.stdout.write("\n  Download completed!\n")


if __name__ == "__main__":
    usgs_list = [
        "12362500",
        "13037500",
        "07099400",
        "06639000",
        "06054500",
        "09361500",
        "09251000",
        "12301933",
        "13202000",
        "12105900",
        "09109000",
        "09050700",
        "09080190",
        "09211150",
        "10128500",
        "11251000",
        "11266500",
        "11446220",
        "12409000",
        "12451000",
        "14181500",
        "09406000",
        "12175000",
        "06259000",
        "08378500",
        "13183000",
    ]
    download_usgs(usgs_list, start_date="1900-01-01", end_date="2023-10-21")
    download_usgs(usgs_list[0:10], data_type="monthly", output_name="raw1")
    download_usgs(usgs_list[10:20], data_type="monthly", output_name="raw2")
    download_usgs(usgs_list[20:], data_type="monthly", output_name="raw3")
