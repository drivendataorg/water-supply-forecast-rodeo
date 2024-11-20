import pandas as pd
from pathlib import Path
from wsfr_download.snotel import *

if __name__ == "__main__":
    ELEMENT_CODES = (
        "WTEQ",  # Snow Water Equivalent (in)
        "SNWD",  # Snow Depth (in)
        "PREC",  # Precipitation Accmulation (in)
        "TMAX",  # Air Temperature Maximum (°F)
        "TMIN",  # Air Temperature Minimum (°F)
        "TAVG",  # Air Temperature Average (°F)
    )
    DURATION = "DAILY"

    df_snotel_sites_elig = pd.read_feather(
        "data/meta/snotel_sites_basins/snotel_sites_flag.feather"
    )
    df_snotel_sites_elig["is_elig"] = (
        df_snotel_sites_elig.filter(like="is_").sum(axis=1) > 0
    )
    df_snotel_sites_elig_filtered = (
        df_snotel_sites_elig.query("is_elig")
        .sort_values(
            ["is_within", "is_near", "is_same_huc", "is_around_200"], ascending=False
        )
        .reset_index(drop=True)
    )

    Path("data/external/snotel").mkdir(exist_ok=True)
    for idx, row in df_snotel_sites_elig_filtered.iterrows():
        snotel_triplet = row["snotel_triplet"]
        try:
            download_station_data(
                snotel_triplet,
                begin_date=pd.to_datetime("1900-10-01"),
                end_date=pd.to_datetime("2023-09-30"),
                out_dir=Path("data/external/snotel"),
                skip_existing=False,
            )
        except:
            print(idx, snotel_triplet)
