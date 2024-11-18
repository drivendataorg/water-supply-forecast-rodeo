"""
Get available sites from runtime
"""

import os
import pandas as pd


def filter_available_snotel_sites(df):
    res = pd.read_csv("data/meta/data.find.txt", header=None, names=["file"])
    res = res[res["file"].str.contains("SNTL")]
    res["snotel_triplet"] = res["file"].apply(os.path.basename)
    res["snotel_id"] = res["snotel_triplet"].str.split("_").apply(lambda x: x[0])

    res["year"] = (
        res["file"]
        .str.split("/")
        .apply(lambda x: x[2])
        .str.replace("FY", "")
        .astype(int)
    )

    res["year_cnt"] = res.groupby("snotel_id")["snotel_id"].transform("count")

    # Filter year_cnt == 10
    res_id = (
        res[res["year_cnt"] == 10][["snotel_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df = pd.merge(df, res_id).reset_index(drop=True)

    return df
