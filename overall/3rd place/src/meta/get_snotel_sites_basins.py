"""
Objective:
- Get pair sites and basins with only active sites and specified criteria

Requirement:
- snotel_sites.feather
- snotel_sites_basins.feather

Output:
- snotel_sites_basins_chosen.feather
"""

import pandas as pd
from loguru import logger

SNOTEL_META_DIR = "data/meta/snotel_sites_basins"

if __name__ == "__main__":
    logger.info("Prepare snotel_basins_chosen.py")
    df_snotel_sites = pd.read_feather(f"{SNOTEL_META_DIR}/snotel_sites.feather")
    df_snotel_sites_basins = pd.read_feather(
        f"{SNOTEL_META_DIR}/snotel_sites_basins.feather"
    )
    df_snotel_sites_basins_agg = (
        df_snotel_sites_basins.query("rnk <= 5 or rnk != rnk")
        .groupby(["site_id", "snotel_id"], as_index=False)
        .agg(method=("method", ",".join), dist=("dist", "min"), rnk=("rnk", "min"))
    )
    df_snotel_sites_basins_chosen = pd.merge(
        df_snotel_sites_basins_agg,
        df_snotel_sites.query(
            'snotel_end_date == "2100-01-01" & snotel_start_date <= "1990-10-01"'
        )[["snotel_id", "snotel_start_date", "snotel_elevation"]],
    )
    df_snotel_sites_basins_chosen.to_feather(
        f"{SNOTEL_META_DIR}/snotel_sites_basins_chosen.feather"
    )

    logger.info(
        f"File is created on `{SNOTEL_META_DIR}/snotel_sites_basins_chosen.feather`"
    )
