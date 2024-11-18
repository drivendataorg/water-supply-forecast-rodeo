import pandas as pd

from src.config import *
from src.data.base import *
from src.features.base import *

if __name__ == "__main__":
    df_cdec_basins = pd.read_csv(f"{CDEC_SWE_DIR}/sites_to_cdec_stations.csv")
    df_cdec_swe = read_cdec_swe(CDEC_SWE_DIR, is_preprocess=True)
    df_cdec_swe = parse_time_features(df_cdec_swe)

    # Only include active CDEC stations with long history data
    df_cdec_swe_agg = df_cdec_swe.groupby("cdec_id").agg(wyear_min=("wyear", "min"), wyear_max=("wyear", "max"))
    df_cdec_swe_agg_chosen = df_cdec_swe_agg.query("wyear_min <= 1990 & wyear_max>=2023")

    df_cdec_basins[df_cdec_basins["station_id"].isin(df_cdec_swe_agg_chosen.index)].reset_index(drop=True).to_feather(
        "data/meta/cdec_sites_chosen.feather"
    )

df_cdec_basins = pd.read_feather("data/meta/cdec_sites_chosen.feather")
