"""
Get HUC6 and HUC8 for each site
"""

import pandas as pd
from dataretrieval import nwis
from src.config import *
from src.data.base import *

if __name__ == "__main__":
    df_meta = read_meta()
    df_meta_extra = pd.read_csv(f"{RAW_DATA_DIR}/supplementary_nrcs_metadata.csv")
    extra_usgs_list = df_meta_extra.usgs_id.tolist()
    usgs_list = df_meta.query("usgs_id==usgs_id").usgs_id.tolist()
    usgs_meta = nwis.get_info(sites=usgs_list)[0]
    _usgs_meta_cols = usgs_meta.isna().sum() / len(usgs_meta)
    _usgs_meta_cols = _usgs_meta_cols[_usgs_meta_cols < 0.5]
    usgs_meta = usgs_meta[_usgs_meta_cols.index.tolist()].drop(columns=["instruments_cd", "gw_file_cd", "land_net_ds"])
    usgs_meta = usgs_meta.rename(columns={"site_no": "usgs_id", "drain_area_va": "drainage_area"})
    usgs_meta.to_feather("data/meta/nwis_usgs_meta.feather")
