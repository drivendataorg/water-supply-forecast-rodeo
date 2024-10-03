"""
Get pair between SNOTEL sites and basins, there are 4 methods:
1. `same_huc`: Same HUC code (HUC-6)
2. `around_200`: Around 200km of basin gauge
3. `within`: Geo, within
4. `near`: Geo, within + near 10km

Output:
1. `snotel_sites.feather`: SNOTEL metadata
1. `snotel_sites_basins.feather`: all pair from all methods
2. `snotel_sites_flag`: list of eligible SNOTEL sites based on the pair
"""

import pandas as pd
from datetime import datetime

from dataretrieval import nwis
from src.utils import *
from src.features.base import *
from src.features.swe import *
from src.data.base import *
from src.config import *
from src.models.postprocess import *
from src.models.lgb import *

import zeep
from wsfr_download.snotel import *


def read_snotel_metadata(df):
    _cols = {
        "stationTriplet": "snotel_triplet",
        "snotel_id": "snotel_id",
        "name": "snotel_station_name",
        "longitude": "snotel_longitude",
        "latitude": "snotel_latitude",
        "elevation": "snotel_elevation",
        "huc6": "huc6",
        "beginDate": "snotel_start_date",
        "endDate": "snotel_end_date",
    }
    df = df.reset_index()
    df["snotel_id"] = df["stationTriplet"].str.split(":").apply(lambda x: x[0])
    df["huc6"] = df["huc"].str[0:6]
    df["beginDate"] = pd.to_datetime(df["beginDate"])
    df["endDate"] = pd.to_datetime(df["endDate"])
    df["is_active"] = (df["endDate"] == "2100-01-01").astype(int)
    _col_numeric = ["longitude", "latitude", "elevation"]
    df[_col_numeric] = df[_col_numeric].astype(float)
    df = df[_cols.keys()].rename(columns=_cols)

    return df


if __name__ == "__main__":
    df_train = read_train(is_forecast=True)
    df_meta = read_meta()
    df_meta_poly, df_meta_point = read_meta_geo()
    usgs_list = df_meta.query("usgs_id==usgs_id").usgs_id.tolist()

    NRCS_AWDB_SOAP_WSDL_URL = (
        "https://wcc.sc.egov.usda.gov/awdbWebService/services?WSDL"
    )
    client = zeep.Client(NRCS_AWDB_SOAP_WSDL_URL)

    df_snotel_sites = get_snotel_station_metadata(client)
    df_snotel_sites = read_snotel_metadata(df_snotel_sites)

    usgs_meta = nwis.get_info(sites=usgs_list)[0]
    df_usgs_sites = usgs_meta[["site_no", "station_nm", "huc_cd"]]
    df_usgs_sites = df_usgs_sites.rename(
        columns={
            "site_no": "usgs_id",
            "station_nm": "usgs_station_name",
            "huc_cd": "huc",
        }
    )
    df_usgs_sites = df_usgs_sites.assign(
        huc=lambda x: x["huc"].astype("str").str.zfill(8),
        huc6=lambda x: x["huc"].str[0:6],
    )

    df_snotel_sites.to_feather("data/meta/snotel_sites_basins/snotel_sites.feather")

    # Buffer for around 10km
    df_meta_poly_buffer = df_meta_poly.copy()
    df_meta_poly_buffer = df_meta_poly_buffer.to_crs(crs=3857)
    df_meta_poly_buffer["geometry"] = df_meta_poly_buffer.buffer(
        10000, cap_style=2, join_style=2
    )
    df_meta_poly_buffer = df_meta_poly_buffer.to_crs(crs=4326)

    # Same HUC code
    print("Join based on HUC-6 code")
    df_meta_snotel_huc = pd.merge(
        df_usgs_sites[df_usgs_sites["usgs_id"].isin(usgs_list)][
            ["usgs_id", "usgs_station_name", "huc6"]
        ],
        df_snotel_sites[["snotel_id", "snotel_station_name", "huc6"]],
    )
    print(df_meta_snotel_huc.usgs_id.nunique(), "USGS sites matched")
    print(df_meta_snotel_huc.snotel_id.nunique(), "SNOTEL sites matched")

    # Around 200km
    print("\nJoin around 200km")
    df_meta_snotel = combine_sites(df_meta, df_snotel_sites, limit=100)
    df_meta_snotel_filtered = (
        df_meta_snotel[["snotel_id", "snotel_longitude", "snotel_latitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df_meta_snotel_poly = gpd.GeoDataFrame(
        df_meta_snotel_filtered,
        geometry=gpd.points_from_xy(
            df_meta_snotel_filtered.snotel_longitude,
            df_meta_snotel_filtered.snotel_latitude,
        ),
    )
    print(df_meta_snotel.snotel_id.nunique(), "SNOTEL sites matched")
    print(
        df_meta_snotel.query("dist < 130").snotel_id.nunique(), "SNOTEL sites matched"
    )
    print(
        df_meta_snotel.query("dist < 130 & rnk <= 10").snotel_id.nunique(),
        "SNOTEL sites matched",
    )
    print(
        df_meta_snotel.query("dist < 200 & rnk <= 10").snotel_id.nunique(),
        "SNOTEL sites matched",
    )

    # Within
    print("Join within basin")
    gdf_snotel_sites = gpd.GeoDataFrame(
        df_snotel_sites,
        geometry=gpd.points_from_xy(
            df_snotel_sites.snotel_longitude, df_snotel_sites.snotel_latitude
        ),
        crs="EPSG:4326",
    )
    gdf_snotel_sites_joined = gpd.sjoin(
        gdf_snotel_sites, df_meta_poly, predicate="within"
    )
    print(gdf_snotel_sites_joined.site_id.nunique(), "Basin sites matched")
    print(gdf_snotel_sites_joined.snotel_id.nunique(), "SNOTEL sites matched")

    # Near
    print("\nJoin near basin")
    gdf_snotel_sites_joined_buffer = gpd.sjoin(
        gdf_snotel_sites, df_meta_poly_buffer, predicate="within"
    )
    print(gdf_snotel_sites_joined_buffer.site_id.nunique(), "Basin sites matched")
    print(gdf_snotel_sites_joined_buffer.snotel_id.nunique(), "SNOTEL sites matched")

    # Consolidate
    based_huc = list(set(df_meta_snotel_huc.snotel_id))
    based_around = list(set(df_meta_snotel.query("dist < 200 & rnk <= 10").snotel_id))
    based_within = list(set(gdf_snotel_sites_joined.snotel_id))
    based_near = list(set(gdf_snotel_sites_joined_buffer.snotel_id))

    df_snotel_sites_elig = df_snotel_sites.assign(
        is_same_huc=lambda x: x["snotel_id"].isin(based_huc).astype(int),
        is_around_200=lambda x: x["snotel_id"].isin(based_around).astype(int),
        is_within=lambda x: x["snotel_id"].isin(based_within).astype(int),
        is_near=lambda x: x["snotel_id"].isin(based_near).astype(int),
    )

    df_snotel_huc = pd.merge(df_meta_snotel_huc, df_meta[["usgs_id", "site_id"]])[
        ["site_id", "snotel_id"]
    ].assign(method="huc")
    df_snotel_around = df_meta_snotel.query("dist < 200 & rnk <= 10")[
        ["site_id", "snotel_id", "dist", "rnk"]
    ].assign(method="around")
    df_snotel_within = gdf_snotel_sites_joined[["site_id", "snotel_id"]].assign(
        method="within"
    )
    df_snotel_near = gdf_snotel_sites_joined_buffer[["site_id", "snotel_id"]].assign(
        method="near"
    )

    df_snotel_sites_basins = pd.concat(
        [df_snotel_huc, df_snotel_around, df_snotel_within, df_snotel_near]
    ).reset_index(drop=True)
    df_snotel_sites_basins.to_feather(
        "data/meta/snotel_sites_basins/snotel_sites_basins.feather"
    )
    df_snotel_sites_elig.filter(regex="is_*|snotel_id|snotel_triplet").to_feather(
        "data/meta/snotel_sites_basins/snotel_sites_flag.feather"
    )
