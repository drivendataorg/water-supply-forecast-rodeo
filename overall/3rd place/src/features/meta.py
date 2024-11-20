import pandas as pd
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from shapely.geometry import box


def fill_missing_drainage_area(df, df_poly):
    df = df.copy()
    df = pd.merge(df, df_poly[["site_id", "area"]])
    if df.drainage_area.isna().sum() > 0:
        m = LinearRegression()
        m.fit(
            df[~df["drainage_area"].isna()][["area"]],
            df[~df["drainage_area"].isna()][["drainage_area"]],
        )
        df.loc[df["drainage_area"].isna(), "drainage_area"] = m.predict(
            df[df["drainage_area"].isna()][["area"]]
        )
    df = df.drop(columns=["area"])

    return df


def apply_buffer(gdf, buffer_size=10):
    gdf = gdf.copy()
    gdf = gdf.to_crs(crs=3857)
    gdf["geometry"] = gdf.buffer(buffer_size * 1000, cap_style=2, join_style=2)
    gdf = gdf.to_crs(crs=4326)

    return gdf


def get_bbox_geometry(gdf):
    gdf = pd.concat([gdf[["site_id"]], gdf.geometry.bounds], axis=1)
    geometry = [
        box(x1, y1, x2, y2)
        for x1, y1, x2, y2 in zip(
            gdf.minx,
            gdf.miny,
            gdf.maxx,
            gdf.maxy,
        )
    ]
    gdf = gpd.GeoDataFrame(gdf, geometry=geometry)

    return gdf
