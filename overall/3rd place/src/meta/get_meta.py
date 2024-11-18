from src.data.base import *
from src.features.meta import *


if __name__ == "__main__":
    df_meta = read_meta()
    df_meta_poly, _ = read_meta_geo()

    df_meta = fill_missing_drainage_area(df_meta, df_meta_poly)
    df_meta["rfc"] = df_meta["rfc"].fillna("Unknown")
    df_meta.to_csv("data/meta/metadata_proc.csv", index=False)
