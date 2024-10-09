# Path
RAW_DATA_DIR = "data/raw/dev"
EXT_DATA_DIR = "data/external"

# External path
SNOTEL_DIR = "D:/Projects/data/wsf/snotel_swe_daily"
SNOTEL_SWE_DIR = "D:/Projects/data/wsf/snotel_swe_daily_2"
USGS_DIR = "D:/Projects/wsf-prod/data/usgs_streamflow"
RCC_ACIS_DIR = "D:/Projects/data/wsf/rcc_acis"
PDSI_DIR = "D:/Projects/data/wsf/pdsi"
TCI_DIR = "D:/Projects/data/wsf/tci"

# Base config
cfg = dict(
    seed=2024,
    target="volume",
    target_prep=dict(diff=True, log=False),
    cat_features=["site_id", "md_id", "rfc"],
    remove_features=[
        "year",
        "volume",
        "md_id",
        "cat",
    ],
    pred_cols=["pred_volume_10", "pred_volume_50", "pred_volume_90"],
    val_years=[[x for x in range(2005, 2024) if x % 2 == 0]],
    test_years=[x for x in range(2004, 2024) if x % 2 == 1],
    synthetic=dict(
        n=0,
        scale_factor=[0.5, 1.5],
    ),
    swe=dict(features=["swe", "prec_cml"], top_k=9, start_lag=2, lags=2, interval=7),
    diff_gap=1,
    mode=dict(cv=True, full_train=False, reg=True),
)

lgb_params = {
    "objective": "quantile",
    "metric": "quantile",
    "learning_rate": 0.05,
    "first_metric_only": True,
    "subsample": 0.9,
    "subsample_freq": 1,
    "num_leaves": 2**7 - 1,
    "min_data_in_leaf": 2**7 - 1,
    "feature_fraction": 0.9,
    "early_stopping_rounds": 50,
    "n_estimators": 2000,
    "seed": cfg["seed"],
    "verbose": -1,
}

lgb_reg_params = {
    "objective": "tweedie",
    "metric": "mae",
    "learning_rate": 0.025,
    "first_metric_only": True,
    "subsample": 0.9,
    "subsample_freq": 1,
    "num_leaves": 2**4 - 1,
    "min_data_in_leaf": 2**4 - 1,
    "feature_fraction": 0.9,
    "early_stopping_rounds": 50,
    "n_estimators": 3000,
    "seed": cfg["seed"],
    "verbose": -1,
}

cfg = {**cfg, "lgb_params": lgb_params, "lgb_reg_params": lgb_reg_params}

# Sites
"""
22 USGS sites, 4 USGS sites are missing
"""
usgs_missing_sites = [
    "american_river_folsom_lake",  # CA
    "fontenelle_reservoir_inflow",
    "ruedi_reservoir_inflow",
    "skagit_ross_reservoir",
]

"""
4 USGS sites have partial data and cannot be used. In the end, we only have 18 USGS sites
"""
usgs_partial_sites = [
    "pueblo_reservoir_inflow",
    "boysen_reservoir_inflow",
    "sweetwater_r_nr_alcova",
    "boise_r_nr_boise",
]

"""
USBR sites, in the end we will have 21 sites, missing sites are:
- skagit_ross_reservoir (only gage height available)
- fontenelle_reservoir_inflow
- ruedi_reservoir_inflow
- boise_r_nr_boise
- sweetwater_r_nr_alcova
"""
td_usbr_sites = [
    "american_river_folsom_lake",  # CA
    "fontenelle_reservoir_inflow",
    "boysen_reservoir_inflow",
    # "pueblo_reservoir_inflow", # no inflow, cannot be used
    # "ruedi_reservoir_inflow", # no inflow, cannot be used
    "taylor_park_reservoir_inflow",
]

td_usgs_sites = [
    "colville_r_at_kettle_falls",
    "animas_r_at_durango",
    "stehekin_r_at_stehekin",
    "pecos_r_nr_pecos",
    "virgin_r_at_virtin",
    "yampa_r_nr_maybell",
    "merced_river_yosemite_at_pohono_bridge",
    "missouri_r_at_toston",
    "weber_r_nr_oakley",
    "green_r_bl_howard_a_hanson_dam",
    # "snake_r_nr_heise" # removed because low correlation
]

tci_features = ["oni", "oni_anom", "pdo_index", "pna_index", "soi"]
