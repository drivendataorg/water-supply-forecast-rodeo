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
    val_years=[x for x in range(2004, 2024) if x % 2 == 0],
    test_years=[x for x in range(2004, 2024) if x % 2 == 1],
    synthetic=dict(
        n=4,
        scale_factor=[0.5, 1.5],
    ),
    swe=dict(
        features=["swe", "prec_cml"],
        top_k=9,
        start_lag=1,
        lags=2,
        interval=7,
        rank_feature="swe",
    ),
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
    "n_estimators": 500,
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
    "n_estimators": 1200,
    "seed": cfg["seed"],
    "verbose": -1,
}

cfg = {**cfg, "lgb_params": lgb_params, "lgb_reg_params": lgb_reg_params}
