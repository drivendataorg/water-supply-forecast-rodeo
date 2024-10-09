from configs.forecast.base import cfg

cfg["diff_gap"] = 0
cfg["filter_snotel_sites"] = True
cfg["target_prep"]["diff_extra"] = True
cfg["lgb_params"]["n_estimators"] = 500
cfg["lgb_reg_params"]["n_estimators"] = 1200
cfg["swe"] = dict(
    features=["swe", "prec_cml"],
    top_k=5,
    lag_list=[3],
    rank_feature="swe",
)
cfg["pdsi"] = dict(features=["pdsi"], lag_list=[5])
