from configs.forecast.base import cfg

cfg["diff_gap"] = 0
cfg["filter_snotel_sites"] = True
cfg["target_prep"]["diff"] = False
cfg["target_prep"]["diff_extra"] = False
cfg["lgb_params"]["n_estimators"] = 500
cfg["lgb_reg_params"]["n_estimators"] = 1200
cfg["swe"] = None
cfg["pdsi"] = dict(features=["pdsi"], lag_list=[5])
