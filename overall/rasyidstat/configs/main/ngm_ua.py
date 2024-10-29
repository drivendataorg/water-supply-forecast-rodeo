from configs.main.base_swe import cfg

cfg["uaswe"] = dict(features=["uaswe_huc6", "uaprec_cml_huc6", "uaswe_huc8", "uaprec_cml_huc8", "uaswe"], lag_list=[1])
cfg.pop("swe", None)
