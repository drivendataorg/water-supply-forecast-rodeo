from configs.main.pdsi_swe import cfg

cfg["cds"] = dict(s51=dict(features=["t2m", "sd", "tprate"], agg_cols=["mean", "p10_p90"], use_same_ens=True))
cfg["uaswe"] = dict(features=["uaswe_huc6", "uaprec_cml_huc6", "uaswe_huc8", "uaprec_cml_huc8", "uaswe"], lag_list=[1])
cfg.pop("swe", None)
