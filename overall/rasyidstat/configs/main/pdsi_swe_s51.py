from configs.main.pdsi_swe import cfg

cfg["cds"] = dict(s51=dict(features=["t2m", "sd", "tprate"], agg_cols=["mean", "p10_p90"], use_same_ens=True))
