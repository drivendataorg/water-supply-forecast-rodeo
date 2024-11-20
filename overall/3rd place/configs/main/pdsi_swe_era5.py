from configs.main.pdsi_swe import cfg

cfg["cds"] = dict(
    era5=dict(
        features=["t2m", "lai_hv", "lai_lv", "asn", "snowc", "sd", "swvl1", "swvl2", "swvl3", "swvl4"],
        hour=0,
    )
)
