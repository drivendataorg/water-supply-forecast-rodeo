import numpy as np
from dataclasses import dataclass
from typing import List, Union

@dataclass
class TimeSeriesSample:
    site_id: str
    site_id_int: int
    forecast_year: int

    season_start_month: int
    season_end_month: int
    season_start_doy: int
    season_end_doy: int

    volume_scale_factor: float

    monthly_volumes: np.ndarray = None
    monthly_volumes_doy: np.ndarray = None

    daily_volumes: np.ndarray = None
    daily_volumes_doy: np.ndarray = None

    daily_usbr_volumes: np.ndarray = None
    daily_usbr_volumes_doy: np.ndarray = None

    snotel_agg_water: np.ndarray = None
    snotel_snowwater: np.ndarray = None
    snotel_doy: np.ndarray = None

    cdec_snowwater: np.ndarray = None
    cdec_doy: np.ndarray = None

    target: int = None
    issue_date: str = None

    def filter_by_forecast_doy(self, forecast_doy: Union[int, List[int], np.ndarray]):
        if isinstance(forecast_doy, int):
            forecast_doy = [forecast_doy]*4

        sample = TimeSeriesSample(
            site_id=self.site_id,
            site_id_int=self.site_id_int,
            forecast_year = self.forecast_year,
            season_start_month=self.season_start_month,
            season_end_month = self.season_end_month,
            season_start_doy=self.season_start_doy,
            season_end_doy=self.season_end_doy,
            volume_scale_factor=self.volume_scale_factor,
            target=self.target)

        if self.monthly_volumes is not None:
            mask = self.monthly_volumes_doy < forecast_doy[0]
            sample.monthly_volumes = self.monthly_volumes[mask]
            sample.monthly_volumes_doy = self.monthly_volumes_doy[mask]

        if self.daily_volumes is not None:
            mask = self.daily_volumes_doy < forecast_doy[1]
            sample.daily_volumes = self.daily_volumes[mask]
            sample.daily_volumes_doy = self.daily_volumes_doy[mask]

        if self.daily_usbr_volumes is not None:
            mask = self.daily_usbr_volumes_doy < forecast_doy[2]
            sample.daily_usbr_volumes = self.daily_usbr_volumes[mask]
            sample.daily_usbr_volumes_doy = self.daily_usbr_volumes_doy[mask]

        if self.snotel_agg_water is not None:
            mask = self.snotel_doy < forecast_doy[3]
            sample.snotel_agg_water = self.snotel_agg_water[mask]
            sample.snotel_snowwater = self.snotel_snowwater[mask]
            sample.snotel_doy = self.snotel_doy[mask]

        return sample
