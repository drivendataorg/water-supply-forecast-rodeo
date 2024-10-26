import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Union, List, Optional
import os
from functools import cache
from tqdm import tqdm
from datetime import date
from libs.data import metadata

@dataclass
class RawData:
    target: pd.DataFrame = None
    monthly_flow: pd.DataFrame = None
    daily_flow: pd.DataFrame = None
    daily_flow_usbr: pd.DataFrame = None
    snotel: pd.DataFrame = None

    metadata: pd.DataFrame = None
    snotel_stations_info: pd.DataFrame = None

@cache
def getAllYears(base_dir = 'data') -> np.ndarray:
    df_cv = pd.read_csv(os.path.join(base_dir, 'final_stage', 'cross_validation_labels.csv')).dropna()
    df_prev = pd.read_csv(os.path.join(base_dir, 'final_stage', 'prior_historical_labels.csv')).dropna()
    df = pd.concat([df_prev, df_cv])
    return np.unique(df.year.values)

def readAllDataForYears(years: Union[np.ndarray, int], base_dir:str = 'data') -> RawData:
    res = RawData()

    if not isinstance(years, np.ndarray):
        years = np.asarray([years], dtype=np.int32)

    res.target = _readTarget(years, base_dir)
    res.metadata = _readMetadata(base_dir)
    res.snotel_stations_info = _readSnotelInfo(base_dir)
    res.monthly_flow = _readMonthlyFlow(years, base_dir)

    site_ids = sorted(list(set(res.target['site_id'].tolist())))

    res.daily_flow = _readDailyFlow(years, site_ids, base_dir)
    res.daily_flow_usbr = _readDailyFlowUSBR(years)
    res.snotel = _readSnotel(years, site_ids=site_ids, base_dir=base_dir)
    return res

def _readTarget(years: np.ndarray, base_dir: str) -> pd.DataFrame:
    df_cv = pd.read_csv(os.path.join(base_dir, 'final_stage', 'cross_validation_labels.csv')).dropna()
    df_prev = pd.read_csv(os.path.join(base_dir, 'final_stage', 'prior_historical_labels.csv')).dropna()
    df = pd.concat([df_prev, df_cv])

    res = df[df.year.isin(years)].copy()
    return res

def _readMetadata(base_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(base_dir, 'final_stage', 'metadata.csv'))
    return df

def _readSnotelInfo(base_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(base_dir, 'snotel', 'sites_to_snotel_stations.csv'))
    return df


def _readMonthlyFlow(years: np.ndarray, base_dir: str) -> pd.DataFrame:
    df_cv = pd.read_csv(os.path.join(base_dir, 'final_stage', 'cross_validation_monthly_flow.csv')).dropna()
    df_prev = pd.read_csv(os.path.join(base_dir, 'final_stage', 'prior_historical_monthly_flow.csv')).dropna()
    df = pd.concat([df_prev, df_cv])

    doys = []
    for _, row in df.iterrows():
        ref_date = date(row.forecast_year - 1, 10, 1)
        dt = metadata.day_end_of_month(row.year, row.month)
        doy = metadata.date_to_doy(ref_date, dt)
        doys.append(doy)

    df['doy'] = doys

    res = df[df.forecast_year.isin(years)].copy()
    return res

def _readDailyFlow(years: np.ndarray, site_ids: List[str], base_dir: str) -> pd.DataFrame:
    res = []

    for year in tqdm(years, desc='Read daily flow data'):
        for site_id in site_ids:
            ds = _daily_volume_samples_site_year(site_id, forecast_year=year, base_dir=base_dir)
            res.append(ds)

    res = pd.concat(res)
    return res

def _readDailyFlowUSBR(years: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv('data/final_stage/USBR_reservoir_inflow.csv')
    df = df.query('forecast_year in @years').copy()
    df.drop_duplicates(subset=['site_id', 'forecast_year', 'date'], inplace=True)

    CUBIC_FOOT_m3 = 0.028316846592
    ACREFOOT_m3 = 1233.48183754752
    CFs_to_KAFd = CUBIC_FOOT_m3 * 3600 * 24 / ACREFOOT_m3 / 1000

    df.value = df.value*CFs_to_KAFd

    doys = []

    for _, row in df.iterrows():
        ref_date = date(row.forecast_year-1, 10, 1)
        dt = date.fromisoformat(row.date)
        doys.append(metadata.date_to_doy(ref_date, dt))
    df['doy'] = doys

    return df



def _readSnotel(years: np.ndarray, site_ids: List[str], base_dir: str) -> pd.DataFrame:
    res = []

    for year in tqdm(years, desc='Read SNOTEL data'):
        for site_id in site_ids:
            snotel_stations = _get_snotel_stations_for_siteid(site_id, base_dir)
            for _, snotel_info in snotel_stations.iterrows():
                df = _snotel_data_station_year(snotel_info['stationTriplet'].replace(':', '_'), forecast_year=year, base_dir=base_dir)
                if df is None:
                    continue

                df['agg_water'] = df['agg_water'].values.astype(np.float32)
                df['curr_snowwater'] = df['curr_snowwater'].values.astype(np.float32)

                df['site_id'] = site_id
                df['in_basin'] = snotel_info['in_basin']
                res.append(df)

    res = pd.concat(res)
    res = res.set_index(['forecast_year', 'snotel_station']).sort_index()
    return res

# Daily flow related functions

@cache
def _daily_volume_samples_site_year(site_id: str, forecast_year: int, base_dir: str) -> pd.DataFrame:
    filename = os.path.join(base_dir, f'usgs_streamflow/FY{forecast_year}/{site_id}.csv')
    if not os.path.exists(filename):
        return pd.DataFrame({})

    col_volume = '00060_Mean'

    ds = pd.read_csv(filename).dropna(subset=[col_volume, 'datetime'])
    ds = ds[ds[col_volume] >= 0]
    if len(ds)==0:
        return pd.DataFrame({})

    ds['site_id'] = site_id
    ds['forecast_year'] = forecast_year
    ds['date'] = ds['datetime'].str[0:10]

    CUBIC_FOOT_m3 = 0.028316846592
    ACREFOOT_m3 = 1233.48183754752
    CFs_to_KAFd = CUBIC_FOOT_m3 * 3600 * 24 / ACREFOOT_m3 / 1000

    ds['volume'] = ds['00060_Mean']*CFs_to_KAFd

    ref_date = date(forecast_year - 1, 10, 1)

    def _calc_doy(dt_str: str):
        dt = date.fromisoformat(dt_str)
        doy = metadata.date_to_doy(ref_date, dt)
        return doy

    ds['doy'] = ds['date'].apply(_calc_doy)


    # ref_date = date(forecast_year - 1, 10, 1)
    # doys = []
    #
    # for _, row in ds.iterrows():
    #     dt = date.fromisoformat(row.date)
    #     doy = metadata.date_to_doy(ref_date, dt)
    #     doys.append(doy)
    #
    # ds['doy'] = doys

    ds = ds[['site_id', 'forecast_year', 'date', 'doy', 'volume']]

    return ds.sort_values('doy')

# SNOTEL related functions

_snotel_aggwater_col_raw = 'PREC_DAILY'
_snotel_curr_swe_col_raw = 'WTEQ_DAILY'

@cache
def _get_snotel_stations_for_siteid(site_id: str, base_dir: str) -> pd.DataFrame:
    df = _readSnotelInfo(base_dir)
    return df[df.site_id==site_id].copy()

@cache
def _snotel_data_station_year(snotel_station: str, forecast_year: int, base_dir: str) -> Optional[pd.DataFrame]:
    df = _read_snotel_df(snotel_station, forecast_year, base_dir)
    if df is None:
        return None

    res = pd.DataFrame({
        'date': df['date'],
        'agg_water': df[_snotel_aggwater_col_raw].values,
        'curr_snowwater': df[_snotel_curr_swe_col_raw].values
    })

    ref_date = date(forecast_year - 1, 10, 1)
    def _calc_doy(dt_str: str):
        dt = date.fromisoformat(dt_str)
        doy = metadata.date_to_doy(ref_date, dt)
        return doy

    res['doy'] = res['date'].apply(_calc_doy)
    res['snotel_station'] = snotel_station
    res['forecast_year'] = forecast_year
    res = res.sort_values('doy')

    return res

def _read_snotel_df(snotel_station: str, year: int, base_dir: str) -> Optional[pd.DataFrame]:
    filename = os.path.join(base_dir, 'snotel', f'FY{year}', f'{snotel_station}.csv')
    if not os.path.exists(filename):
        return None

    df = pd.read_csv(filename)
    if (_snotel_aggwater_col_raw not in df.columns) or (_snotel_curr_swe_col_raw not in df.columns):
        return None

    df = df.dropna(subset=[_snotel_aggwater_col_raw, _snotel_curr_swe_col_raw])
    if len(df) == 0:
        return None

    return df