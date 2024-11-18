import pickle

import numpy as np
import pandas as pd
from functools import cache
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from queue import PriorityQueue
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from copy import deepcopy

from libs.data.rawdata import RawData, _snotel_aggwater_col_raw, _snotel_curr_swe_col_raw
from libs.multiscaler import MultiScaler

class NormalizedData:
    def __init__(self):
        self.raw: RawData = None
        self.scaler = None
        self.siteid_to_idx_dict = dict()
        self.idx_to_siteid_dict = dict()

    def setRawData(self, raw: RawData):
        self.raw = raw

        site_ids = sorted(list(set(raw.target['site_id'].tolist())))
        for idx, siteid in enumerate(site_ids):
            self.siteid_to_idx_dict[siteid] = idx
            self.idx_to_siteid_dict[idx] = siteid

    def initNormalization(self):
        if self.raw is None:
            return

        if self.scaler is None:
            self.scaler = MultiScaler()
            self.scaler.fit(self.raw.target)

        self.init_snotel_normalization()

    def siteid_to_idx(self, site_id: str) -> int:
        return self.siteid_to_idx_dict[site_id]

    def saveNormalizationModelsToFile(self, filename: str):
        normParams = {'scaler': self.scaler,
                      'best_snotel_stations_siteid': self.best_snotel_stations_siteid,
                      'snotel_normalization_model_siteid': self.snotel_normalization_model_siteid,
                      'siteid_to_idx_dict': self.siteid_to_idx_dict,
                      'idx_to_siteid_dict': self.idx_to_siteid_dict
                      }
        pickle.dump(normParams, open(filename, 'wb'))

    def loadNormalizationModelsFromFile(self, filename:str):
        normParams = pickle.load(open(filename, 'rb'))
        self.scaler = normParams['scaler']
        self.best_snotel_stations_siteid = normParams['best_snotel_stations_siteid']
        self.snotel_normalization_model_siteid = normParams['snotel_normalization_model_siteid']
        self.siteid_to_idx_dict = normParams['siteid_to_idx_dict']
        self.idx_to_siteid_dict = normParams['idx_to_siteid_dict']

    def saveNormalizedDataToFile(self, filename: str):
        pass


    def loadNormalidedDataFromFile(self, filename: str):
        pass

    def all_site_ids(self) -> List[str]:
        return list(self.siteid_to_idx_dict.keys())

    @cache
    def season_months_for_site_id(self, site_id):
        metadata = self.raw.metadata
        metadata_site = metadata[metadata.site_id == site_id]
        return metadata_site.season_start_month.iloc[0], metadata_site.season_end_month.iloc[0]

    @cache
    def train_data_normalized(self) -> pd.DataFrame:
        return self.scaler.transform_df(self.raw.target)

    @cache
    def train_monthly_flow_normalized(self) -> pd.DataFrame:
        return self.scaler.transform_df(self.raw.monthly_flow)

    def train_monthly_volume_samples_site_year_normalized(self, site_id: str, forecast_year: int) -> pd.DataFrame:
        df = self.raw.monthly_flow
        df_site_year = df[(df.site_id==site_id) & (df.forecast_year==forecast_year)].copy()
        if len(df_site_year):
            return self.scaler.transform_df(df_site_year)
        else:
            return df_site_year

    def daily_volume_samples_site_year_normalized(self, site_id: str, forecast_year: int) -> pd.DataFrame:
        df = self.raw.daily_flow
        df_site_year = df[(df.site_id==site_id) & (df.forecast_year==forecast_year)].copy()
        if len(df_site_year):
            return self.scaler.transform_df(df_site_year)
        return df_site_year

    def daily_usbr_volume_samples_site_year_normalized(self, site_id: str, forecast_year: int) -> pd.DataFrame:
        df = self.raw.daily_flow_usbr
        df_site_year = df[(df.site_id==site_id) & (df.forecast_year==forecast_year)].copy()
        if len(df_site_year):
            return self.scaler.transform_df(df_site_year, value_column='value')
        return df_site_year

    def volume_scale_factor(self, site_id: str) -> float:
        return self.scaler.scalers[site_id].scale_[0]

    @cache
    def snotel_stations_for_site_id(self, site_id: str) -> List[str]:
        df = self.raw.snotel_stations_info
        res = df[df.site_id == site_id].stationTriplet.tolist()
        res = [x.replace(':', '_') for x in res]

        return res

    def init_snotel_normalization(self):
        self.best_snotel_stations_siteid = dict()
        self.snotel_normalization_model_siteid = dict()

        train_df = self.train_data_normalized()

        # form dict site_id->station->year->max_agg_water
        agg_waters = {}
        all_years = train_df["year"].unique().tolist()
        for site_id in tqdm(train_df["site_id"].unique(), desc='Load raw SNOTEL values for end of season'):
            agg_waters[site_id] = self._read_all_snotels_max_by_year(site_id, all_years)

        for site_id, train_siteid in tqdm(train_df.groupby('site_id'), total=26, desc='init SNOTEL normalization', disable=True):
            best_snotel_stations = None
            best_snotel_lr_model = None
            min_years = 30

            years = train_siteid["year"].values
            targets = train_siteid["volume"].values
            q = PriorityQueue()

            def add_lr_by_stations(stations: List[str]) -> None:
                xs = np.array([
                    [
                        agg_waters[site_id][station].get(year)
                        for station in stations
                    ]
                    for year in years
                ])
                mask = [np.all([subx is not None for subx in x]) for x in xs]
                xs = xs[mask]
                ys = targets[mask]
                if len(xs) < min_years:
                    return

                lr = RANSACRegressor(random_state=0)
                lr.fit(xs, ys)
                pred = lr.predict(xs)
                rmse = math.sqrt(mean_squared_error(ys, pred))
                q.put((rmse, stations, lr))

            # print("="*20)
            for station in agg_waters[site_id].keys():
                add_lr_by_stations([station])

            if len(q.queue):
                for _ in range(5):
                    best_lr_rmse, best_snotel_stations, best_snotel_lr_model = deepcopy(q.queue[0])
                    print(f"{site_id=} {best_snotel_stations=}, {best_lr_rmse=}")
                    # print(f"{best_snotel_stations=}, {best_lr_rmse=}, coefs: {best_snotel_lr_model.coef_}, {best_snotel_lr_model.intercept_}")
                    station_lists = [deepcopy(el[1]) for el in q.queue[1:]]
                    for stations in station_lists:
                        new_stations = list(set(best_snotel_stations + stations))
                        if len(new_stations) > len(best_snotel_stations) and len(new_stations) > len(stations):
                            try:
                                add_lr_by_stations(new_stations)
                            except:
                                pass    # happens when the combination was already checked

                best_lr_rmse, best_snotel_stations, best_snotel_lr_model = q.queue[0]
                print(f"{best_snotel_stations=}, {best_lr_rmse=}")
                # print(f"{best_snotel_stations=}, {best_lr_rmse=}, coefs: {best_snotel_lr_model.coef_}, {best_snotel_lr_model.intercept_}")

            self.best_snotel_stations_siteid[site_id] = best_snotel_stations
            self.snotel_normalization_model_siteid[site_id] = best_snotel_lr_model

        all_used_snotel_stations = []
        for _, v in self.best_snotel_stations_siteid.items():
            all_used_snotel_stations.extend(v)
        self.raw.snotel = self.raw.snotel.query('snotel_station in @all_used_snotel_stations').copy()
        self.raw.snotel.sort_index(inplace=True)



    def _read_all_snotels_max_by_year(
            self,
            site_id: str,
            years: List[int],
            filter_year: Optional[int] = None
    ) -> Dict[str, Dict[int, float]]:
        res = {}
        snotel_stations = self.snotel_stations_for_site_id(site_id)

        if filter_year is not None:
            snotel_stations = list(
                filter(
                lambda s: self.read_snotel_df(s, filter_year) is not None,
                snotel_stations
                )
            )

        snotel = self.raw.snotel
        # snotel_subset = snotel[snotel.snotel_station.isin(snotel_stations) & snotel.forecast_year.isin(years)]
        snotel_subset = snotel.query('forecast_year in @years & snotel_station in @snotel_stations')
        # snotel_subset = snotel.query()

        for (station, year), group in snotel_subset.groupby(['snotel_station', 'forecast_year']):
            if station not in res:
                res[station] = {}
            res[station][year] = group['agg_water'].values.max()

        return res

    def read_snotel_df(self, snotel_station: str, year: int) -> Optional[pd.DataFrame]:
        df = self.raw.snotel
        df = df[(df['snotel_station']==snotel_station) & df['forecast_year']==year].copy()

        if len(df) == 0:
            return None

        return df

    def snotel_data_siteid_year_normalized(self, site_id: str, year: int) -> pd.DataFrame:
        snotel_stations = self.best_snotel_stations_siteid[site_id]
        doys, agg, curr = self.snotel_data_multistations_year(snotel_stations, year)
        if doys is None:
            res = pd.DataFrame({})
            return res

        lr = self.snotel_normalization_model_siteid[site_id]
        snotel_data = pd.DataFrame({
            "doy": doys,
            'agg_water': lr.predict(agg),
            'curr_snowwater': lr.predict(curr)
        })
        return snotel_data

    def snotel_data_multistations_year(self, snotel_stations: List[str], year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        snotel = self.raw.snotel
        # snotel_subset = snotel[(snotel.forecast_year==year) & snotel.snotel_station.isin(snotel_stations)]
        if year not in snotel.index:
            return None, None, None
        snotel_subset = snotel.loc[year].query('snotel_station in @snotel_stations')

        dfs = [
            # snotel_subset[snotel_subset.snotel_station==station]
            snotel_subset.query('snotel_station == @station')
            for station in snotel_stations
        ]

        for df in dfs:
            if (df is None) or (len(df)==0):
                return None, None, None

        dfs = [
            df[['date', 'agg_water', 'curr_snowwater', 'doy']]
            for df in dfs
        ]

        # add doys
        all_doys = []
        all_agg = []
        all_curr = []
        for df in dfs:
            df = df.sort_values('doy')
            all_doys.append(df["doy"].values)
            all_agg.append(df["agg_water"].values)
            all_curr.append(df["curr_snowwater"].values)

        res_doy = []
        res_agg = []
        res_curr = []

        def next_pos(pos: List[int]) -> Optional[List[int]]:
            min_d = None
            min_st_incr = None
            for i in range(len(pos)):
                if pos[i] < len(all_doys[i])-1:
                    if min_d is None or min_d > all_doys[i][pos[i]+1]:
                        min_d = all_doys[i][pos[i]+1]
                        min_st_incr = i
            if min_d is None:
                return None
            new_pos = deepcopy(pos)
            new_pos[min_st_incr] += 1
            return new_pos

        curr_pos = [0 for _ in snotel_stations]
        while curr_pos is not None:
            res_doy.append(np.max([l[curr_pos[i]] for i, l in enumerate(all_doys)]))
            res_agg.append([l[curr_pos[i]] for i, l in enumerate(all_agg)])
            res_curr.append([l[curr_pos[i]] for i, l in enumerate(all_curr)])
            curr_pos = next_pos(curr_pos)

        return np.array(res_doy), np.array(res_agg), np.array(res_curr)

