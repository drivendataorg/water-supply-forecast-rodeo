import numpy as np
import pandas as pd
from libs.multiscaler import MultiScaler
from functools import cache
from sklearn.model_selection import KFold, RepeatedKFold
from datetime import date, timedelta
import calendar
import os
from copy import deepcopy
import pickle
from typing import List, Dict, Union, Optional, Tuple
from tqdm import tqdm
import math
from queue import PriorityQueue
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def day_end_of_month(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])

def date_to_doy(ref_date: date, curr_date: date):
    return (curr_date - ref_date).days


class DataHandler:
    _snotel_aggwater_col_raw = 'PREC_DAILY'
    _snotel_curr_swe_col_raw = 'WTEQ_DAILY'

    _cdec_sensor_type = 'sensorType'
    _cdec_date = 'date'
    _cdec_value = 'value'
    _cdec_SNOWWC = 'SNOW WC'
    _cdec_SNOADJ = 'SNO ADJ'

    def __init__(
            self,
            snotel_pretrained_path: Optional[str] = None,
            base_directory: str = "data"
    ):
        self.scaler = None
        self.siteid_to_idx_dict = dict()
        self.idx_to_siteid_dict = dict()
        self.base_dir = base_directory

        if snotel_pretrained_path is not None:
            self.load_snotel_models(snotel_pretrained_path)
        self.snotel_normalized_cache = {}

    @cache
    def train_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.base_dir, 'train.csv')).dropna()

        site_ids = sorted(list(set(df['site_id'].tolist())) )
        for idx, siteid in enumerate(site_ids):
            self.siteid_to_idx_dict[siteid] = idx
            self.idx_to_siteid_dict[idx] = siteid

        return df

    def save_scaler(self, path: str) -> None:
        self.scaler.save_to_file(path)

    def load_scaler(self, path: str) -> None:
        self.scaler = MultiScaler.from_file(path)

        if len(self.siteid_to_idx_dict) == 0:
            site_ids = sorted(list(set(self.scaler.scalers.keys())))
            for idx, siteid in enumerate(site_ids):
                self.siteid_to_idx_dict[siteid] = idx
                self.idx_to_siteid_dict[idx] = siteid

    @cache
    def train_data_normalized(self) -> pd.DataFrame:
        df = self.train_data()

        if self.scaler is None:
            self.scaler = MultiScaler()
            self.scaler.fit(df)

        return self.scaler.transform_df(df)

    @cache
    def metadata(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.base_dir, 'metadata.csv'))
        return df

    @cache
    def train_monthly_flow(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.base_dir, 'train_monthly_naturalized_flow.csv')).dropna()
        return df

    @cache
    def train_monthly_flow_normalized(self) -> pd.DataFrame:
        df = self.train_monthly_flow()
        return self.scaler.transform_df(df)

    @cache
    def test_monthly_flow_normalized(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.base_dir, 'test_monthly_naturalized_flow.csv')).dropna()
        return self.scaler.transform_df(df)

    @cache
    def kfold_train_test_spits_by_years(self, folds: int = 5, repeats: int = 1):
        train_df = self.train_data_normalized()
        unique_years = np.unique(train_df['year'])

        kfold = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=0)
        splits = []
        for i, (train_index, test_index) in enumerate(kfold.split(unique_years)):
            train_years = unique_years[train_index]
            val_years = unique_years[test_index]

            splits.append((train_years, val_years))

        return splits

    @cache
    def season_months_for_site_id(self, site_id):
        metadata = self.metadata()
        metadata_site = metadata[metadata.site_id==site_id]

        return metadata_site.season_start_month.iloc[0], metadata_site.season_end_month.iloc[0]

    def train_monthly_volume_samples_site_year(self, site_id: str, forecast_year: int) -> pd.DataFrame:
        df = self.train_monthly_flow()
        df_site_year = df[(df.site_id == site_id) & (df.forecast_year == forecast_year)].copy()

        ref_date = date(forecast_year - 1, 10, 1)
        doys = []

        for _, row in df_site_year.iterrows():
            dt = day_end_of_month(row.year, row.month)
            doy = date_to_doy(ref_date, dt)
            doys.append(doy)

        df_site_year['doy'] = doys
        return df_site_year.sort_values('doy')

    def train_monthly_volume_samples_site_year_normalized(self, site_id: str, forecast_year: int) -> pd.DataFrame:
        df = self.train_monthly_volume_samples_site_year(site_id, forecast_year)

        if len(df)==0:
            return df

        return self.scaler.transform_df(df)

    def test_monthly_volume_samples_site_year_normalized(self, site_id: str, forecast_year: int) -> pd.DataFrame:
        df = self.test_monthly_flow_normalized()
        df_site_year = df[(df.site_id==site_id) & (df.forecast_year==forecast_year)].copy()

        ref_date = date(forecast_year-1, 10, 1)
        doys = []

        for _, row in df_site_year.iterrows():
            dt = day_end_of_month(row.year, row.month)
            doy = date_to_doy(ref_date, dt)
            doys.append(doy)

        df_site_year['doy'] = doys
        return df_site_year.sort_values('doy')

    @cache
    def train_target_percentile_for_site(self, site_id: str, percentile: float) -> float:
        df = self.train_data_normalized()
        df_site = df[df.site_id==site_id]

        return np.percentile(df_site['volume'], percentile)

    def daily_volume_samples_site_year(self, site_id: str, forecast_year: int) -> pd.DataFrame:
        filename = os.path.join(self.base_dir, f'usgs_streamflow/FY{forecast_year}/{site_id}.csv')
        if not os.path.exists(filename):
            return pd.DataFrame({})

        ds = pd.read_csv(filename).dropna()
        if len(ds)==0:
            return pd.DataFrame({})

        ds['site_id'] = site_id
        ds['date'] = ds['datetime'].str[0:10]

        CUBIC_FOOT_m3 = 0.028316846592
        ACREFOOT_m3 = 1233.48183754752
        CFs_to_KAFd = CUBIC_FOOT_m3 * 3600 * 24 / ACREFOOT_m3 / 1000

        ds['volume'] = ds['00060_Mean']*CFs_to_KAFd

        ref_date = date(forecast_year - 1, 10, 1)
        doys = []

        for _, row in ds.iterrows():
            dt = date.fromisoformat(row.date)
            doy = date_to_doy(ref_date, dt)
            doys.append(doy)

        ds['doy'] = doys

        return ds.sort_values('doy')

    def daily_volume_samples_site_year_normalized(self, site_id: str, forecast_year: int) -> pd.DataFrame:
        ds = self.daily_volume_samples_site_year(site_id, forecast_year)
        if len(ds):
            ds = self.scaler.transform_df(ds)
        return ds

    def siteid_to_idx(self, site_id: str) -> int:
        return self.siteid_to_idx_dict[site_id]

    def volume_scale_factor(self, site_id: str) -> float:
        return self.scaler.scalers[site_id].scale_[0]

    @cache
    def snotel_stations_for_site_id(self, site_id: str) -> List[str]:
        df = pd.read_csv(os.path.join(self.base_dir, 'snotel/sites_to_snotel_stations.csv'))
        res = df[df.site_id==site_id].stationTriplet.tolist()
        res = [x.replace(':', '_') for x in res]

        return res
    
    def read_snotel_df(self, snotel_station: str, year: int) -> Optional[pd.DataFrame]:
        filename = os.path.join(self.base_dir, 'snotel', f'FY{year}', f'{snotel_station}.csv')
        if not os.path.exists(filename):
            return None

        df = pd.read_csv(filename)
        if (self._snotel_aggwater_col_raw not in df.columns) or (self._snotel_curr_swe_col_raw not in df.columns):
            return None

        df = df.dropna(subset=[self._snotel_aggwater_col_raw, self._snotel_curr_swe_col_raw])
        if len(df) == 0:
            return None

        return df

    def snotel_data_station_year(self, snotel_station: str, year: int, add_doy: bool) -> Optional[pd.DataFrame]:
        df = self.read_snotel_df(snotel_station, year)
        if df is None:
            return None

        res = pd.DataFrame({
            'date': df['date'],
            'agg_water': df[self._snotel_aggwater_col_raw].values,
            'curr_snowwater': df[self._snotel_curr_swe_col_raw].values
        })

        if add_doy:
            ref_date = date(year - 1, 10, 1)
            doys = []

            for _, row in res.iterrows():
                dt = date.fromisoformat(row.date)
                doy = date_to_doy(ref_date, dt)
                doys.append(doy)

            res['doy'] = doys
            res = res.sort_values('doy')

        return res

    def _read_all_snotels_max_by_year(
            self,
            site_id: str,
            years: List[int]
    ) -> Dict[str, Dict[int, float]]:
        res = {}
        snotel_stations = self.snotel_stations_for_site_id(site_id)
        for station in snotel_stations:
            res[station] = {}
            for year in years:
                df = self.read_snotel_df(station, year)
                if df is not None:
                    res[station][year] = df[self._snotel_aggwater_col_raw].values.max()

        return res

    def init_snotel_normalization(self):
        self.best_snotel_stations_siteid = dict()
        self.snotel_normalization_model_siteid = dict()

        train_df = self.train_data_normalized()

        # form dict site_id->station->year->max_agg_water
        agg_waters = {}
        all_years = train_df["year"].unique().tolist()
        for site_id in tqdm(train_df["site_id"].unique()):
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

                lr = RANSACRegressor()
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
                    # print(f"{best_snotel_stations=}, {best_lr_rmse=}")
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

    def save_snotel_models(self, filename: str):
        pickle.dump(
            {
                "stations": self.best_snotel_stations_siteid,
                "models": self.snotel_normalization_model_siteid
            },
            open(filename, 'wb')
        )

    def load_snotel_models(self, filename: str):
        loaded_models = pickle.load(open(filename, 'rb'))
        self.best_snotel_stations_siteid = loaded_models["stations"]
        self.snotel_normalization_model_siteid = loaded_models["models"]

    def snotel_data_multistations_year(self, snotel_stations: List[str], year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dfs = [
            self.read_snotel_df(station, year)
            for station in snotel_stations
        ]

        if not np.all([df is not None for df in dfs]):
            return None, None, None

        dfs = [
            pd.DataFrame({
                'date': df['date'],
                'agg_water': df[self._snotel_aggwater_col_raw].values,
                'curr_snowwater': df[self._snotel_curr_swe_col_raw].values
            })
            for df in dfs
        ]

        # add doys
        ref_date = date(year - 1, 10, 1)
        all_doys = []
        all_agg = []
        all_curr = []
        for df in dfs:
            doys = []
            for _, row in df.iterrows():
                dt = date.fromisoformat(row.date)
                doy = date_to_doy(ref_date, dt)
                doys.append(doy)

            df['doy'] = doys
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

    def snotel_data_siteid_year_normalized(self, site_id: str, year: int) -> pd.DataFrame:
        if site_id not in self.snotel_normalized_cache:
            self.snotel_normalized_cache[site_id] = {}

        if year in self.snotel_normalized_cache[site_id]:
            return self.snotel_normalized_cache[site_id][year]

        snotel_stations = self.best_snotel_stations_siteid[site_id]
        doys, agg, curr = self.snotel_data_multistations_year(snotel_stations, year)
        if doys is None:
            res = pd.DataFrame({})
            self.snotel_normalized_cache[site_id][year] = res
            return res

        lr = self.snotel_normalization_model_siteid[site_id]
        snotel_data = pd.DataFrame({
            "doy": doys,
            'agg_water': lr.predict(agg),
            'curr_snowwater': lr.predict(curr)
        })
        self.snotel_normalized_cache[site_id][year] = snotel_data
        return snotel_data

    @cache
    def cdec_stations_for_site_id(self, site_id: str, in_basin_only = True) -> List[str]:
        df = pd.read_csv(os.path.join(self.base_dir, 'cdec/sites_to_cdec_stations.csv'))
        if in_basin_only:
            df = df[df.in_basin==True]

        res = df[df.site_id == site_id]['station_id'].tolist()

        return res

    def read_cdec_df(self, cdec_station: str, year: int) -> Optional[pd.DataFrame]:
        filename = os.path.join(self.base_dir, 'cdec', f'FY{year}', f'{cdec_station}.csv')
        if not os.path.exists(filename):
            return None

        df = pd.read_csv(filename)
        if len(df)==0:
            return None

        if self._cdec_sensor_type not in df.columns:
            return None

        dataType = df[self._cdec_sensor_type].tolist()
        if (self._cdec_SNOADJ not in dataType) and ((self._cdec_SNOWWC not in dataType)):
            return None

        df = df.dropna(subset=[self._cdec_sensor_type, self._cdec_value])
        df = df[df[self._cdec_value]>=0]
        if len(df) == 0:
            return None

        return df

    def cdec_data_station_year(self, cdec_station: str, year: int, add_doy: bool) -> Optional[pd.DataFrame]:
        df = self.read_cdec_df(cdec_station, year)
        if df is None:
            return None

        if self._cdec_SNOADJ in df[self._cdec_sensor_type].tolist():
            df = df[df[self._cdec_sensor_type] == self._cdec_SNOADJ]
        elif self._cdec_SNOWWC in df[self._cdec_sensor_type].tolist():
            df = df[df[self._cdec_sensor_type] == self._cdec_SNOWWC]
        else:
            return None

        res = pd.DataFrame({
            'date': df['date'],
            'curr_snowwater': df[self._cdec_value].values
        })

        if add_doy:
            ref_date = date(year - 1, 10, 1)
            doys = []

            for _, row in res.iterrows():
                date_str = row.date
                if ' ' in date_str:
                    date_str = date_str.split(' ')[0]
                date_tokens = date_str.split('-')
                dt = date(int(date_tokens[0]), int(date_tokens[1]), int(date_tokens[2]) )

                doy = date_to_doy(ref_date, dt)
                doys.append(doy)

            res['doy'] = doys
            res = res.sort_values('doy')

        return res

    def init_cdec_normalization(self):
        pass

    def cdec_data_siteid_year_normalized(self, site_id: str, year: int) -> pd.DataFrame:
        cdec_stations = self.cdec_stations_for_site_id(site_id)
        if len(cdec_stations)==0:
            return pd.DataFrame({})

        df = self.cdec_data_station_year(cdec_stations[0], year, add_doy=True)
        if df is None:
            return pd.DataFrame({})

        return df
