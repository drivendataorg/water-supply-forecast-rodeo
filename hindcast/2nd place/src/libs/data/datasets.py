import os.path
from typing import Dict, Any, List, Tuple, Sequence, Optional, Literal
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
import pandas as pd

from libs.data.TimeSeriesSample import TimeSeriesSample
from libs.DataHandler import DataHandler
from libs.data.timeseries import getTimeseries, prepare_validation_timeseries, getTestTimeseries


class TimeseriesBasedDataset(Dataset):
    def __init__(self,
                 samples: List[TimeSeriesSample],
                 tot_samples: Optional[int] = None,
                 augment_doys: bool = False,
                 num_sites: int = 26,
                 min_doy=0 ):
        super(TimeseriesBasedDataset).__init__()

        self.timeseries_samples = samples
        self.tot_samples = tot_samples
        self.augment_doys = augment_doys
        self.num_sites = num_sites
        self.min_doy = min_doy

    def __len__(self):
        return self.tot_samples if self.tot_samples is not None else len(self.timeseries_samples)

    def __getitem__(self, item):
        if self.tot_samples is not None:
            item = np.random.randint(0, len(self.timeseries_samples))

        timeseries = self.timeseries_samples[item]
        if self.augment_doys:
            curr_doy = np.random.randint(self.min_doy, timeseries.season_end_doy)
            #range_min, range_max = max(self.min_doy, curr_doy-15), min(curr_doy+15, timeseries.season_end_doy)
            #curr_doy = np.random.randint(range_min, range_max, size=3)

            timeseries = timeseries.filter_by_forecast_doy(curr_doy)

        siteid_onehot = np.zeros(self.num_sites, dtype=np.float32)
        siteid_onehot[timeseries.site_id_int] = 1

        features = []

        # monthly features
        if (timeseries.monthly_volumes_doy is not None) and len(timeseries.monthly_volumes_doy):
            waterseason_tot_volume = np.sum(timeseries.monthly_volumes)
            waterseason_data_avaiable_ratio = (timeseries.monthly_volumes_doy[-1] + 1) / (timeseries.season_end_doy + 1)

            season_volumes, season_doys = self._filter_season_data(timeseries.monthly_volumes,
                                                                  timeseries.monthly_volumes_doy,
                                                                  timeseries.season_start_doy)

            season_tot_volume = np.sum(season_volumes)
            if len(season_doys):
                season_data_avaiable_ratio = (season_doys[-1] + 1 - timeseries.season_start_doy) / (timeseries.season_end_doy - timeseries.season_start_doy + 1)
            else:
                season_data_avaiable_ratio = 0

            features.extend([waterseason_tot_volume, waterseason_data_avaiable_ratio, season_tot_volume, season_data_avaiable_ratio])
        else:
            features.extend([0, 0, 0, 0])


        # daily features
        if (timeseries.daily_volumes_doy is not None) and len(timeseries.daily_volumes_doy):
            waterseason_tot_volume = np.sum(timeseries.daily_volumes)
            waterseason_data_avaiable_ratio = (timeseries.daily_volumes_doy[-1] + 1) / (timeseries.season_end_doy + 1)

            season_volumes, season_doys = self._filter_season_data(timeseries.daily_volumes,
                                                                  timeseries.daily_volumes_doy,
                                                                  timeseries.season_start_doy)

            season_tot_volume = np.sum(season_volumes)
            if len(season_doys):
                season_data_avaiable_ratio = (season_doys[-1] + 1 - timeseries.season_start_doy) / (
                            timeseries.season_end_doy - timeseries.season_start_doy + 1)
            else:
                season_data_avaiable_ratio = 0

            features.extend([waterseason_tot_volume, waterseason_data_avaiable_ratio, season_tot_volume,
                             season_data_avaiable_ratio])
        else:
            features.extend([0, 0, 0, 0])

        # SNOTEL features
        if (timeseries.snotel_doy is not None) and len(timeseries.snotel_doy):
            aggwater = timeseries.snotel_agg_water.max()
            snoweater_curr = timeseries.snotel_snowwater[-1]
            snowwater_max = timeseries.snotel_snowwater.max()
            waterseason_ratio = (timeseries.snotel_doy[-1] + 1) / (timeseries.season_end_doy + 1)
            if timeseries.snotel_doy[-1] >= timeseries.season_start_doy:
                season_ratio = (timeseries.snotel_doy[-1] + 1 - timeseries.season_start_doy) / (
                            timeseries.season_end_doy - timeseries.season_start_doy + 1)
            else:
                season_ratio = 0

            features.extend([1, aggwater, snoweater_curr, snowwater_max, waterseason_ratio, season_ratio])
        else:
            features.extend([0, 0, 0, 0, 0, 0])

        features = np.asarray(features, dtype=np.float32)
        X = np.concatenate([siteid_onehot, np.asarray(features, dtype=np.float32)])

        res = {'X': X,
               'volume_scale_factor': timeseries.volume_scale_factor}
        if timeseries.target is not None:
            res['target'] = timeseries.target

        return res

    def _filter_season_data(self, volumes: np.ndarray, doys: np.ndarray, season_start_doy: int) -> Tuple[np.ndarray,np.ndarray]:
        mask = doys >= season_start_doy
        return volumes[mask], doys[mask]


def _create_dataloader(opt: Dict[str, Any]) -> DataLoader:
    ds_type = opt.pop("type", "raw")

    batch_size = opt.pop("batch_size")
    shuffle = opt.pop("shuffle", False)
    workers = opt.pop("workers", 0)
    drop_last = opt.pop("drop_last", False)

    if ds_type == "timeseries":
        ds = TimeseriesBasedDataset(**opt)
    else:
        raise ValueError(f"Unknown dataset type {ds_type}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        #num_workers=0,
        drop_last=drop_last,
    )


def filter_timeseries(all_data: List[TimeSeriesSample], years: np.ndarray) ->List[TimeSeriesSample]:
    return list(filter(lambda x: x.forecast_year in years, all_data))


def create_dataloaders(opt: Dict[str, Any]) -> List[DataLoader]:
    result = []

    snotel_path = opt["snotel_path"]
    scaler_path = opt.get("scaler_path", None)
    src_dir = opt.get("src_dir", None)
    if src_dir is not None:
        snotel_path = os.path.join(src_dir, snotel_path)
        if scaler_path is not None:
            scaler_path = os.path.join(src_dir, scaler_path)

    if hasattr(create_dataloaders, "datahandler"):
        data = create_dataloaders.datahandler
    else:
        base_dir = opt.get("base_dir", "data")
        if "test_data" in opt:
            data = DataHandler(snotel_pretrained_path=snotel_path, base_directory=base_dir)
        else:
            data = DataHandler(base_directory=base_dir)
            data.init_snotel_normalization()
            data.save_snotel_models(snotel_path)

        create_dataloaders.datahandler = data

    full_train_timeseries = None
    train_folds = None
    cross_val_parts = opt.get("cross_val_parts", 5)
    cross_val_repeats = opt.get("cross_val_repeats", 1)
    cv_fold_pos = opt.get("cv_fold_pos", 0)

    if "train_data" in opt:
        if full_train_timeseries is None:
            if hasattr(create_dataloaders, "full_train_timeseries"):
                full_train_timeseries = create_dataloaders.full_train_timeseries
            else:
                full_train_timeseries = getTimeseries(data, 'train')
                create_dataloaders.full_train_timeseries = full_train_timeseries
        if train_folds is None:
            train_folds = data.kfold_train_test_spits_by_years(cross_val_parts, cross_val_repeats)

        train_years, _ = train_folds[cv_fold_pos]
        train_timeseries = filter_timeseries(full_train_timeseries, train_years)
        if scaler_path and cv_fold_pos==0:
            data.save_scaler(scaler_path)

        conf = deepcopy(opt["train_data"])
        len_mult = conf.pop("len_mult", 100)
        native_samples = conf.pop("native_samples", False)
        conf["samples"] = train_timeseries
        conf["tot_samples"] = None if native_samples else len(train_timeseries) * len_mult
        conf["augment_doys"] = True
        result.append(_create_dataloader(conf))

    if "val_data" in opt:
        if full_train_timeseries is None:
            if hasattr(create_dataloaders, "full_train_timeseries"):
                full_train_timeseries = create_dataloaders.full_train_timeseries
            else:
                full_train_timeseries = getTimeseries(data, 'train')
                create_dataloaders.full_train_timeseries = full_train_timeseries
        if train_folds is None:
            train_folds = data.kfold_train_test_spits_by_years(cross_val_parts, cross_val_repeats)

        _, val_years = train_folds[cv_fold_pos]
        val_timeseries = filter_timeseries(full_train_timeseries, val_years)
        val_timeseries = prepare_validation_timeseries(val_timeseries)

        conf = deepcopy(opt["val_data"])
        len_mult = conf.pop("len_mult", 10)
        native_samples = conf.pop("native_samples", True)
        conf["samples"] = val_timeseries
        conf["tot_samples"] = None if native_samples else len(val_timeseries) * len_mult
        conf["augment_doys"] = False
        result.append(_create_dataloader(conf))

    if "test_data" in opt:
        data.load_scaler(scaler_path)
        test_timeseries = getTestTimeseries(data)

        conf = deepcopy(opt["test_data"])
        conf["samples"] = test_timeseries
        conf["tot_samples"] = None
        conf["augment_doys"] = False
        result.append(_create_dataloader(conf))

    return result
