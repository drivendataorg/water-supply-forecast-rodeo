import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict, Union, Optional, Literal
from datetime import date
from libs.DataHandler import DataHandler, day_end_of_month, date_to_doy
from libs.data.TimeSeriesSample import TimeSeriesSample


def getTimeseries(data: DataHandler, mode: Literal['train', 'test']) -> List[TimeSeriesSample]:
    res = []

    if mode=='train':
        df = data.train_data_normalized()

    for site_id, site_df in tqdm(df.groupby('site_id'), total=26):
        season_start_month, season_end_month = data.season_months_for_site_id(site_id)

        for forecast_year, site_year_df in site_df.groupby('year'):
            ref_date = date(forecast_year - 1, 10, 1)
            season_start_date = date(forecast_year, season_start_month, 1)


            sample = TimeSeriesSample(site_id=site_id,
                                      site_id_int=data.siteid_to_idx(site_id),
                                      forecast_year=forecast_year,
                                      season_start_month=season_start_month,
                                      season_end_month=season_end_month,
                                      season_start_doy=date_to_doy(ref_date, season_start_date),
                                      season_end_doy=date_to_doy(ref_date, day_end_of_month(forecast_year, season_end_month)),
                                      volume_scale_factor=data.volume_scale_factor(site_id)
                                      )

            if mode == 'train':
                sample.target = site_year_df['volume'].iloc[0] if mode == 'train' else None

            # monthly timeseries
            if mode=='train':
                monthly_volume_samples = data.train_monthly_volume_samples_site_year_normalized(site_id, forecast_year)

            if len(monthly_volume_samples):
                sample.monthly_volumes = monthly_volume_samples['volume'].values
                sample.monthly_volumes_doy = monthly_volume_samples['doy'].values

            # daily timeseries
            daily_volumes_sample = data.daily_volume_samples_site_year_normalized(site_id, forecast_year)
            if len (daily_volumes_sample):
                sample.daily_volumes = daily_volumes_sample['volume'].values
                sample.daily_volumes_doy = daily_volumes_sample['doy'].values

            #SNOTEL timeseries
            snotel_sample = data.snotel_data_siteid_year_normalized(site_id, forecast_year)
            if len(snotel_sample):
                sample.snotel_agg_water = snotel_sample['agg_water'].values
                sample.snotel_snowwater = snotel_sample['curr_snowwater'].values
                sample.snotel_doy = snotel_sample['doy'].values

            #CDEC Timeseries
            # cdec_sample = data.cdec_data_siteid_year_normalized(site_id, forecast_year)
            # if len(cdec_sample):
            #     sample.cdec_snowwater = cdec_sample['curr_snowwater'].values
            #     sample.cdec_doy = cdec_sample['doy'].values

            res.append(sample)

    return res

def prepare_validation_timeseries(timeseries: List[TimeSeriesSample]) -> List[TimeSeriesSample]:
    res = []

    for ts in timeseries:
        ref_date = date(ts.forecast_year-1, 10, 1)

        for month in range(1, ts.season_end_month+1):
            for day in [1, 8, 15, 22]:
                dt = date(ts.forecast_year, month, day)
                doy =date_to_doy(ref_date, dt)
                res.append(ts.filter_by_forecast_doy(doy))

    return res


def getTestTimeseries(data: DataHandler) -> List[TimeSeriesSample]:
    submission_format = pd.read_csv(os.path.join(data.base_dir, 'submission_format.csv'))
    timeseries = []

    cache = dict()

    for _, row in tqdm(submission_format.iterrows(), total=len(submission_format)):
        site_id = row.site_id
        issue_date_str = row.issue_date[0:10]
        forecast_year = int(issue_date_str[0:4])

        ref_date = date(forecast_year - 1, 10, 1)
        issue_date = date.fromisoformat(issue_date_str)
        issue_doy = date_to_doy(ref_date, issue_date)

        k = (site_id, forecast_year)
        if k in cache:
            sample_full = cache[k]
        else:
            season_start_month, season_end_month = data.season_months_for_site_id(site_id)

            season_start_date = date(forecast_year, season_start_month, 1)

            sample_full = TimeSeriesSample(site_id=site_id,
                                      site_id_int=data.siteid_to_idx(site_id),
                                      forecast_year=forecast_year,
                                      season_start_month=season_start_month,
                                      season_end_month=season_end_month,
                                      season_start_doy=date_to_doy(ref_date, season_start_date),
                                      season_end_doy=date_to_doy(ref_date, day_end_of_month(forecast_year, season_end_month)),
                                      volume_scale_factor=data.volume_scale_factor(site_id)
                                      )

            # monthly timeseries
            monthly_volume_samples = data.test_monthly_volume_samples_site_year_normalized(site_id, forecast_year)
            if len(monthly_volume_samples):
                sample_full.monthly_volumes = monthly_volume_samples['volume'].values
                sample_full.monthly_volumes_doy = monthly_volume_samples['doy'].values
            else:
                print(f"No monthly data for {site_id}, {forecast_year}")

            # daily timeseries
            daily_volumes_sample = data.daily_volume_samples_site_year_normalized(site_id, forecast_year)
            if len(daily_volumes_sample):
                sample_full.daily_volumes = daily_volumes_sample['volume'].values
                sample_full.daily_volumes_doy = daily_volumes_sample['doy'].values
            else:
                print(f"No daily data for {site_id}, {forecast_year}")

            # SNOTEL timeseries
            snotel_sample = data.snotel_data_siteid_year_normalized(site_id, forecast_year)
            if len(snotel_sample):
                sample_full.snotel_agg_water = snotel_sample['agg_water'].values
                sample_full.snotel_snowwater = snotel_sample['curr_snowwater'].values
                sample_full.snotel_doy = snotel_sample['doy'].values
            else:
                print(f"No snotel data for {site_id}, {forecast_year}")

            # # CDEC Timeseries
            # cdec_sample = data.cdec_data_siteid_year_normalized(site_id, forecast_year)
            # if len(cdec_sample):
            #     sample_full.cdec_snowwater = cdec_sample['curr_snowwater'].values
            #     sample_full.cdec_doy = cdec_sample['doy'].values

            cache[k] = sample_full

        sample = sample_full.filter_by_forecast_doy(issue_doy)
        timeseries.append(sample)

    return timeseries

def getTestTimeseries_siteid_date(data: DataHandler, site_id: str, issue_date:date) -> List[TimeSeriesSample]:
    forecast_year = issue_date.year if issue_date.month<10 else issue_date.year+1
    ref_date = date(forecast_year - 1, 10, 1)
    issue_doy = date_to_doy(ref_date, issue_date)

    season_start_month, season_end_month = data.season_months_for_site_id(site_id)

    season_start_date = date(forecast_year, season_start_month, 1)

    sample_full = TimeSeriesSample(site_id=site_id,
                                   site_id_int=data.siteid_to_idx(site_id),
                                   forecast_year=forecast_year,
                                   season_start_month=season_start_month,
                                   season_end_month=season_end_month,
                                   season_start_doy=date_to_doy(ref_date, season_start_date),
                                   season_end_doy=date_to_doy(ref_date,
                                                              day_end_of_month(forecast_year, season_end_month)),
                                   volume_scale_factor=data.volume_scale_factor(site_id)
                                   )

    # monthly timeseries
    monthly_volume_samples = data.test_monthly_volume_samples_site_year_normalized(site_id, forecast_year)
    if len(monthly_volume_samples):
        sample_full.monthly_volumes = monthly_volume_samples['volume'].values
        sample_full.monthly_volumes_doy = monthly_volume_samples['doy'].values
    else:
        print(f"No monthly data for {site_id}, {forecast_year}")

    # daily timeseries
    daily_volumes_sample = data.daily_volume_samples_site_year_normalized(site_id, forecast_year)
    if len(daily_volumes_sample):
        sample_full.daily_volumes = daily_volumes_sample['volume'].values
        sample_full.daily_volumes_doy = daily_volumes_sample['doy'].values
    else:
        print(f"No daily data for {site_id}, {forecast_year}")

    # SNOTEL timeseries
    snotel_sample = data.snotel_data_siteid_year_normalized(site_id, forecast_year)
    if len(snotel_sample):
        sample_full.snotel_agg_water = snotel_sample['agg_water'].values
        sample_full.snotel_snowwater = snotel_sample['curr_snowwater'].values
        sample_full.snotel_doy = snotel_sample['doy'].values
    else:
        print(f"No snotel data for {site_id}, {forecast_year}")

    sample = sample_full.filter_by_forecast_doy(issue_doy)

    return [sample]
