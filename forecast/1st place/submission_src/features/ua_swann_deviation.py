import ssl
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

site_basins = {
    'libby_reservoir_inflow': '17010101',
    'american_river_folsom_lake': '18020111',
    'ruedi_reservoir_inflow': '14010004',
    'skagit_ross_reservoir': '17110005',
    'snake_r_nr_heise': '17040104',
    'missouri_r_at_toston': '10030101',
    'san_joaquin_river_millerton_reservoir': '18040001',
    'hungry_horse_reservoir_inflow': '17010209',
    'boysen_reservoir_inflow': '10080005',
    'boise_r_nr_boise': '17050114',
    'fontenelle_reservoir_inflow': '14040101',
    'yampa_r_nr_maybell': '14050002',
    'owyhee_r_bl_owyhee_dam': '17050110',
    'detroit_lake_inflow': '17090005',
    'merced_river_yosemite_at_pohono_bridge': '18040008',
    'stehekin_r_at_stehekin': '17020009',
    'pueblo_reservoir_inflow': '11020002',
    'green_r_bl_howard_a_hanson_dam': '17110013',
    'animas_r_at_durango': '14080104',
    'colville_r_at_kettle_falls': '17020003',
    'dillon_reservoir_inflow': '14010002',
    'sweetwater_r_nr_alcova': '10180006',
    'weber_r_nr_oakley': '16020101',
    'taylor_park_reservoir_inflow': '14020001',
    'virgin_r_at_virtin': '15010008',
    'pecos_r_nr_pecos': '13060001',
}


def generate_ua_swann_deviation(src_dir: Path,
                                test_features: pd.DataFrame,
                                issue_date: str) -> pd.DataFrame:
    logger.info("Generating UA Swann deviation features.")

    swann_base_url = 'https://climate.arizona.edu/snowview/csv/Download/Watersheds/'
    ua_swann = []
    for site_id in test_features.site_id.unique():
        logger.info(f"retrieving UA Swann file for {site_id}")
        site_basin = site_basins[site_id]
        site_file_name = f'{site_basin}.csv'
        s = requests.get(f"{swann_base_url}{site_file_name}", verify=False).text
        site_ua_swann = pd.read_csv(StringIO(s))
        site_ua_swann.columns = ['date', 'acc_water', 'ua_swe']
        site_ua_swann['date'] = pd.to_datetime(site_ua_swann['date'])
        site_ua_swann['site_id'] = site_id
        ua_swann.append(site_ua_swann)

    ua_swann_pd = pd.concat(ua_swann)
    ua_swann_pd = ua_swann_pd[ua_swann_pd['date'] < issue_date]
    ua_swann_pd['issue_date'] = (ua_swann_pd['date'] + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')

    ua_swann_deviation = ua_swann_pd
    ua_swann_max_date = ua_swann_deviation['date'].max()
    logger.info(f"Max UA Swann date: {ua_swann_max_date}")
    ua_swann_deviation['month_day'] = ua_swann_deviation['date'].dt.strftime("%m%d")
    grouped_ua_swann = ua_swann_deviation.groupby(['site_id', 'month_day'])[['acc_water', 'ua_swe']].agg(
        [np.mean, np.std]).reset_index()
    grouped_ua_swann.columns = ['site_id', 'month_day', 'acc_water_mean', 'acc_water_std', 'ua_swe_mean', 'ua_swe_std']
    ua_swann_deviation = pd.merge(ua_swann_deviation, grouped_ua_swann, on=['site_id', 'month_day'])

    acc_water = ua_swann_deviation['acc_water']
    acc_water_mean = ua_swann_deviation['acc_water_mean']
    acc_water_std = ua_swann_deviation['acc_water_std']
    ua_swann_deviation['acc_water_deviation'] = (acc_water - acc_water_mean) / acc_water_std

    ua_swe = ua_swann_deviation['ua_swe']
    ua_swe_mean = ua_swann_deviation['ua_swe_mean']
    ua_swe_std = ua_swann_deviation['ua_swe_std']
    ua_swann_deviation['ua_swe_deviation'] = (ua_swe - ua_swe_mean) / ua_swe_std
    ua_swann_deviation = ua_swann_deviation[
        ['issue_date', 'site_id', 'acc_water', 'ua_swe', 'ua_swe_deviation', 'acc_water_deviation']]


    test_features = pd.merge(test_features, ua_swann_deviation, on=['site_id', 'issue_date'], how='left')

    logger.info(test_features.groupby('site_id')[['ua_swe_deviation', 'acc_water_deviation']].mean().round(3))

    return test_features
