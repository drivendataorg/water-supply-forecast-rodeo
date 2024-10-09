import json
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

def generate_acis_deviation(preprocessed_dir: Path,
                            df_features: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating acis climate features.")

    logger.info("Generating acis precipitation features.")
    precip_file = preprocessed_dir / "train_acis_precip.csv"
    temp_file = preprocessed_dir / "train_acis_temp.csv"
    if precip_file.is_file() and temp_file.is_file():
        logger.info("Acis files already exists, pulling existing.")
        acis_precip = pd.read_csv(precip_file)
        acis_temp = pd.read_csv(temp_file)
    else:
        all_precips = []
        all_temps = []
        for site_id in df_features['site_id'].unique():
            site_list = []
            date_list = []
            station_list = []
            precip_list = []
            maxt_list = []
            for year in df_features['year'].unique():
                sdate = f"{year - 1}-10-01"
                edate = f"{year}-07-21"
                input_dict = {
                    'basin': site_basins[site_id],
                    'sdate': sdate,
                    'edate': edate,
                    'meta': 'name, sids',
                    'elems': [{
                        'name': 'pcpn',
                        'interval': 'dly',
                        'duration': 1,
                        'reduce': {'reduce': 'sum'},
                        'maxmissing': 3
                    }, {
                        'name': 'maxt',
                        'interval': 'dly',
                        'duration': 1,
                        'reduce': {'reduce': 'sum'},
                        'maxmissing': 3
                      }]
                }
                dates = list(pd.date_range(sdate, edate, freq='D'))
                sites = [site_id] * len(dates)
                params = {'params': json.dumps(input_dict)}
                headers = {'Accept': 'application/json'}
                req = requests.post('http://data.rcc-acis.org/MultiStnData', data=params, headers=headers)
                response = req.json()
                acis_data = response['data']

                for v in range(0, len(acis_data)):
                    site_list += sites
                    date_list += dates
                    stations = [acis_data[v]['meta']['name']] * len(dates)
                    station_list += stations
                    precip_list += list(np.array(acis_data[v]['data'])[:, 0])
                    maxt_list += list(np.array(acis_data[v]['data'])[:, 1])
            acis_precip = pd.DataFrame({
                'site_id': site_list,
                'date': date_list,
                'station': station_list,
                'precip': precip_list,
            })
            all_precips.append(acis_precip)
            acis_temp = pd.DataFrame({
                'site_id': site_list,
                'date': date_list,
                'station': station_list,
                'maxt': maxt_list,
            })
            all_temps.append(acis_temp)

        acis_precip = pd.concat(all_precips)
        acis_precip.to_csv(precip_file, index=False)

        acis_temp = pd.concat(all_temps)
        acis_temp.to_csv(temp_file, index=False)

    acis_precip['date'] = pd.to_datetime(acis_precip["date"])
    acis_precip["month"] = acis_precip["date"].dt.month
    acis_precip["year"] = acis_precip["date"].dt.year
    acis_precip["forecast_year"] = acis_precip.apply(lambda rw: rw["year"] if rw["month"] < 8 else rw["year"] + 1,
                                                     axis=1)
    acis_precip['month_day'] = acis_precip['date'].dt.strftime("%m%d")
    acis_precip['precip'] = acis_precip['precip'].replace('T', 0).replace('M', np.nan)
    acis_precip['precip'] = acis_precip['precip'].astype(float)
    acis_precip = acis_precip.sort_values(["station", "date"])
    precip_30 = acis_precip.groupby(['forecast_year', 'site_id', 'station'])['precip'].rolling(window=30).agg(
        [np.mean]).reset_index(level=[0, 1, 2], drop=True)
    precip_30.columns = ["precip_30"]
    acis_precip = acis_precip.join(precip_30)
    precip_90 = acis_precip.groupby(['forecast_year', 'site_id', 'station'])['precip'].rolling(window=90).agg(
        [np.mean]).reset_index(level=[0, 1, 2], drop=True)
    precip_90.columns = ["precip_90"]
    acis_precip = acis_precip.join(precip_90)
    precip_grouped = acis_precip.groupby(["site_id", "station", "month_day"])[["precip", "precip_30", "precip_90"]].agg(
        [np.mean, np.std]).reset_index()
    precip_grouped.columns = ["site_id", "station", "month_day", "precip_mean", "precip_std", "precip_30_mean",
                              "precip_30_std", "precip_90_mean", "precip_90_std"]
    acis_precip = pd.merge(acis_precip, precip_grouped, on=["site_id", "station", "month_day"])
    acis_precip['precip_deviation'] = (acis_precip['precip'] - acis_precip['precip_mean']) / acis_precip['precip_std']
    acis_precip['precip_deviation'] = acis_precip['precip_deviation'].replace(np.inf, np.nan).replace(-np.inf, np.nan)
    acis_precip['precip_30_deviation'] = (acis_precip['precip_30'] - acis_precip['precip_30_mean']) / acis_precip[
        'precip_30_std']
    acis_precip['precip_90_deviation'] = (acis_precip['precip_90'] - acis_precip['precip_90_mean']) / acis_precip[
        'precip_90_std']
    precip_deviation = acis_precip.groupby(['site_id', 'date'])[
        ['precip', 'precip_30', 'precip_deviation', 'precip_30_deviation', 'precip_90_deviation']].mean().reset_index()
    # Set issue_date to be one day ahead, so we only join with past data
    precip_deviation["issue_date"] = (precip_deviation["date"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")


    def rolling_precip_deviation(rw, window=30, agg='mean'):
        rw_year = rw['year']
        sub_df = precip_deviation[(precip_deviation['site_id'] == rw['site_id']) &
                                  (precip_deviation['date'] < rw['issue_date']) &
                                  (precip_deviation['date'] >= f"{rw_year - 1}-10-01")]

        rw[f'precip_deviation_{window}_{agg}'] = sub_df[-window:]['precip_deviation'].agg(agg)
        rw[f'precip_deviation_season_{agg}'] = sub_df['precip_deviation'].agg(agg)

        return rw

    df_features = df_features.apply(lambda rw: rolling_precip_deviation(rw), axis=1)
    precip_columns = ["precip_30_deviation", "precip_90_deviation"]
    try:
        df_features = df_features.drop(precip_columns, axis=1)
    except:
        logger.info("Trying to drop precipitation columns that don't exist.")

    df_features = pd.merge(df_features, precip_deviation[
        ["site_id", "issue_date", "precip_30_deviation", "precip_90_deviation"]
    ], on=["site_id", "issue_date"], how="left")

    logger.info(df_features[['precip_deviation_30_mean', 'precip_deviation_season_mean']].describe())

    logger.info("Generating acis temperature features.")

    acis_temp['maxt'] = acis_temp['maxt'].replace('T', 0).replace('M', np.nan)
    acis_temp['maxt'] = acis_temp['maxt'].astype(float)
    acis_temp['date'] = pd.to_datetime(acis_temp['date'])
    acis_temp["month_day"] = acis_temp["date"].dt.strftime("%m%d")
    temp_grouped = acis_temp.groupby(["site_id", "station", "month_day"])[["maxt"]].agg(
        [np.mean, np.std]).reset_index()
    temp_grouped.columns = ['site_id', 'station', 'month_day', 'maxt_mean', 'maxt_std']
    temp_grouped.to_csv(preprocessed_dir / 'acis_temp_parameters.csv', index=False)
    acis_temp = pd.merge(acis_temp, temp_grouped, on=["site_id", "station", "month_day"])
    acis_temp['maxt_deviation'] = (acis_temp['maxt'] - acis_temp['maxt_mean']) / acis_temp['maxt_std']
    acis_temp['maxt_deviation'] = acis_temp['maxt_deviation'].replace(np.inf, np.nan).replace(-np.inf, np.nan)
    # acis_temp = acis_temp.dropna(subset='maxt_deviation')
    temp_deviation = acis_temp.groupby(['site_id', 'date'])[
        ['maxt', 'maxt_deviation']].mean().reset_index()
    temp_deviation = temp_deviation.sort_values(['site_id', 'date'])

    def rolling_temp_deviation(rw, window=30, agg='mean'):
        year = rw['year']
        sub_df = temp_deviation[(temp_deviation['site_id'] == rw['site_id']) &
                                (temp_deviation['date'] < rw['issue_date']) &
                                (temp_deviation['date'] >= f"{year - 1}-10-01")]

        rw[f'maxt_deviation_{window}_{agg}'] = sub_df[-window:]['maxt_deviation'].agg(agg)
        rw[f'maxt_deviation_season_{agg}'] = sub_df['maxt_deviation'].agg(agg)

        return rw

    df_features = df_features.apply(lambda rw: rolling_temp_deviation(rw), axis=1)

    logger.info(df_features[['maxt_deviation_30_mean', 'maxt_deviation_season_mean']].describe())

    return df_features
