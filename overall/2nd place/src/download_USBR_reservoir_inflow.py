import requests
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

def fetch_timeseries(site_id:str, forecast_year:int, item:int, url:str) -> pd.DataFrame:
    r = requests.get(url, headers={'accept': 'application/vnd.api+json'})
    if r.status_code != requests.codes.ok:
        print(f'Impossible to fetch timeseries for {site_id= } {forecast_year= } {item= }')
        return None

    data = r.json()['data']
    all_dates = []
    all_values = []
    for dt in data:
        all_dates.append(dt['attributes']['dateTime'][0:10])
        all_values.append(dt['attributes']['result'])

    res = pd.DataFrame({'site_id': site_id,
                         'forecast_year': forecast_year,
                         'item': item,
                         'date': all_dates,
                         'value': all_values})

    return res



if __name__ == '__main__':
    siteid_to_item = {
        'taylor_park_reservoir_inflow': 794,
        'fontenelle_reservoir_inflow': 4281,
        'american_river_folsom_lake': 10773,
        'boysen_reservoir_inflow': 197
    }

    siteid_start_years = {
        'taylor_park_reservoir_inflow': 1962,
        'fontenelle_reservoir_inflow': 1966,
        'american_river_folsom_lake': 1956,
        'boysen_reservoir_inflow': 1911
    }

    tasks = []

    for site_id, item in siteid_to_item.items():
        for forecast_year in range(siteid_start_years[site_id]+1, 2023+1):
            url = f'https://data.usbr.gov/rise/api/result?itemsPerPage=2000&order%5BdateTime%5D=ASC&itemId={item}&dateTime%5Bafter%5D={forecast_year-1}1001&dateTime%5Bstrictly_before%5D={forecast_year}0722'
            tasks.append(dict(site_id=site_id, forecast_year=forecast_year, item=item, url=url))

    all_data = Parallel(n_jobs=8, prefer='threads')(delayed(fetch_timeseries)(**task) for task in tqdm(tasks))
    all_data = pd.concat(all_data)
    all_data.to_csv('data/final_stage/USBR_reservoir_inflow.csv', index=False)

