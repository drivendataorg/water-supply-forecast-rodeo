from pathlib import Path
from solution import preprocess, predict
import pandas as pd

data_dir = Path('data/forecast_stage')

assets = preprocess(Path("."), data_dir, Path("tmp"))

site_ids = sorted(list(set(pd.read_csv('data/forecast_train.csv')['site_id'].tolist())))


def call_predict(site_id, date):
    res = predict(site_id, date, assets, Path("."), data_dir, Path("tmp"))
    print(f"{site_id=}, {date=}, {res=}")

for site_id in site_ids:
    print('start prediction for', site_id)
    call_predict(site_id, "2024-01-01")
    print('-----------------')

