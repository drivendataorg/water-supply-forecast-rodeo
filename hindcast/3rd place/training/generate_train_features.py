#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[0]))

from loguru import logger


def main(data_dir: Path,
         preprocessed_dir: Path,
         train_since: int = 1960):
    logger.info(f"Beginning generation of train features from {train_since}.")

    update_metadata(data_dir, preprocessed_dir)

    metadata = pd.read_csv(preprocessed_dir / 'metadata.csv', dtype={"usgs_id": "string"})

    train = pd.read_csv(DATA_DIR / 'train.csv')

    site_list = []
    date_list = []
    year_list = []
    month_list = []
    day_list = []
    for year in train['year'].unique():
        for idx, rw in metadata.iterrows():
            for month in [1, 2, 3, 4, 5, 6, 7]:
                for day in [1, 8, 15, 22]:
                    site_list.append(rw['site_id'])
                    date_list.append(f"{year}-{month:02d}-{day:02d}")
                    year_list.append(year)
                    month_list.append(month)
                    day_list.append(day)

    train_idx = pd.DataFrame(
        {'site_id': site_list, 'issue_date': date_list, 'year': year_list, 'month': month_list, 'day': day_list})

    train_features = pd.merge(train_idx, train, on=['site_id', 'year'], how='left')
    train_features['day_of_year'] = pd.to_datetime(train_features['issue_date']).dt.day_of_year

    train_features = train_features[train_features['year'] > train_since]
    print(f'train_features shape, {train_features.shape}')
    metadata_keep_cols = ['site_id', 'elevation', 'latitude', 'longitude', 'drainage_area',
                          'season_start_month', 'season_end_month']
    train_features = pd.merge(train_features,
                              metadata[metadata_keep_cols],
                              on=['site_id'], how='left')

    site_elevations = generate_elevations(data_dir, preprocessed_dir)
    train_features = pd.merge(train_features, site_elevations, on='site_id')

    train_features = generate_monthly_naturalized_flow(data_dir, preprocessed_dir, train_features, metadata)
    print(train_features.shape)
    train_features = generate_acis_deviation(preprocessed_dir, train_features)
    train_features = generate_snotel_deviation(data_dir, preprocessed_dir, train_features)
    train_features = generate_cdec_deviation(data_dir, preprocessed_dir, train_features)
    train_features = generate_streamflow_deviation(preprocessed_dir, train_features)

    train_features['combined_swe_deviation_30'] = train_features[['snotel_wteq_deviation_30', 'cdec_deviation_30']].mean(
        axis=1)
    train_features['combined_swe_deviation_0'] = train_features[['snotel_wteq_deviation_0', 'cdec_deviation_0']].mean(
        axis=1)

    logger.info(train_features.columns)

    train_features.to_csv(preprocessed_dir / "train_features.csv", index=False)


def update_metadata(data_dir: Path,
                    preprocessed_dir: Path) -> None:
    metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={"usgs_id": "string"})
    metadata = metadata.set_index('site_id')

    metadata.loc['ruedi_reservoir_inflow', 'usgs_id'] = '09080400'
    metadata.loc['fontenelle_reservoir_inflow', 'usgs_id'] = '09211200'
    metadata.loc['american_river_folsom_lake', 'usgs_id'] = '11446500'
    metadata.loc['skagit_ross_reservoir', 'usgs_id'] = '12181000'
    metadata.loc['skagit_ross_reservoir', 'drainage_area'] = 999.0
    metadata.loc['boysen_reservoir_inflow', 'usgs_id'] = '06279500'
    metadata.loc['boise_r_nr_boise', 'usgs_id'] = '13185000'
    metadata.loc['sweetwater_r_nr_alcova', 'usgs_id'] = '06235500'

    metadata.to_csv(preprocessed_dir / 'metadata.csv')


if __name__ == "__main__":
    from pathlib import Path

    import os
    import sys
    import warnings
    import pandas as pd

    sys.path.append(str(Path(__file__).parent.resolve()))

    DATA_DIR = Path.cwd() / "training/train_data"
    os.environ["WSFR_DATA_ROOT"] = str(DATA_DIR)
    PREPROCESSED_DIR = Path.cwd() / "training/preprocessed_data"

    warnings.filterwarnings('ignore')

    from features.acis import generate_acis_deviation
    from features.monthly_naturalized_flow import generate_monthly_naturalized_flow
    from features.streamflow_deviation import generate_streamflow_deviation
    from features.snotel_deviation import generate_snotel_deviation
    from features.cdec_deviation import generate_cdec_deviation
    from features.glo_elevations import generate_elevations


    main(DATA_DIR, PREPROCESSED_DIR)
