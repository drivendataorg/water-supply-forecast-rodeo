#!/usr/bin/env python
# coding: utf-8

"""
This script downloads data and creates features for water supply forecasting.

It processes various data sources including SNOTEL, SWANN, and monthly flow data
to generate a comprehensive feature set for machine learning models.

The script is designed to be run from a Makefile and uses functions from custom modules
in the src directory.
"""

import os
import datetime
from pathlib import Path
import geopandas as gpd
import pandas as pd

from data.preprocess import preprocess, create_issue_dates
from data.swann import compute_swann_features
from data.snotel import compute_snotel_features
from data.other import compute_antecedent_flow, compute_offset

pre_processed_dir = Path('./pre-processed/')


def create_huc_mapping(data_dir):
    """
    Create a mapping from site_id to HUC (Hydrologic Unit Code).

    Args:
        data_dir (Path): Path to the data directory.

    Returns:
        None
    """
    gdf1 = gpd.read_file(data_dir / 'huc' / 'huc250k.shp')
    gdf2 = gpd.read_file(data_dir / 'geospatial.gpkg')
    gdf1 = gdf1.to_crs(gdf2.crs)
    gdf_overlayed = gpd.sjoin(gdf1, gdf2, how='inner', predicate='intersects')
    huc_mapping = gdf_overlayed[['CAT','site_id']]
    huc_mapping = huc_mapping.rename({'CAT': 'huc8'}, axis=1)
    huc_mapping.to_csv(data_dir / 'huc-to-site.csv', index=False)

def create_feature_dataframe(labels, assets):
    """
    Create a feature dataframe from various data sources.

    Args:
        labels (pd.DataFrame): DataFrame containing year, site_id, and volume data.
        assets (dict): Dictionary containing preprocessed data assets.

    Returns:
        pd.DataFrame: DataFrame containing all features.
    """
    year_site_pairs = labels[['year', 'site_id', 'volume']].dropna()

    df_list = []
    for _, row in year_site_pairs.iterrows():
        df = create_issue_dates(row['year'])
        df['forecast_year'] = df['issue_date'].dt.year
        df['day_in_year'] = df['issue_date'].dt.dayofyear
        df['volume'] = row['volume']
        df['site_id'] = row['site_id']
        df_list.append(df)

    dates_df = pd.concat(df_list)
    # Removing detroit lake rows out of season
    condition = ~((dates_df['issue_date'].dt.month == 7) & (dates_df['site_id'] == 'detroit_lake_inflow'))
    dates_df = dates_df[condition]

    print('Creating SNOTEL Features')
    snotel_features = dates_df.apply(lambda row: compute_snotel_features(assets['snotel'][row['site_id']], row['site_id'], row['issue_date']), axis=1)

    print('Creating SWANN Features')
    swann_features = dates_df.apply(lambda row: compute_swann_features(assets['swann'][row['site_id']], row['site_id'], row['issue_date']), axis=1)

    print('Computing Offsets')
    offset = dates_df.apply(lambda row: compute_offset(assets['monthly_flow'], row['site_id'], row['issue_date']), axis=1)

    print('Creating Antecedent Features')
    antecedent_flow_features = dates_df.apply(lambda row: compute_antecedent_flow(assets['monthly_flow'], row['site_id'], row['issue_date']), axis=1)

    print('Merging Everything')
    features = pd.concat([
        dates_df,
        offset,
        antecedent_flow_features,
        snotel_features,
        swann_features,
    ], axis=1)

    return features

def main():
    """
    Main function to orchestrate the data download and feature creation process.
    """
    start_time = datetime.datetime.now()

    # Define directories
    src_dir = Path('./src')
    data_dir = Path('./data/')
    preprocessed_dir = Path('./pre-processed/')

    # Load labels
    loocv_labels = pd.read_csv(data_dir / 'cross_validation_labels.csv')
    historic_labels = pd.read_csv(data_dir / 'prior_historical_labels.csv')
    labels = pd.concat([loocv_labels, historic_labels])

    # Create HUC mapping
    create_huc_mapping(data_dir)

    # Preprocess assets
    assets = preprocess(src_dir, data_dir, preprocessed_dir)

    # Create feature dataframe
    features = create_feature_dataframe(labels, assets)

    # Save features
    features.to_csv(pre_processed_dir / 'features.csv', index=False)

    # Play sound to indicate completion
    os.system("printf '\a'")

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    minutes, seconds = divmod(duration.total_seconds(), 60)
    print(f'Data creation took {int(minutes)}:{int(seconds):02}')

if __name__ == "__main__":
    main()
