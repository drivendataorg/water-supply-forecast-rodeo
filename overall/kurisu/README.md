# Solution - Water Supply Forecast Rodeo: Final Prize Stage

Team: Christoph Molnar (User: kurisu)

## Summary

This repository contains code for training machine learning models to forecast streamflow.
For each quantile (10%, 50%, and 90%) and year from 2004 to 2024, I trained an ensemble of 10 XGBoost models, yielding 3 x 10 x 20 = 600 models.
The models use features based on SWANN, SNOTEL, and antecedent flow data.

## Setup

1. Install the prerequisites

The code is based on Python 3.12.
Python packages are managed via Conda version 24.7.1.

2. Install the required Python packages

```sh
conda env create -f config/environment.yml --name wsf`
```

## Hardware

The solution was run on a MacBook Air (2020) with an Apple M1 chip and 16GB of RAM.

- Downloading SNOTEL data: <1 hour
- Downloading Swann and creating features: ~4 minutes.
- Training time: ~ 2 minutes
- Inference time: < 1 minute

## Train the models

All commands should be run from the root directory of this repository.

### Download competition data

Download competition data and put it into `./data/`:

- `geospatial.gpkg`
- `cross_validation_labels.csv`
- `cross_validation_monthly_flow.csv`
- `cross_validation_submission_format.csv`
- `prior_historical_labels.csv`
- `prior_historical_monthly_flow.csv`

### Download HUC

Go to [sciencebase.gov](https://www.sciencebase.gov/catalog/item/631405c4d34e36012efa315f) and download huc250k_shp.zip.
Unzip it and move all the files into the `./data/huc/` directory.
Extract only the files, not the directory structure they are in.

### Download SNOTEL data

To download the SNOTEL data, you can use the `wsfr_download` module.
If the module is already installed, simply run the following from the root directory of this repository.

```sh
python3 -m wsfr_download bulk config/wsfr_config.yml
```

If you haven't installed it:

```sh
git clone git@github.com:drivendataorg/water-supply-forecast-rodeo-runtime.git
python3 -m venv wsfr-download
. wsfr-download/bin/activate
pip install water-supply-forecast-rodeo-runtime/data_download/
python3 -m wsfr_download bulk config/wsfr_config.yml
deactivate
```

Outcome:

- SNOTEL data in `./data/snotel`

### Download SWANN data and create features

```sh
conda activate wsf
python3 src/create-features.py
```

Outcomes:

- SWANN data in `./pre-processed/swann/
- Features dataframe in `./pre-processed/features.csv`

### Train the XGBoost models

```
conda activate wsf
python3 src/train.py
```

This trains the XGBoost models using a leave-one-year-out approach.
For each year, an ensemble of 10 XGBoost models is trained for each quantile (10%, 50%, and 90%), resulting in 30 models per year.

Outcomes:

- The 3x10 quantile models for each year are saved in `./pre-processed/models_{year}.pkl`. These files also contain the adjustment factors for the 10% and 90% quantiles based on conformal prediction (cp_adjustments).
- Leave-one-year-out predictions are saved in `./pre-processed/forecasts.csv`.
