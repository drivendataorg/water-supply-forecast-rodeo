# Solution - Water Supply Forecast Rodeo: Explainability and Communication

Team: Christoph Molnar (User: kurisu)

## Summary

This repository contains code for training machine learning models to forecast streamflow.
For each quantile (10%, 50%, and 90%), I trained an ensemble of XGBoost models.
The models use features based on SWANN, SNOTEL, and antecedent flow data.
In addition, the repository includes code to generate reports that explain the forecasts.
The explanations are based on Shapley values and What-If plots.
I used Python to train the models, and Quarto (with both Python and R) to produce the reports.

## Setup

1. Install the prerequisites

The code is based on Python 3.12 and on R version 4.4.1.
Python and R packages are managed via Conda version 24.7.1.

2. Install the required Python packages

```sh
conda env create -f config/environment.yml --name wsf`
```

3. Install shaprpy

The shaprpy library, used for computing grouped Shapley values, must be installed separately:

```sh
conda activate wsf
git clone https://github.com/NorskRegnesentral/shapr.git
cd shapr
git checkout ddd32c7c92ba9f37c8505720129d7e10979ccc4c
Rscript python/install_r_packages.R
pip install -e python/.
conda deactivate
```

## Hardware

The solution was run on a MacBook Air (2020) with an Apple M1 chip and 16GB of RAM.

- Downloading SNOTEL data: <1 hour
- Downloading Swann and creating features: ~4 minutes.
- Training time: ~ 2 minutes
- Inference time: < 1 minute
- Report creation: ~40 seconds per report

## Train the models

The trained models for 2023 are already included in this repository `./pre-processed/models_2023.pkl`.
If you are only interested in the explainability reports, you can skip the training.

All commands should be run from the root directory of this repository.

### Download competition data

Download competition data and put it into `./data/`:

- geospatial.gpkg
- cross_validation_labels.csv
- cross_validation_monthly_flow.csv
- cross_validation_submission_format.csv
- prior_historical_labels.csv
- prior_historical_monthly_flow.csv

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

```sh
conda activate wsf
python3 src/train.py
```

This trains the XGBoost models using a leave-one-year-out approach.
For each year, an ensemble of 10 XGBoost models is trained for each quantile (10%, 50%, and 90%), resulting in 30 models per year.
However, only the 30 models for 2023 are stored which we are going to use for creating the communication outputs.

Outcomes:

- The 3x10 quantile models for 2023 are saved in `./pre-processed/models_2023.pkl`
- Leave-one-year-out predictions are saved in `./pre-processed/forecasts.csv`

## Create communication outputs

Install the Quarto CLI [from this website](https://quarto.org/docs/get-started/).
For the forecast reports, I used version 1.5.6.

Run the following:

```sh
conda activate wsf
python src/generate_report.py --issue_date 2023-03-15 --site_id owyhee_r_bl_owyhee_dam --target_dir ./forecast_reports
python src/generate_report.py --issue_date 2023-05-15 --site_id owyhee_r_bl_owyhee_dam --target_dir ./forecast_reports
python src/generate_report.py --issue_date 2023-03-15 --site_id pueblo_reservoir_inflow --target_dir ./forecast_reports
python src/generate_report.py --issue_date 2023-05-15 --site_id pueblo_reservoir_inflow --target_dir ./forecast_reports

```

Outcome:

- 4 reports in `./forecast_reports/`
  - Pueblo Reservoir 2023-03-15 and 2023-05-15
  - Owyhee Reservoir 2023-03-15 and 2023-05-15
