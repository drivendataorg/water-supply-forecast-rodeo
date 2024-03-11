# Water Supply Forecast Rodeo Hindcast

This repository contains code to train models that predict the 0.1, 0.5, 0.9 quantiles of naturalized streamflow for stream sites
in the Western United States.

The repository also contains a Docker runtime that uses the trained models to predict streamflow on a set of holdout years.

This repository is based off of https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime

## Repo Organization

```
.
├── README.md                             <- You are here!
├── Streamflow_Hindcast_Report.pdf        <- Reference for the solution
├── requirements-train.txt                <- File with the required packages to train the models
├── data                                  <- Directory with base data for running inference on the trained models
├── data_download                         <- Directory with code required for downloading raw data
├── data_reading                          <- Directory with helper code for reading raw data
├── runtime                               <- Code and Dockerfile for running inference workflow
└── submission_src                        <- Code to be packaged and run inferenece workflow
      └── feature_parameters              <- Directory for pre-computed feature paramaters calculated during training
      └── features                        <- Directory with helper code for generating feature data from raw data
         ├── acis.py                      <- Code for creating climate features from ACIS data
         ├── cdec_deviation.py            <- Code for creating snow pack features from CDEC data
         ├── feature_engineering.py       <- Code for end-to-end test feature generation
         ├── monthly_naturalized_flow.py  <- Code for creating naturalized flow features from NRCS data
         ├── snotel_deviation.py          <- Code for creating snow pack features from Snotel data
         └── streamflow_deviation.py      <- Code fore creating streamflow features from USGS data
      ├── models                          <- Directory for trained models
      ├── wsfr_download                   <- Directory with download code to be used in inference runtime
      ├── generate_predictions.py         <- Code to generate predictions and save them to a submission file
      └── solution.py                     <- Code to run the inference solution
└── training                              <- Code for running the training workflow
    └── features                          <- Directory with helper code for generating feature data from the raw data
         ├── acis.py                      <- Code for creating climate features from ACIS data
         ├── cdec_deviation.py            <- Code for creating snow pack features from CDEC data
         ├── glo_elevations.py            <- Code for creating elevation features from GLO Copernicus data
         ├── monthly_naturalized_flow.py  <- Code for creating naturalized flow features from NRCS data
         ├── snotel_deviation.py          <- Code for creating snow pack features from Snotel data
         └── streamflow_deviation.py      <- code fore creating streamflow features from USGS data
    ├── models                            <- Directory for trained models
    ├── preprocessed_data                 <- Directory to store intermediate data files
    ├── train_data                        <- Directory to store data used for model training
    ├── wsfr_download_train               <- Directory with download code to be used during model training
    ├── generate_train_features.py        <- Code for end-to-end traing feature generation
    ├── train_model.py                    <- Code for end-to-end model training
    ├── train_monthly_model.py            <- Code to train the monthly Catboost models
    └── train_yearly_model.py             <- Code to train the yearly Catboost models
```

## Directions for running the solution end-to-end
The steps below should be followed in order to reproduce the competition submission.

### 0. Get competition data files

Download the following competition data files and place them in `data/`:

- `cdec_snow_stations.csv`
- `cpc_climate_divisions.gpkg`
- `geospatial.gpkg`
- `metadata.csv`
- `smoke_submission_format.csv`
- `submission_format.csv`
- `test_monthly_naturalized_flow.csv`
- `train_monthly_naturalized_flow.csv`
- `train.csv`

### 1. Requirements and installation

Requires Python 3.10. To install with the exact dependencies that will be used by DrivenData, create a new virtual environment and run

```bash
pyenv install 3.10
pyenv virtualenv 3.10 water-supply-hindcast
pyenv activate water-supply-hindcast
pip install -r ./data_download/requirements.txt
pip install ./data_download/
pip install ./data_reading/
pip install -r requirements-train.txt
```

### 2. Data Download

Download all train and test raw data using the `bulk` command. From the repository root as your working directory, run:

```bash
WSFR_DATA_ROOT=training/train_data python -m wsfr_download bulk data_download/hindcast_train_config.yml
python -m wsfr_download bulk data_download/hindcast_test_config.yml
```

The train data will be downloaded to the directory `training/train_data` and the test data will be downloaded to the `data` directory.

### 3. Generate the train features file

From the repository root as your working directory, run:

```bash
python training/generate_train_features.py
```

### 4. Train the models

From the repository root as your working directory, run:

```bash
python training/train_model.py
```

### 5. Copy files produced during training over to the test runtime environment

From the repository root as your working directory, run:

```bash
cp training/preprocessed_data/site_elevations.csv submission_src/feature_parameters/site_elevations.csv
cp training/preprocessed_data/train_acis_precip.csv submission_src/feature_parameters/train_acis_precip.csv
cp training/preprocessed_data/train_acis_temp.csv submission_src/feature_parameters/train_acis_temp.csv
cp training/preprocessed_data/train_cdec.csv submission_src/feature_parameters/train_cdec.csv
cp training/preprocessed_data/train_monthly_nat_group.csv submission_src/feature_parameters/train_monthly_nat_group.csv
cp training/preprocessed_data/train_snotel.csv submission_src/feature_parameters/train_snotel.csv
cp training/preprocessed_data/train_streamflow.csv submission_src/feature_parameters/train_streamflow.csv

cp training/models/cat_10_monthly_model.txt submission_src/models/cat_10_monthly_model.txt
cp training/models/cat_10_yearly_model.txt submission_src/models/cat_10_yearly_model.txt
cp training/models/cat_50_monthly_model.txt submission_src/models/cat_50_monthly_model.txt
cp training/models/cat_50_yearly_model.txt submission_src/models/cat_50_yearly_model.txt
cp training/models/cat_90_monthly_model.txt submission_src/models/cat_90_monthly_model.txt
cp training/models/cat_90_yearly_model.txt submission_src/models/cat_90_yearly_model.txt
```

### 6. Pack and test the submission to run the inference code environment and produce a submission.csv

From the repository root as your working directory, run:

```bash
make pull
make build
make pack-submission
make test-submission
```

The code submission will be located at `submission/submission.zip`

The submission file will be located at `submission/submission.csv`
