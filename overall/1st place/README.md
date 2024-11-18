# Water Supply Forecast Rodeo Final Cross Validation and Explainability Solution

This repository contains code to train models that predict the 0.1, 0.5, 0.9 quantiles of naturalized streamflow for stream sites
in the Western United States.

The repository also contains a Docker runtime that uses the trained models to predict streamflow on a set of holdout years.

This repository includes code from https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime

## Repo Organization

```
.
├── README.md                                           <- You are here!
├── reports/
│   ├── report.pdf                                      <- Report about solution from Final Stage
│   └── competition_winner_solution_documentation.pdf   <- Winner documentation
├── requirements-train.txt                              <- File with the required packages to train the models
├── data_download/                                      <- Directory with code required for downloading raw data
├── data_reading/                                       <- Directory with helper code for reading raw data
└── training                                            <- Code for running the training workflow
    ├── features                                        <- Directory with helper code for generating feature data from the raw data
    │    ├── acis.py                                    <- Code for creating climate features from ACIS data
    │    ├── cdec_deviation.py                          <- Code for creating snow pack features from CDEC data
    │    ├── drought_deviation.py                       <- Code for creating drought features from PDSI data
    │    ├── feature_utils.py                           <- Utility code for creating feature data
    │    ├── glo_elevations.py                          <- Code for creating elevation features from GLO Copernicus data
    │    ├── monthly_naturalized_flow.py                <- Code for creating naturalized flow features from NRCS data
    │    ├── snotel_deviation.py                        <- Code for creating snow pack features from Snotel data
    │    ├── streamflow_deviation.py                    <- code fore creating streamflow features from USGS data
    │    └── ua_swann_deviation.py                      <- code fore creating SWE features from UA Swann data
    ├── preprocessed_data                               <- Directory to store intermediate data files
    │    └── feature_corrs                              <- Directory to store correlations for stations or measurement locations for each CV year
    │    ├── models                                     <- Directory for trained models for each CV year
    │    └── plots                                      <- Directory where site explainability plots are saved
    ├── train_data                                      <- Directory to store data used for model training
    │    └── cb_2018_us_state_500k                      <- USA State Shape File Available at - https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip
    ├── wsfr_download_train                             <- Directory with download code to be used during model training
    ├── generate_train_features.py                      <- Code for end-to-end traing feature generation
    ├── predict_model.py                                <- Code to run prediction with previously trained models
    ├── site_explainability.py                          <- Code to generate explainability plots for a given site and issue_date
    ├── streamflow_cross_validation.py                  <- Code to run cross validation (CV) end-to-end
    ├── train_monthly_model.py                          <- Code to train the monthly Catboost models
    └── train_yearly_model.py                           <- Code to train the yearly Catboost models
```

## Directions for running the solution end-to-end

The steps below should be followed in order to reproduce the competition submission.

### 1. Requirements and installation

Requires Python 3.10. To install with the exact dependencies that will be used by DrivenData, create a new Python 3.10 virtual environment. For example, if using `pyenv`

```bash
pyenv install 3.10
pyenv virtualenv 3.10 water-supply-cv
pyenv activate water-supply-cv
```

to install dependencies, run:

```bash
pip install -r ./data_download/requirements.txt
pip install ./data_download/
pip install ./data_reading/
pip install -r requirements-train.txt
```

### 2. Data Download

Download all train and test raw data using the `bulk` command. From the repository root as your working directory, run:

```bash
WSFR_DATA_ROOT=training/train_data python -m wsfr_download bulk data_download/cv_config.yml
```

The train data will be downloaded to the directory `training/train_data` and the test data will be downloaded to the `data` directory.

### 3. Generate the train features file

From the repository root as your working directory, run:

```bash
python training/generate_train_features.py
```
The primary outputs of this script are two files located in `training/preprocessed_data`:
* `cv_features.csv` - a file containing all of the features used for training each cross-validation year.
* `cv_test_features.csv` - a file containing all of the test features used for predicting each cross-validation year.

The script also outputs feature correlations used for explainability visualizations in `training/preprocessed_data/feature_corrs`.

This process can take 1.5 - 2 hours. It also relies on access to https://climate.arizona.edu/snowview/csv/Download/Watersheds/ which has intermittent outages.

### 4. Train the models

From the repository root as your working directory, run:

```bash
python training/streamflow_cross_validation.py
```

The submission file will be located in `training/preprocessed_data/final_submission.csv`.
For each cross-validation year the models are saved in `training/preprocessed_data/models`.
This process also prints to the screen score output for each cross-validation year and, upon conclusion, an overall score for the cross-validation.


## Directions for generating explainability plots

### 1. Run cross-validation solution end-to-end above to produce the data files necessary for explainability plots.

Make sure the `water-supply-cv` environment you created is activated.

### 2. Generate the explainability plots for a specific site_id and issue_date

```bash
python training/site_explainability -s site_id -d issue_date
```

```bash
usage: site_explainability.py [-h] [-s SITE_ID] [-d ISSUE_DATE]

Generate explainability summary for previous forecast

options:
  -h, --help            show this help message and exit
  -s SITE_ID, --site-id SITE_ID
                        The site-id you would like to run explainability for.
  -d ISSUE_DATE, --issue-date ISSUE_DATE
                        The issue-date you would like to run explainbility for.
```

The plots will be saved to a directory corresponding to the site_id and issue_date: `preprocessed_data\plots\{site_id}\{issue_date}`.
