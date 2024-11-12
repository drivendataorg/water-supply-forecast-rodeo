# water_supply_forecast_rodeo_competition

## Introduction

The repository contains a full Python solution for the [Water Supply Forecast Rodeo](https://www.drivendata.org/competitions/group/reclamation-water-supply-forecast/) competition for its Final Stage.
The competition aims at predicting water supply for 26 western US hydrologic sites. Forecasts are made for different quantiles - 0.1, 0.5 and 0.9. Additionally, the predictions are made for different
issue dates. For most hydrologic sites, the predictions are made for the volume of water flow between April and July, but the forecasts are issued in different months (4 weeks from January, February, March, April,
May, June, July).

The solution makes predictions based on the approach of creating LightGBM models for different months and averaging their results with distribution estimates from historical data.

In the third stage ([Final Prize Stage](https://www.drivendata.org/competitions/262/reclamation-water-supply-forecast-final/), the objective was to make predictions using LOOCV
(Leave-One-Out Cross Validation) with 20 folds where one of the years between 2004-2023 is a test set. It is different from the other stages in the way that using this fold CV year data for
creating aggregations is prohibited, so each fold's data has to be processed separately.

A 12-page summary of the solution is in `reports/report.pdf`.

## Content

1. Scripts to run to create and test models
	1. `cds_downloads.py` - it downloads CDS data (CDS monthly data and seasonal forecasts). However, using notebooks/CDS downloads.ipynb is recommended to easier keep track of already downloaded data.
	2. `data_processing.py` - it processes the data and saves a DataFrame to be used in `model_training.py`.
	3. `model_params.py` - contains model hyperparameters and features for different months to be used in `model_training.py`. It saves these parameters as files for simplicity.
	4. `distribution_estimates.py` - it fits data for each hydrologic site to different distributions, optimizes parameters for the distribution, selects distributions with the best fit to data,
		adds manually chosen amendments to best distribution fits and saves results for each LOOCV fold (excluding given LOOCV year from the processing).
	5. `model_training.py` - trains the models. There are different parameters to choose from for this script. By default, `RUN_FINAL_SOLUTION` is set to `True`, which ignores other parameters and
	trains all models required to obtain the final solution. Models' hyperparameters were already read from `model_params.py`. Still, creating hyperparameters by yourself is also supported to check
	if hardcoded hyperparameters are indeed the outputs of running the function. Keep in mind that hyperparameters optimization takes long (20-50 hours).
2. Auxiliary scripts
	1. `utils.py` - general utilities, functions/list/dictionary that could be used for different tasks.
	2. `feature_engineering.py` - functions to facilitate feature engineering.
	3. `train_utils.py` - functions dedicated for model training.
	4. `cv_and_hyperparams_opt.py` - functions with main cv/hyperparameters tuning logic.

## Generated files

1. Distribution estimates (`data/distr`). Contains 4 types of output files from `distribution_estimates.py`:
	1. `distr_per_site_[year]` - all fitted distributions with site_id-distribution_name-distribution_fit-parameter_values combinations from a given year.
	2. `distr_per_site_best_[year]` - one best fit for each site_id from a given year (site_id-distribution_name-parameter_values combinations).
	3. `distr_amend_[year]` - amendments to make to distr_per_site_best_[year]. It contains (site_id-distribution_name-parameter_values combinations) of site_ids with distributions to change
	for a given year.
	4. `distr_final_[year]` - final distributions used in the model. It is the result of merging distr_per_site_best_[year] with `distr_amend_[year]` amendments for selected site_ids for a given year.
2. Results (`results/`)
	1. Contains `submission_2024_03_28.csv` with final results submitted to the competition.
	2. Contains hyperparameters tuning results for the Final Stage of the competition, one per month (different quantiles were optimized together).
5. Notebooks (`notebooks/`)
	1. Additional analyses.
	2. CDS data download. It was provided in a notebook to facilitate keeping track of download progress.

## How to run

The solution was created using Python version 3.10.13 on Windows 11.

*Keep in mind that results won't be exactly the same as those from models/ repo directory when downloading data again, as some of the datasets could be updated (it happened for example
with USGS streamflow. There was a data update in 2024 but data available on 2023-11-10 was used in the solution, to not take into account future update that wasn't available at a time
when the predictions would have been made if it was run real-time).*

1. Install dependencies:
	1. Install Python packages from requirements (`pip install -r requirements.txt`).
	2. If you run into problems with using eccodes package, try to install it with conda (`conda install -c conda-forge eccodes==2.33.0`)
2. Follow the official guidelines to use CDS API https://cds.climate.copernicus.eu/api-how-to. It requires creating an account, saving API key and agreeing to the Terms of Use
	of every dataset that you intend to download. When running CDS download for the first time, a link to agreeing to the Terms should be accessible within code output.
3. Create `data/` directory within the project if needed. All data should be downloaded to this directory.
4. Download data from the [competition website](https://www.drivendata.org/competitions/262/reclamation-water-supply-forecast-final/data/). The following files are needed:
	- `cross_validation_labels.csv`
	- `cross_validation_monthly_flow.csv`
	- `geospatial.gpkg`
	- `metadata.csv`
	- `prior_historical_monthly_flow.csv`
	- `prior_historical_labels.csv`
5. Download feature data using the competition-provided `wsfr_read` package with the `hindcast_test_config.yml` file in this project. See the instructions from the Data download section from the [water-supply-forecast-rodeo-runtime repo](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime?tab=readme-ov-file#data-download) for further detail if needed.
6. Download CDS data. There are 2 options to achieve that:
	1. [Recommended] Use notebooks/CDS downloads.ipynb (https://github.com/progin2037/water_supply_forecast_rodeo_competition/blob/main/notebooks/CDS%20downloads.ipynb).
	2. Use `cds_downloads.py` (https://github.com/progin2037/water_supply_forecast_rodeo_competition/blob/main/cds_downloads.py). This way, it will be harder to keep track of already downloaded data.
6. Run `data_processing.py`.
7. Run `model_params.py`.
8. Run `distribution_estimates.py`. Output from this script is already saved in this repository in data/distr, as running the script takes about 4-6 hours.
9. Run `model_training.py`.
