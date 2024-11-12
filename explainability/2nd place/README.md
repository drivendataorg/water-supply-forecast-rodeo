# Solution - Water Supply Forecast Rodeo

**Username:** kamarain

## Summary

This repository contains the code to reproduce the solution of kamarain, outlining data processing and modeling steps as described in the Final Prize Stage report.

## Setup Instructions

1. **Download competition data files**

These files should be downloaded to `src/data/`

- `src/data/cdec_snow_stations.csv`
- `src/data/cross_validation_submission_format.csv`
- `src/data/cross_validation_labels.csv`
- `src/data/cross_validation_monthly_flow.csv`
- `src/data/prior_historical_labels.csv`
- `src/data/prior_historical_monthly_flow.csv`
- `src/data/geospatial.gpkg`
- `src/data/metadata.csv`

2. **Install the environment dependencies**

Install the necessary Python packages using `environment-kamarain-cpu.yml` and activate it:

```bash
conda env create -f environment-kamarain-cpu.yml
conda activate watersupply_kamarain
```

Note that this environment file includes installing the [competition-provided data download tool](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/tree/main).

## Hardware

The solution was run on [Puhti CRAY UNIX supercomputer](https://docs.csc.fi/computing/systems-puhti/) using 13 computing nodes per one request, fitting all models for one target month. A total of seven requests were needed to optimize and fit models for all issue dates and sites.

Each of the Xeon Gold 6230 nodes contain 2 x 20 cores @ 2,1 GHz and have 192 GB of RAM. A workload of one computing node comprised two target sites and four issue dates, totalling eight tasks. Each task was run with five CPU cores and 23 GB of RAM.

Running the code was also tested on a Dell Precision 3561 laptop (GPU was not used).

The heaviest version of the code, used in the Final Prize Stage and Explainability Stage, takes about 10 hours per month to fit on the Puhti supercomputer on average, including optimization and fitting of the SHAP analysis models. Out of all months, January was the most time consuming, taking over 16 hours to fit.

The most memory-intensive phase of the processing is the dimensionality increasing part, i.e. the calculation of the moving averages, lags, and differences. It can peak the memory consumption temporarily, especially for the PDSI and ECMWF gridded data sets containing multiple grid cells per target domain. The exact memory requirement is not clear, but 23 GB is known to be enough: the peak consumption is most likely in the range of 8–16 GB per task.

## Run training

> [!IMPORTANT]
> All of the following instructions use `src/` as the working directory.

### Download official data

Download either the full competition data or a subset using the provided `hindcast_train_config_kamarain.yml` and the official runtime tool.

```bash
python -m wsfr_download bulk hindcast_train_config_kamarain.yml
```

> [!NOTE]
> By default, this will downlaod to `./data`. If needed to change, ensure the environment variable `WSFR_DATA_ROOT` is correctly defined, and if not, redefine it:
>
> ```bash
> export WSFR_DATA_ROOT=/path/to/competition/data
> ```

### USGS neighborhood sites metadata

Metadata files for downloading additional USGS data are provided at `src/usgs_neighborhood_sites_<site>.csv`. These files contain identification codes for those neighbouring sites/stations that have 1) long observational records 2) contain no (or only small) gaps in their time series and 3) are still operational currently. All these requirements are critical for fitting and especially for successful real time forecasting.

If you need to recreate them, run the script `prepare_neighborhood_metadata.py`. The logic of this process is as follows:

- First download all possible data for all years around the target sites.
- Then, for each relevant variable (streamflow, water level in wells, temperature, ...) and statistic code (mean, median, min, max, ...) separately, concatenate the data you got to a multi-year (dates as index), multi-site (neighboring station names as columns) dataframe for each target site. Do not forget to include full NAN rows for dates that do not contain data at all.
- Based on this dataframe, apply rules to pick the suitable neighboring stations: e.g., must contain data at least for the period 1990–2023, must not contain more than 15% of missing data, must contain data for the day of retrieval.
- Finally collect the filtered site codes, variables, and statistic codes to the `usgs_neighborhood_sites_<site>.csv` files.

### Download additional data

> [!WARNING]
> Please note the CDS providing the ECMWF seasonal forecasts is about to change to [new system in September 2024](https://forum.ecmwf.int/t/the-new-climate-data-store-beta-cds-beta-is-now-live/3315).
>
> Therefore the downloading script provided for that might not work properly or at all after the change. Additionally, downloading the ECMWF data can be very slow.

Download and process additional data with:

```bash
bash run_get_process_data.bash
```

### Check Data Directory Structure

Check that your `data/` contains the following folders with data inside them:

 ```bash
cdec
cds
models
pdsi
snotel
swann
teleconnections
teleindices
usgs_neighborhood
usgs_streamflow
```

### Model fitting

> [!IMPORTANT]
> All of the following instructions use `src/` as the working directory.

The provided version of the code and models differ slightly from the Final Prize Stage version code and models. The differences:

- The number of optimization rounds was reduced from 200 to 50 to speed up the process
- The number of fitted estimators was reduced from 500 to 100 to save disk space and ease fitting and application of the SHAP analysis models

These changes do not impact the accuracy of the models significantly. The submission volume becomes smaller, making it easier to send and handle. See `fig_combined_combined_kfold3_loocv.png` for details of the performance of the provided model files, and compare it with Final Prize Stage report Figure 6.

For running the file `run_fit_optim.bash` to fit the models check the input settings in the beginning of the file:

 ```bash
- issue_month               Defines the target forecast month to fit and/or optimize. Format is '0X' where X is the month.
- experiment                Defines the cross-validation strategy. Can be either 'kfold3' (fast to fit) or 'loocv' (slow, but replicates the Final Prize Stage approach)
- optimize                  Logical switch to define whether to optimize the model hyperparameters. Either 'False' or 'True'
- use_preopt_results        Logical switch to define whether to use previously optimized hyperparameters. Either 'False' or 'True'. Should not be 'True' when 'optimize' is 'True'.
- fit_models                Logical switch to define whether to fit the actual forecasting models. Either 'False' or 'True'.
- fit_shap                  Logical switch to define whether to fit and apply the SHAP analysis models for plotting the figures in the Explainability Reports. Either 'False' or 'True'.
- use_slurm                 Logical switch to define whether to use the SLURM supercomputing environment. Either 0 or 1.

- code_dir                  The exact path to the code to be executed, eg. `/src`
- data_dir                  The exact data path, eg. `data/`. This path will store also the model files and other output from the code.

- eval "$(path/to/conda shell.bash hook)" Installation path of Conda
 ```

Preoptimized hyperparameters are included in the model files.

To replicate the full Final Prize Stage code, including the fittings of the SHAP models for Explainability Reports, the settings would look like this:

```bash
issue_month='01'            # '01 # '02' # '03' # '04' # '05' # '06' # '07'
experiment='loocv'          # 'kfold3' # 'loocv'
optimize='True'             # 'False' # 'True'
use_preopt_results='False'  # 'False' # 'True'
fit_models='True'           # 'False' # 'True'
fit_shap='True'             # 'False' # 'True'
use_slurm=1                 # 0 # 1
```

Run the `run_fit_optim.bash` script seven times, changing the `issue_month` parameter between runs. The output is directed to `data/`.

## Run inference

> [!IMPORTANT]
> All of the following instructions use `src/` as the working directory.

### Download official and additional data

_If data was not already downloaded for training..._

Download and preprocess all data as explained previously. To save time, only data for years 2004–2023 is required and they can be downloaded by modifying the year ranges in the files `hindcast_train_config.yml` and `run_get_process_data.bash`.

### Real-time inference

For real-time inference and forecasting, use the `solution.py` file with a suitable submission format file, as defined at the Forecast Stage of the competition. This file also downloads and processes the most up-to-date data. Note that it has some global variables in the beginning which need to be adjusted (`DOWNLOAD_DATA` and `FOLDS`). This file can not forecast historical dates.

### Inference of the historical forecasts

The scripts `run_fit_optim.bash` and `analyse_combined_results.py` can be used for inference of the historical forecasts in the cross-validation framework. The `run_fit_optim.bash` calls the script `fit_optim_qregress_multi.py` which performs all optimization, fitting, and inference tasks of the Final Prize Stage. That script can also handle real-time inference: for that, change the year range in the `all_valid_years` parameter.

Set up the run file parameters in `run_fit_optim.bash` like this:

```bash
issue_month='01'            # '01 # '02' # '03' # '04' # '05' # '06' # '07'
experiment='loocv'          # 'kfold3' # 'loocv'
optimize='False'            # 'False' # 'True'
use_preopt_results='False'  # 'False' # 'True'
fit_models='False'          # 'False' # 'True'
fit_shap='False'            # 'False' # 'True'
use_slurm=1                 # 0 # 1
```

This setup assumes the provided model files, which have been fitted with LOOCV cross-validation.

Run the `run_fit_optim.bash` seven times, changing the `issue_month` parameter between runs. The output is directed to `data/`.

After all runs completed, run the file `analyse_combined_results.py`. It will collect the forecasts from `run_fit_optim.bash` to one output file. Some helpful metric information is plotted as well. Please check the name of the experiment ('loocv') in the beginning of the file, and change it to 'kfold3' if using that cross-validation approach. The script can handle multiple experiments defined in the 'experiments' list, and it will output the cross-validated historical forecasts as CSV files in the format of the Final Prize Stage. The result files are named `cross_validation_submission_format_{experiment}.csv` and `cross_validation_submission_format_withObs_{experiment}.csv`, the latter also containing the observations of the target streamflow for reference.

## Plot figures for the Explainability and Communication Bonus Track

Run the `explain_forecasts.py` script. It requires the forecast date and the target site as input parameters:

```bash
python explain_forecasts.py 2023-03-15 owyhee_r_bl_owyhee_dam &
python explain_forecasts.py 2023-05-15 owyhee_r_bl_owyhee_dam &
wait
```

The script makes its own inference, so it is not dependent on the previous stages (fitting or inference) except the data download/preprocessing stage.

The provided model files include the SHAP models for all sites and issue times, so it is possible to generate Explainability figures everywhere and for any issue date.

The output is directed to `data/` with the filename pattern `fig_explainability_*.png`.
