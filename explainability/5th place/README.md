# Water Supply Forecast Rodeo (Explainability)

Username: Atabek Umirbekov, Changxing Dong

## Summary

The solution is designed to handle data preprocessing, model training, prediction generation, and explainability for forecasting seasonal water volume in basins across the western US. It integrates hydrological and climate data, including SNOTEL SWE, SWANN SWE, NRCS flow data, and climate indices like SOI, PNA, and PDO. The predictions are made using Partial Least Squares (PLS) regression models, and the results are complemented with visualizations to explain the key inputs driving each forecast.

### Key Features

- **Self-explanatory**: Each code chunk is accompanied by a clear explanation of its function.
- **Dynamic Data Handling**: The script checks for the availability of the most recent data prior to each forecast issue date, ensuring that it works with the latest available inputs. Training and inference are conducted simultaneously, with up-to-date information, allowing for real-world operational use.
- **Data Integration**: The model integrates multiple data sources (SWE, precipitation, monthly flows, and climate indices) and identifies the most representative inputs that have the strongest association with seasonal flow volume, ensuring relevant predictors are used.
- **Forecasting and Explainability**: The script uses PLS regression models with Leave-One-Out Cross-Validation (LOOCV) to generate predictions for the 10th, 50th, and 90th percentiles.  It conducts variance-based variable importance analysis and  produces visual explanations of the forecasts by illustrating how different inputs (SWE, precipitation, flow) compare with historical data.
- **Parallel Processing**: The script takes advantage of multiple CPU cores to efficiently process and forecast for all basins and issue dates.
- **Post-processing of the outputs**: The script checks the consistency of the predictions and constrains any unrealistic outcomes. Since PLS regression tend to extrapolate linearly, extreme predictions, particularly in very dry years, may result in negative water volumes. To prevent this, the script imposes constraints, ensuring that predictions do not fall below the historical minimum values for each basin.

### Snotel Station Selection

The station selection process identifies SNOTEL stations with the highest correlation between cumulative precipitation (PRCP) or snow water equivalent (SWE) and seasonal streamflow volume. The code comprises four parts:

1. Metadata Preparation: Sorts station IDs, joins with metadata, filters by data availability, downloads SNOTEL data, and merges it with streamflow data.

2. Representative Stations: Calculates correlations between SNOTEL-derived PRCP and SWE values and streamflow volume for each basin. For each season, stations with over 20 observations and a median correlation above 0.3 are considered. Up to four high-correlation stations are selected per basin.

3. Donor Stations: Identifies basins with fewer than three high-correlation stations, finding nearby donor stations within the same HUC to fill gaps.

4. Final Station List: Combines selected and donor stations into a final list, saved as `data/selected_snotel_meta.csv`, to support seasonal streamflow forecasting for each basin.

### Model Training and Prediction

This script operates in several key stages:

1.	Package Installation and Setup: It first installs and loads the necessary R packages for handling the data and conducting the forecasts.
2.	Data Preprocessing: Preprocessed inputs, such as SWE and flow data, are loaded and filtered for the relevant forecast issue dates.
3.	Inference and Model Training: The script sets up a parallel processing loop, training a PLS regression model for each basin and forecast issue date using LOOCV.
4.	Prediction and Explainability: It generates forecasts and creates visualizations to explain the role of key inputs, like SWE, precipitation, and antecedent flows.
5.	Postprocessing: The forecast results are checked for consistency, with adjustments made to ensure logical constraints are respected, such as limiting negative predictions using historical minimum thresholds.

## Setup

1. Operating system tested for our code is Ubuntu version 24.04. The following packages should be installed with root access rights (e.g. sudo) through shell script **OSpackages.sh**.
    - build-essential
    - r-base (R version 4.3)
    - git
    - python3
    - python3-dev
    - gdal-bin
    - libgdal-dev

  Alternatively, you can use conda with the `environment.yml` file:

  ```bash
  conda env create -f environment.yml
  conda activate iamo-team
  ```

2. Install the R-packages through R-script file (`Rscript **Rpackages.R**`). If you installed R with administrator rights, you made need to use `sudo`.
    - readr
    - dplyr
    - tidyr
    - lubridate
    - data.table
    - terra
    - reshape2
    - scales
    - doParallel
    - caret
    - pls
    - vip
    - ggplot2
    - knitr
    - pdp
    - gridExtra

## Data setup

### Competition data

Download the following competition data files to the following locations:

- `solution/data/geospatial.gpkg`
- `solution/data/metadata.csv`
- `solution/data/cross_validation_supplementary_nrcs_monthly_flow.csv`
- `solution/data/prior_historical_supplementary_nrcs_monthly_flow.csv`
- `solution/cross_validation_labels.csv`
- `solution/cross_validation_monthly_flow.csv`
- `solution/cross_validation_submission_format.csv`
- `solution/prior_historical_labels.csv`
- `solution/prior_historical_monthly_flow.csv`

### SNOTEL

Download the Snow Telemetry (**SNOTEL**, https://wcc.sc.egov.usda.gov/awdbRestApi/services) using the competition data download tool ([`wsfr_download`](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/tree/main/data_download)) and the `download_snotel.yml` configuration file:

```bash
WSFR_DATA_ROOT=./solution/data python -m wsfr_download bulk download_snotel.yml
```

### SWANN

Download Snow Water Artificial Neural Network (**SWANN**, https://climate.arizona.edu/data/UA_SWE/WYData_4km) data from the University of Arizona website. Water years 1983 through 2023 are needed.

## Hardware

The solution was run on Dell notebook G3 15 with 32 GB memory, CPU i7-9750H.

Training time: about 50 Min

Inference time: several seconds


## Run the Code

First, you should **cd** into the **`solution/`** directory. All of the following scripts use this as the working directory.

### Cross-Validation Submission

#### SNOTEL Station Selection

Run in a terminal

```bash
Rscript basin_snotel_selection.R
```

will produce a file **selected_snotel_meta.csv** in the **data/** directory


#### Submission file of the cross-validation

Run in a terminal

```bash
Rscript FINAL.R
```

Several intermediate files are save in the directory **explain/**, which will be used for the explainability communication run.

## Explainability and Communication Submission

This script generates forecast reports in the **reports/** directory for each basin and issue date specified in submission_format_explainability.csv.

To run the code in a terminal:

```bash
Rscript EXPLAINABILITY.R
```

In this example, the code generates PDF reports with graphical results for two basins on two issue dates: **pueblo_reservoir_inflow** and **owyhee_r_bl_owyhee_dam** on **2023-03-15** and **2023-05-15**:

```
pueblo_reservoir_inflow_2023-03-15.pdf
pueblo_reservoir_inflow_2023-05-15.pdf
owyhee_r_bl_owyhee_dam_2023-03-15.pdf
owyhee_r_bl_owyhee_dam_2023-05-15.pdf
```
