# Solution - Water Supply Forecast Rodeo: Forecast Stage

Team: ck-ua

- Roman Chernenko (username: RomanChernenko)
- Vitaly Bondar (username: johngull)

## Summary

Our approach centered on using a Multi-Layer Perceptron (MLP) neural network
with four layers and residual connection for the season water level, which proved to be the most effective within the given
constraints.
The network was constructed to simultaneously predict the 10th, 50th, and 90th
percentile targets of water volume distribution. We experimented with various
network enhancements and dropout regularization but observed no substantial
improvement in the model's performance.

For our data sources, we relied on the NRCS and RFCs monthly naturalized flow,
USGS streamflow, and NRCS SNOTEL data, all of which were meticulously normalized
and encoded to serve as features for our training process. We propose a novel
approach for using SNOTEL data by training specialized RANSAC mini-models for
each site separately. For each of these mini-models the list of the used SNOTEL
stations are selected by heuristic approach.

Our validation strategy involved a repeated k-fold cross-validation based on
years, to both avoid overfitting and to reinforce the robustness of our model.
This led us to train 25 models with distinct training years and, upon inference,
we employed an ensemble of all models to determine the median value for
predictions of each percentile. We also employed data augmentation due to the
limited size of our training dataset, which allowed us to artificially expand
our sample set.

Inference result file used as-is without any modifications for the submission.

## Setup

We used Ubuntu 23.10 OS for training/inference.

1. Install conda (or mamba) python package manager.
2. Install the required python packages from `environment.yml` file using conda or mamba package manager.
For example, you can do it with next command:

    ```bash
    conda env create -f environment.yml
    ```

3. Activate `water_forecast_forecast` environment using next command:

    ```bash
    conda activate water_forecast_forecast
    ```


## Hardware

The solution was run on the PC with next specifications:

- CPU: AMD Ryzen 7 5700G
- RAM: 32 GB
- OS: Ubuntu 23.10

The GPU is not mandatory for training and inference.
Training/inference benchmark time was measured without GPU.

Training time: 29m 23s

Inference time: 1m 43s for all sites at 2024-01-01

## Run training

1. Set the `src/` folder as the working directory.

2. Download the following files from the competition into the `src/data` folder:

    - `metadata.csv`
    - `forecast_train_monthly_naturalized_flow.csv`
    - `forecast_train.csv`
    - `geospatial.gpkg`

3. Need to download SNOTEL and USGS Streamflow data for all available years that
allowed for training in `data/` folder. Please run script `download_train_data.sh` to download all training data automatically.
After downloading the data you should receive next directory structure in `src` folder:
    ```
    ├── configs
    ├── data
    │   ├── snotel
    │   │   ├── FY1964
    .....
    │   │   ├── FY2013
    │   │   ├── FY2014
    │   │   ├── FY2015
    │   │   ├── FY2016
    │   │   ├── FY2017
    │   │   ├── FY2018
    │   │   ├── FY2019
    │   │   ├── FY2020
    │   │   ├── FY2021
    │   │   ├── FY2022
    │   │   ├── FY2023
    │   └── usgs_streamflow
    │       ├── FY1890
    .....
    │   │   ├── FY2014
    │   │   ├── FY2015
    │   │   ├── FY2016
    │   │   ├── FY2017
    │   │   ├── FY2018
    │   │   ├── FY2019
    │   │   ├── FY2020
    │   │   ├── FY2021
    │   │   ├── FY2023
    │   │   ├── FY2023
    ├── data_download
    ├── libs
    │   ├── data
    │   ├── losses
    │   ├── models
    │   └── optimizers
    └── results
    ```

4. Remove any existing trained model or submission result files from `results/forecast` folder.

5. Run training script using next command:

    ```bash
    ./train.sh
    ```

    The script not used any pretrained data-source models.
    Training process not require the network access except for data downloading at step #1.
    Model weights will be saved in `results/forecast` folder.
    All weights model files require only approx. 5 MB of disk space.

## Run inference

The repository contains file `src/solution.py` with `predict` and `preprocess` functions according to
the official code submission format.

The solution uses only the next data sources:

- monthly naturalized flow
- USGS streamflow
- NRCS SNOTEL data

All needed data files for 2024 forecast year should be prepared before running the solution. The function `preprocess`
doesn't download any external data and provided only for compatibility with the official API.

You can read more about code submission format
[here](https://www.drivendata.org/competitions/259/reclamation-water-supply-forecast/page/828/)
at DrivenData website.

Alternatively, you can use the file `src/test_solution_forecast.py` to run the inference for all sites for single date. This file is not required and provided only for test purposes.

This script require to has all input data for 2024 forecast year in `src/data/forecast_stage`. See the [runtime repository](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/tree/main?tab=readme-ov-file#data-download) from the challenge organizers for instructions, and example download configuration files [here](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/blob/main/data_download/forecast_config-2024-01-08.yml).
