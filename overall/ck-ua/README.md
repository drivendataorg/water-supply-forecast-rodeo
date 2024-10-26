# Solution - Water Supply Forecast Rodeo: Final Prize Stage

Team: ck-ua
- Roman Chernenko (username: RomanChernenko)
- Vitaly Bondar (username: johngull)

## Summary

Our approach centered on using a Multi-Layer Perceptron (MLP) neural network
with five layers and residual connection for the season water level, which
proved to be the most effective within the given constraints.
The network was constructed to simultaneously predict the 10th, 50th, and 90th
percentile targets of water volume distribution. We experimented with various
network enhancements and dropout regularization but observed no substantial
improvement in the model's performance.

For our data sources, we relied on the NRCS and RFCs monthly naturalized flow,
USGS streamflow, USBR reservoir inflow, and NRCS SNOTEL data, all of which
were meticulously normalized
and encoded to serve as features for our training process. We propose a novel
approach for using SNOTEL data by training specialized RANSAC mini-models for
each site separately. For each of these mini-models the list of the used SNOTEL
stations are selected by heuristic approach.

In the final solution, we use a single model for each test cross-validation fold.
The data from the test year in each cross-validation split was removed from the
dataset and not used during the training/evaluation process in any way. All data
for the remaining years was randomly split into the train/validation datasets based on
years to prevent overfitting and make the model more robust.
We also employed data augmentation due to the
the limited size of our training dataset, which allowed us to artificially expand
our sample set.

The inference result file is used as-is without any modifications for the submission.

## Setup
We used the Ubuntu 23.10 OS for training/inference.

1. Install conda (or mamba) python package manager.
2. Install the required python packages from `environment.yml` file
using conda or mamba package manager.
For example, you can do it with the next command:

    ```bash
    conda env create -f environment.yml
    ```

3. Activate `water_forecast_final` environment using the next command:

    ```bash
    conda activate water_forecast_final
    ```

## Hardware

The solution was run on the PC with the following specifications:

- CPU: AMD Ryzen 7 5700G
- RAM: 32 GB
- OS: Ubuntu 23.10

The GPU is not mandatory for training and inference.
Training/inference benchmark time was measured without GPU.

Training time: 1h 35m

Inference time: 1m 20s

## Run training

1. Set the `src/` folder as the working directory.

2. Download the following competition data files into the `src/data`:

    - `geospatial.gpkg`
    - `metadata.csv`

3. Download the following competition data files into the `src/data/final_stage` folder:

    - `cross_validation_labels.csv`
    - `cross_validation_monthly_flow.csv`
    - `cross_validation_submission_format.csv`
    - `geospatial.gpkg`
    - `metadata.csv`
    - `prior_historical_labels.csv`
    - `prior_historical_monthly_flow.csv`

4. Download SNOTEL, USGS Streamflow, and USBR reservoir inflow data for
all available years that allowed for training in the `data` folder.
To download all training data automatically please run script `download_data.sh`.
After downloading the data you should receive the next directory structure in `src` folder:

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
    │   └── final_stage
    │       ├── USBR_reservoir_inflow.csv
    │       ├── ...
    ├── data_download
    ├── libs
    │   ├── data
    │   ├── losses
    │   ├── models
    │   └── optimizers
    └── results
    ```

    The USBR reservoir inflow data will be stored into `data/final_stage/USBR_reservoir_inflow.csv` file.

> [!NOTE]
> We had trouble downloading USBR reservoir inflow data from Ukraine due to API access problems.
> To fix this, we used a cloud instance in the USA to download the data instead.
> So we recommend running a data download script from the USA-located machine or cloud instance.

5. Remove any existing trained model files or submission result files from `results/final` folder.

6. Run the training script using the next command:

    ```bash
    ./train.sh
    ```

    The script not use any pretrained data-source models.
    The training process does not require network access except for data downloading through step #4.
    Model weights will be saved in the `results/final` folder.
    All weights model files require only approximately 16 MB of disk space.

## Run inference

You can run the inference without performing the training process.

1. Download all required data per Steps #1-#4 from training above.

2. Set the `src/` folder as the working directory.

3. Run the inference script using the next command:

    ```bash
    ./predict.sh
    ```

    The script automatically loads all needed files and performs an end-to-end inference process.
    The results of the inference will be saved in the `results/final/final_predict_submission_mlp_sumres.csv`
    file. There are no additional preprocessing or postprocessing steps are required. The script doesn't
    generate any intermediate files during the inference process except the final results in .csv format.
    The inference process does not require network access except for data downloading at step #1.
