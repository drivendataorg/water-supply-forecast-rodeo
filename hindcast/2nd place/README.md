# Solution - Water Supply Forecast Rodeo: Hindcast Evaluation

Team: ck-ua

- Roman Chernenko (username: RomanChernenko)
- Vitaly Bondar (username: johngull)

## Summary

Our approach centered on using a Multi-Layer Perceptron (MLP) neural network
with four layers, which proved to be the most effective within the given
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

We used the Ubuntu 23.10 OS for training/inference.

### Environment

> [!NOTE]
> The provided `environment.yml` requirements file may not resolve for OSes other than Linux.

1. Install conda (or mamba) python package manager.
2. Install the required python packages from `environment.yml` file using conda or mamba package manager.
For example, you can do it with next command:

    ```bash
    conda env create -f environment.yml
    ```
4. Activate `water_forecast_hindcast` environment using next command:

    ```bash
    conda activate water_forecast_hindcast
    ```
5. Change your working directory to the `src/` directory

### Data

Download the following competition data files:

- `metadata.csv` -> `src/data/metadata.csv`
- `submission_format` -> `src/data/submission_format.csv`
- `train_monthly_naturalized_flow.csv` -> `src/data/train_monthly_naturalized_flow.csv`
- `train.csv` -> `src/data/train.csv`

## Hardware

The solution was run on the PC with next specifications:

- CPU: AMD Ryzen 7 5700G
- RAM: 32 GB
- OS: Ubuntu 23.10

The GPU is not mandatory for training and inference.
Training/inference benchmark time was measured without GPU.

Training time: 23 m

Inference time: 27 s

## Run training

The following instructions use `src/` as the working directory.

1. Need to download SNOTEL and USGS Streamflow data for all available years that
allowed for training in `src/data` folder. Please use the competition's [`wsfr_download` tool](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime?tab=readme-ov-file#data-download) (installed as part of the requirements) with next files:

    ```bash
    python -m wsfr_download bulk data_download/hindcast_train_config_snotel.yml
    python -m wsfr_download bulk data_download/hindcast_train_config_usgs_streamflow.yml
    ```

    After downloading the data you should receive next directory structure in `src` folder:

    ```
    ├── configs
    ├── data
    │   ├── snotel
    │   │   ├── FY1964
    .....
    │   │   ├── FY2004
    │   │   ├── FY2006
    │   │   ├── FY2008
    │   │   ├── FY2010
    │   │   ├── FY2012
    │   │   ├── FY2014
    │   │   ├── FY2016
    │   │   ├── FY2018
    │   │   ├── FY2020
    │   │   ├── FY2022
    │   └── usgs_streamflow
    │       ├── FY1890
    .....
    │   │   ├── FY2004
    │   │   ├── FY2006
    │   │   ├── FY2008
    │   │   ├── FY2010
    │   │   ├── FY2012
    │   │   ├── FY2014
    │   │   ├── FY2016
    │   │   ├── FY2018
    │   │   ├── FY2020
    │   │   ├── FY2022
    ├── data_download
    ├── libs
    │   ├── data
    │   ├── losses
    │   ├── models
    │   └── optimizers
    └── results
    ```

2. Run training script using next command:

    ```bash
    ./train.sh
    ```

    The script not used any pretrained data-source models.
    Training process not require the network access except for data downloading at step #1.
    Model weights will be saved in `results` folder.
    All weights model files require only approx. 5 MB of disk space.

## Run inference

The following instructions use `src/` as the working directory. To run inference locally:

1. Download the following competition data files:

    - `test_monthly_naturalized_flow.csv` -> `src/data/test_monthly_naturalized_flow.csv`


2. Need to download SNOTEL and USGS Streamflow data for all available test years to the `data/` folder.
Please use the competition's `wsfr_download` tool with next files:

    ```bash
    python -m wsfr_download bulk data_download/hindcast_test_config_snotel.yml
    python -m wsfr_download bulk data_download/hindcast_test_config_usgs_streamflow.yml
    ```

    After downloading the data you should receive next directory structure in `src` folder:

    ```
    ├── configs
    ├── data
    │   ├── snotel
    │   │   ├── FY2005
    │   │   ├── FY2007
    │   │   ├── FY2009
    │   │   ├── FY2011
    │   │   ├── FY2013
    │   │   ├── FY2015
    │   │   ├── FY2017
    │   │   ├── FY2019
    │   │   ├── FY2021
    │   │   └── FY2023
    │   └── usgs_streamflow
    │       ├── FY2005
    │       ├── FY2007
    │       ├── FY2009
    │       ├── FY2011
    │       ├── FY2013
    │       ├── FY2015
    │       ├── FY2017
    │       ├── FY2019
    │       ├── FY2021
    │       └── FY2023
    ├── data_download
    ├── libs
    │   ├── data
    │   ├── losses
    │   ├── models
    │   └── optimizers
    └── results
    ```

3. Run inference script using next command:

    ```bash
    ./predict.sh
    ```

    The script automatically loaded all needed files and performed end-to-end inference process.
    Results of the inference will be saved in the `results/predict_submission_mlp_sumres_cv.csv` file.
    No any additional preprocessing or postprocessing steps are required.
    The script doesn't generate any intermediate files during inference process except the final results in .csv format.
    Inference process not require the network access except for data downloading at step #1.

## Code submission

The repository also contains `src/solution.py` file that was used during code submission stage.

To package the code submission:

```bash
zip submission.zip -r solution.py results libs configs
```
