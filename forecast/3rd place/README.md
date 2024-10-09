# Water Supply Forecast Rodeo - Forecast Stage

## Summary

We use ensembles of LightGBM models with Tweedie loss for point forecast and quantile loss for 0.10 and 0.90 quantile forecast. We mainly use lagged data of SNOTEL SWE and cumulative precipitation averaged within and near the basins from top 𝐾 = 9 sites as the input of the model. Using additional features can cause overfitting easily due to limited training sample size for each site and lack of signal/information loss due to data aggregation. We use synthetic data generation to mitigate small sample size problems and increase training sample size by 5x, which significantly improves forecast skill, prediction interval reliability, and generalizability. We also incorporate daily USGS and USBR observed flow from sites with minimal impairment. In addition, we also incorporate gridMET PDSI features, which improves forecast skill around 1 KAF compared to previous model.

## Changes compared to Hindcast Stage

### Training

* Add PDSI features
* Retrain with additional data from odd years 2004-2023
* Exclude anomalous years for training (2011 and 2015)
* Scale SNOTEL SWE and cumulative precipitation with 0-max so averaging still can be done in case of missing SNOTEL sites
* Only include recent lag t-1 or t-3 for SNOTEL data features to reduce dependency
* Reduce random seeds iteration for ensemble (9 → 3) to reduce model size
* Inner CV ensemble is based on recent 3 validation years (2020-2022) rather than all 20 validation years to reduce model size

### Inference

* Corner case handling if data not available
    * Minimum 65% of SNOTEL sites is available for a given site, otherwise forecast skill decline will be higher than 3 KAF
    * Backup models with K=5 SNOTEL sites and lag t-3 to reduce dependency in case of missing stations
    * In case less than 65% of SNOTEL sites are available for a single site or any dependent features are missing, the inference code will always pick the ensemble of models with the latest data available
* Update inference code to filter date based on issue date for clarity (in fact, it does not change the result since we use lagged variables as the input and this already guarantees no future data leak)
* Update inference code to preprocess data for each issue date rather than in bulk like in the Hindcast Stage

## Setup

* Python 3.10.13
* Packages:

    ```bash
    pip install "setuptools==73.0.1" cython "flit-core<4"
    pip install -r requirements.txt --no-build-isolation
    ```

## Data preparation

1. Prepare metadata
    * Run `python -m src.meta.get_meta` to fill missing drainage area from original metadata and generate `metadata_proc.csv`
    * Run `python -m src.meta.get_snotel_sites_basins_all` to generate pair between SNOTEL stations and forecast sites
    * Run `python -m src.meta.get_snotel_sites_basins` to generate filtered pair between SNOTEL stations and forecast sites
    * Copy `data.find.txt` from [competition runtime](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/blob/main/data.find.txt) to `data/meta`. This file is used to filter only available SNOTEL stations from the runtime for the training. There's only few SNOTEL stations which are not available on the runtime and the score difference is negligible.
2. Download raw data for all training years
    * SNOTEL SWE -> run `python -m src.data.snotel` to download SNOTEL data based on defined pair between SNOTEL sites and drainage sites
    * USGS Streamflow -> run `python -m src.data.usgs_streamflow` (rdb version) or use [competition runtime data download](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/tree/main/data_download) to retrieve USGS data
    * USBR Streamflow -> run `python -m src.data.usbr` to download USBR data from selected 4 sites
    * gridMET PDSI -> run `python -m src.data.pdsi` to download PDSI data
* You also can update the directory for these 4 data sources on `src/config.py`

## Training

Run `train.sh` to train all models for different seeds for each validation years (2020-2022) iteration. For each single experiment or config training model run, the output will be:

* `models` directory of generated models, reused for the inference code submission
* `snotel_sites_basins` pair between SNOTEL stations and forecast sites for each validation years iteration, reused for the inference code submission
* `pred.csv` all predictions for OOF validation years (one for each site and issue date) and test years (ten for each site and issue date)
* `scale_factor` scale factor used to generate synthetic data for each validation years iterations

## Inference

The source code for an inference submission is in `submission/`. To add the trained model artifacts:

- Copy `data/meta/metadata_proc.csv` to `submission/data/meta/metadata_proc.csv`
- Copy `data/meta/snotel_sites_basins/snotel_sites.feather` to `submission/data/meta/snotel_sites.feather`.
- Copy the `runs/` output directory to `submission/runs/`.
- Run `median.py` and copy `data/meta/1991_2020_train_stats.csv` to `submission/data/meta/1991_2020_train_stats.csv`. These historical values are used for logging diagnostic information.

## Machine Specifications

* CPU: Core i5
* RAM: 8GB
* Training duration: approximately 45 minutes
* Inference duration: less than 2 minutes
