# Water Supply Forecast Rodeo - Final Stage

## Summary

We use ensembles of LightGBM models with Tweedie loss for point forecast and quantile loss for 0.10 and 0.90 quantile forecast. We mainly use lagged data of SNOTEL SWE and cumulative precipitation averaged within and near the basins from top 𝐾 = 9 sites as the input of the model. Using additional features can cause overfitting easily due to limited training sample size for each site and lack of signal/information loss due to data aggregation. We use synthetic data generation to mitigate small sample size problems and increase training sample size by 5x, which significantly improves forecast skill, prediction interval reliability, and generalizability. We also incorporate daily USGS and USBR observed flow from sites with minimal impairment. In addition, we also incorporate gridMET PDSI, UA/SWANN SWE, ERA5-Land, and seasonal forecast products from ECMWF SEAS51 features, which improves forecast skill around 5 KAF compared to previous model.

## Changes compared to Hindcast Stage

### Training

#### Forecast Stage

* Add PDSI features
* Retrain with additional data from odd years 2004-2023
* Exclude anomalous years for training (2011 and 2015)
* Scale SNOTEL SWE and cumulative precipitation with 0-max so averaging still can be done in case of missing SNOTEL sites
* Only include recent lag t-1 or t-3 for SNOTEL data features to reduce dependency
* Reduce random seeds iteration for ensemble (9 → 3) to reduce model size
* Inner CV ensemble is based on recent 3 validation years (2020-2022) rather than all 20 validation years to reduce model size

#### Final Stage

* Add CDEC SWE features
* Add UA-SWANN SWE features
* Add seasonal forecast ECMWF SEAS51 features
* Add ERA-5 land features
* Scale target variable with 0-max
* Ensemble
    * Disable inner CV ensemble
    * Disable different random seeds variation (3 → 1)
* Synthetic generation function is adjusted for features with negative values or negative correlation as we add new features (e.g. temperature)

## Setup

* Python 3.10.13
* Packages:

    ```bash
    pip install "setuptools==73.0.1" cython "flit-core<4"
    pip install -r requirements.txt --no-build-isolation
    ```

### Data preparation
0. Download the following competition data files to `data/raw`:
    * `metadata.csv`
    * `geospatial.gpkg`
    * `supplementary_nrcs_metadata.csv`
1. Prepare metadata
    * Run `python -m src.meta.get_meta` to fill missing drainage area from original metadata and generate `metadata_proc.csv`
    * Run `python -m src.meta.get_snotel_sites_basins_all` to generate pair between SNOTEL stations and forecast sites
    * Run `python -m src.meta.get_snotel_sites_basins` to generate filtered pair between SNOTEL stations and forecast sites
    * Copy `data.find.txt` from [competition runtime](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/blob/main/data.find.txt) to `data/meta`. This file is used to filter only available SNOTEL stations from the runtime for the training. There's only few SNOTEL stations which are not available on the runtime and the score difference is negligible.
    * Run `python -m src.meta.get_usgs_huc` to generate USGS metadata containing HUC level used for join with UA/SWANN data
2. Download raw data for all training years
    * SNOTEL SWE -> run `python -m src.data.snotel` to download SNOTEL data based on defined pair between SNOTEL sites and drainage sites
    * CDEC SWE -> run `WSFR_DATA_ROOT=data/raw python download_cdec.py` to retrieve CDEC data and move the files to `data/external/cdec`
    * USGS Streamflow -> run `python -m src.data.usgs_streamflow` (rdb version) or use [competition runtime data download](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/tree/main/data_download) to retrieve USGS data
    * USBR Streamflow -> run `python -m src.data.usbr` to download USBR data from selected 4 sites
    * gridMET PDSI -> run `python -m src.data.pdsi` to download PDSI data
    * UA/SWANN SWE -> run `python -m src.data.uaswann` to download UA/SWANN SWE data
    * ERA5-Land and SEAS51 Forecast -> set CDS API key in `.env` file and run `python -m src.data.cds` to download daily D-1 ERA5-Land data and monthly SEAS51 forecast data. For ERA5-Land data, you also need to unzip the files within the directory with `bash unzip_era5.sh`.
3. Post-process CDEC metadata
    * Run `python -m src.meta.get_cdec_sites_basins` to generate pair between CDEC stations and forecast sites

## Training

Run `train.sh` to train all nine models for all validation years (2004-2023). For each single experiment or config training model run, the output will be:

* `models` directory of generated models
* `snotel_sites_basins` pair between SNOTEL/CDEC stations and forecast sites for each validation years iteration
* `pred.csv` all predictions for all validation years (2004-2023)
* `scale_factor` scale factor used to generate synthetic data for each validation years iterations
* `train_stats` saved historical avg/min/max of target for each validation year for normalization
* `swe_stats` saved historical avg/min/max of SWE for each validation year for normalization
* `features` cached feature input for each validation year (only cached for models with all features: `pdsi_swe_era5_s51` and `ngm_pdsi_ua_era5_s51`)

To generate ensemble predictions from those models, you can refer to `notebooks/eval_cv.ipynb` which also report cross validation score for all 20 years along with each individual models.

## Additional information

* As per 26th September 2024, new CDP API is fully operated ([source](https://forum.ecmwf.int/t/goodbye-legacy-climate-data-store-hello-new-climate-data-store-cds/6380)). If you are redownloading the data and using the new CDS, you can use `is_new_cds=True` in download and preprocessing functions of ERA5 and SEAS51 data
* You also can update the directory for these data sources on `src/config.py`
* You can remove some unnecessary files generated from training process to save space storage by running `src/cleanup.py`
* We recommend to have minimum 16-20 GB of free space storage to store raw data and artefacts of model training
* `train_ext.sh` is used to extend CV training and evaluation from 1991 to 2003
* `baseline_median.py` generates `data/sub/pred_median_dp.feather`, which is baseline median used to calculate relative skill score of averaged mean pinball loss

## Explainability

To generate a report for a single issue date and location, you can use `generate_report` function from `explainability/report.py` file. This function requires feature input and models from the training process. It will generate explainability outputs such as:

* Table -> summary of SHAP feature contribution along with contextual input and previous issue date for comparison
* Plot -> plot of 10th, 50th and 90th forecast along with mean/median/min/max for additional context

In addition, you also can refer to `notebooks/explainability_report.ipynb` for the example.

## Machine Specifications

* CPU: Core i5 1135G7
* RAM: 8GB
* Training data preprocessing for all years 1981-2023: 3-4 hours (not including
data download time, e.g. retrieving data from CDS API can take ~1 hour for a single date or month)
* Training duration: ~3 hours for all 720 models (20-fold years x 9 model variants x 4
losses)
* Inference duration: less than 3 minute for a single issue date and 26 sites (not including
data download time)

## Optimization tips

There are ways to reduce training and processing time as follow:

* Exclude MAE loss since it’s not needed and we use Tweedie loss for the final solution for 50th percentile forecast
* Cache some data processing steps
    * Selection of SNOTEL sites
    * Data features input (only executed once using models with all features: `pdsi_swe_era5_s51` for models with SNOTEL/CDEC data and `ngm_pdsi_ua_era5_s51` for models without SNOTEL/CDEC data)
* Only use several ensemble members and adjust based on dependency and processing complexity. For example, we can exclude models with ERA5-Land data because forecast skill improvement is minimal based on CV score (~1 KAF) and processing time is longer especially if we include data download time from CDS API
