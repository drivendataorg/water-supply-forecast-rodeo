# Water Supply Forecast Rodeo - Hindcast Stage 1st Place Solution

# Summary

We use ensembles of LightGBM models with Tweedie loss for point forecast and quantile loss
for 0.10 and 0.90 quantile forecast. We mainly use lagged data of SNOTEL SWE and
cumulative precipitation averaged within and near the basins from top 𝐾 = 9 sites as the input
of the model. Using additional features can cause overfitting easily due to limited training
sample size for each site and lack of signal/information loss due to data aggregation. We use
synthetic data generation to mitigate small sample size problems and increase training sample
size by 5x, which significantly improves forecast skill, prediction interval reliability, and
generalizability. We also incorporate daily USGS and USBR observed flow from sites with
minimal impairment. In addition, LOOCV for even years 2004-2023 is used for evaluating the
model internally and it shows good consistency between validation and test years set.

# Setup

* Python 3.10.13
* Packages: `pip install -r requirements.txt`
* `wsfr_download` from [competition repo](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/tree/main) also needs to be installed

## Data preparation

1. Prepare metadata
    * Run `python src/meta/get_meta.py` to fill missing drainage area from original metadata and generate `metadata_proc.csv`
    * Run `python src/meta/get_snotel_sites_basins_all.py` to generate pair between SNOTEL stations and forecast sites
    * Run `python src/meta/get_snotel_sites_basins.py` to generate filtered pair between SNOTEL stations and forecast sites
    * Copy `data.find.txt` from [competition runtime](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/blob/main/data.find.txt) to `data/meta`. This file is used to filter only available SNOTEL stations from the runtime for the training. There's only few SNOTEL stations which are not available on the runtime and the score difference is negligible.
2. Download raw data for all training and testing years
    * SNOTEL SWE -> run `python src/data/snotel.py` to download SNOTEL data based on defined pair between SNOTEL sites and drainage sites
    * USGS Streamflow -> run `python src/data/usgs_streamflow.py` (rdb version) or use [competition runtime data download](https://github.com/drivendataorg/water-supply-forecast-rodeo-runtime/tree/main/data_download) to retrieve USGS data
    * USBR Streamflow -> run `python src/data/usbr.py` to download USBR data from selected 4 sites
    * You also can update the directory for these 3 data sources on `src/config.py`

## Training and predict

Run `train.sh` to train all models for different seeds for each validation years (even years 2004-2023) iteration. For each single experiment or config training model run, the output will be:

* `models` directory of generated models, reused for the inference code submission
* `snotel_sites_basins` pair between SNOTEL stations and forecast sites for each validation years iteration, reused for the inference code submission
* `pred.csv` all predictions for OOF validation years (one for each site and issue date) and test years (ten for each site and issue date)
* `scale_factor` scale factor used to generate synthetic data for each validation years iterations

## Evaluate and generate CSV submission

Run `eval_hindcast_ensemble.ipynb` to:

1. Generate the ensemble predictions
2. Evaluate the predictions for all validation years (even years 2004-2023) 
3. Generate submission for test years (odd years 2004-2023) with required format

## Preparing the code submission

Copy the following static metadata files into `submission/data/meta/`:

- `data/meta/metadata_proc.csv` -> `submission/data/meta/`
- `data/meta/snowtel_sites_basins/snotel_sites.feather` -> `submission/data/meta/snowtel_sites.feather`

Copy the fitted model directories from `runs/hindcast_stage/` to `submission/runs/`. For each model, the `models/` and `snotel_sites_basins/` subdirectories are necessary. 

Then pack up a ZIP archive of the submission contents:

```bash
(cd submission && zip -r ../submission.zip ./*)
```

# Machine Specifications

* CPU: Core i5
* RAM: 8GB
* Training duration: ~2 hours for all 360 models (10-fold years x 9 random seeds x 4
losses)
* Inference duration: less than 5 minutes for 10 years test set (not including data
download time)
