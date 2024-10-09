
# Hindcast

From https://github.com/rasyidstat/wsf/tree/v1.1-hindcast

* Update config.py path, raw data location
* Update .gitignore
* Remove directory: `data/clean`
* Simplify `src/data/base.py`
* Include all years from `src/data/usbr.py` main
* Move `src/adhoc/snotel_basins.py` to `src/meta/get_snotel_sites_basins_all.py`
* All SNOTEL sites basins related move to special directory
* Add `src/data/snotel.py` to get SNOTEL data
* Simplify `src/features/base.py`
* Add description on `get_month_tf`
* Add `src/meta/get_meta.py`
* Remove pred_iters (GAJADI)

# Forecast

From https://github.com/rasyidstat/wsf/tree/v2.0-forecast

* Download and preprocess PDSI data ([Kaggle - WSF: Preprocess V2 (PDSI)](https://www.kaggle.com/code/rasyidstat/wsf-preprocess-v2-pdsi))
    * Add `src/data/pdsi.py` to get PDSI data
    * Add `src/features/pdsi.py` to preprocess PDSI data
* Update `train_cv_fcst_mm.py`
    * Delete commented lines of code
    * Preprocess PDSI if file not exists
    * Disable `eval_output`
* Existing changes from Hindcast
    * Update code for reading GT and monthly code
    * Update `expand_date` by including `max_ymd` option
    * Add `fill_missing_value`
* Update `lgb.py`
    * Add `eval_output` and `feature_output`