The changelog was created to keep track of most important changes between different stages of the competition (Hindcast Stage, Forecast Stage, Final Prize Stage).

# Changes between Forecast Stage and Final Prize Stage
* added new features:
	* PREC_DAILY_prev (previous available value of PREC_DAILY from SNOTEL),
	* PREC_DAILY_Apr_mean (average value since April of PREC_DAILY from SNOTEL),
	* PREC_DAILY_Apr_prev_diff (difference between previous available value of PREC_DAILY and a value from Mar 30 from SNOTEL),
	* WTEQ_DAILY_Jun_prev_diff (difference between previous available value of WTEQ_DAILY and a value from May 30 from SNOTEL),
	* WTEQ_DAILY_Jul_prev_diff (difference between previous available value of WTEQ_DAILY and a value from Jun 29 from SNOTEL),
	* sd_prev (previous available value of Snow depth from CDS)
* added volume residuals prediction for May-Jul models for 23 site ids (new function lgbm_cv_residuals)
* added May-Jul predictions for 3 site ids without nat flow previous values from a given water year, model is still built for all site ids but only 3 are evaluated (incorporated into lgbm_cv function)
* added better DISTR_PERC for February and March predictions
* added final_preds pipeline (predictions from LOOCV)
* added pdsi_prev_to_last_month_diff
* added CDS Copernicus forecasts processing, both with and without late Jun/Jul forecasts
* added CDS data processing
* added different hyperparms range to optimize for early months (1, 2)
* added optimizing RMS of results in CV and hyperparams tuning
* added additional clipping with nat_flow_sum since April
* added pdsi_prev, removed nat_flow_11_to_10_ratio features
* removed outliers removal from full data processing (kept for distribution estimates and min_max_site_id_dict_final)
* added min_max_site_id_dict_final - min/max values for site_id without a year from the given LOOCV fold (20 DataFrames in a dictionary, each for one LOOCV year)
* added fitted distributions to data/distr directory
* added distribution fitting separately for all LOOCV years, amendments added to distribution estimation pipeline
* corrected CV results (added less weight for July)

# Changes between Hindcast Stage and Forecast Stage
* added 20-fold LOOCV with one changing year
* changed settings for hyperparams optimization and model training
* added interval coverage
* updated features to train
* updated distribution estimates based on full available data (with outliers removal)
* added new competition data that include Hindcast test years with their observed values
* changed DISTR_PERC to distr_perc_dict to assign different distribution weights for different months
* changed YEAR_SINCE from 1930 to 1965
* added discharge_cfs_mean_since_Oct_std
