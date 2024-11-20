"""This is a template for the expected code submission format. Your solution must
implement the 'predict' function. The 'preprocess' function is optional."""





import itertools, glob, sys, time, datetime, itertools
import numpy as np
import pandas as pd
import xarray as xr



import multiprocessing
import concurrent.futures
import subprocess


import warnings
warnings.filterwarnings("ignore")

from joblib import load


from datetime import timedelta



from collections.abc import Hashable
from pathlib import Path
from typing import Any


FOLDS = np.arange(20) + 1 # Use this with LOOCV
FOLDS = np.arange(3)  + 1 # Use this with KFold, K=3

DOWNLOAD_DATA = True


def predict(
    site_id: str,
    issue_date: str,
    assets: dict[Hashable, Any],
    src_dir: Path,
    data_dir: Path,
    preprocessed_dir: Path,) -> tuple[float, float, float]:
    """A function that generates a forecast for a single site on a single issue
    date. This function will be called for each site and each issue date in the
    test set.
    
    Args:
        site_id (str): the ID of the site being forecasted.
        issue_date (str): the issue date of the site being forecasted in
            'YYYY-MM-DD' format.
        assets (dict[Hashable, Any]): a dictionary of any assets that you may
            have loaded in the 'preprocess' function. See next section.
        src_dir (Path): path to the directory that your submission ZIP archive
            contents are unzipped to.
        data_dir (Path): path to the mounted data drive.
        preprocessed_dir (Path): path to a directory where you can save any
            intermediate outputs for later use.
    Returns:
        tuple[float, float, float]: forecasted values for the seasonal water
            supply. The three values should be (0.10 quantile, 0.50 quantile,
            0.90 quantile).
    """
    
    '''
    src_dir             = Path('/home/kamarain/prize-winner-template/src')
    data_dir            = Path('/media/kamarain/varmuus/streamflow_data')
    preprocessed_dir    = Path('/media/kamarain/varmuus/streamflow_data')
    
    issue_date = '2022-05-01'
    site_id = 'ruedi_reservoir_inflow'
    site_id = 'green_r_bl_howard_a_hanson_dam'
    '''
    
    
    t1 = stopwatch('start')
    
    
    # Create a dictionary to store results
    all_results = []
    
    trgt_qtls = ['q0p1', 'q0p5', 'q0p9']
    
    # Predict all quantiles and ensemble members (folds) in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        # Use submit to asynchronously execute the function for each set of arguments
        futures = {executor.submit(predict_one_fold, fold, assets, site_id, issue_date): (fold)
                   for fold in FOLDS}
        
        # Wait for all tasks to complete and collect results
        concurrent.futures.wait(futures)
        
        # Retrieve results from completed tasks and populate the dictionary
        for future, fold in futures.items():
            result = future.result()
            all_results.append(result)
    
    
    all_results = np.sort(np.array(all_results), axis=1)
    
    q0p1_predictions = all_results[:,0]
    q0p5_predictions = all_results[:,1]
    q0p9_predictions = all_results[:,2]
    
    
    # The q0p1 and q0p9 have been modeled based on the error distribution of q0p5
    #for i in range(len(FOLDS)):
    #    q0p1_predictions[i] = q0p5_predictions[i] - np.abs(q0p1_predictions[i])
    #    q0p9_predictions[i] = q0p5_predictions[i] + np.abs(q0p9_predictions[i])
    
    print(f'Prediction time for {issue_date} {site_id}', stopwatch('stop', t=t1),'\n')
    
    # The best guess is the median of all predictions. Ensure non-negative output
    return np.clip(np.nanmedian(q0p1_predictions), a_min=0, a_max=None), \
           np.clip(np.nanmedian(q0p5_predictions), a_min=0, a_max=None), \
           np.clip(np.nanmedian(q0p9_predictions), a_min=0, a_max=None)



def execute_script(script_args):
    script_path, *args = script_args
    command = ['python', str(script_path)] + args
    subprocess.run(command)


def previous_month(timestamp):
    timestamp = pd.to_datetime(timestamp)
    previous_month = timestamp - pd.DateOffset(months=1)
    return str(previous_month.month).zfill(2), str(previous_month.year)


def preprocess(src_dir: Path, data_dir: Path, preprocessed_dir: Path) -> dict[Hashable, Any]:
    """An optional function that performs setup or processing.
    
    Args:
        src_dir (Path): path to the directory that your submission ZIP archive
            contents are unzipped to.
        data_dir (Path): path to the mounted data drive.
        preprocessed_dir (Path): path to a directory where you can save any
            intermediate outputs for later use.
    
    Returns:
        (dict[Hashable, Any]): a dictionary containing any assets you want to
            hold in memory that will be passed to to your 'predict' function as
            the keyword argument 'assets'.
    
    
    src_dir             = Path('/home/kamarain/prize-winner-template/src')
    data_dir            = Path('/media/kamarain/varmuus/streamflow_data')
    preprocessed_dir    = Path('/media/kamarain/varmuus/streamflow_data')
    
    issue_time = '07-22'
    site = 'green_r_bl_howard_a_hanson_dam'
    site = 'hungry_horse_reservoir_inflow'
    site = 'american_river_folsom_lake'
    
    """
    
    # Read own functions
    sys.path.append(str(src_dir))
    import functions as fcts
    
    t1 = stopwatch('start')
    
    print('\n\nDownloading data\n\n')
    
    sites  = fcts.all_sites()
    
    
    date_now = datetime.datetime.now()
    month_now, year_now = str(date_now.month).zfill(2), str(date_now.year)
    month_prv, year_prv = previous_month(date_now)
    
    issue_time_now = str(date_now.month).zfill(2)+'-'+str(date_now.day).zfill(2)
    #issue_times = fcts.all_issue_times()
    
    # Find closest predefined issue date to find the correct model
    issue_time = closest_issue_time(issue_time_now)
    
    
    if DOWNLOAD_DATA:
        
        # Define paths to download/preprocessing scripts
        script_paths = []
        script_paths.append([src_dir / 'preprocess_teleindices.py',         ])
        
        # These are for the pre-downloaded data
        script_paths.append([src_dir / 'preprocess_snotel.py',              year_now])
        script_paths.append([src_dir / 'preprocess_pdsi.py',                year_now])
        script_paths.append([src_dir / 'preprocess_cdec.py',                year_now])
        
        # These are for downloading and processing own data
        script_paths.append([src_dir / 'get_neighborhood_streamflow.py',    year_now])
        script_paths.append([src_dir / 'get_seasonal_from_cds.py',          year_now, month_now])
        script_paths.append([src_dir / 'get_seasonal_from_cds.py',          year_prv, month_prv])
        script_paths.append([src_dir / 'get_process_SWANN.py',              year_now])
        
        # Use ThreadPoolExecutor for parallelization
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit script execution tasks to the executor
            futures = [executor.submit(execute_script, script_args) for script_args in script_paths]
            
            # Wait N minutes for tasks to complete
            concurrent.futures.wait(futures, timeout=600)
        
        
        # Preprocess ECMWF seasonal data
        subprocess.run(['python', src_dir / 'preprocess_ecmwf.py', year_now])
        
        
        print('\n\nData downloaded and partly preprocessed\n')
        print('Downloading/preprocessing time:', stopwatch('stop', t=t1),'\n')
    
    print('\nPreprocessing downloaded data further\n\n')
    
    
    
    
    
    # Create a dictionary to store intermediate data 
    data_dict = {}
    
    # Read all data in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        
        # Use submit to asynchronously execute the function for each site
        futures = [executor.submit(read_X_for_sites, site, [int(year_now)], src_dir, data_dir, preprocessed_dir) for site in sites]
        
        # Wait for all tasks to complete and collect results
        for future in concurrent.futures.as_completed(futures):
            site, data_folds = future.result()
            data_dict[site] = data_folds
    
    
    
    # Create another dictionary to store final processed results
    data_out = {}
    
    for site, fold in itertools.product(sites, FOLDS): 
        
        if len(data_dict[site]) == 0: 
            print(site,'NOT OK')
            continue
        
        # X from the previous reading
        X_all = data_dict[site][fold]['X_all'].copy()
        
        # List to store futures for this iteration
        iteration_futures = []
        
        # Read and process data further in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            
            # Use submit to asynchronously execute the function for each set of arguments
            futures = {executor.submit(process_site_issue_time, site, issue_time, fold, X_all, [int(year_now)], 
                        src_dir, data_dir, preprocessed_dir): (site, fold, issue_time) for issue_time in [issue_time]}
            
            # Store the futures for this iteration
            iteration_futures.extend(futures)
        
        # Wait for all tasks to complete and collect results in order
        for future in concurrent.futures.as_completed(iteration_futures):
            site, fold, issue_time = futures[future]  # Retrieve the corresponding site, fold and issue_time
            result = future.result()
            data_out.setdefault(site, {}).setdefault(fold, {}).setdefault(issue_time, {}).update(result)
    
    
    print('\n\nData preprocessed\n')
    print('Total downloading + preprocessing time:', stopwatch('stop', t=t1),'\n')
    
    # data_out['virgin_r_at_virtin'][1]['01-01']['X_test']
    
    return data_out



def closest_issue_time(target_date):
    from datetime import datetime
    
    # Convert target_date to a datetime object
    target_datetime = datetime.strptime(target_date, '%m-%d')

    # List of issue times
    issue_times = ['01-01', '01-08', '01-15', '01-22', '02-01', '02-08', '02-15', '02-22',
                   '03-01', '03-08', '03-15', '03-22', '04-01', '04-08', '04-15', '04-22',
                   '05-01', '05-08', '05-15', '05-22', '06-01', '06-08', '06-15', '06-22',
                   '07-01', '07-08', '07-15', '07-22']

    # Convert issue_times to datetime objects
    issue_times = [datetime.strptime(issue_time, '%m-%d') for issue_time in issue_times]

    # Find the closest issue time
    closest_issue_time = min(issue_times, key=lambda x: abs(x - target_datetime))

    # Convert closest_issue_time back to string format
    closest_issue_time_str = closest_issue_time.strftime('%m-%d')

    return closest_issue_time_str





def read_X_for_sites(site, test_years, src_dir, data_dir, preprocessed_dir):
    
    # Read own functions
    sys.path.append(str(src_dir))
    import functions as fcts
    
    pp_model_file = glob.glob(f'{str(data_dir)}/models/preprocessing_models_{site}.joblib')
    pp_models = load(pp_model_file[0])
    
    data_folds = {}
    for fold in FOLDS:
        try:
            print('\nReading data for fold #', fold, site)
            
            pca_models = pp_models[f'pca_models_fold={fold}']
            
            predictor_data, _, _ = fcts.read_process_everything(str(data_dir)+'/', 
                                    str(preprocessed_dir)+'/', site, test_years, 
                                    train_or_test='test', pca_models=pca_models)
            
            X_all = pd.concat(predictor_data.values(), axis=1)
            
            data_folds[fold] = {'pca_models': pca_models.copy(), 'X_all': X_all.copy()}
            
            print_ram_state()
        except:
            print('Reading data for #', fold, site, 'FAILED')
            
    return (site, data_folds)





from joblib import load
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")
def process_site_issue_time(site, issue_time, fold, X_all, test_years, src_dir, data_dir, preprocessed_dir):
    
    
    
    issue_dates = []
    for year in test_years:
        issue_dates.append(f'{str(year)}-{issue_time}')
    
    actual_issue_dates = pd.to_datetime(issue_dates)
    
    models = None
    
    model_file = glob.glob(f'{str(data_dir)}/models/models_{site}_issuetime={issue_time}.joblib')
    if len(model_file) > 0:
        models = load(model_file[0])
    
    else:
        # In case of problems try to find a model near the issue date 
        issue_times = all_issue_times()
        idx = issue_times.index(issue_time)
        idx_range = list(np.clip((idx-2, idx+3), 0, None))
        issue_times = issue_times[idx_range[0]:idx_range[1]]
        
        
        # Check if some model can be found
        some_model_was_found = False 
        while not some_model_was_found and len(issue_times) > 0:
            
            some_issue_time = issue_times.pop()
            model_file = glob.glob(f'{str(data_dir)}/models/models_{site}_issuetime={some_issue_time}.joblib')
            
            if len(model_file) > 0:
                models = load(model_file[0])
                some_model_was_found = True
                print(issue_time, model_file[0])
    
    
    data = None
    if models is not None:
        
            
        # Extract the model
        prediction_models = models[f'model_fold={fold}']
        
        x_columns = X_all.columns
        X_test = pd.DataFrame(index=actual_issue_dates, columns=x_columns).astype(float)
        X_test = X_test.rename_axis('time')
        
        common_steps = np.intersect1d(X_all.index.values, X_test.index.values)
        X_test.loc[common_steps] = X_all.ffill().loc[common_steps].values
        
        data = {'model': prediction_models, 'X_test': X_test}
        print(X_test)
    
    
    print_ram_state()
    
    return data



def predict_one_fold(fold, assets, site_id, issue_date):
    
    issue_mnth = issue_date.split('-')[1]
    issue_day  = issue_date.split('-')[2]
    
    issue_time = issue_mnth+'-'+issue_day
    
    X_test = None; model = None
    
    # assets['virgin_r_at_virtin'][1]['01-01']['q0p9']['X_test']
    try:
        X_test      = assets[site_id][fold][issue_time]['X_test']
        model       = assets[site_id][fold][issue_time]['model']
    
    except:
        # In case of problems try to find a model/data pair near the issue date (BUT BEFORE it to not break rules)
        issue_times = all_issue_times()
        idx = issue_times.index(issue_time)
        idx_range = list(np.clip((idx-3, idx+1), 0, None))
        issue_times = issue_times[idx_range[0]:idx_range[1]]
        
        
        some_member_was_found = False
        
        
        # Check if some member can be found in the assets
        while not some_member_was_found and len(issue_times) > 0:
            
            some_issue_time = issue_times.pop()
            
            for some_fold in FOLDS:
                if (some_fold) in assets[site_id][some_issue_time]:
                    
                    X_test = assets[site_id][some_issue_time][(some_fold)]['X_test']
                    model = assets[site_id][some_issue_time][(some_fold)]['model']
                    
                    some_member_was_found = True
                    break    
    
    
    if X_test is not None and model is not None:
        prediction_raw = model.predict(X_test)
        df_pred = pd.DataFrame(index=X_test.index, data=prediction_raw, columns=FOLDS)
        
        prediction = df_pred.loc[issue_date].values
    
    else:
        prediction = np.full_like(np.ones(len(FOLDS)), np.nan)
        
    print('Prediction',issue_date,fold,'for',site_id,'=',prediction)
    
    return prediction




def stopwatch(start_stop, t=-99):
    import time
    
    if start_stop=='start':
        t = time.time()
        return t
    
    if start_stop=='stop':
        elapsed = time.time() - t
        return time.strftime("%H hours, %M minutes, %S seconds",time.gmtime(elapsed))



def print_ram_state():
    import resource
    
    # Get memory usage in kilobytes
    ram_state_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    # Convert kilobytes to megabytes
    ram_state_mb = mb(ram_state_kb)
    
    print(f"RAM: {ram_state_mb:.2f} MB", flush=True)


def mb(nbytes):
    return nbytes / 1024



def all_issue_times():
    
    issue_times = []
    for month,day in itertools.product( ['01', '02', '03', '04', '05', '06', '07'], 
                                        ['01', '08', '15', '22']):
        
        issue_times.append(month+'-'+day)
    
    return issue_times





"""

zip -r /media/kamarain/varmuus/submission.zip submission_src/* 

zip -r submission.zip * 


import pandas as pd
from pathlib import Path
import importlib

src_dir             = Path('/users/kamarain/streamflow_water_supply')
data_dir            = Path('/fmi/scratch/project_2002138/streamflow_water_supply/water-supply-forecast-rodeo-runtime/data')
preprocessed_dir    = data_dir

src_dir             = Path('/home/kamarain/water-supply-forecast-rodeo-runtime/submission_src')
data_dir            = Path('/home/kamarain/water-supply-forecast-rodeo-runtime/data')
preprocessed_dir    = Path('/home/kamarain/water-supply-forecast-rodeo-runtime/preprocessed')

src_dir             = Path('/home/kamarain/prize-winner-template/src')
data_dir            = Path('/media/kamarain/varmuus/streamflow_data')
preprocessed_dir    = Path('/media/kamarain/varmuus/streamflow_data')



import solution as sol

assets = sol.preprocess(src_dir, data_dir, preprocessed_dir)




subm_format = pd.read_csv(data_dir / 'submission_format.csv')

for i, row in subm_format.iterrows():
    
    site_id = row.site_id
    issue_date = row.issue_date
    
    try:
        q0p1, q0p5, q0p9 = sol.predict(site_id,issue_date,assets,src_dir,data_dir,preprocessed_dir)
        
        subm_format.loc[i, 'volume_10'] = q0p1
        subm_format.loc[i, 'volume_50'] = q0p5
        subm_format.loc[i, 'volume_90'] = q0p9
    except: pass


subm_format.to_csv('result_test_years.csv', index=False)

"""

