import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
from tqdm import tqdm
import pickle
from loguru import logger

from utils import ReadAllData, years_cv, get_outliers, all_distr_dict,\
    distr_param_values_to_dict

import time

start = time.time()

###############################################################################
#Initialize variables and parameters
###############################################################################
#Get train df from dfs list
dfs = ReadAllData()
df = dfs.train.copy()
#Get unique site ids
site_ids_unique = dfs.site_ids_unique.copy()
#Remove missing values
df = df[df.volume.notna()].reset_index(drop = True)
#Keep train data from all years
df_all_years = df.copy()

#Remove troublesome distributions
distr_trouble = ['kstwo', 'rv_continuous', 'rv_histogram', 'studentized_range',
                 'vonmises_fisher', 'vonmises', 'gausshyper', 'powerlaw',
                 'vonmises_line', 'ncx2', 'arcsine', 'kappa4', 'erlang']
for distr in distr_trouble:
    del all_distr_dict[distr]

#Create data\distr directory if it doesn't exist. Distribution estimates will
#be stored in this path
distr_path = Path('data/distr')
distr_path.mkdir(parents = True, exist_ok = True)

###############################################################################
#Fit distributions to data and save
###############################################################################
#Iterate over different years from LOOCV folds. For a given fold, remove
#its year from the DataFrame and fit distributions to data without that year
for year_cv in tqdm(years_cv):
    print('\n', year_cv)
    #Remove rows from given CV year
    df = df_all_years[df_all_years.year != year_cv].reset_index(drop = True)
    #Remove outliers from the DataFrame
    OUT_THRES = 2.5
    zscores_outliers = get_outliers(df, OUT_THRES)
    df = df.iloc[~df.index.isin(zscores_outliers.index)]

    #Fit distributions
    distr_results = []
    for site_id in tqdm(site_ids_unique):
        print(site_id)
        for distr_name, distr in all_distr_dict.items():
            #Keep best estimate of distribution parameters in distr_results and
            #best parameters in distr_params
            args = distr._fitstart(df[df.site_id == site_id].volume)
            x0, func, restore, args = distr._reduce_func(args, {})
            try:
                res = minimize(func, x0, args=(df[df.site_id == site_id].volume,),
                               method='Nelder-Mead')
                distr_results.append([site_id, distr_name, res.fun, tuple(res.x)])
            except:
                logger.info(f'There was an error with {distr_name} for {site_id}. Ignoring the distribution and trying the next one.')
    #Save all distributions
    with open(Path("data") / "distr" / f"distr_per_site_{year_cv}", "wb") as fp:
        pickle.dump(distr_results, fp)
    #Keep only best distribution fit for each site_id
    distr_results_best = pd.DataFrame(distr_results).sort_values(2).drop(2, axis = 1).\
        groupby(0).head(1).values.tolist()
    #Change to appropriate format
    distr_results_best = [[x[0], {x[1]: distr_param_values_to_dict(x[1], x[2])}]
                          for x in distr_results_best]
    #Save best distributions
    with open(Path("data") / "distr" / f"distr_per_site_best_{year_cv}", "wb") as fp:
        pickle.dump(distr_results_best, fp)

###############################################################################
#Make amendments to distributions and save final results after the revisions
###############################################################################
#Create site_id: distribution_name pairs to add to amendments. For each LOOCV
#year, the same distributions are changed
amendments_site_to_distr_dict = {'american_river_folsom_lake': 'johnsonsb',
                                 'colville_r_at_kettle_falls': 'rice',
                                 'fontenelle_reservoir_inflow': 'genhalflogistic',
                                 'missouri_r_at_toston': 'gennorm',
                                 'owyhee_r_bl_owyhee_dam': 'gompertz',
                                 'san_joaquin_river_millerton_reservoir': 'triang',
                                 'sweetwater_r_nr_alcova': 'triang',
                                 'virgin_r_at_virtin': 'loguniform',
                                 'detroit_lake_inflow': 'anglit'}
for year_cv in years_cv:
    #Load all distributions from given year
    with open(Path("data") / "distr" / f"distr_per_site_{year_cv}", "rb") as fp:
        distr_results = pickle.load(fp)
    distr_results = np.array(distr_results, dtype="object")
    distr_results = pd.DataFrame(distr_results)
    distr_results.columns = ['site_id', 'distribution', 'score', 'params']
    distr_results = distr_results.\
        sort_values(['site_id', 'score', 'distribution']).reset_index(drop = True)
    #Iterate over amendment pairs and add them to a joint list.
    #The format of the output is adjusted to the best distributions format
    amend_list_this_year = []
    for site_id, distr in amendments_site_to_distr_dict.items():
        #Get parameters fitted to the distribution
        params = distr_results.loc[(distr_results.site_id == site_id) &
                                   (distr_results.distribution == distr),
                                   'params'].iloc[0]
        #Add parameter names
        params_dict = distr_param_values_to_dict(distr,
                                                 params)
        amend_list_this_year.append([site_id, {distr: params_dict}])
    #Save amendments from given year
    with open(Path("data") / "distr" / f"distr_amend_{year_cv}", "wb") as fp:
        pickle.dump(amend_list_this_year, fp)

    ###########################################################################
    #Append amendments to best fits
    ###########################################################################
    #Read best fits from this year
    with open(Path("data") / "distr" / f"distr_per_site_best_{year_cv}", "rb") as fp:
        distr_results_best = pickle.load(fp)
    #Get site_ids that require distribution change
    sites_distr_to_change = [x[0] for x in amend_list_this_year]
    #Get site_ids from distr_results_best
    site_ids_distr_per_site = [x[0] for x in distr_results_best]
    #Get indexes to change from site_ids_distr_per_site (distr_per_site)
    idxs_to_change = [site_ids_distr_per_site.index(x) for x in sites_distr_to_change]
    #Get indexes to change distr_per_site values. Those indexes are from distr_to_change
    idxs_change_with = list(range(len(sites_distr_to_change)))
    #Change values for specified indexes
    for idx_to_change, idx_change_with in zip(idxs_to_change, idxs_change_with):
        distr_results_best[idx_to_change] = amend_list_this_year[idx_change_with]

    ###########################################################################
    #Save final distributions from given year
    ###########################################################################
    with open(Path("data") / "distr" / f"distr_final_{year_cv}", "wb") as fp:
        pickle.dump(distr_results_best, fp)

end = time.time()
elapsed = end - start
