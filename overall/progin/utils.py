import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy import stats
import geopandas as gpd
import xarray as xr
import pickle
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from pathlib import Path

from wsfr_read.streamflow.usgs_streamflow import read_usgs_streamflow_data

def display_first_n_last_n_rows(data: pd.DataFrame,
                                n: int = 5):
    """
    Display n first and last rows of the dataframe.

    Args:
        data (pd.DataFrame): A DataFrame with rows to display
        n (int): How many first and last rows to display
    """
    display(pd.concat([data.head(n), data.tail(n)]))


def merge_usgs_streamflow(site_ids_unique: list,
                          years: list) -> tuple[pd.DataFrame, list]:
    """
    Merge together USGS streamflow data obtained from read_usgs_streamflow_data
    function. Create additionally a list for not found year-site_id combinations.

    Args:
        site_ids_unique (list): Unique site_ids
        years (list): Years to iterate over
    Results:
        streamflow (pd.DataFrame): Merged USGS streamflow data
        not_found_usgs (list): An auxiliary check of what data was missing for
            given year-site_id combination.
    """
    streamflow = pd.DataFrame()
    not_found_usgs = []
    for site_id in site_ids_unique:
        for year in years:
            #Get all observations before issue_date (YYYY-07-22 format)
            issue_max = str(year) + '-07-22'
            try:
                #Read data from given site_id-year combination. A year contains
                #all available data for this year, in this case in days.
                one_site_year = read_usgs_streamflow_data(site_id, issue_max)
                one_site_year['year'] = year
                one_site_year['site_id'] = site_id
                #Append rows from the given site_id-year combination
                streamflow = pd.concat([streamflow, one_site_year])
            except FileNotFoundError:
                #Create a list with not found year-site_id combinations
                not_found_usgs.append([year, site_id])
    return streamflow, not_found_usgs


def get_paths(path_dir: str,
              extension: str) -> list:
    """
    Get all paths with a provided extension from a given directory and its
    subfolders.

    Args:
        path_dir (str): A main directory path
        extension (str): An extension of file names to iterate over
    Returns:
        paths (list): All paths within the given directory
    """
    paths = []
    for root, dirs, files in os.walk(Path(path_dir)):
        for file in files:
            full_path = os.path.join(root, file)
            if full_path.endswith(extension):
                #Append the file name to the list
                paths.append(os.path.join(root, file))
    return paths


def read_snotel(dir_name: str) -> pd.DataFrame:
    """
    Read SNOTEL data (https://www.nrcs.usda.gov/wps/portal/wcc/home/aboutUs/monitoringPrograms/automatedSnowMonitoring/).
    If a single file with full SNOTEL data is not yet created, read all
    separate data files, merge them and save to a .csv file.

    Args:
        dir_name (str): A directory with SNOTEL files. The algorithm will go through
            any subfolders coming from that directory to search for files. It is
            also a directory where one .csv file for all SNOTEL files is created.
    Returns:
        snotel_df (pd.DataFrame): SNOTEL data transformed to one DataFrame
    """
    snotel_full_path = Path(f'{dir_name}/snotel_full.csv')
    if os.path.isfile(snotel_full_path):
        #Read data from already created snotel_full.csv
        snotel_df = pd.read_csv(snotel_full_path)
    else:
        #If snotel_full.csv wasn't created, iterate over all paths
        paths = get_paths(dir_name, '.csv')
        paths = pd.Series(paths)
        paths_meta = paths[paths.str.contains('station_metadata|sites_to_snotel_stations')]
        paths_data = paths[~paths.index.isin(paths_meta.index)]

        #Create an empty df with column names and types
        snotel_df = pd.DataFrame({'date': pd.Series(dtype = 'str'),
                                  'PREC_DAILY': pd.Series(dtype = 'float'),
                                  'TAVG_DAILY': pd.Series(dtype = 'float'),
                                  'TMAX_DAILY': pd.Series(dtype = 'float'),
                                  'TMIN_DAILY': pd.Series(dtype = 'float'),
                                  'WTEQ_DAILY': pd.Series(dtype = 'float'),
                                  'STATION': pd.Series(dtype = 'str')})
        logger.info("Creating snotel_full.csv file. It should take 15-40 minutes.\
 This operation must be done only when reading SNOTEL data for the first time.")
        for path in tqdm(paths_data):
            #Get station name
            station = Path(path).stem
            #Read a file
            df_to_add = pd.read_csv(path)
            #Append station
            df_to_add['STATION'] = station
            #Append to the DataFrame
            snotel_df = pd.concat([snotel_df, df_to_add], axis = 0, ignore_index = True)
        #Save full snotel data
        snotel_df.to_csv(snotel_full_path, index = False)
    return snotel_df


def get_pdsi_aggs(dir_name: str,
                  geospatial: pd.DataFrame) -> pd.DataFrame:
    """
    Get statistics from PDSI data. PDSI is Palmer Drought Severity Index data
    published every 5 days since 1980.

    Args:
        dir_name: A directory where PDSI aggr data as .pkl file should be saved
        geospatial: geospatial.gpkg file from
            https://www.drivendata.org/competitions/254/reclamation-water-supply-forecast-dev/data/
            with station ids and their precise coordinates in POLYGON format
    Returns:
        pdsi_df (pd.DataFrame): PDSI data with statistics on different dates
            and state ids
    """
    pdsi_full_path = Path(f'{dir_name}') / 'pdsi_full.pkl'
    if os.path.isfile(pdsi_full_path):
        pdsi_df = pd.read_pickle(pdsi_full_path)
    else:
        logger.info("No aggregated PDSI data found. Starting creating a DataFrame with PDSI data.")
        stats_pdsi = []
        #Get all paths with pdsi data
        paths_pdsi = get_paths('data/pdsi', '.nc')
        if not paths_pdsi:
            logger.info("It seems that PDSI data wasn't downloaded. Download it and run ReadAllData again.")
        for path in paths_pdsi:
            print('\n', path)
            #Read data from a specific year (forecast year)
            pdsi_one_year = xr.open_dataset(path)
            #Change encoding of coordinates
            pdsi_one_year.rio.write_crs("epsg:4326", inplace=True)
            #Iterate over rows from geospatial.gpkg
            for _, geo in geospatial.iterrows():
                site_id = geo.site_id
                print(site_id)
                #Keep only information from specific site_id
                pdsi_data = pdsi_one_year.rio.clip([geo.geometry], pdsi_one_year.rio.crs)
                #Get number of days from the year to iterate over them
                num_days = pdsi_data.dims['day']
                for num_day in range(num_days):
                    #Get date
                    date = pdsi_data.day[num_day].values
                    #Mean, max, min, median values per day and site_id
                    mean_val = np.nanmean(pdsi_data.daily_mean_palmer_drought_severity_index[num_day])
                    max_val = np.nanmax(pdsi_data.daily_mean_palmer_drought_severity_index[num_day])
                    min_val = np.nanmin(pdsi_data.daily_mean_palmer_drought_severity_index[num_day])
                    med_val = np.nanmedian(pdsi_data.daily_mean_palmer_drought_severity_index[num_day])
                    #Append results from a specific forecast year - site_id combination
                    stats_pdsi.append([site_id, date, mean_val, max_val, min_val, med_val])
        #Create a dataframe from a list and save it to .pkl
        if paths_pdsi:
            pdsi_df = pd.DataFrame(stats_pdsi,
                                   columns = ['site_id',
                                              'pdsi_date',
                                              'pdsi_mean',
                                              'pdsi_max',
                                              'pdsi_min',
                                              'pdsi_median'])
            pdsi_df.to_pickle(Path('data') / 'pdsi_full.pkl')
    return pdsi_df


def create_cds_dataframe(path: str,
                         geospatial: gpd.geodataframe.GeoDataFrame,
                         output_name: str,
                         all_touched: bool):
    """
    Read .nc file, create a DataFrame from it and save it.

    Args:
        path (str): A path to .nc file
        geospatial (gpd.geodataframe.GeoDataFrame): site_id data that maps
            site_ids to polygons that localize them
        output_name (str): File name of created .pkl CDS DataFrame
        all_touched (bool): Specifies if centers of pixels should be within
            polygon (True) all just any small areas that touch the pixel (False).
            True is used if CDS data is more detailed (coordinates with step
            0.1 that come from ERA5-Land) and False is used if step of
            coordinates is by 1 (Seasonal meteorological forecasts from Copernicus)
    """
    #Create data\cds directory if it doesn't exist
    models_path = Path('data') / 'cds'
    models_path.mkdir(parents = True, exist_ok = True)
    stats_cds = []
    #Read data from a specific year (forecast year)
    cds_one_month = xr.open_dataset(path)
    #Change encoding of coordinates
    cds_one_month.rio.write_crs("epsg:4326", inplace=True)
    #Get variable names
    cds_vars = list(cds_one_month.keys())
    #Iterate over rows from geospatial.gpkg
    for _, geo in geospatial.iterrows():
        site_id = geo.site_id
        print(site_id)
        #Keep only information from specific site_id
        if all_touched == True:
            cds_data = cds_one_month.rio.clip([geo.geometry], all_touched = True)
        else:
            cds_data = cds_one_month.rio.clip([geo.geometry])
        #Get dates to iterate over
        dates = cds_data.dims['time']
        #Iterate over different variables
        for var in cds_vars:
            for date in range(dates):
                #Get date
                date_val = cds_data.time[date].values
                #Get average value for given var-date combination. This
                #operation just averages variable's values over different
                #locations adequate for a given site_id
                mean_val = np.nanmean(cds_data[var][date])
                #Append results
                stats_cds.append([var, site_id, date_val, mean_val])
    #Store stats_cds data in a DataFrame
    stats_cds = pd.DataFrame(stats_cds)
    #Set site_id, date and CDS variables' averages as columns
    stats_cds.columns = ['cds_var', 'site_id', 'date', 'mean_value']
    stats_cds = pd.pivot_table(stats_cds,
                               values = 'mean_value',
                               index = ['site_id', 'date'],
                               columns = 'cds_var').reset_index()
    #Add date columns
    stats_cds['year'] = stats_cds.date.dt.year
    stats_cds['month'] = stats_cds.date.dt.month
    stats_cds['day'] = stats_cds.date.dt.day
    stats_cds['hour'] = stats_cds.date.dt.hour
    #Add year_forecast
    stats_cds['year_forecast'] = stats_cds.year
    stats_cds.loc[stats_cds['month'].astype(int).between(10, 12),
                  'year_forecast'] = stats_cds.year_forecast + 1
    #Save file, so the processing doesn't have to be done every time
    stats_cds.to_pickle(f'data/cds/{output_name}.pkl')


class ReadAllData:
    """
    A class that loads and stores all data.
    """
    #Main data
    train_hist = pd.read_csv('data/prior_historical_labels.csv')
    train_cv = pd.read_csv('data/cross_validation_labels.csv')
    train = pd.concat([train_hist, train_cv]).reset_index(drop = True)

    meta = pd.read_csv('data/metadata.csv', dtype={"usgs_id": "string"})
    submission_format = pd.read_csv('data/cross_validation_submission_format.csv')

    prior_historical_monthly_flow = pd.read_csv('data/prior_historical_monthly_flow.csv')
    cross_validation_monthly_flow = pd.read_csv('data/cross_validation_monthly_flow.csv')
    train_monthly_naturalized_flow = pd.concat([prior_historical_monthly_flow,
                                                cross_validation_monthly_flow])

    geospatial = gpd.read_file('data/geospatial.gpkg')

    site_ids_unique = list(train['site_id'].unique())
    years = [2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023]
    years_all = [1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899,
                 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909,
                 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919,
                 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929,
                 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939,
                 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949,
                 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959,
                 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969,
                 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979,
                 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
                 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                 2000, 2001, 2002, 2003, 2004, 2006, 2008, 2010, 2012, 2014,
                 2016, 2018, 2020, 2022,
                 #Hindcast test years
                 2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023]
    #Get streamflow data
    streamflow, not_found_usgs = merge_usgs_streamflow(site_ids_unique, years_all)
    #Get snotel data. Takes ~30 minutes for the first time
    snotel = read_snotel(Path('data/snotel'))
    sites_to_snotel_stations = pd.read_csv(Path('data/snotel/sites_to_snotel_stations.csv'))
    snotel_meta = pd.read_csv(Path('data/snotel/station_metadata.csv'))
    #Get PDSI data. Should take 10-20 minutes for the first time
    pdsi = get_pdsi_aggs('data', geospatial)


def flatten_pandas_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten column names to get rid of MultiIndex. It happens when aggregations
    are being made on many columns at once

    Args:
        df (pd.DataFrame): A DataFrame with columns MultiIndex columns
    Returns:
        df (pd.dataFrame): A DataFrame with columns transformed from MultiIndex
            to a string name with different indexes separated with '_'
    """
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    df.columns = df.columns.str.replace(' ', '_')
    return df


def get_outliers(df: pd.DataFrame,
                 out_thres: float) -> pd.DataFrame:
    """
    Get outliers based on z-score and its threshold.

    Args:
        df (pd.DataFrame): Data with outliers to find
        out_thres (float): Threshold for outliers (higher finds less outliers)
    Returns:
        zscores_outliers (pd.DataFrame): Outliers found in the df
    """
    zscores = df.groupby('site_id')['volume'].apply(lambda x: stats.zscore(x))
    zscores_outliers = zscores[zscores.abs() >= out_thres]
    zscores_outliers = zscores_outliers.reset_index().set_index('level_1')
    return zscores_outliers


def distr_param_values_to_dict(distr: str,
                               distr_params: tuple) -> dict:
    """
    Get names of distribution parameters. Based on that, create a dictionary
    with those names and their estimated values from distr_params.

    Args:
        distr (str): A distribution name
        distr_params (tuple): Fitted parameter values of the distribution
    Returns:
        distr_dict (dict): Dictionary mapping (parameter names to values)
    """
    #Get names of parameters from the distribution
    distr_params_names = getattr(stats, distr)
    distr_params_names = distr_params_names.shapes
    if distr_params_names:
        #It should be in str format or in case of many parameters, str format
        #with params separated by ', '. Store different params in a list
        distr_params_names = distr_params_names.split(', ')
        #Add 'loc' and 'scale' params as they aren't included but exist for all distributions
        distr_params_names = distr_params_names + ['loc'] + ['scale']
    else:
        #Case when there aren't any arguments except for 'loc' and 'scale'
        distr_params_names = ['loc'] + ['scale']
    #Store param names and estimated values in a dictionary
    distr_dict = dict(zip(distr_params_names, distr_params))
    return distr_dict


def print_summary_outliers_distribution(df: pd.DataFrame,
                                        site_ids_unique: list,
                                        zscores_outliers: pd.DataFrame,
                                        bins: int=30):
    """
    Print distribution summary with its outliers. Grouped by different site_ids,
    it contains a table with outliers, plotted distribution and the most
    important summary statistics for the given site_id.

    Args:
        df (pd.DataFrame): Data to examine
        site_ids_unique (list): Unique site_ids to iterate over
        zscores_outliers (pd.DataFrame): Found outliers
        bins (int): Informs on how many parts distribution should be divided
    """
    for site_id in site_ids_unique:
        print(f'Table with outliers from {site_id}:')
        display(df.iloc[zscores_outliers[zscores_outliers.site_id == site_id].index])
        sns.histplot(df.loc[df.site_id == site_id, 'volume'], kde = True, bins = bins)
        plt.title(f'Distribution of volume from {site_id}')
        plt.show()
        print(f'Distribution summary of {site_id}:')
        print(df[df.site_id == site_id]['volume'].describe())
        print('\n')


#Create a list of LOOCV years
years_cv = [2004,
            2005,
            2006,
            2007,
            2008,
            2009,
            2010,
            2011,
            2012,
            2013,
            2014,
            2015,
            2016,
            2017,
            2018,
            2019,
            2020,
            2021,
            2022,
            2023]


#Create a dictionary with distributions to be fitted for distribution estimates
all_distr_dict = {'alpha': stats.alpha,
                  'anglit': stats.anglit,
                  'arcsine': stats.arcsine,
                  'argus': stats.argus,
                  'beta': stats.beta,
                  'betaprime': stats.betaprime,
                  'bradford': stats.bradford,
                  'burr': stats.burr,
                  'burr12': stats.burr12,
                  'cauchy': stats.cauchy,
                  'chi': stats.chi,
                  'chi2': stats.chi2,
                  'cosine': stats.cosine,
                  'crystalball': stats.crystalball,
                  'dgamma': stats.dgamma,
                  'dweibull': stats.dweibull,
                  'erlang': stats.erlang,
                  'expon': stats.expon,
                  'exponnorm': stats.exponnorm,
                  'exponpow': stats.exponpow,
                  'exponweib': stats.exponweib,
                  'f': stats.f,
                  'fatiguelife': stats.fatiguelife,
                  'fisk': stats.fisk,
                  'foldcauchy': stats.foldcauchy,
                  'foldnorm': stats.foldnorm,
                  'gamma': stats.gamma,
                  'gausshyper': stats.gausshyper,
                  'genexpon': stats.genexpon,
                  'genextreme': stats.genextreme,
                  'gengamma': stats.gengamma,
                  'genhalflogistic': stats.genhalflogistic,
                  'genhyperbolic': stats.genhyperbolic,
                  'geninvgauss': stats.geninvgauss,
                  'genlogistic': stats.genlogistic,
                  'gennorm': stats.gennorm,
                  'genpareto': stats.genpareto,
                  'gibrat': stats.gibrat,
                  'gompertz': stats.gompertz,
                  'gumbel_l': stats.gumbel_l,
                  'gumbel_r': stats.gumbel_r,
                  'halfcauchy': stats.halfcauchy,
                  'halfgennorm': stats.halfgennorm,
                  'halflogistic': stats.halflogistic,
                  'halfnorm': stats.halfnorm,
                  'hypsecant': stats.hypsecant,
                  'invgamma': stats.invgamma,
                  'invgauss': stats.invgauss,
                  'invweibull': stats.invweibull,
                  'johnsonsb': stats.johnsonsb,
                  'johnsonsu': stats.johnsonsu,
                  'kappa3': stats.kappa3,
                  'kappa4': stats.kappa4,
                  'ksone': stats.ksone,
                  'kstwo': stats.kstwo,
                  'kstwobign': stats.kstwobign,
                  'laplace': stats.laplace,
                  'laplace_asymmetric': stats.laplace_asymmetric,
                  'levy': stats.levy,
                  'levy_l': stats.levy_l,
                  'levy_stable': stats.levy_stable,
                  'loggamma': stats.loggamma,
                  'logistic': stats.logistic,
                  'loglaplace': stats.loglaplace,
                  'lognorm': stats.lognorm,
                  'loguniform': stats.loguniform,
                  'lomax': stats.lomax,
                  'maxwell': stats.maxwell,
                  'mielke': stats.mielke,
                  'moyal': stats.moyal,
                  'nakagami': stats.nakagami,
                  'ncf': stats.ncf,
                  'nct': stats.nct,
                  'ncx2': stats.ncx2,
                  'norm': stats.norm,
                  'norminvgauss': stats.norminvgauss,
                  'pareto': stats.pareto,
                  'pearson3': stats.pearson3,
                  'powerlaw': stats.powerlaw,
                  'powerlognorm': stats.powerlognorm,
                  'powernorm': stats.powernorm,
                  'rayleigh': stats.rayleigh,
                  'rdist': stats.rdist,
                  'recipinvgauss': stats.recipinvgauss,
                  'reciprocal': stats.reciprocal,
                  'rel_breitwigner': stats.rel_breitwigner,
                  'rice': stats.rice,
                  'rv_continuous': stats.rv_continuous,
                  'rv_histogram': stats.rv_histogram,
                  'semicircular': stats.semicircular,
                  'skewcauchy': stats.skewcauchy,
                  'skewnorm': stats.skewnorm,
                  'studentized_range': stats.studentized_range,
                  't': stats.t,
                  'trapezoid': stats.trapezoid,
                  'trapz': stats.trapz,
                  'triang': stats.triang,
                  'truncexpon': stats.truncexpon,
                  'truncnorm': stats.truncnorm,
                  'truncpareto': stats.truncpareto,
                  'truncweibull_min': stats.truncweibull_min,
                  'tukeylambda': stats.tukeylambda,
                  'uniform': stats.uniform,
                  'vonmises': stats.vonmises,
                  'vonmises_fisher': stats.vonmises_fisher,
                  'vonmises_line': stats.vonmises_line,
                  'wald': stats.wald,
                  'weibull_max': stats.weibull_max,
                  'weibull_min': stats.weibull_min,
                  'wrapcauchy': stats.weibull_min}


def get_quantiles_from_distr(data: pd.DataFrame,
                             min_max_site_id: pd.DataFrame,
                             all_distr_dict: dict,
                             path_distr: str) -> pd.DataFrame:
    """
    Calculate Q0.1 and Q0.9 quantile values from distribution for given site_id.

    It's based on the following logic:
    1. Get LightGBM model prediction (x)
    2. Calculate in what point on CDF function x appears on (q0.5)
        * if model returned 3600 volume and it is on 0.64 quantile, then we
            assume that q(0.5) = 0.64)
    3. Calculate in what point on CDF function max possible historical data for
        site_id (max(site_id)) appears on (qmax)
        * if 5000 value (maximum per site_id from historical data) is on 0.98
            quantile, then qmax = 0.98
    3. Set x to max possible value (max(site_id)) if x > max(site_id). Predicted
        value can't be greater than max value.
    4. Calculate a difference on CDF between qmax and q0.5
    5. This difference becomes a base for calculating quantile 0.9 (q0.9).
        The quantile is calculated based on proportions.
            qmax - q0.5 = value
            q0.9 = q0.5 + 4/5 * value (4/5 as thanks to that the distance is
                                       proportional; it would be 3/5 for q0.8)
    6. Do similarly for min value
        1. Calculate in what point on CDF function min possible historical data
            for site_id (min(site_id)) appears on (qmin)
        2. Set x to min possible value (min(site_id)) if x < min(site_id)
        3. Calculate a difference on CDF between q0.5 and qmin
        4. This difference becomes a base for calculating quantile0.1 (q0.1).
            The quantile is calculated based on proportions.
                q0.5 - qmin = value
                q0.1 = q0.5 - 4/5 * value

    Thanks to this approach, quantile values are continuous (except LightGBM
        predictions that exceed min/max site_id value) and will never exceed
        min and max values range.

    Args:
        data (pd.DataFrame): Data to process, quantiles will be added to this df
        min_max_site_id (pd.DataFrame): Minimum and maximum historical volumes
            for given site_id. It is a part of min_max_site_id_dict dictionary
            with already specified LOOCV year
        all_distr_dict (dict): Available distributions
        path_distr (str): Path to values of distribution estimate parameters
            per each site_id already with amendments to distributions. Correct
            LOOCV year was already passed to the argument, so it is the path to
            distribution from the given year.
    Returns:
        data (pd.DataFrame): A DataFrame with appended quantile 0.1 and 0.9
        values from distributions
    """
    #Read a dictionary with site_id - distribution matches
    with open(path_distr, "rb") as fp:
        distr_per_site = pickle.load(fp)

    #Iterate over different site_ids
    for site_id, site_params in distr_per_site:
        #Get distribution name
        distr = next(iter(site_params.keys()))
        #Get distribution parameters
        distr_args = site_params[distr]

        #Center distribution on our volume_50 prediction
        site_cdf_50 = data[data.site_id == site_id]['volume_50'].apply(
            lambda x: all_distr_dict[distr].cdf(x, **distr_args))

        #Get site_cdf_10 and site_cdf_90 based on proportions
        #Get min and max values for site_id volume
        max_site = min_max_site_id.loc[min_max_site_id.index == site_id, 'max']
        min_site = min_max_site_id.loc[min_max_site_id.index == site_id, 'min']

        #Get CDFs of min and max volume for current site_id
        max_cdf = all_distr_dict[distr].cdf(max_site, **distr_args).item()
        min_cdf = all_distr_dict[distr].cdf(min_site, **distr_args).item()

        #Fill values greater than max_cdf with max_cdf
        site_cdf_50.loc[site_cdf_50 > max_cdf] = max_cdf
        #Fill values less than min_cdf with min_cdf
        site_cdf_50.loc[site_cdf_50 < min_cdf] = min_cdf
        #Calculate site_cdf_90 based on proportions
        site_cdf_90 = site_cdf_50 + (max_cdf - site_cdf_50) * 0.8
        #Calculate site_cdf_10 based on proportions
        site_cdf_10 = site_cdf_50 - (site_cdf_50 - min_cdf) * 0.8

        #Get 0.9 quantile values
        q_90 = site_cdf_90.apply(lambda x: all_distr_dict[distr].ppf(q = x,
                                                                     **distr_args))
        #Get 0.1 quantile values
        q_10 = site_cdf_10.apply(lambda x: all_distr_dict[distr].ppf(q = x,
                                                                     **distr_args))
        #Fill distribution 0.1 and 0.9 volume for given site_id
        data.loc[data.site_id == site_id, 'volume_90_distr'] = q_90
        data.loc[data.site_id == site_id, 'volume_10_distr'] = q_10
    return data
