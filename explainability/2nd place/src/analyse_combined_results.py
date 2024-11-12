#!/usr/bin/env python




import sys, glob
import pandas as pd
import numpy as np

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from wsfr_download.config import DATA_ROOT






experiments = ['kfold3', 'loocv']
experiments = ['kfold3']




data_dir = DATA_ROOT 


'''
# Command line arguments 
experiments = sys.argv[1:]
'''

# Read own functions
import functions as fcts



print('Analyzing results for experiments:',experiments)



def classify_data(df, column_name, classes, class_column):
    # Calculate the 33.3rd and 66.6th percentiles
    low_high_thresholds = df[column_name].quantile([0.333, 0.666]).values
    
    # Function to classify each value
    def classify(value):
        if value <= low_high_thresholds[0]:
            return classes[0]
        elif value <= low_high_thresholds[1]:
            return classes[1]
        else:
            return classes[2]
    
    # Apply the classification
    df[class_column] = df[column_name].apply(classify)
    
    return df[class_column]





# Metadata
df_metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={'usgs_id': 'string'}, index_col='site_id')

sites = df_metadata.index.values.astype(str)






# Define the metrics
metrics = ['r', 'r2', 'mql', 'l10', 'l50', 'l90', 'ic']

# Create subplots for each metric
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))

# Flatten the axes array for easier iteration
axes = axes.flatten()


colors = ['black',  'tab:red', 'tab:cyan', 'tab:orange', ]


#crosval_files = {}
for e, experiment in enumerate(experiments):
    #crosval_files[experiment] = np.sort(glob.glob(f'{data_dir}combined_*_crosval_{experiment}*csv'))
    crosval_files = np.sort(glob.glob(f'{data_dir}/combined_*_crosval_{experiment}*csv'))
    
    print('Cross-validation files:',crosval_files)
    
    # Look for the target CV file, if not exists, read the original format
    cv_subm_format = pd.read_csv(data_dir / 'cross_validation_submission_format.csv')
    cv_subm_format[['volume_10',  'volume_50',  'volume_90']] = np.nan
    cv_subm_format['observed'] = np.nan
    
    # Read and substitute
    for fle in crosval_files: #[experiment]:
        df = pd.read_csv(fle, index_col=False, header=0)
        
        for i, row in df.iterrows(): 
            idx = (cv_subm_format['issue_date'] == row['issue_date']) & (cv_subm_format['site_id'] == row['site_id'])
            
            cv_subm_format.loc[idx, ['volume_10','volume_50','volume_90','observed']] = \
                                row[['volume_10','volume_50','volume_90','observed']].values
    
    
    
    # Write to CSV
    cv_subm_format.to_csv(data_dir / f'cross_validation_submission_format_withObs_{experiment}.csv', index=False)
    cv_subm_format.drop(columns=['observed']).to_csv(data_dir / f'cross_validation_submission_format_{experiment}.csv', index=False)
    
    
    # Analyse the results in detail for the final report
    dropped = cv_subm_format.dropna()
    dropped['issue_date'] = pd.to_datetime(dropped['issue_date'])
    dropped['issue_time'] = dropped['issue_date'].dt.strftime('%m-%d')
    
    for site in sites:
        site_idx = dropped['site_id'] == site
        
        dropped.loc[site_idx, 'elevation'] = df_metadata.loc[site, 'elevation']
        dropped.loc[site_idx, 'drainage_area'] = df_metadata.loc[site, 'drainage_area']
        dropped.loc[site_idx, 'meanflow'] = np.mean(dropped.loc[site_idx, 'observed'])
        
        dropped.loc[site_idx, 'annual_wetness'] = classify_data(dropped.loc[site_idx], 'observed', ['dry','normal','wet'], 'annual_wetness')
    
    dropped.loc[:, 'typical_flow'] = classify_data(dropped, 'meanflow', ['small','medium','large'], 'typical_flow')
    dropped.loc[:, 'altitude'] = classify_data(dropped, 'elevation', ['low','medium','high'], 'altitude')
    dropped.loc[:, 'catchment_size'] = classify_data(dropped, 'drainage_area', ['small','medium','large'], 'catchment_size')
    

    
    
    

    site_metrics = []

    for site in dropped.site_id.unique():
        data = dropped.where(dropped.site_id == site).dropna()
        data.groupby('issue_time').apply(lambda df: fcts.calc_pbloss_all(df['observed'].values, 
                                                        df[['volume_10', 'volume_50', 'volume_90']].values, [0.1,0.5,0.9]))




    # Calculate results for Figure 6 of the report
    metrics_files = np.sort(glob.glob(f'{data_dir}/combined_*_metrics_{experiment}*csv'))


    print('Metrics files:',metrics_files)



    # Read and append
    output = []
    for fle in metrics_files:
        #issue_time = fle.split('_')[-1].split('.')[0]
        
        df = pd.read_csv(fle, index_col=0, header=0)
        #df['issue_time'] = issue_time
        output.append(df)
        #df_mean = df.mean()
        
        #output[issue_time] = df #_mean


    # Merge into one
    merged_df = pd.concat(output).reset_index().rename(columns={'index': 'site'})

    # Reshape the dataframe to long format
    merged_df = pd.melt(merged_df, id_vars=['issue_time', 'site_id'], 
                        value_vars=['r', 'r2', 'ic', 'mql', 'l10', 'l50', 'l90'], var_name='metric')

    # Display the resulting dataframe
    print(merged_df)



    # Plot Figure 6 of the report
    # Set seaborn style
    sns.set_style('whitegrid')
    
    # Iterate over each metric and plot
    for i, metric in enumerate(metrics):
        
        # Create lineplot with shading for percentile ranges
        #sns.lineplot(data=merged_df[merged_df['metric'] == metric], x='issue_time', y='value', estimator='mean', ax=axes[i], color=colors[e], errorbar=('ci', 100), alpha=0.03)
        sns.lineplot(data=merged_df[merged_df['metric'] == metric], x='issue_time', y='value', estimator='mean', ax=axes[i], color=colors[e], errorbar=('ci', 98), alpha=0.05)
        #sns.lineplot(data=merged_df[merged_df['metric'] == metric], x='issue_time', y='value', estimator='mean', ax=axes[i], color=colors[e], errorbar=('ci', 50), alpha=0.3)
        sns.lineplot(data=merged_df[merged_df['metric'] == metric], x='issue_time', y='value', estimator='mean', ax=axes[i], color=colors[e], errorbar=None, label=experiment)
        axes[i].legend(loc='lower left')
        
        mean_val   = np.round(merged_df[merged_df['metric'] == metric]['value'].mean(), 3)
        median_val = np.round(merged_df[merged_df['metric'] == metric]['value'].median(), 3)
        
        # Annotate plot with median value
        xlocs = [0.05, 0.65, 0.35]
        ylocs = [0.80, 0.50, 0.35]
        axes[i].annotate(f'Mean:    {mean_val:.2f}\nMedian: {median_val:.2f}',  
                         xy=(xlocs[e], ylocs[e]), xycoords='axes fraction', fontsize=10, color=colors[e],
                         bbox=dict(facecolor='w', edgecolor='lightgray', boxstyle='round', alpha=0.5))
        
        # Set plot title
        axes[i].set_title(f'{metric.upper()}')
        axes[i].set_xlabel('Issue time'); axes[i].tick_params(axis='x', labelrotation=90)
        axes[i].set_ylabel('')
        if i!=0:
            axes[i].get_legend().remove()
        

    # Hide the empty subplot(s)
    if len(metrics) < len(axes):
        for j in range(len(metrics), len(axes)):
            axes[j].axis('off')




plt.suptitle(f'Experiments: {experiments}')
plt.tight_layout()

output_file = data_dir / f'fig_combined_combined_{"_".join(experiments)}.png'
fig.savefig(output_file, dpi=200)





