#!/usr/bin/env python




import sys, glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



print('Combining results')

#code_dir='/users/kamarain/streamflow_water_supply/'
data_dir='/fmi/scratch/project_2002138/streamflow_water_supply/water-supply-forecast-rodeo-runtime/data/'
experiment='testloocv_kfold3'
issue_month='01'






# Command line arguments 
#code_dir = str(sys.argv[1])+'/'
data_dir = str(sys.argv[1])+'/'
experiment = str(sys.argv[2])
issue_month = str(sys.argv[3])





issue_times = [issue_month+'-01', issue_month+'-08', issue_month+'-15', issue_month+'-22']


print('Input args:',data_dir,experiment,issue_month)

metrics_file_base = 'metrics_'+experiment
crosval_file_base = 'crosval_'+experiment

metrics_files = []
crosval_files = []

for issue_time in issue_times:
    metrics_files.extend(np.sort(glob.glob(data_dir+metrics_file_base+'*'+issue_time+'*csv')))
    crosval_files.extend(np.sort(glob.glob(data_dir+crosval_file_base+'*'+issue_time+'*csv')))


print('Result files:',metrics_files,crosval_files)





# Read and append metrics
output = []
for fle in metrics_files:
    
    df = pd.read_csv(fle, index_col=0, header=0)
    output.append(df)


# Merge into one
df_metrics = pd.concat(output, axis=0).rename_axis('site_id').reset_index()
df_metrics = df_metrics.sort_values(['site_id', 'issue_time']).set_index('site_id')


print(df_metrics)





# Read and append predictions
output = []
for fle in crosval_files:
    
    df = pd.read_csv(fle, index_col=0, header=0)
    output.append(df)


# Merge into one
df_crosval = pd.concat(output, axis=0).reset_index() #, ignore_index=True)
df_crosval = df_crosval.sort_values(['site_id', 'issue_date']).set_index('site_id')

print(df_crosval)



# Write to CSV
df_metrics.to_csv(data_dir+'combined_'+issue_month+'_'+metrics_file_base+'.csv') #, index=False)
df_crosval.to_csv(data_dir+'combined_'+issue_month+'_'+crosval_file_base+'.csv') #, index=False)






# Create a figure for the subplots
fig, axes = plt.subplots(1, 7, figsize=(20, 6))  # Adjust the figure size as needed
fig.suptitle('Experiment: '+metrics_file_base)

metrics_columns = ['r', 'r2', 'ic', 'mql', 'l10', 'l50', 'l90']

for i, column in enumerate(metrics_columns):
    ax = axes[i]
    # Create boxplots for each issue_time within the current metric's subplot
    data_to_plot = [df_metrics[df_metrics['issue_time'] == issue_time][column] for issue_time in issue_times]
    boxprops = dict(facecolor="lightyellow", color="gray")  # Custom box properties
    whiskerprops = dict(color="gray")
    capprops = dict(color="gray")
    medianprops = dict(color="gray")
    flierprops = dict(markeredgecolor="gray")

    bp = ax.boxplot(data_to_plot, positions=range(1, len(issue_times) + 1), patch_artist=True, 
                    boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, 
                    medianprops=medianprops, flierprops=flierprops)

    ax.set_title(column)
    ax.set_xticks(range(1, len(issue_times) + 1))
    ax.set_xticklabels(issue_times)
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

    # Annotate quantiles for each boxplot
    for j, issue_time in enumerate(issue_times):
        group_data = df_metrics[df_metrics['issue_time'] == issue_time][column]
        q50 = group_data.quantile(0.50)
        q25 = group_data.quantile(0.25)
        q75 = group_data.quantile(0.75)
        ax.text(j + 1, q50, f'Q50:\n{q50:.2f}', va='center', ha='center', fontsize=8)
        ax.text(j + 1, q25, f'Q25:\n{q25:.2f}', va='center', ha='center', fontsize=8, color='blue')
        ax.text(j + 1, q75, f'Q75:\n{q75:.2f}', va='center', ha='center', fontsize=8, color='blue')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust the layout
output_file = data_dir+'fig_combined_'+issue_month+'_'+metrics_file_base+'.png'
fig.savefig(output_file)



'''

# Define the metrics and issue times
metrics = ['r', 'r2', 'ic', 'mql', 'l10', 'l50', 'l90']
issue_times = df_metrics['issue_time'].unique()  # ['01-01', '01-08', '01-15', '01-22']

# Create subplots for each metric
fig, axes = plt.subplots(1, 7, figsize=(20, 6))  # Adjust figsize as needed
fig.suptitle('Experiment: '+metrics_file_base)

for i, metric in enumerate(metrics):
    ax = axes[i]
    
    # Create a list to hold the data for each issue_time group for the current metric
    data_to_plot = [df_metrics[df_metrics['issue_time'] == issue_time][metric] for issue_time in issue_times]
    
    # Plot the boxplot for each issue_time group side by side
    ax.boxplot(data_to_plot, labels=issue_times)
    ax.set_title(metric)
    ax.set_xlabel('Issue Time')
    ax.set_ylabel(metric)
    

plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust the layout
output_file = data_dir+'fig_combined_'+issue_month+'_'+metrics_file_base+'.png'
fig.savefig(output_file)






fig, axes = plt.subplots(1, 7, figsize=(12, 8))  # Adjust the figure size as needed
fig.suptitle('Experiment: '+metrics_file_base)

# Flatten the axes array for easy iteration
axes_flat = axes.flatten()

for i, column in enumerate(['r', 'r2', 'ic', 'mql', 'l10', 'l50', 'l90']):
    df_metrics.boxplot(column, ax=axes_flat[i])
    axes_flat[i].set_title(column)
    axes_flat[i].set_ylabel('')
    axes_flat[i].set_xlabel('')
    
    # Calculating statistics
    q50 = df_metrics[column].quantile(0.50)
    q25 = df_metrics[column].quantile(0.25)
    q75 = df_metrics[column].quantile(0.75)
    
    # Annotation for quantiles
    axes_flat[i].text(1, q50, f'Q50: {q50:.2f}', va='bottom', ha='center',)# backgroundcolor=None)
    axes_flat[i].text(1, q25, f'Q25: {q25:.2f}', va='bottom', ha='center',)# backgroundcolor=None)
    axes_flat[i].text(1, q75, f'Q75: {q75:.2f}', va='bottom', ha='center',)# backgroundcolor=None)
    


# Hide any unused subplots
for i in range(len(df_metrics.columns), len(axes_flat)):
    fig.delaxes(axes_flat[i])

plt.tight_layout() #rect=[0, 0.03, 1, 0.97])  # Adjust the layout to make room for the main title
output_file = data_dir+'fig_combined_'+issue_month+'_'+metrics_file_base+'.png'
fig.savefig(output_file)
#plt.show()

'''


