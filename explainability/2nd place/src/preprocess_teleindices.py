#!/usr/bin/env python



# Read modules
import sys, datetime, requests, os
import numpy as np
import pandas as pd


from pandas.tseries.offsets import MonthEnd

from pathlib import Path



from wsfr_download.config import DATA_ROOT




trgt_dir = DATA_ROOT / 'teleindices'
data_dir = DATA_ROOT 





# Create the output directory if it doesn't exist
os.makedirs(str(trgt_dir), exist_ok=True)











# Oceanic Niño Index (ONI)



oni = pd.read_fwf(data_dir / 'teleconnections' / 'oni.txt')


# create a dictionary to map SEAS to month number
seas_map = {'DJF': '01', 'JFM': '02', 'FMA': '03', 'MAM': '04', 'AMJ': '05', 'MJJ': '06',
            'JJA': '07', 'JAS': '08', 'ASO': '09', 'SON': '10', 'OND': '11', 'NDJ': '12'}

# create a new column that combines YR and mapped month number as a string
oni['YR_MONTH_DY'] = oni['YR'].astype(str) + oni['SEAS'].map(seas_map) + '01'
oni['time'] = pd.to_datetime(oni['YR_MONTH_DY'], format='%Y%m%d') + MonthEnd()

oni.set_index('time', inplace=True)
oni = oni.rename(columns={'TOTAL':'oni_total', 'ANOM':'oni_anom'})

oni = oni[['oni_total', 'oni_anom']]





soi_path = data_dir / 'teleconnections' / 'soi.txt'
with open(soi_path, 'r') as file:
    # Read all lines from the file and store them in a list
    lines = file.readlines()




soi_data = []
soi_anom = []

data_read = False; anom_read = False; 
for row in lines:
    #print(row)
    
    year = -1e5
    if 'ANOMALY' in row: 
        anom_read = True
        data_read = False
    
    if 'STANDARDIZED' in row: 
        anom_read = False
        data_read = True
    
    
    line = row.replace('-999.9', '   NaN')
    line = line.split()
    #print(data_read, anom_read, line)
    
    if anom_read and not data_read and len(line)==13:
        soi_anom.append(line)
    
    if not anom_read and data_read and len(line)==13:
        soi_data.append(line)


soi_data = np.array(soi_data)
soi_data = pd.DataFrame(soi_data[1:,1:], columns=soi_data[0,1:], index=soi_data[1:,0]).astype(float)

soi_data = soi_data.stack().reset_index(name='soi_data')
soi_data['year'] = soi_data['level_0'].values.astype(str); 
soi_data['month'] = soi_data['level_1'].values.astype(str)
soi_data['day'] = '01'

soi_data['time'] = soi_data['year']+'-'+soi_data['month']+'-'+soi_data['day']
soi_data['time'] = pd.to_datetime(soi_data['time']) + MonthEnd()
soi_data.set_index('time', inplace=True)

soi_data = soi_data[['soi_data']]




soi_anom = np.array(soi_anom)
soi_anom = pd.DataFrame(soi_anom[1:,1:], columns=soi_anom[0,1:], index=soi_anom[1:,0]).astype(float)


soi_anom = soi_anom.stack().reset_index(name='soi_anom')
soi_anom['year'] = soi_anom['level_0'].values.astype(str); 
soi_anom['month'] = soi_anom['level_1'].values.astype(str)
soi_anom['day'] = '01'

soi_anom['time'] = soi_anom['year']+'-'+soi_anom['month']+'-'+soi_anom['day']
soi_anom['time'] = pd.to_datetime(soi_anom['time']) + MonthEnd()
soi_anom.set_index('time', inplace=True)

soi_anom = soi_anom[['soi_anom']]







pna = pd.read_fwf(data_dir / 'teleconnections' / 'pna.txt', index_col=0).rename_axis('year').astype(float)
pna = pna.stack().reset_index(name='pna')
pna['month'] = pna.level_1.values.astype(str); pna['day'] = '01'

pna['time'] = pna['year'].astype(str)+'-'+pna['month']+'-'+pna['day']
pna['time'] = pd.to_datetime(pna['time']) + MonthEnd()
pna.set_index('time', inplace=True)

pna = pna[['pna']]





pdo = pd.read_fwf(data_dir / 'teleconnections' / 'pdo.txt', index_col=0, skiprows=1, na_values=['99.99']).rename_axis('year').astype(float)
pdo = pdo.stack().reset_index(name='pdo')
pdo['month'] = pdo.level_1.values.astype(str); pdo['day'] = '01'

pdo['time'] = pdo['year'].astype(str)+'-'+pdo['month']+'-'+pdo['day']
pdo['time'] = pd.to_datetime(pdo['time']) + MonthEnd()
pdo.set_index('time', inplace=True)

pdo = pdo[['pdo']]



indices_monthly = pd.concat([oni, pdo, pna, soi_data, soi_anom], axis=1)




indices_monthly.to_csv(trgt_dir / 'teleindices_monthly.csv')







mjo = pd.read_fwf(data_dir / 'teleconnections' / 'mjo.txt', skiprows=1, parse_dates=[0], index_col='PENTAD').rename_axis('time')
mjo = mjo.replace({'*****':'NaN'}).astype(float)
for col in mjo.columns:
    mjo = mjo.rename(columns={col:'mjo_'+col})



mjo.to_csv(trgt_dir / 'mjo_index.csv')

print('Preprocessing TELEINDICES completed')


