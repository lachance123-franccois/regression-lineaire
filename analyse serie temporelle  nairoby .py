# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 07:38:07 2025

@author: AWOUNANG
"""

import pandas as pd
import numpy as np 
from  statsmodels.tsa.seasonal import seasonal_decompose
#from datetime import datetime 
import matplotlib.pyplot as plt

df = pd.read_csv("nairobi_air_data2018.csv", sep=',',low_memory=False)
df['value'] = pd.to_numeric(df['value'], errors='coerce')
#print(df.head())
df = df.dropna(subset=['value']) 
print(df['value'].dtype)
print(df['value'].head())

df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')
data=df.set_index('timestamp')



#df = df.asfreq('D')  # fréquence qu

data_df = data.pivot_table(
    index='timestamp',
    columns='value_type',
    values='value',
    aggfunc='mean'  # ou 'sum', 'max', 'min', selon ce que tu veux
)



df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')
data=df.set_index('timestamp')

colonnes=['lat', 'lon','P', 'P2', 'temperature', 'humidity']
print(data_df.isna().sum() )# nombre de NaN par co

data_df=data_df[colonnes]

data_int= data_df.interpolate(method='time')

#data_int1 = data_int.asfreq('D') 
print("alpha entre 0 et 1  plus alpha est grand, moins le lissage est fort")
data_smooth = data_int.ewm(alpha=0.3).mean()
print("etape 2")
data_smooth = data_smooth.fillna(data_smooth.mean())
print("etape3")
data_smoothf= data_smooth.fillna(method='bfill').fillna(method='ffill')  # ou .interpolate()
#data_smooth = data_smooth.fillna(data_smooth.mean())

print(data_smoothf.isna().sum() )
print("Nombre total de NaN :", data_smooth.isna().sum().sum())
print("Valeurs infinies :", (~np.isfinite(data_smooth)).sum())
print(data_smooth.head())

#plt.show()


#lissage des valeur  moyenne mobile 
TSA=seasonal_decompose(data_smooth['P1'],model='additive', period=7)
TSA.plot()
#lissage des valeur  moyenne mobile 
TSA2=seasonal_decompose(data_smooth['P1'],model='multiplicative', period=7)
TSA2.plot('r')

plt.show()