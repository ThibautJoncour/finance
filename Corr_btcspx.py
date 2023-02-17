# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 19:55:17 2023

@author: tibo9
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings
warnings.filterwarnings('ignore', '.*do not.*' )
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
print("No Warning Shown")
plt.style.use('ggplot')



# Scrapping on Yahoo Finance
def data(x):
    df =yf.download(tickers=x,   period='300mo', interval='1d')

    
    df = df['Close']
    df = df[:]
    
    return df


df = yf.download(tickers='ES=F',  period='300mo', interval='1d')
df = df.iloc[:,3:4]
liste = ['BTC-USD']
col = ['SP500','BTC']
for i in liste:
    df[i] = data(i)
df.columns = col
np.shape(df)

df = df.fillna(method='ffill')
# Calcul des rendements 
df1 = df.pct_change()*100

# Rolling covariance 

df1['cov BTC/SPX'] = df1['BTC'].rolling(32).cov(df1['SP500'])

# Rolling Corrélation

df1['corr BTC/SPX'] = df1['BTC'].rolling(32).corr(df1['SP500'])

# Rolling Variance of BTC
df1['variance BTC'] = df1['SP500'].rolling(32).var()

# Rolling Beta

df1['beta'] = df1['cov BTC/SPX']/df1['variance BTC']
df1

for i in np.arange(-0.5,0.0,0.1):
    print('Average BTC return when correlation is below',round(i,2),'is', round(df1[(df1['corr BTC/SPX'])<i]['BTC'].mean(),2),"%")

for i in np.arange(0.0,0.5,0.1):
    print('Average BTC return when correlation is above',round(i,2),'is', round(df1[(df1['corr BTC/SPX'])>i]['BTC'].mean(),2),"%")


sns.distplot(df1[(df1['corr BTC/SPX'])>0.2]['BTC'], color="red")
sns.distplot(df1[(df1['corr BTC/SPX'])<-0.2]['BTC'], color='green')
plt.legend(labels=['Corrélation > 0.2','Corrélation < -0.2'])


