#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random
import pandas as pd
import random
import pandas as pd
from yahoo_fin.stock_info import get_analysts_info
from statsmodels.tsa.arima_model import ARIMA

from yahoo_fin.stock_info import *
from yahoo_fin import stock_info as si
ticker = "BTC-USD"
date_now = time.strftime("%Y-%m-%d")
import numpy as np
import pandas as pd

#Data Source
import yfinance as yf




df = yf.download(tickers='BTC-USD', period='100mo', interval='1d')
#Print data
df = df['Close']
df = df
df
train = df[:-100]
diff = 100*df.pct_change().dropna()

test = df[-100:].values

print(np.shape(test))
sig = diff.std()/100
m = diff.mean()/100
print(np.where(diff<-10))
train[46]


# In[217]:


test


# In[207]:


S0 = test[:1]
S0


# In[208]:


seuil_95 =S0 - (S0 * 1.96*sig*np.sqrt(40))
seuil_95_2 =(S0 * 1.96*sig*np.sqrt(40))+S0


# In[215]:





# In[218]:


import numpy as np
import matplotlib.pyplot as plt
 

# drift coefficent
mu = m
# number of steps
n = 100
# time in years
T = 100
# number of sims
M = 100
# initial stock price
S0 = test[:1]
# volatility
sigma = sig

# calc each time step
dt = T/n

# simulation using numpy arrays
St = np.exp(
    (mu - sigma ** 2 / 2) * dt
    + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T
)

# include array of 1's
St = np.vstack([np.ones(M), St])

# multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0). 
St = S0 * St.cumprod(axis=0)
 


# Define time interval correctly 
time = np.linspace(0,100,n+1)

# Require numpy array that is the same shape as St
tt = np.full(shape=(M,n+1), fill_value=time).T
 

plt.figure(figsize=(20,15))


plt.plot(tt, St)
plt.axhline(seuil_95, color = "black")
plt.axhline(seuil_95_2, color = "black")

plt.axhline(S0, color = "black")
plt.plot(test, color = 'black')
plt.xlabel("jours $(t)$")
plt.ylabel("Stock Price $(S_t)$")
plt.title(
    "Realizations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(S0, mu, sigma)
)
plt.show()


# In[ ]:




