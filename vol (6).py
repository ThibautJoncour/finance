#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas_datareader.data as web
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random
import pandas as pd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.irf import IRAnalysis as IRF
from statsmodels.tsa.vector_ar.var_model import FEVD
from statsmodels.tsa.stattools import grangercausalitytests as granger

# packages grahiques
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
sns.set_style('darkgrid')
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

df1 = yf.download(tickers='BTC-USD', period='24mo', interval='1d')
#Print data


window_size = 10
df_c  = df1['Close'].pct_change().dropna()*100
vols =df_c.rolling(window_size).std()#*(252**0.5)
#plt.plot(vols)
#plt.plot(df['Close'])
vols


df2 = yf.download(tickers='^VIX', period='24mo', interval='1d')
df1['Vix'] = df2['Close'].pct_change().dropna()*100

df2 = df1['Vix'].fillna(0)

df2


# In[62]:


returns = vols.pct_change().dropna()*100


# rolling_predictions = []
# test_size = 200
# 
# for i in range(test_size):
#     train = returns[:-(test_size-i)]
#     model = arch_model(train, p=1, q=10)
#     model_fit = model.fit(disp='off')
#     pred = model_fit.forecast(horizon=1)
#     rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
# 

# X = df2[-201:-4].values
# y = returns[-201:-4].values
# y1 = rolling_predictions[-200:-3]
# X3 = np.transpose(np.array([y,X,y1]))
# 
# 
# model = VAR(X3)
# 
# var2 = model.fit(ic='aic', verbose=True)
# 
# 
# 
# model1 = VAR(X3)
# 
# # le modèle est estimé : l'option verbose sert à afficher le nombre de retard selectionné par la machine
# res = model1.fit(ic='bic', verbose=True)
# results = model1.fit(maxlags=10,ic='aic')
# lag_order = results.k_ar
# x = results.forecast(X3[-lag_order:],5)
# x = x[:,:1]
# x
# 
# plt.plot(x[:4], color='red')
# plt.plot(returns[-4:].values)

# In[ ]:





# 
# rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-200:])
# plt.figure(figsize=(10,4))
# true, = plt.plot(returns[-200:])
# preds, = plt.plot(rolling_predictions)
# plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
# plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)
# 

# In[ ]:





# In[64]:


from statsmodels.tsa.arima_model import ARIMA



#p,d,q  p = periods taken for autoregressive model
#d -> Integrated order, difference
# q periods in moving average model
model_arima = ARIMA(vols[-200:],order=(15, 1, 1))
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)
predictions= model_arima_fit.forecast(steps=10)[0]
predictions

#pred = pd.Series(np.sqrt(pred1.variance.values[-1,:]))
#pred = pred.values
rolling_f = np.concatenate((vols[-200:],predictions))
np.shape(rolling_f)


# In[ ]:





# In[70]:


rolling_f1 = pd.DataFrame(rolling_f)


# In[71]:


rolling_f = rolling_f1.pct_change().dropna()*100


# In[72]:



rolling_f = rolling_f.values
rolling_f = rolling_f.reshape(-1,)
returns = returns[-209:]
np.shape(returns)


# In[ ]:





# In[73]:



#pred = pd.Series(np.sqrt(pred1.variance.values[-1,:]))
#pred = pred.values

X3 = np.transpose(np.array([returns,df2[-209:]]))


model = VAR(X3)

var2 = model.fit(ic='aic', verbose=True)



model1 = VAR(X3)

# le modèle est estimé : l'option verbose sert à afficher le nombre de retard selectionné par la machine
res = model1.fit(ic='bic', verbose=True)
results = model1.fit(maxlags=12,ic='aic')

lag_order = results.k_ar
x = results.forecast(X3[-lag_order:],10)
x = x[:,:1]
#results.plot_forecast(10)

n = vols[-1:]

s = []
for liste in x:
    
    n = (n)*(1+liste/100)
    s.append(n)
    
    s

future_dates = [returns.index[-1] + timedelta(days=i) for i in range(1,11)]

d = pd.Series((predictions), index=future_dates)
plt.figure(figsize=(10,4))


plt.plot(d,color ='green')

plt.plot(vols[-200:])
v = pd.Series((s), index=future_dates)
plt.plot(v,color='red')
print(x)
plt.title('rate of volatility Prediction - Next 7 Days', fontsize=20)
plt.legend(['ARMA','volatility', 'VAR'], fontsize=16)


# train = returns[10:]
# model = arch_model(train, p=1, q=10)
# model_fit = model.fit(disp='off')
# pred1 = model_fit.forecast(horizon=7)
# future_dates = [returns.index[-1] + timedelta(days=i) for i in range(1,8)]
# pred = pd.Series(np.sqrt(pred1.variance.values[-1,:]), index=future_dates)
# plt.figure(figsize=(10,4))
# plt.plot(pred)
# plt.title('Volatility of volatility Prediction - Next 7 Days', fontsize=20)
# #plt.plot(x[:4], color='red')
# 
# 
# 
# 

# In[77]:


d1 = d[-1:].values


# In[83]:


price = df1.iloc[-1:,:1].values
price


# In[84]:


x11 =price * (d1/100) * 1.03643 + price
x12 =abs(price * (d1/100) * 1.03643- price)
print(x11)
print(x12)


# In[85]:


x1 =price * (d1/100) * 1.96 + price
x2 =abs(price * (d1/100) * 1.96- price)
print(x1)
print(x2)


# In[86]:


plt.figure(figsize=(10,4))

plt.plot(df1['Close'][-100:])
plt.axhline(x11, color = "red")
plt.axhline(x12, color = "red")
plt.axhline(x1, color = "black")
plt.axhline(x2, color = "black")

plt.legend(['seuil 90% de confiance', 'seuil 80% de confiance'], fontsize=16)


# In[ ]:




