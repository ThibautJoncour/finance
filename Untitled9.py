#!/usr/bin/env python
# coding: utf-8

# In[539]:


#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.irf import IRAnalysis as IRF
from statsmodels.tsa.vector_ar.var_model import FEVD
from statsmodels.tsa.stattools import grangercausalitytests as granger
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
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.irf import IRAnalysis as IRF
from statsmodels.tsa.vector_ar.var_model import FEVD
from statsmodels.tsa.stattools import grangercausalitytests as granger
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
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.irf import IRAnalysis as IRF
from statsmodels.tsa.vector_ar.var_model import FEVD
from statsmodels.tsa.stattools import grangercausalitytests as granger
from datetime import datetime

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
start = datetime.datetime(2015, 1, 1)

end = datetime.datetime.today()


# La variable expliqué est le chomage on recupere les données via le site FRED


# Les variables explicatives sont le PIB, l'inflation, et les taux directeurs


df1 = yf.download(tickers='BTC-USD', period='45mo', interval='1d')
#Print data




# In[540]:


data = df1


# In[541]:


short_window = 50
long_window = 100

# Calculate moving averages
data['short_mavg'] = data['Close'].rolling(short_window).mean()
data['long_mavg'] = data['Close'].rolling(long_window).mean()

# Plot close price and moving averages
plot_data = data[-500:]
plt.figure(figsize=(10, 5))
plt.title('Long and Short Moving Averages', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Price')

plt.plot(plot_data['Close'], label='Close')
plt.plot(plot_data['short_mavg'], label='50-Day Moving Average')
plt.plot(plot_data['long_mavg'], label='200-Day Moving Average')

plt.legend()


# In[542]:


# Take long positions
data['long_positions'] = np.where(data['short_mavg'] > data['long_mavg'], 1, 0)

# Take short positions
data['short_positions'] = np.where(data['short_mavg'] < data['long_mavg'], -1, 0)

data['positions'] = data['long_positions'] + data['short_positions'] 

# Plot close price and moving averages
plot_data = data[-1000:]
plt.figure(figsize=(10, 7))
plt.title('Long and Short Signal', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Price')

plt.plot(plot_data['Close'], label='Close')
plt.plot(plot_data['short_mavg'], label='50-Day Moving Average')
plt.plot(plot_data['long_mavg'], label='200-day Moving Average')


plt.plot(plot_data[(plot_data['long_positions'] == 1) &
                       (plot_data['long_positions'].shift(1) == 0)]['short_mavg'],
         '^', ms=15, label='Buy Signal', color='green')


plt.plot(plot_data[(plot_data['short_positions'] == -1) &
                       (plot_data['short_positions'].shift(1) == 0)]['short_mavg'],
         '^', ms=15, label='Sell Signal', color='red')

plt.legend()
plt.show()


# In[543]:


data


# In[544]:


# Calculate daily returns
data['returns'] = data['Close'].pct_change()

# Calculate strategy returns
data['strategy_returns'] = data['returns'] * data['positions'].shift(1)

# Plot cumulative returns
cumulative_returns = (data['strategy_returns'] + 1).cumprod()
cumulative_returns.plot(figsize=(10, 7))
plt.title('Cumulative Strategy Returns')
plt.show()


# In[1288]:



def var(x):
    df = yf.download(tickers=x, period='34mo', interval='1d')
#Print data
    df = df['Close'].pct_change().dropna()*100
    df = df[-200:]
    
    return df

df1 = var('BTC-USD')
df2 = var('NDAQ')
df3 = var('^GSPC')
df4 = var('GOLD')

df1tr = df1[:-20]
df2tr = df2[:-20]
df3tr = df3[:-20]
df4tr = df4[:-20]

df1t = df1[-20:]
df2t = df2[-20:]
df3t = df3[-20:]
df4t = df4[-20:]

def transpose(df):
    liste = ['BTC-USD','TESLA','^GSPC','USOIL']
    
    for i in liste:
        df = var(i)
    
    
    #X5 = np.transpose(np.array([df1tr,df2tr,df3tr,df4tr]))


# In[1302]:


df = yf.download(tickers='BTC-USD', period='34mo', interval='1d')
print(df.iloc[:-30,3:4])


# In[1303]:



def data(x):
    df = yf.download(tickers=x, period='134mo', interval='1d')
#Print data
    df = df['Close'].pct_change().dropna()*100
    df =df[-1000:]
    
    return df


# In[1304]:


data('BTC-USD')
df.iloc[df.index.get_loc(datetime(2021,12,12),method='nearest')]


# In[1306]:


df


# In[1308]:


df = []
x = []


liste = ['BTC-USD','TSLA','NDAQ','^GSPC']

def transpose():
    
    for i in liste:
        
        df.append(data(i))
          
    return np.array(df)


df =  np.transpose(transpose())
df =pd.DataFrame(df, columns=liste)
df = df.iloc[500:,:]
df


# In[ ]:





# In[1294]:


matrice = df.corr()
matrice.iloc[:1,1:]


# In[1310]:


idx = pd.date_range('2019-02-27	', '2021-12-27')

df = df.reindex(idx)
df


# In[ ]:





# In[1295]:


np.shape(df)


# In[1299]:


pd.Series(df, index=pd.date_range("500", freq="D"))


# In[1296]:


df


# In[ ]:





#  volatilité inferieur à 0 
#  
#  array([[200, 197, 196, 194, 192, 191, 188, 187, 183, 180, 179, 178, 177,
#         176, 175, 174, 172, 171, 170, 168, 167, 166, 165, 163, 162, 159,
#         156, 153, 152, 151, 150, 145, 144, 139, 134, 133, 132, 131, 128,
#         127, 121, 120, 116, 115, 114, 113, 110, 108, 106, 104, 103, 102,
#         101,  95,  94,  91,  89,  87,  86,  85,  81,  80,  78,  76,  73,
#          72,  71,  69,  67,  66,  64,  60,  59,  58,  57,  55,  54,  53,
#          52,  51,  47,  46,  45,  44,  41,  40,  37,  34,  30,  29,  28,
#          27,  26,  23,  22,  21,  20,  14,  12,   9,   8,   7,   6,   5,
#           4,   3,   2,   1]])

#  88,  74,  68

# In[1198]:


df = np.array(df.iloc[:,:])


# In[1199]:


np.shape(df)


# In[1200]:



df_train = df[:-21,:]
df_train1 = df[:-32,:]
df_train2 = df[:-40,:]


# In[1206]:





n = []

n = yf.download(tickers='BTC-USD', period='34mo', interval='1d')
n = n['Open'].values
n1 =  n[-22:-21]
n2 = n[-33:-32]
n3 = n[-41:-40]
n1


# In[1207]:


print(np.shape(df_train))
print(np.shape(df_train1))
print(np.shape(df_train2))


# In[ ]:





# In[ ]:





# In[ ]:





# In[1220]:


lst = [df_train,df_train1,df_train2]
x1 = []
def varx(x):
    
    model = VAR(x)
    res = model.fit(ic='bic', verbose=True)
    for i in range(1,36):
        
        results = model.fit(maxlags=i,ic='aic')
    lag_order = results.k_ar
    x1 = results.forecast(x[-lag_order:],5)
    x1 = x1[:,:1]
    x_m2 = x1.mean()
    x12 = (x1-x_m2)/x1.std()
    df3_cr2 = (x12*df_train.std())+df_train.mean()
    df3_cr2
    np.shape(df3_cr2)
    x1 = df3_cr2


    
    return x1
    


# In[1221]:


n1


# In[ ]:





# In[1222]:


t1 = varx(df_train)
t2 = varx(df_train1)
t3 = varx(df_train2)


# In[1223]:


print(t1)
print(t2)
print(t3)


# In[1224]:



s = []
s1 = []
s2 = []
def prix(t,v,s):
    
    for liste in t:
        v = (v)*(1+liste/100)
        s.append(v)
        
    return s

target = np.array(prix(t1,n1,s))
target1 = np.array(prix(t2,n2,s1))
target2 = np.array(prix(t3,n3,s2))


# In[1225]:


from datetime import datetime


# In[1226]:


t3


# In[1227]:


df_train2[-1:,:]


# In[1228]:



n3


# In[1229]:


target2


# In[1230]:


n3


# In[1231]:


#target 1 = 10
#target2 = 20
#trget 3 = 30


# 
# df_train = df[:-21,:]
# df_train1 = df[:-48,:]
# df_train2 = df[:-61,:]

# In[1232]:


88,  74,  68


# In[1260]:


def plot(x):
    df = yf.download(tickers=x, period='34mo', interval='1d')
    s = target.reshape(-1,)
    s1 = target1.reshape(-1,)
    s2 = target2.reshape(-1,)
    

    
    future_dates = [df.index[-21] + timedelta(days=i) for i in range(0,5)]
    future_dates1 = [df.index[-32] + timedelta(days=i) for i in range(0,5)]
    future_dates2 = [df.index[-40] + timedelta(days=i) for i in range(0,5)]


    s = pd.Series((s), index=future_dates)
    s1 = pd.Series((s1), index=future_dates1)
    s2 = pd.Series((s2), index=future_dates2)
    plt.figure(figsize=(20,10))



    plt.plot(s, color='black')
    plt.plot(s1)
    plt.plot(s2)

    

    #plt.plot(df['Close'][-50:])
    plt.plot(df.iloc[-58:,3:4], '--', color='red')
    plt.title('Vector Autoregressif Model - Price Prediction - Next 30 Days', fontsize=20)
    plt.legend(['VAR','Prix'], fontsize=16)

    plt.legend(loc="upper left")
    


# In[1267]:


s1


# In[1262]:


from datetime import timedelta
plot('BTC-USD')
plt.figure(figsize=(20,10))

dfv = yf.download(tickers='BTC-USD', period='34mo', interval='1d')
dfv = dfv['Close'].pct_change()

window_size=10

vols =dfv.rolling(window_size).std()#*(252**0.5)
vols = vols.pct_change()*100

plt.plot(vols[-58:])


# In[1279]:


vols.iloc[vols.index.get_loc(datetime(2022,12,12),method='nearest')]


# In[1266]:






df.iloc[df.index.get_loc(datetime.datetime(2021,11,22),method='nearest')]


# In[1263]:


vols = vols[-100:]
vols


# In[1259]:


d = np.where(vols)
d = np.array(d)
x = 100-42
#vols[31]
vols[x]


# In[552]:


vols[8]


# In[495]:


dfv = yf.download(tickers='^VIX', period='34mo', interval='1d')
plt.plot(dfv.iloc[-50:,:1])


# In[ ]:





# In[ ]:




