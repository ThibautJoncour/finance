#!/usr/bin/env python
# coding: utf-8

# trois période de bear market:
# 
# de decembre 2017 à fevrier 2019
# de juin 2019 au 1er Avril 2020
# de decembre 2021 à juillet 2022

# In[165]:



import seaborn as sns
import math
import yfinance as yf
import numpy as np
import warnings
warnings.filterwarnings('ignore', '.*do not.*' )
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
print("No Warning Shown")

ticker = [['^VIX']]
#df = df1.pct_change().dropna()*100
#


list_ticker = [i for x in ticker for i in x]
len(list_ticker)

def Vol(ticker):
    
    #data = yf.download(tickers=ticker,start='2017-12-01', end='2019-02-01')
    #data = yf.download(tickers=ticker,start='2019-06-01', end='2020-06-01')
    data = yf.download(tickers=ticker,start='2019-01-01', end='2021-01-01')

    data['Ret_Close'] = data['Close'].pct_change().fillna(method='ffill').dropna()*100
    data['Volatility'] = data['Ret_Close'].rolling(window=10).std()
    data['ret_vol'] = data['Volatility'].pct_change()*100
    data['ticker']=str(ticker)
    

    return data



def buy_sell(data): 
    sig_buy = []
    sig_sell = []
    flag = -1
    stack = []
    for i in range(1,len(data)):
        if data["Volatility"][i] > data["Volatility"][i-1] and data["Volatility"][i-1] > data["Volatility"][i-2] and data['Close'][i]<40:
            if flag != 1:
                sig_buy.append(data['Close'][i])
                sig_sell.append(np.nan)
                flag = 1
            else :
                sig_buy.append(np.nan)
                sig_sell.append(np.nan)
        elif data["Volatility"][i] < data["Volatility"][i-1] and data["Volatility"][i-1] < data["Volatility"][i-2]:
            if flag != 0:
                sig_buy.append(np.nan)
                sig_sell.append(data['Close'][i])
                flag = 0
            else :
                sig_buy.append(np.nan)
                sig_sell.append(np.nan)   
        else :                                      # cas oÃ¹ il ne faut ni vendre ni acheter
            sig_buy.append(np.nan)
            sig_sell.append(np.nan)
            
    data = data.iloc[1:,:]
    data['sell_sig'] = sig_sell
    data['buy_sig'] = sig_buy
    fig = plt.figure(figsize=(25,15))

#ax1 = fig.add_subplot(111, ylabel='price')
    plt.plot(data['Close'], color='black')

#ax1.plot(df[df['signal long',]==1]['BTC-USD'],'^', markersize=20, color='g')
#ax1.plot(df[df['signal short',]==-1]['BTC-USD'],'v', markersize=20, color='r')

    plt.plot(data["buy_sig"], label='Buy', marker='^',markersize=20, c='g')
    plt.plot(data["sell_sig"], label='Sell', marker='v',markersize=20, c='r')    
    
    if data[data['buy_sig']>0].index[-1] >data[data['sell_sig']>0].index[-1]:
        data['buy_sig'] = data['buy_sig'].drop(data[data['buy_sig']>0].index[-1])
    if data[data['sell_sig']>0].index[0] <  data[data['buy_sig']>0].index[0]:
        data['sell_sig'] = data['sell_sig'].drop(data[data['sell_sig']>0].index[0])  
    data['buy_sig'] = data['buy_sig'].fillna(0)
    data['sell_sig'] = data['sell_sig'].fillna(0)
    
    return data

def name_ticker(df):
    return str(df.iloc[-1:,-1:].values)

def BT(df):
    d = df[df["buy_sig"]!=0]['Close']
    c = df[df["sell_sig"]!=0]['Close']
    

    

    per_rsi = np.vstack((d,c))
    per_rsi = np.transpose(per_rsi)
    np.shape(per_rsi)

    per_rsi = pd.DataFrame(per_rsi, columns=['buy_sig','sell_sig'])
    per_rsi['ticker'] = str(df.iloc[-1:,-3:-2].values)
    



        
        
    return per_rsi

def hold_return(df):
    hold = ((df['Close'][-1:].values - df['Close'][:1].values)/(df['Close'][:1].values))*100
    return print('La strategie de hold donne :',str(df.iloc[-1:,-3:-2].values) ,hold)

def profit(per_rsi):
        
    profit_rsi = []
    for i in range(len(per_rsi)):
        profit_rsi.append((per_rsi['sell_sig'][i]-per_rsi['buy_sig'][i])/per_rsi['buy_sig'][i])
        profit_rsi
        profit_rsi = pd.DataFrame(profit_rsi)
        profit_rsi = profit_rsi*100
        
        return print('Pour'+  per_rsi['ticker'][-1:] + 'le returns du backtest est',profit_rsi.sum())
    
    
    
for i in range(0,len(list_ticker)):
    profit(BT(buy_sell(Vol(ticker[0][i]))))

for i in range(0,len(list_ticker)):
    hold_return(buy_sell(Vol(ticker[0][i])))
    


# In[146]:


df = buy_sell(Vol(ticker[0][0]))
def hold_return(df):
    hold = ((df['Close'][-1:].values - df['Close'][:1].values)/(df['Close'][:1].values))*100
    return print(hold)
df


# In[147]:


import pandas as pd


import statsmodels.api as sm
import numpy as np
import pandas as pd


import pandas_datareader as pdr
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
import seaborn as sns
import math
liste = [['BTC-USD']]#
import yfinance as yf


ticker = [['BTC-USD','^TNX','^VIX','^GSPC']]
#df = df1.pct_change().dropna()*100



list_ticker = [i for x in ticker for i in x]
len(list_ticker)

def Vol(ticker):
    
    data = yf.download(tickers=ticker,period='10mo', interval='1wk')
    data['Ret_Close'] = data['Close'].pct_change().fillna(method='ffill').dropna()*100
    data['Volatility'] = data['Ret_Close'].rolling(window=2).std()*100
    data['ret_vol'] = data['Volatility'].pct_change()*100
    data['ticker']=str(ticker)
    

    return data




stack = []

def buy_sell(data) : 
    sig_buy = []
    sig_sell = []
    flag = -1
    stack = []
    for i in range(1,len(data)):
        if data["Volatility"][i] > data["Volatility"][i-1] and data["Volatility"][i-1] > data["Volatility"][i-2] :
            if flag != 1:
                data['Position']='Long' 
                flag = 1
            else :
                sig_buy.append(np.nan)
                sig_sell.append(np.nan)
        elif data["Volatility"][i] < data["Volatility"][i-1] and data["Volatility"][i-1] < data["Volatility"][i-2]:
            if flag != 0:
                data['Position']= 'Short'
                flag = 0
            
     
    
    stack.append((data[['ticker','Position']][-1:]))    
       
    return print(stack)




for i in range(0,len(list_ticker)):
    (buy_sell((Vol(ticker[0][i]))))


# In[ ]:


# Stratégie écart type


# In[ ]:





# In[ ]:





# In[127]:


import pandas_datareader as pdr
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
import seaborn as sns
import math
# Changer BTC-USD par le nom du titre dont vous voulez la perf

liste = [['^VIX','^GSPC','^TNX']]#
#MAM TRANSITION DURABLE OBLIGATIONS

#MAM SHORT TERM BONDS ESG
#MAM HIGH YIELD ESG
#MAM FLEXIBLE BONDS ESG
#MAM OBLI CONVERTIBLES ESG
#MAM TAUX VARIABLES ESG
liste2 = ['Close','CL=F','^TNX']

df = []
for i in liste:
    df = data.DataReader(i, 
                       start='2019-10-01', 
                       end='2022-04-20', 
                       data_source='yahoo')['Close']

idx= df.index
df = df.values
df = pd.DataFrame(df,columns = liste2, index = idx)
df = df.fillna(method='ffill')
df = df.dropna()
df['date'] = df.index
df
df['Ret_Close'] = df['Close'].pct_change().fillna(method='ffill').dropna()
df['Ret_tnx'] = df['^TNX'].pct_change().fillna(method='ffill').dropna()
df['Ret_CL'] = df['CL=F'].pct_change().fillna(method='ffill').dropna()
X = df.iloc[1:,-2:]
import statsmodels.api as sm
import numpy as np
import pandas as pd
y = df['Ret_Close'].dropna()

x = sm.add_constant(X)
 
# performing the regression
# and fitting the model
result = sm.OLS(y, x).fit()
y_pred = result.predict(x)
y_pred
yt = y.values
y_predt = y_pred.values
sns.distplot(yt-y_predt)
sig = df['Ret_Close'].std()
sig = sig
mu = df['Ret_Close'].mean()
mu = mu
df['pred'] = (y_pred*sig)+mu
df['erreur']= df['Ret_Close'].values - df['pred'].values
df


# In[142]:



# Nouvelle strategie

std_erreur = df['erreur'].std()

x = std_erreur
x = float(x)


new_df=df.assign(stderr_positif=x)
df =pd.DataFrame(new_df)
df
new_df=df.assign(stderr_negatif=-x)
df =pd.DataFrame(new_df)
df




# In[136]:


def buy_sell(data):
    sig_buy = []
    sig_sell = []
    flag = -1      # sert Ã  signaler Ã  l'algo de ne pas procÃ©der deux fois Ã  une vente/achat
    
    for i in range(len(data)):
        if df['erreur'][i]<df['stderr_negatif'][i]: # rÃ¨gle de dÃ©cision pour l'achat
            if flag != 1:
                sig_buy.append(df['Close'][i])
                sig_sell.append(np.nan)
                flag = 1
            else :
                sig_buy.append(np.nan)
                sig_sell.append(np.nan)
        elif df['erreur'][i]>df['stderr_positif'][i]: # rÃ¨gle de dÃ©cision pour la vente
            if flag != 0:
                sig_buy.append(np.nan)
                sig_sell.append(df['Close'][i])
                flag = 0
            else :
                sig_buy.append(np.nan)
                sig_sell.append(np.nan)   
        else :                                      # cas oÃ¹ il ne faut ni vendre ni acheter
            sig_buy.append(np.nan)
            sig_sell.append(np.nan)   
            
            
    return sig_sell, sig_buy
buy_sell = buy_sell(df)
df["sell_sig"] = buy_sell[1]
df["buy_sig"] = buy_sell[0]


# In[137]:



fig = plt.figure(figsize=(25,15))

#ax1 = fig.add_subplot(111, ylabel='price')
plt.plot(df['Close'], color='black')

#ax1.plot(df[df['signal long',]==1]['BTC-USD'],'^', markersize=20, color='g')
#ax1.plot(df[df['signal short',]==-1]['BTC-USD'],'v', markersize=20, color='r')

plt.plot(df["buy_sig"], label='Buy', marker='^',markersize=20, c='g')
plt.plot(df["sell_sig"], label='Sell', marker='v',markersize=20, c='r')    
    


# In[138]:


if df[df['buy_sig']>0].index[-1] >df[df['sell_sig']>0].index[-1]:
        df['buy_sig'] = data['buy_sig'].drop(df[df['buy_sig']>0].index[-1])
if df[df['sell_sig']>0].index[0] <  df[df['buy_sig']>0].index[0]:
        df['sell_sig'] = df['sell_sig'].drop(df[df['sell_sig']>0].index[0])  
df['buy_sig'] = df['buy_sig'].fillna(0)
df['sell_sig'] = df['sell_sig'].fillna(0)


# In[139]:


BT(df)


# In[140]:


hold_return(df)


# In[141]:


profit(BT(df))


# In[ ]:




