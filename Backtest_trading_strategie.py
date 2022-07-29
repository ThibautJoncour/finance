import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
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
import pandas_datareader as pdr
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
import seaborn as sns

ticker = [['ORP.PA']]
#df = df1.pct_change().dropna()*100
#


list_ticker = [i for x in ticker for i in x]
len(list_ticker)

def Vol(ticker):
    
    data = yf.download(tickers=ticker,period='24mo', interval='1d')
    data['Ret_Close'] = data['Close'].pct_change().fillna(method='ffill').dropna()*100
    data['Volatility'] = data['Ret_Close'].rolling(window=20).std()
    data['ticker']=str(ticker)
    data['ma100'] = data['Close'].rolling(100).mean()
    data['ma200'] = data['Close'].rolling(200).mean()

    

    return data



def buy_sell(data): 
    sig_buy = []
    sig_sell = []
    flag = -1
    stack = []
    for i in range(1,len(data)):
        if data['ma200'][i] < data['ma100'][i] and data["Volatility"][i] > data["Volatility"][i-1] and data["Volatility"][i-1] > data["Volatility"][i-2]:
            if flag != 1:
                sig_buy.append(data['Close'][i])
                sig_sell.append(np.nan)
                flag = 1
            else :
                sig_buy.append(np.nan)
                sig_sell.append(np.nan)
        elif data['ma200'][i] < data['ma100'][i] and data["Volatility"][i] < data["Volatility"][i-1] and data["Volatility"][i-1] < data["Volatility"][i-2]:
            if flag != 0:
                sig_buy.append(np.nan)
                sig_sell.append(data['Close'][i])
                flag = 0
            else :
                sig_buy.append(np.nan)
                sig_sell.append(np.nan)
        elif data['ma200'][i] > data['ma100'][i] and data["Volatility"][i] < data["Volatility"][i-1] and data["Volatility"][i-1] < data["Volatility"][i-2]:
            if flag != 1:
                sig_buy.append(data['Close'][i])
                sig_sell.append(np.nan)
                flag = 1
            else :
                sig_buy.append(np.nan)
                sig_sell.append(np.nan)
                
        elif data['ma200'][i] > data['ma100'][i] and data["Volatility"][i] > data["Volatility"][i-1]:
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

    plt.plot(data['Close'], color='black')



    plt.plot(data["buy_sig"], label='Buy', marker='^',markersize=20, c='g')
    plt.plot(data["sell_sig"], label='Sell', marker='v',markersize=20, c='r')    
    plt.plot(data['ma100'])
    plt.plot(data['ma200'])

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
    

    

    perf = np.vstack((d,c))
    perf = np.transpose(perf)
    np.shape(perf)

    perf = pd.DataFrame(perf, columns=['buy_sig','sell_sig'])
    perf['ticker'] = str(df.iloc[-1:,-5:-4].values)
    



        
        
    return perf

def hold_return(df):
    hold = ((df['Close'][-1:].values - df['Close'][:1].values)/(df['Close'][:1].values))*100
    return print('La strategie de hold donne pour  '+ df.iloc[-1:,-5:-4].values ,hold)

def profit(perf):
        
    profit = []
    for i in range(len(perf)):
        profit.append((perf['sell_sig'][i]-perf['buy_sig'][i])/perf['buy_sig'][i])
        profit = pd.DataFrame(profit)
        profit = profit*100
        
        return print('Pour'+  perf['ticker'][-1:] + 'le returns du backtest est',profit.sum())
    
    
    
for i in range(0,len(list_ticker)):
    profit(BT(buy_sell(Vol(ticker[0][i]))))

for i in range(0,len(list_ticker)):
    hold_return(buy_sell(Vol(ticker[0][i])))
    
