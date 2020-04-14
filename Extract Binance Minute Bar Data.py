#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from binance.client import Client
import time
import collections

client = Client("", "")

tickers = client.get_ticker() #get all the tickers and related info from binance 
tic_symbol =[]
for ticker in tickers:        #extract only the symbol 
    if ticker['symbol'][-3:] == 'BTC':   #extract only the symbol ends with BTC
        tic_symbol.append(ticker['symbol'])
#sorting all the symbols based on monthly trading volume
result = {}
for symbol in tic_symbol:
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MONTH, "1 month ago UTC")
    if klines == []:
        print('no data for', symbol)
        continue
    volume = float(klines[0][5])*(float(klines[0][2])+float(klines[0][3]))/2
    result[symbol] = volume
    
coin = collections.Counter(result)
coin = coin.most_common(70)        #select the top X coins based on monthly volume
coin_symbol = [i[0] for i in coin]



#get the daily price for the selected coins from binance
coin_pool = pd.DataFrame()
for j in coin_symbol:
    info = client.get_historical_klines(j, Client.KLINE_INTERVAL_1HOUR, "01 Jan, 2018", "22 Aug, 2019")
    day = []
    price = []
    for i in info:
        date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(i[0]/1000))[:20]
        close_price = float(i[4])
        day.append(date)
        price.append(close_price)
    df = pd.DataFrame([day,price]).T
    df.drop_duplicates(inplace=True) #drop, by default, when all columns are the same, here, some dates are the same, so we drop the duplicates before setting the date as index
    df.columns = ('Date', j)
    df = df.set_index('Date')
    df=df[~df.index.duplicated(keep='first')]  #some have duplicated index, I do not know why, considering it's hourly data, we just drop it for simplicity  
    coin_pool = pd.concat([coin_pool,df], axis = 1, join = 'outer', ignore_index= False, sort=False)   #sometimes concat fails when there are duplciated values in index, so we drop the duplicates in the previous command
    
coin_pool.index.name = 'Date' #set index name
coin_pool = coin_pool.loc['2017-12-31 19:00:00':'2019-08-21 20:00:00']
coin_pool.to_csv('Top70_Hourly_Data_from_binance_based_on_monthly_volume.csv')

