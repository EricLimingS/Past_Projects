#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from cmc import coinmarketcap
from datetime import datetime
from coinmarketcap import Market
import pandas as pd
import time
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import collections
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
import requests
from bs4 import BeautifulSoup
import pickle
import pandas_datareader.data
import itertools


# In[ ]:


#method 1: read the table info directly
df = pd.read_html('https://coinmarketcap.com/currencies/volume/monthly/')[0]


#find the top 100 trading volume pairs
#method 2: use coinmarketcap package
coin_list = []
namelist = Market().listings()
converter = {} #convert the symbol to website_slug
coin_symbol = df[:100]['Symbol']
for name in namelist['data']:
    converter[name['symbol']] = name['website_slug']
for i in coin_symbol:
    try:        #some coins retrieved are not included in Coinmarketcap
        coin_list.append(converter[i])
    except KeyError:
        print('KeyError:', i)
        
start, end = datetime(2017,6,1), datetime(2019,8,15)
df_bitcoin = pd.DataFrame()
for name in coin_list:
    df = coinmarketcap.getDataFor(name, start, end, fields = 'Close')
    df_bitcoin = pd.concat([df_bitcoin,df],axis = 1, sort = False)
    time.sleep(10)
print(df_bitcoin)

