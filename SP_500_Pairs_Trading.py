#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from bs4 import BeautifulSoup
import pickle
import requests
import numpy as np
import pandas as pd
import statsmodels
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
import pandas_datareader.data
import collections
import datetime as dt
import itertools
from datetime import datetime
from urllib.request import urlopen
from urllib.error import HTTPError #in  sp500_ticker_seperate(), sometimes there is HTTPError


# In[ ]:


def save_sp500_tickers(): #collect sp500 ticker names
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker[:(len(ticker)-1)]
        ticker = ticker.replace('.','-')  
        tickers.append(ticker)

    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
    return tickers

def SP500_ticker_separate(ticker):
    tickers_sector = {}
    for ticker in tickers:
        try:
            url = "https://finance.yahoo.com/quote/" + ticker + "/profile?p=" + ticker
            response = urlopen(url)
            soup = BeautifulSoup(response, 'html.parser') #give the html to beautifulsoup to parse and get a soup object
            links = soup.find_all('span', attrs = {'class':"Fw(600)"})
            sector = links[0].text
            if sector in tickers_sector:
                tickers_sector[sector].append(ticker)
            else:
                tickers_sector[sector] = [ticker]
        except IndexError:      #some stock does not have profile info on yahoo finance, resulting indexError
            print("indexError:", ticker)
            continue
        except HTTPError:     #sometimes when server is busy, there could be a HTTPError
            print("HTTPError")
    return tickers_sector

def find_cointegrated_pairs(train_star, train_end, tickers):
    
    train_price = full_price.iloc[train_star:train_end,:][tickers]        
    n = train_price.shape[1] # return the number of stocks
    keys = train_price.keys() # return the tickers of stocks
    pairs = []
    for i in range(n-1):  #the last key is "Date", remove
        for j in range(i+1, n):
            S1 = train_price[keys[i]]
            S2 = train_price[keys[j]]
            if (adfuller(S1)[1] > 0.05) and (adfuller(S2)[1] > 0.05): #dickey fuller test
                result = coint(S1, S2)
                pvalue = result[1]
                if pvalue < 0.05:
                    pairs.append(((keys[i], keys[j]),pvalue))
    return pairs


# In[ ]:


def pairs_trade(s1, s2, window1, window2, init_value, entry, stop_gain, stop_loss):
    
    # we need to compute the corresponding z score in each day:
    ratio = s1/s2
    mavg_w1 = ratio.rolling(window = window1).mean()
    mavg_w2 = ratio.rolling(window = window2).mean()
    std_w2 = ratio.rolling(window = window2).std()
    z_score_col = []
    z_score = (mavg_w1-mavg_w2)/std_w2
    
   
    #simulate the trade, start with 10000 money position for each pair
    cash = init_value  # cash on hands
    s1_pos = 0  #stock position
    s2_pos = 0  #
    port_value = [] #accumulated pairs_trade value
    indicator = 0 #0 indicates no position on hand, 1 means currently hold a position 
    margin_acc = 0  #
    stock_value = 0  #  
    short_value = 0 #
    borrow_cost = 0
    margin_acc_interest = 0
    total_value = cash 
    prev_sl_point1 = 100  #any large number
    prev_sl_point2 = -100  #any small number
    counter = 0
    #just for test
    for i in range((window2-1),len(ratio)): 
        if indicator == 0:
            z_score = (mavg_w1[i]-mavg_w2[i])/std_w2[i]
        else:
            z_score = (mavg_w1[i]-mavg_w2[jj])/std_w2[jj]
        
            
        #short z score, short ratio, short s1 and long s2
        #if set smaller value rather than 0.95, may miss investment opportunity
        if  ((prev_sl_point1*0.95 > z_score >= entry)or (z_score > prev_sl_point1*1.2)) and indicator == 0:# or (prev_sl_point*0.95) > avoid starting a trade right after stop loss
            margin = cash * 1/3 # cash put in the margin account, assume 50% of the short sell
            cash = cash/1.5   # cash available for long stocks
            s2_share = (cash)//s2[i]
            s1_share = round(s2_share / ratio[i])
            cash -= s2[i] * s2_share
            stock_value = s2[i] * s2_share
            short_value = s1[i] * s1_share
            margin_acc += short_value + margin # assume 1.5 margin account requirement
            s1_pos -= s1_share
            s2_pos += s2_share
            commission = min((s1_share + s2_share) * 0.005,25) 
            bid_ask_spread = (stock_value + short_value) * 0.0003 #commission fee + bid-ask spread
            cash -= (commission + bid_ask_spread)
            total_value = cash + stock_value + margin_acc - short_value 
            borrow_cost = short_value * 0.02 * (1/(252)) # assume interest rate for stock loan is 2%
            margin_acc_interest = margin_acc * 0.015 * (1/(252)) #assume interest rate for margin account is 1.5%
            indicator = 1
            enter_point = z_score
            counter +=1
            jj = i
            print('enter1:')

        #long z score, long ratio
        elif (((prev_sl_point2 * 0.95) < z_score <= -entry)or z_score < prev_sl_point2 * 1.2) and indicator == 0: #or z_score < prev_sl_point2 * 2.25;(prev_sl_point*0.9) >
            margin = cash * 1/3
            cash = cash/1.5
            s1_share = (cash)//s1[i]
            s2_share = round(s1_share * ratio[i]) #make sure shares are integer
            cash -= s1_share * s1[i]
            stock_value = s1_share * s1[i]
            short_value = s2[i] * s2_share
            margin_acc += short_value + margin
            s1_pos += s1_share
            s2_pos -= s2_share
            commission = min((s1_share + s2_share) * 0.005,25)  
            bid_ask_spread = (stock_value + short_value) * 0.0003
            cash -= (commission + bid_ask_spread)
            total_value = cash + stock_value + margin_acc - short_value 
            borrow_cost = short_value * 0.02 * (1/(252)) # assume interest rate for stock loan is 2%
            margin_acc_interest = margin_acc * 0.015 * (1/(252)) #assume interest rate for margin account is 1.5%
            indicator = 1
            enter_point = z_score
            counter +=1
            jj = i
            print('enter2:')

        #stop gain 
        elif (indicator == 1 and enter_point > 0 and z_score < enter_point * stop_gain) or (indicator == 1 and enter_point < 0 and z_score > enter_point * stop_gain):
            cash += s1_pos * s1[i] + s2_pos * s2[i] + margin_acc #"""margin account should change"""
            stock_value = 0
            short_value = 0
            commission = min((abs(s1_pos) + abs(s2_pos)) * 0.005,25) 
            bid_ask_spread = (abs(s1_pos) * s1[i] + abs(s2_pos) * s2[i]) * 0.0003
            cash -= (commission + bid_ask_spread)
            borrow_cost = 0
            margin_acc_interest = 0
            s1_pos = 0
            s2_pos = 0
            indicator = 0
            margin_acc = 0
            total_value = cash + stock_value + margin_acc - short_value
            #exit_point = z_score[i]
            print('sg,')
        #stop loss 
        elif indicator == 1 and enter_point > 0 and z_score > enter_point * stop_loss:
            cash += s1_pos * s1[i] + s2_pos * s2[i] + margin_acc 
            stock_value = 0
            short_value = 0
            commission = min((abs(s1_pos) + abs(s2_pos)) * 0.005,25) 
            bid_ask_spread = (abs(s1_pos) * s1[i] + abs(s2_pos) * s2[i]) * 0.0003
            cash -= (commission + bid_ask_spread)
            borrow_cost = 0
            margin_acc_interest = 0
            s1_pos = 0
            s2_pos = 0
            indicator = 0
            margin_acc = 0
            total_value = cash + stock_value + margin_acc - short_value 
            prev_sl_point1 = z_score  #mark the stop loss exit point
            print('sl:')
        #stop loss
        elif indicator == 1 and enter_point < 0 and z_score < enter_point * stop_loss:
            cash += s1_pos * s1[i] + s2_pos * s2[i] + margin_acc 
            stock_value = 0
            short_value = 0
            commission = min((abs(s1_pos) + abs(s2_pos)) * 0.005,25) 
            bid_ask_spread = (abs(s1_pos) * s1[i] + abs(s2_pos) * s2[i]) * 0.0003
            cash -= (commission + bid_ask_spread)
            borrow_cost = 0
            margin_acc_interest = 0
            s1_pos = 0
            s2_pos = 0
            indicator = 0
            margin_acc = 0
            total_value = cash + stock_value + margin_acc - short_value 
            prev_sl_point2 = z_score  #mark the stop loss exit point
            print('sl:')
        else:    #put money in Vanguard Prime Money Market Fund
            stock_value = max(s1_pos,0)*s1[i] + max(s2_pos,0)*s2[i]
            short_value = abs(min(s1_pos,0)*s1[i] + min(s2_pos,0)*s2[i])
            cash = cash * (1+0.023) ** (1/(252))
            total_value = cash + stock_value + margin_acc - short_value
            #print('z_score:', z_score)
        
        z_score_col.append(z_score)
        total_value -= (borrow_cost - margin_acc_interest) 
        print(s1.index[i],"total_value:",total_value,'z_score', z_score, s1[i], s2[i])
        port_value.append([str(mavg_w2.index[i])[:10], total_value]) 
    return port_value


def find_invest_uni_sharp(vali_star, vali_end, coin_pair, init_value, full_price):
    inv_uni = [] #investment universe
    max_sharp = 0
    mavg1 = 2
    mavg = list(range(30,90,10))   #list(range(60,181,5))
    enter = list(np.arange(1.5, 2.0, 0.1))
    
    multiplier1 = list(np.arange(1.2,1.35,0.05)) #sl multiplier
    multiplier2 = list(np.arange(0.8,0.55,-0.05)) #sg multiplier
    
    comb = itertools.product(enter, multiplier1, multiplier2, mavg)
    info = []
    
    for k in comb:
        print(coin_pair)
        
        vali_price = full_price.iloc[(vali_star-k[3]+1):vali_end]   #??? change
        port_value = pairs_trade(vali_price[coin_pair[0]],vali_price[coin_pair[1]], mavg1, k[3],init_value,k[0],k[2],k[1]) #? problem!!!!! ✝️ entry:1.5,stop loss:1.7 都不行  
        print(port_value)
        Val_end = port_value[-1][1]
        Val_star = init_value    #assign 10000 to each pair at the begining

        #Annualized Yield
        if Val_end < Val_star:
            continue
        ann_return = Annual_yield(vali_star, vali_end, Val_star, Val_end)

        #standard devidation
        port_value = pd.DataFrame(port_value, columns = ['Date', coin_pair])
        port_value= port_value.set_index('Date')
        port_return = port_value/port_value.shift(1)-1
        port_return = port_return.values
        port_return[0] = 0 #change the first data from nan to 1
        std = np.array(port_return).std()
        annual_std = std * ((252) ** 0.5)

        #check max drawdown
        max_dd = Mdd(port_value)

        #Calmar ratio
        calmar_rate = ann_return / max_dd

        if calmar_rate < 1:
            continue
        #if max_dd > 0.30: #0.35
            #continue
        if annual_std > 0.3: #0.35      #when this number is too small, it excludes out valuable pairs
            continue

        rf_return = 0.022
        sharp = sharp_ratio(ann_return, rf_return, annual_std)
        if sharp > max_sharp + 0.2:
            max_sharp = sharp
            info = [mavg1, k[3], k[0], k[2], k[1]] #w1, w2, enter, sg, sl
        else:
            continue
    
    return(info, max_sharp)
def Mdd(port_value):
    xs = np.array(port_value)
    i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
    if i == 0:  #no draw down situation 
        return 0
    j = np.argmax(xs[:i]) # start of period
    mdd = (xs[i]-xs[j])/xs[j]
    return abs(mdd)


#check for annualized yeild
from datetime import datetime
def Annual_yield(T_star, T_end, Val_star, Val_end):
    T_diff = T_end - T_star
    Ann_y = (Val_end/Val_star)**(252/(T_diff))-1   # -59 for 60 days moving ave ??????should we -59 here, the date already adjusted?
    return Ann_y

#check for sharp ratio
def sharp_ratio(port_return, rf_return, std_port):
    return (port_return-rf_return)/std_port

def out_sample_test(T_star, T_end, pair, info, init_value,full_price):
    oos_price = full_price.iloc[(T_star-info[1]+1):T_end]   
    performances={}
    pair_performance = pairs_trade(oos_price[pair[0]],oos_price[pair[1]], info[0], info[1],init_value,info[2],info[3],info[4])
    return pair_performance


# In[ ]:


tickers = save_sp500_tickers()
trad_start = '2014-08-26'
trad_end = '2019-08-26'
full_data = {ticker:pandas_datareader.data.get_data_yahoo(ticker, trad_start, trad_end) for ticker in tickers}
full_data = pd.DataFrame({ticker:data['Adj Close'] for ticker, data in full_data.items()}).replace(np.nan, 0, regex=False) #replace nan to 0
full_data['Date'] = full_data.index

collect = []       #collect z_score info
industry_separed_tickers = SP500_ticker_separate(tickers)
invest = 100000    #total investment 
length = len(full_data)
final_outcome = pd.DataFrame()
rolling_return = []   #record rolling annualized return for each period
rolling_std = []      #record rolling std for each period
rolling_sharp = []    #record rolling sharp for each period


# In[ ]:


t1 = 0 #starting at day 0
t2 = 580
while t2< length:
    #test cointegration:
    
    full_price = full_data.iloc[t1:t2,:]
    coin_pairs_separated = []
    CIT_start = 0
    CIT_end = 454  #2 year to test cointegration

    #need to take out nan values
    full_price = full_price[full_price.columns[~np.any(pd.isna(full_price), axis = 0)]]
    
    for key, value in industry_separed_tickers.items():
        pairs = find_cointegrated_pairs(CIT_start,CIT_end,value)    
        coin_pairs_separated += pairs

    coint_pairs = dict()
    for i in coin_pairs_separated:
        coint_pairs[i[0]] = i[1]
    coint_pairs = collections.Counter(coint_pairs).most_common()[-1:-21:-1]
    coint_pairs = [i[0] for i in coint_pairs]

    vali_star = 328 #must be larger than maximum of mavg window2
    vali_end = 454
    T_star = 454
    T_end = 580

    init_value = 10000  #just for validation period
    inv_uni = {}
    for coin_pair in coint_pairs:
        info, sharp = find_invest_uni_sharp(vali_star, vali_end, coin_pair, init_value, full_price)
        if sharp not in (None, 0):
            inv_uni[(coin_pair, tuple(info))] = sharp
    rank = collections.Counter(inv_uni)
    rank = rank.most_common(10)
    inv_uni = [i[0] for i in rank]
    collect.append(inv_uni)

    
    #3. out of sample test
    init_value = invest/len(inv_uni)  #allocate equally to each pair
    port_performance = pd.DataFrame()
    for i in inv_uni:
        pair = i[0]
        info = i[1]
        pair_performance = out_sample_test(T_star, T_end, pair, info, init_value,full_price)
        pair_performance = pd.DataFrame(pair_performance, columns = ['Date', i[0]])
        pair_performance = pair_performance.set_index('Date')
        port_performance = pd.concat([port_performance, pair_performance], ignore_index=False, axis = 1)
    outcome = port_performance.sum(axis = 1)
    final_outcome = pd.concat([final_outcome, outcome], ignore_index=False, axis = 0)

    Val_end = outcome.iloc[-1]
    Val_star = invest # pay attention to this
    invest = Val_end
    ann_return = Annual_yield(T_star, T_end, Val_star, Val_end)
    rolling_return.append(ann_return)
    #standard devidation
    port_return = pd.DataFrame(outcome)/pd.DataFrame(outcome).shift(1)-1
    port_return = port_return.values
    port_return[0] = 0 #change the first data from nan to 1
    std = np.array(port_return).std()
    annual_std = std * ((252) ** 0.5)
    rolling_std.append(annual_std)
    rf_return = 0.02
    sharp = sharp_ratio(ann_return, rf_return, annual_std)
    rolling_sharp.append(sharp)
    t1 += 126
    t2 += 126




print("rolling_sharp", rolling_sharp, "rolling_return", rolling_return, "rolling_std", rolling_std)
print(final_outcome)
print(collect)
final_outcome.to_csv("/Users/limingsun/Acceleration Capital Group/SP_final_outcome.csv")

