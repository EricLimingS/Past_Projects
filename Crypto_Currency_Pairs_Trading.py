#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#use johansen or Engle Granger + Linear Regression to improve, to achieve beta market neutral
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

def pairs_trade(s1, s2, window1, window2, init_value, entry, stop_gain, stop_loss):
    
    # we need to compute the corresponding z score in each day:
    ratio = s1/s2
    mavg_w1 = ratio.rolling(window = window1).mean()
    mavg_w2 = ratio.rolling(window = window2).mean()
    std_w2 = ratio.rolling(window = window2).std()
    z_score_col = []
    z_score = (mavg_w1-mavg_w2)/std_w2
    
   
    
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
            borrow_cost = short_value * 0.02 * (1/(365*24)) # assume interest rate for stock loan is 2%
            margin_acc_interest = margin_acc * 0.015 * (1/(365*24)) #assume interest rate for margin account is 1.5%
            indicator = 1
            enter_point = z_score
            counter +=1
            jj = i
            print('enter1:')
            collect_zscore.append(['enter1',z_score])

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
            borrow_cost = short_value * 0.02 * (1/(365*24)) # assume interest rate for stock loan is 2%
            margin_acc_interest = margin_acc * 0.015 * (1/(365*24)) #change 365*24 based on hourly data, minute data or others 
            indicator = 1
            enter_point = z_score
            counter +=1
            jj = i
            print('enter2:')
            collect_zscore.append(['enter2',z_score])

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
            print('sg,')
            collect_zscore.append(['sg',z_score])
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
            collect_zscore.append(['sl',z_score])
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
            collect_zscore.append(['sl',z_score])
        else:    #put money in Vanguard Prime Money Market Fund
            stock_value = max(s1_pos,0)*s1[i] + max(s2_pos,0)*s2[i]
            short_value = abs(min(s1_pos,0)*s1[i] + min(s2_pos,0)*s2[i])
            cash = cash * (1+0.023) ** (1/(365*24))
            total_value = cash + stock_value + margin_acc - short_value
        
        z_score_col.append(z_score)
        total_value -= (borrow_cost - margin_acc_interest) 
        print(s1.index[i],"total_value:",total_value,'z_score', z_score, s1[i], s2[i])
        port_value.append([str(mavg_w2.index[i])[:10], total_value]) 
    return port_value


def find_invest_uni_sharp(vali_star, vali_end, coin_pair, init_value, full_price):
    inv_uni = [] #investment universe
    max_sharp = 0
    mavg1 = 1
    mavg = list(range(720,1440,240))   #list(range(60,181,5))
    enter = list(np.arange(3.0, 6.3, 0.3))
    
    multiplier1 = list(np.arange(1.1,1.2,0.05)) #sl multiplier
    multiplier2 = list(np.arange(0.9,0.6,-0.05)) #sg multiplier
    
    comb = itertools.product(enter, multiplier1, multiplier2, mavg)
    info = []
    
    for k in comb:
        print(coin_pair)
        
        vali_price = full_price.iloc[(vali_star-k[3]+1):vali_end]   
        port_value = pairs_trade(vali_price[coin_pair[0]],vali_price[coin_pair[1]], mavg1, k[3],init_value,k[0],k[2],k[1]) #? problem!!!!! ✝️ entry:1.5,stop loss:1.7 都不行  
        print(port_value)
        Val_end = port_value[-1][1]
        Val_star = init_value    

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
        annual_std = std * ((365*24) ** 0.5)

        #check max drawdown
        max_dd = Mdd(port_value)

        #Calmar ratio
        calmar_rate = ann_return / max_dd

        if calmar_rate < 1:
            continue
        #if max_dd > 0.30: #0.35
            #continue
        if annual_std > 0.4: #0.35      #when this number is too small, it excludes out valuable pairs
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
    Ann_y = (Val_end/Val_star)**(365*24/(T_diff))-1 
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


df_bitcoin = pd.read_csv('/Users/limingsun/Acceleration Capital Group/Top70_Hourly_Data_from_binance_based_on_monthly_volume.csv')
df_bitcoin = df_bitcoin.set_index('Date')


# In[ ]:


#find cointegration relationship
collect_zscore = []
collect = []
df_bitcoin = df_bitcoin.loc['2018-03-01 20:00:00':'2019-09-01 20:00:00',:]

invest = 100000
final_outcome = pd.DataFrame()
rolling_return = []
rolling_std = []
rolling_sharp = []


f_star = 0     #'2018-09-01 20:00:00'
f_end = 4320
t1 = 1440
t2 = 8640
for k in range(0,2): # two large rolling periods, each period contain 3 small rolling periods. 
    init_value = 30000
       #'2019-03-01 20:00:00'

    f_data = df_bitcoin.iloc[f_star:f_end,:]

    coint_pairs = []
    coin_names = list(f_data.columns)
    coin_name_pairs = itertools.combinations(coin_names, 2)
    counter = 0
    for i in coin_name_pairs:
        n1 = (~pd.isna(f_data[i[0]])).sum()
        n2 = (~pd.isna(f_data[i[1]])).sum()
        n3 = min(n1,n2)
        if n3 < 4310:
            continue
        S1 = f_data[i[0]][-n3:]
        S2 = f_data[i[1]][-n3:]
        if pd.isna(S1).sum() !=0 or pd.isna(S2).sum() !=0:
            continue
        if (adfuller(S1)[1] < 0.05) or (adfuller(S2)[1] < 0.05):
            continue
        p_value = coint(S1,S2)[1]
        if p_value < 0.05:
            coint_pairs.append([i,p_value])  

    Top_coint_pairs = {}
    for pair in coint_pairs:
        Top_coint_pairs[pair[0]] = pair[1]
    top15_coint_pairs = collections.Counter(Top_coint_pairs).most_common()[-1:-21:-1]    



    #validation period

    ans = [i[0] for i in top15_coint_pairs]
    df = df_bitcoin.iloc[t1:t2,:] # 2018-05-01 to 2019-03-01 #must consider the period for mavg
    #2. validation period
    vali_star = 1440 #must be larger than maximum of mavg window2
    vali_end = 2880
    T_star = 2880
    T_end = 4320

    while T_end <= len(df):
        init_value = 30000  #randomly choose number, just for validation period
        inv_uni = {}
        for coin_pair in ans:
            info, sharp = find_invest_uni_sharp(vali_star, vali_end, coin_pair, init_value, df)
            if sharp not in (None, 0):
                inv_uni[(coin_pair, tuple(info))] = sharp
        rank = collections.Counter(inv_uni)
        rank = rank.most_common(10)
        inv_uni = [i[0] for i in rank]

        init_value = invest/len(inv_uni)
        
        #3. out of sample test
        port_performance = pd.DataFrame()
        collect.append(inv_uni)
        for i in inv_uni:
            pair = i[0]
            info = i[1]
            pair_performance = out_sample_test(T_star, T_end, pair, info, init_value,df)
            pair_performance = pd.DataFrame(pair_performance, columns = ['Date', i[0]])
            pair_performance = pair_performance.set_index('Date')
            port_performance = pd.concat([port_performance, pair_performance], ignore_index=False, axis = 1)
        outcome = port_performance.sum(axis = 1)
        final_outcome = pd.concat([final_outcome, outcome], ignore_index=False, axis = 0)

        Val_end = outcome.iloc[-1]
        Val_star = invest # pay attention to this
        ann_return = Annual_yield(T_star, T_end, Val_star, Val_end)
        rolling_return.append(ann_return)
        #standard devidation
        port_return = pd.DataFrame(outcome)/pd.DataFrame(outcome).shift(1)-1
        port_return = port_return.values
        port_return[0] = 0 #change the first data from nan to 1
        std = np.array(port_return).std()
        annual_std = std * ((365*24) ** 0.5)
        rolling_std.append(annual_std)
        rf_return = 0.02
        sharp = sharp_ratio(ann_return, rf_return, annual_std)
        rolling_sharp.append(sharp)

        vali_star += 1440
        vali_end += 1440

        T_star += 1440
        if T_end < len(df):  
            T_end = min(T_end + 1440, len(df))
        else:
            T_end += 1440
        invest = Val_end

    f_star += 4320   #rolling to next 12 month period
    f_end += 4320
    t1 += 4320
    t2 += 4320
    
#print out z_score, investment uni
print("rolling_sharp", rolling_sharp, "rolling_return", rolling_return, "rolling_std", rolling_std)
print(final_outcome)
print(collect)
final_outcome.to_csv("/Users/limingsun/Acceleration Capital Group/final_outcome.csv")

