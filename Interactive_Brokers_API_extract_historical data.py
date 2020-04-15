#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('./pythonclient')  #add another search directory, there might be better way to do this, check later

# to retrieve live data: need market data subscription + access in account management
# retrieve historical data

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

import pandas as pd

data_collected = []
column = []

class TestApp(EWrapper, EClient):   #inheritate from EWrapper and Eclient classes
    def __init__(self):
        EClient.__init__(self,self)
        
    def error(self, reqId, errorCode, errorString):  #we overwrite the original one
        print("Error: ", reqId, " ", errorCode, " ", errorString)
    
    def historicalData(self, reqId, bar):
        global data_collected, column
        data_collected.append([reqId, bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.barCount, bar.average])
        column = ["HistoricalData", "Date", "Open", "High", "Low", "Close", "Volume",  "Count", "WAP"]
        #print("HistoricalData. ", reqId, "Date: ", bar.date, "Open: ", bar.open, "High: ", bar.high, "Low: ", bar.low, "Close: ", bar.close, "Volume: ", bar.volume, "Count: ", bar.barCount, "WAP: ", bar.average)
        
def main():
    global data_collected, column
    data_collected = []
    app = TestApp()
    
    app.connect("127.0.0.1", 7496, 0)
    
    #create a contract object
    contract = Contract()

    contract.symbol = "EUR"
    contract.secType = "CASH"
    contract.exchange = "IDEALPRO"   #if you use smart, the API look through all available exchanges
    contract.currency = "USD"
    
    #http://interactivebrokers.github.io/tws-api/historical_bars.html
    app.reqHistoricalData(1, contract, "20200127 23:59:59", "1 D", "1 min", "MIDPOINT", 1, 1, False, [])
    #3rd parameter is "end day", if empty space, it means current day, duration,bar size, regular trading hour, format of date, keep_up_the_date, always [] reserved for internal use
    
    app.run()
    df = pd.DataFrame(data_collected, columns = column)
    df.to_csv("historical_data")
    print (df)


    
if __name__ == "__main__":
    main()







