#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('./pythonclient')  #add another search directory, there might be better way to do this, check later


# In[ ]:


'''
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class TestApp(EWrapper, EClient):   #inheritate from EWrapper and Eclient classes
    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self,self)
    def error(self, reqId, errorCode, errorString):  #we overwrite the original one, the errorId -1 means connection is successful
        print("Error: ", reqId, " ", errorCode, " ", errorString)
    def contractDetails(self, reqId, contractDetails):     #EWrapper function, this receives and handles the returning data
        print("contractDetails: ", reqId, " ", contractDetails)

def main():
    app = TestApp()
    
    app.connect("127.0.0.1", 7496, 0) #IP, port_number, client ID
    
    #represents trading instruments such as a stocks, futures or options.
    #Every time a new request that requires a contract (i.e. market data, order placing, etc.) is sent to TWS, the platform will 
    #try to match the provided contract object with a single candidate. 
    #to find the description of contract: TWS--tickername--double click--expand--financial instrument info--description
    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"   #if you use smart, the API look through all the exchanges
    contract.currency = "USD"
    contract.primaryExchange = "NASDAQ"
    
    app.reqContractDetails(1, contract)    #EClient function, through which, the client can send messages to TWS. 
                                           #1st parameter is "request ID", which should be unique for each request
    #Once the client is connected, a reader thread will be automatically created to handle incoming messages and 
    #put the messages into a message queue for further process. User is required to trigger Client::run() below,
    #where the message queue is processed in an infinite loop and the EWrapper call-back functions are automatically 
    #triggered.
    
    app.run() #call the run loop, which is used to process the messages in the message queue, recieved from TWS
    
if __name__ == "__main__":
    main()
'''    


# In[ ]:


'''
for reqMktData function:
4 data types: live data(default), frozen(after market closed, the last available data), delayed(if has no subscription), delayed-frozen

'''
'''
#demo for marketData.py
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum

class TestApp(EWrapper, EClient):   #inheritate from EWrapper and Eclient classes
    def __init__(self):
        EClient.__init__(self,self)
        
    def error(self, reqId, errorCode, errorString):  #we overwrite the original one
        print("Error: ", reqId, " ", errorCode, " ", errorString)
        
    def tickPrice(self, reqId, tickType, price, attrib):
        print("Tick Price, Ticker ID: ", reqId, "tickType: ", TickTypeEnum.to_str(tickType), "Price: ", price, end=' ')
    
    def tickSize(self, reqId, tickType, size):
        print("Tick Size, Ticker ID: ", reqId, "tickType: ", TickTypeEnum.to_str(tickType), "size: ", size)

        
def main():
    app = TestApp()
    
    app.connect("127.0.0.1", 7496, 0)
    
    #create a contract object
    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"   #if you use smart, the API look through all available exchanges
    contract.currency = "USD"
    contract.primaryExchange = "NASDAQ"
    
    app.reqMarketDataType(4)    #switch to delayed-frozen dta if live not available
    app.reqMktData(1, contract, "", False, False, [])  #1st parameter is request ID, 
    
    app.run()



if __name__ == "__main__":
    main()
    
'''


# In[2]:


# to retrieve live data: need market data subscription + access in account management
# retrieve historical data

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class TestApp(EWrapper, EClient):   #inheritate from EWrapper and Eclient classes
    def __init__(self):
        EClient.__init__(self,self)
        
    def error(self, reqId, errorCode, errorString):  #we overwrite the original one
        print("Error: ", reqId, " ", errorCode, " ", errorString)
    
    def historicalData(self, reqId, bar):
        print("HistoricalData. ", reqId, "Date: ", bar.date, "Open: ", bar.open, "High: ", bar.high, "Low: ", bar.low, "Close: ", bar.close, "Volume: ", bar.volume, "Count: ", bar.barCount, "WAP: ", bar.average)
        
def main():
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
    
    #app.run()

    
if __name__ == "__main__":
    main()


# In[ ]:




