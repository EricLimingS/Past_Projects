#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import collections

path_driver = '/Users/limingsun/Python/Web Crawling/chromedriver'
url = 'https://www.hbg.com/zh-cn/markets/#btc'
element_list = [[], [], []]

driver = webdriver.Chrome(path_driver)
driver.get(url)
time.sleep(3) #leave some time for brower to load data

element3 = driver.find_element_by_xpath('//*[@id="scroll_head"]/dt/span[8]/em')
time.sleep(3) #leave some time for brower to load data
element3.click()  #I feel click is not stable enought, sometime it meets problem for no reason

#find the top 100
for i in range(0, 100):
    body1 = '//*[@id="symbol_list"]/dd[' + str(i+1) + ']/span[2]'
    body2 = '//*[@id="symbol_list"]/dd[' + str(i+1) + ']/span[8]'
    body3 = '//*[@id="symbol_list"]/dd[' + str(i+1) + ']/span[3]/b'  #in bitcoin

    b1 = driver.find_elements_by_xpath(body1)[0].text
    b2 = driver.find_elements_by_xpath(body2)[0].text
    #b2 = b2.replace(',','') #change the price to int
    #b2 = float(b2[1:])
    b3 = driver.find_elements_by_xpath(body3)[0].text

    element_list[0].append(b1)
    element_list[1].append(b2)
    element_list[2].append(b3)
driver.quit()

df = pd.DataFrame(element_list, index = ['Symbol','24H Volume','Price']).T
df['Symbol'] = df['Symbol'].apply(lambda x: x.split(' ')[0])
df = df.set_index('Symbol')
print(df)


# In[ ]:





# In[ ]:




