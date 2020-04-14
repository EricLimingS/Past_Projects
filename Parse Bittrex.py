#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://www.cnblogs.com/derek1184405959/p/8450130.html
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import collections


# In[18]:


#extract data from bittrex
from selenium.webdriver import ActionChains
path_driver = '/Users/limingsun/Python/Web Crawling/chromedriver'
url = 'https://bittrex.com/home/markets'
element_list = [[], [], []]

driver = webdriver.Chrome(path_driver)
driver.get(url)
time.sleep(3) #leave some time for brower to load data


#find the top 120
for j in range(0,5): #5 pages
    for i in range(0, 20): #20 items per page
        try: # if the total number of rows on the web is less than 100, there will be indexerror, so, just omit it
            body1 = '//*[@id="home-wrapper"]/div[2]/div[3]/table/tbody/tr[' + str(i+1) + ']/td[1]/a/span' #coin name
            body2 = '//*[@id="home-wrapper"]/div[2]/div[3]/table/tbody/tr[' + str(i+1) + ']/td[3]'   #volume
            body3 = '//*[@id="home-wrapper"]/div[2]/div[3]/table/tbody/tr[' + str(i+1) + ']/td[5]' #last price in bitcoin

            b1 = driver.find_elements_by_xpath(body1)[0].text
            b2 = driver.find_elements_by_xpath(body2)[0].text
            b2 = float(b2.replace(',',''))   #change the price to int
            b3 = driver.find_elements_by_xpath(body3)[0].text
            

            element_list[0].append(b1)
            element_list[1].append(b2)
            element_list[2].append(b3)
        except IndexError:
            break
            
    target = driver.find_element_by_xpath('//*[@id="home-wrapper"]/div[2]/div[3]/nav/ul/li[4]/a') #click next page
    time.sleep(3)
    #target.click()
    ActionChains(driver).move_to_element(target).click(target).perform()
    time.sleep(3)

driver.quit()
df = pd.DataFrame(element_list, index = ['pair','volume','last_price']).T
df['pair'] = df['pair'].apply(lambda x: x.split('-')[1])
df = df.set_index('pair')
print(df)


# In[ ]:





# In[ ]:




