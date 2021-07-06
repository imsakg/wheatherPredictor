#!/usr/bin/env python3

import chromedriver_binary
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import datetime

firstDate = datetime.datetime(2020,2,23)
driver = webdriver.Chrome()

while (firstDate + datetime.timedelta(days=1) < datetime.datetime.today()):
    url = f'https://www.wunderground.com/history/daily/tr/osmangazi/LTBR/date/{firstDate.date()}'
    driver.get(url)
    try:
        tables = WebDriverWait(driver,20).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table")))
        for table in tables:
            newTable = pd.read_html(table.get_attribute('outerHTML'))
            if newTable:
                newTable = pd.DataFrame(data=newTable[0])
                newTable.to_csv(f"datas/{firstDate.date()}.csv")
        print(firstDate.date())
    except:
        print("error on: ",firstDate.date())
    firstDate += datetime.timedelta(days=1)
