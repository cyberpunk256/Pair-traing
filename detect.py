#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import datetime as dt
import os
import pycopg2
import seaborn
import functools
import matplotlib.pyplot as plt

import base as bs


# In[ ]:


def main():
    skip_etfs = True
    db_credential_info_p = "\\" + "database_info.txt"
    
    db_host, db_user, db_password, db_name = cm.load_db_credential_info(db_credential_info_p)
    conn = psycopg2.connect(host=db_host,database=db_name, user=db_user, password=db_password)
    
    year_array = list(range(2004, 2015))
    
    for yr in year_array:
        year = yr
        end_year = year + 2
        last_tr_day_start = cm.fetch_last_day_mth(year, conn)
        last_tr_day_end = cm.fetch_last_day_mth(end_year, conn)
        start_dt = datetime.date(year,12,last_tr_day_start)
        end_dt = datetime.date(end_year,12,last_tr_day_end)
        start_dt_str = start_dt.strftime("%Y%m%d")
        end_dt_str = end_dt.strftime("%Y%m%d")
        
        list_of_stocks = cm.load_db_tickers_sectors(start_dt, conn)
        sector_dict = cm.build_dict_of_arrays(list_of_stocks)
        passed_pairs = {}
        
        for sector, ticker_arr in sector_dict.items():
            if skip_etfs and sector != "ETF":
                
                ticker_arr.append('SPY')

                data_array_of_dfs = cm.load_df_stock_data_array(ticker_arr, start_dt, end_dt, conn)
                merged_data = cm.data_array_merge(data_array_of_dfs)
                

                scores, pvalues, pairs = cm.find_cointegrated_pairs(merged_data)
                
                new_pairs = cm.remove_ticker('SPY', pairs)
                passed_pairs[sector] = new_pairs
                print("Complete sector {0} for date range: {1}-{2}".format(sector, start_dt_str, end_dt_str))
                
        f_name = "coint_method_pairs_{0}".format(end_dt_str)    
        cm.write_dict_text(f_name, passed_pairs)
    
    
if __name__ == "__main__":
    main()

