#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import functools
import statsmodels.api as sm
from itertools import combinations

def dict_arrays(list_tups):
    sector_dict = {}
    
    for stock_sector in list_tups:
        sector = stock_sector[1]
        ticker = stock_sector[0]
        
        if sector not in sector_dict:
            sector_dict[sector] = [ticker]
        else:
            sector_dict[sector].append(ticker)
            
    return sector_dict

def data_array(data_array):
    merged_df = functools.reduce(lambda left,right: pd.merge(left,right,on='Date'), data_array)
    merged_df.set_index('Date', inplace=True)
    return merged_df

def fetch_last_day(year, conn):
     
    cur = conn.cursor()
    cur.execute(SQL, [year,year])        
    data = cur.fetchall()
    cur.close()
    last_day = int(data[0][0])
    return last_day

def fetch_last_day_any_mth(year, mth, conn):
    cur = conn.cursor()
    cur.execute(SQL, [year,mth, year, mth])        
    data = cur.fetchall()
    cur.close()
    last_day = int(data[0][0])
    return last_day

def find_cointegrated_pairs(data, p_value=0.01):
     n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = ts.coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < p_value:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs
   
def load_db_credential_info(f_name_path):
    cur_path = os.getcwd()
    f = open(cur_path + f_name_path, 'r')
    lines = f.readlines()[1:]
    lines = lines[0].split(',')
    return lines

def load_db_tickers_start_date(start_date, conn):
     date_string = start_date.strftime("%Y-%m-%d")
    
    cur = conn.cursor()
    cur.execute(SQL, (date_string,))        
    data = cur.fetchall()
    return data

def load_db_tickers_sectors(start_date, conn):
    date_string = start_date.strftime("%Y-%m-%d")
    cur = conn.cursor()
    cur.execute(SQL, (date_string,))        
    data = cur.fetchall()
    return data

def load_df_stock_data_array(stocks, start_date, end_date, conn):
    array_pd_dfs = []    

    cur = conn.cursor()
    for ticker in stocks:
        cur.execute(SQL, (ticker,))
        results = cur.fetchall()
        stock_data = pd.DataFrame(results, columns=['Date', ticker])
        stock_data = stock_data.sort_values(by=['Date'], ascending = True)
        stock_data[ticker] = stock_data[ticker].astype(float)
        mask = (stock_data['Date'] > start_date) & (stock_data['Date'] <= end_date)
        stock_data = stock_data.loc[mask]
        stock_data = stock_data.reset_index(drop=True)
        array_pd_dfs.append(stock_data)
        
    return array_pd_dfs

def load_pairs_stock_data(pair, start_date, end_date, conn):
     array_pd_dfs = []    

    cur = conn.cursor()
    for ticker in pair:
        cur.execute(SQL, (ticker,))
        results = cur.fetchall()
        stock_data = pd.DataFrame(results, columns=['Date', 'Adj_Close'])
        stock_data = stock_data.sort_values(by=['Date'], ascending = True)
        stock_data['Adj_Close'] = stock_data['Adj_Close'].astype(float)
        mask = (stock_data['Date'] > start_date) & (stock_data['Date'] <= end_date)
        stock_data = stock_data.loc[mask]
        stock_data = stock_data.reset_index(drop=True)
        array_pd_dfs.append(stock_data)
        
    return array_pd_dfs

def pair_data_verifier(array_df_data, pair_tickers, threshold=10):
    stock_1 = pair_tickers[0]
    stock_2 = pair_tickers[1]
    df_merged = pd.merge(array_df_data[0], array_df_data[1], left_on=['Date'], right_on=['Date'], how='inner')
    
    new_col_names = ['Date', stock_1, stock_2] 
    df_merged.columns = new_col_names
    df_merged[stock_1] = df_merged[stock_1].round(decimals = 2)
    df_merged[stock_2] = df_merged[stock_2].round(decimals = 2)
    
    new_size = len(df_merged.index)
    old_size_1 = len(array_df_data[0].index)
    old_size_2 = len(array_df_data[1].index)

    if (old_size_1 - new_size) > threshold or (old_size_2 - new_size) > threshold:
        print("This pair {0} and {1} were missing data.".format(stock_1, stock_2))
        return False
    else:
        return df_merged

def plot_price_series(df, ts1, ts2, start_date, end_date):
    months = mdates.MonthLocator() 
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2], label=ts2)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Prices' % (ts1, ts2))
    plt.legend()
    plt.show()

def plot_scatter_series(df, ts1, ts2):
    plt.xlabel('%s Price ($)' % ts1)
    plt.ylabel('%s Price ($)' % ts2)
    plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
    plt.scatter(df[ts1], df[ts2])
    plt.show()

def plot_residuals(df):
    months = mdates.MonthLocator() 
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals")
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('Residual Plot')
    plt.legend()
    plt.plot(df["res"])
    plt.show()
    
def remove_ticker(ticker, array_pairs_to_clean):
     clean_pairs = []
    
    for pair in array_pairs_to_clean:
        if ticker in pair:
            continue
        else:
            clean_pairs.append(pair)
    return clean_pairs

def write_dict_text(f_name, dict_):

    f_name = f_name + ".txt"
    file_to_write = open(f_name, 'w')
    
    for sector, ticker_arr in dict_.items():
        for ele in ticker_arr:
            new_str = (sector + "," + str(ele)).replace("(","").replace(")","").replace("'","").replace(" ","")
            file_to_write.write("%s\n" % (new_str,)) 

    print("{0} file created.".format(f_name))
    
def write_results_text_file(f_name, sub_array):
    f_name = f_name + ".txt"
    file_to_write = open(f_name, 'w')

    for ele in sub_array:
        file_to_write.write("%s\n" % (ele,))

