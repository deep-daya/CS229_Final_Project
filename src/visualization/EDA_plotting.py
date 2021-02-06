#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:01:27 2020

@author: joshuageiser
"""

import os
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt

def get_datetime(dt_str):
    
    # year = int(dt_str.split('-')[0])
    # month = int(dt_str.split('-')[1])
    # day = int(dt_str.split('-')[2].split(' ')[0])
    
    # hour = int(dt_str.split(' ')[1].split(':')[0])
    # minute = 0
    # second = 0
    
    dt_obj = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
    
    return dt_obj

def to_dt(dt_dataframe):
    
    N = len(dt_dataframe)
    
    return_arr = [''] * N
    
    for i in range(N):
        return_arr[i] = get_datetime(dt_dataframe[i])
    
    return return_arr

def normalize(col):
    
    normalized_col = (col-col.min())/(col.max()-col.min())
    
    return normalized_col

def get_data(data_dir):
    
    csvs = [csv for csv in os.listdir(data_dir) if '_first_pass.csv' in csv]
    
    data = {}
    for csv in csvs:
        city = csv.split('_')[1]
        data[city] = pd.read_csv(os.path.join(data_dir, csv))
        
    return data


def main():
    
    data_dir = os.path.join(os.getcwd(), '..', '..', 'data', 'interim')
    data_dir = os.path.abspath(data_dir)
    data = get_data(data_dir)
    
    city = 'houston'
    
    i_start = 4344
    i_end = 4512
    
    fig, ax1 = plt.subplots()
    
    dates_arr = to_dt(data[city]['date_time'])
    load_norm = data[city]['load'][i_start:i_end]
    tmpc_norm = data[city]['tmpc'][i_start:i_end]

    
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temp (C)', color=color)
    ax1.plot_date(dates_arr[i_start:i_end], tmpc_norm, '--', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Load (MW)', color=color)  # we already handled the x-label with ax1
    ax2.plot_date(dates_arr[i_start:i_end], load_norm, '--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Houston 1-Week Load/Temp Data')
    plt.show()
    
    plt.figure()
    for city in data.keys():
        plt.plot_date(to_dt(data[city]['date_time'])[::168], data[city]['load'][::168], ':', label=city)
        #plt.plot_date(to_dt(data[city]['date_time'])[i_start:i_end], data[city]['load'][i_start:i_end], ':', label=city)
    
    plt.ylim([1000, 18000])
    plt.xticks(rotation=45)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Load (MW)')
    plt.title('Seasonal Load Magnitude by City')
    
    plt.figure()
   
    
    return

if __name__ == "__main__":
    main()