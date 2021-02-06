#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:59:18 2020

@author: joshuageiser
"""

import os
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def get_datetime(dt_str):
    
    dt_obj = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    
    return dt_obj

def to_dt(dt_arr):
    
    N = len(dt_arr)
    
    return_arr = [''] * N
    
    j = 0
    for i in range(N):
        return_arr[j] = get_datetime(dt_arr[i])
        j+=1
    
    return return_arr

def plot_houston_test(filepath): 
    
    df = pd.read_csv(filepath)
    
    dates_arr = to_dt(df['time'])
    
    y_true = df['load']
    y_pred = df['load_pred']
    
    plt.figure()
    
    start_date_hour = '2020-02-01 00:00:00'
    end_date_hour = '2020-02-15 00:00:00'
    
    start_ind = df[df['time']==start_date_hour].index.values[0]
    end_ind = df[df['time']==end_date_hour].index.values[0] # 24*7*2 # 2 weeks
    
    plt.plot_date(dates_arr[start_ind:end_ind], y_true[start_ind:end_ind], 'b:', label='True')
    plt.plot_date(dates_arr[start_ind:end_ind], y_pred[start_ind:end_ind], 'g:', label='Predicted')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Load (MW)')
    plt.title('Single-City Test Set 2-Week Predicted Load')
    plt.legend()
    plt.show()
    
    return

def plot_all_cities_test(filepath): 
    
    mapping = {'boston': 0,
               'chicago': 1,
               'houston': 2,
               'kck': 3, 
               'nyc': 4}
    
    df = pd.read_csv(filepath)
    
    dates_arr = to_dt(df['time'])
    
    y_true = df['load']
    y_pred = df['load_pred']
    
    plt.figure()
    
    start_date_hour = '2020-02-01 00:00:00'
    end_date_hour = '2020-02-15 00:00:00'
    
    start_indices = df[df['time']==start_date_hour].index.values
    end_indices = df[df['time']==end_date_hour].index.values # 24*7*2 # 2 weeks
    
    # Boston
    ind = mapping['boston']
    plt.plot_date(dates_arr[start_indices[ind]:end_indices[ind]], y_true[start_indices[ind]:end_indices[ind]], 'b:', label='True')
    plt.plot_date(dates_arr[start_indices[ind]:end_indices[ind]], y_pred[start_indices[ind]:end_indices[ind]], 'g:', label='Predicted')
    plt.annotate('Boston', (mdates.date2num(dates_arr[end_indices[ind]-28]), 3000), xytext=(3, 12), 
            textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
    
    # Houston
    ind = mapping['houston']
    plt.plot_date(dates_arr[start_indices[ind]:end_indices[ind]], y_true[start_indices[ind]:end_indices[ind]], 'b:', label='True')
    plt.plot_date(dates_arr[start_indices[ind]:end_indices[ind]], y_pred[start_indices[ind]:end_indices[ind]], 'g:', label='Predicted')
    #plt.text(mdates.date2num(dates_arr[end_indices[ind]]), 13000, 'Houston')
    plt.annotate('Houston', (mdates.date2num(dates_arr[end_indices[ind]-38]), 12700), xytext=(5, 15), 
            textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
    
    # NYC
    ind = mapping['nyc']
    plt.plot_date(dates_arr[start_indices[ind]:end_indices[ind]], y_true[start_indices[ind]:end_indices[ind]], 'b:', label='True')
    plt.plot_date(dates_arr[start_indices[ind]:end_indices[ind]], y_pred[start_indices[ind]:end_indices[ind]], 'g:', label='Predicted')
    plt.annotate('NYC', (mdates.date2num(dates_arr[end_indices[ind]-24]), 6500), xytext=(5, 15), 
            textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
    
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Load (MW)')
    plt.title('Multi-City Test Set 2-Week Predicted Load')
    plt.legend(('True','Predicted'))
    plt.show()
    
    return

def plot_all_cities_compare(filepath):
    
    mapping = {'boston': 0,
               'chicago': 1,
               'houston': 2,
               'kck': 3, 
               'nyc': 4}
    
    df = pd.read_csv(filepath)
    
    dates_arr = to_dt(df['time'])
    
    y_true = df['load']
    y_pred = df['load_pred']
    
    
    ##########################################################################
    plt.figure()
    
    start_date_hour = '2020-03-02 00:00:00'
    end_date_hour = '2020-04-02 00:00:00'
    
    start_indices = df[df['time']==start_date_hour].index.values
    end_indices = df[df['time']==end_date_hour].index.values # 24*7*2 # 2 weeks
    
    # NYC
    ind = mapping['nyc']
    plt.plot_date(dates_arr[start_indices[ind]:end_indices[ind]], y_true[start_indices[ind]:end_indices[ind]], 'b:', label='True')
    plt.plot_date(dates_arr[start_indices[ind]:end_indices[ind]], y_pred[start_indices[ind]:end_indices[ind]], 'g:', label='Predicted')
    
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Load (MW)')
    plt.title('Effect of COVID on Load in NYC - Short Term')
    plt.legend()
    #plt.ylim([3700, 6900])
    plt.show()
    
    ##########################################################################
    plt.figure()
    
    start_date_hour = '2020-03-02 00:00:00'
    end_date_hour = '2020-09-02 00:00:00'
    
    start_indices = df[df['time']==start_date_hour].index.values
    end_indices = df[df['time']==end_date_hour].index.values # 24*7*2 # 2 weeks
    
    # NYC
    ind = mapping['nyc']
    ibp = 18 # interval between points 
    plt.plot_date(dates_arr[start_indices[ind]:end_indices[ind]:ibp], y_true[start_indices[ind]:end_indices[ind]:ibp], 'b:', label='True')
    plt.plot_date(dates_arr[start_indices[ind]:end_indices[ind]:ibp], y_pred[start_indices[ind]:end_indices[ind]:ibp], 'g:', label='Predicted')
    
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Load (MW)')
    plt.title('Effect of COVID on Load in NYC - Long Term')
    plt.legend()
    plt.show()    
    
    return


def main():
    
    houston_test_path = os.path.join(os.path.dirname(__file__), 'CS_229_Final_Project_Results', 
                                     'run_111620_223149_houston_test', 'val_pred_array.csv')
    
    #plot_houston_test(houston_test_path)
    
    ##########################################################################
    
    all_cities_test_path = os.path.join(os.path.dirname(__file__), 'CS_229_Final_Project_Results', 
                                     'run_111620_222259_allcities_test', 'val_pred_array.csv')
    
    #plot_all_cities_test(all_cities_test_path)
    
    ##########################################################################
    
    all_cities_compare_path = os.path.join(os.path.dirname(__file__), 'CS_229_Final_Project_Results', 
                                     'run_111620_222319_allcities_compare', 'val_pred_array.csv')
    
    plot_all_cities_compare(all_cities_compare_path)
    

    
    return

if __name__ == "__main__":
    main()