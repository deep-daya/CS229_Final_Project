#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 19:24:18 2020

@author: joshuageiser
"""

import os
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
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


def get_data(data_dir, region, city):
    
    # second pass file (might need to be cleaned a little in the future)
    filepath = os.path.join(data_dir, f'{region}_{city}_third_pass.csv')
    df = pd.read_csv(filepath)
    
    # Constants corresponding to start/end rows of CSV (assuming not LA datafile)
    start_index = 0 # First line where all features have data
    end_index = 27552 # End at last day/hour of 2019
    
    # Get list of columns that will be used for X dataset
    x_cols = list(df.columns)
    x_cols.remove('date_hour')
    x_cols.remove('load')
    x_cols.remove('hour')
    x_cols.remove('weekday')
    x_cols.remove('week_of_year')
    
    # Get X dataset as numpy array
    X = df[x_cols][start_index:end_index]
    X = np.array(X)
    
    # Get y dataset as numpy array
    y_cols = ['load']
    y = df[y_cols][start_index:end_index]
    y = np.array(y)
    
    # Split off December 2019 into a separate validation set for timeseries
    # data visualization
    slice_index = 26112 - start_index
    X_test = X[slice_index:]
    y_test = y[slice_index:]
    X = X[:slice_index]
    y = y[:slice_index]
    
    # Also need the dates range for plotting timeseries
    dates_range_test = np.array(df['date_hour'][slice_index+start_index:end_index])
    
    return X, y, X_test, y_test, dates_range_test


def main():
    
    # Path to data directory contaning CSVs 
    data_dir = os.path.join(os.getcwd(), '..', '..', 'data', 'interim')
    data_dir = os.path.abspath(data_dir)
    
    # For now, just run on one city at a time
    region = 'ercot'
    city = 'houston'
    
    # Get design matrix and labels
    X,y,X_test,y_test,dates_range_test = get_data(data_dir, region, city)
    
    # Split into train and test sets
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train = X
    y_train = y
    
    # Train model and predict 
    mdl = linear_model.LinearRegression()
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    
    # Scatter plot of predictions vs true values
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Load (MW)')
    plt.ylabel('Predicted Load (MW)')
    plt.title('Linear Regression Predicted vs True Load')
    plt.show()
    
    # Error statistics
    MSE = mean_squared_error(y_test, y_pred)
    R2_score = r2_score(y_test, y_pred)
    MAPE = (sum(abs(y_test-y_pred) / y_test) / len(y_test))[0]
    
    print(f'Mean Squared Error: {MSE:.2f}')
    print(f'R2 Score: {R2_score:.3f}')
    print(f'MAPE: {MAPE:.4f}')
    
    ##########################################################################
    
    #y_test_pred = mdl.predict(X_test)
    dates_arr = to_dt(dates_range_test)
    
    plt.figure()
    
    start_ind = 0
    end_ind = 350
    plt.plot_date(dates_arr[start_ind:end_ind], y_test[start_ind:end_ind], 'b:', label='True')
    plt.plot_date(dates_arr[start_ind:end_ind], y_pred[start_ind:end_ind], 'g:', label='Predicted')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Load (MW)')
    plt.title('Linear Regression 2-Week Predicted vs True Load')
    plt.legend()
    plt.show()
    
    return

if __name__ == "__main__":
    main()
    