#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:54:04 2020

@author: joshuageiser
"""

import os
import pandas as pd
import numpy as np
import datetime
import json
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

def calc_metrics(y_true, y_pred, dates_arr, hyperparams, save_outputs=True, 
                 train_y_true=None, train_y_pred=None, train_dates_arr=None):
    '''
    Calculates some metrics and plots for current run of an LSTM and saves them
    to a folder under the root/models subdirectory

    Parameters
    ----------
    y_true : numpy array of shape (m,1)
        True true load values
    y_pred : numpy array of shape (m,1)
        Predicted load values
    dates_arr : numpy array of shape (m,1) 
        Each entry in numpy array should contain a unique date_hour string in
        the format: 'YYYY-MM-DD HH:MM:SS'
    hyperparams : dictionary 
        Contains hyperparameter specifications
    save_outputs : bool, optional
        If True, will save outputs to a generated timestamped directory under
        the root/models subdirectory

    Returns
    -------
    None.

    '''
    
    output_dir = '' # Placeholder if save_outputs is False
    
    if save_outputs:
        # Path to data directory contaning CSVs 
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        output_dir = os.path.abspath(output_dir)
        
        # Current date/time
        run_datetime = datetime.datetime.now().strftime('%m%d%y_%H%M%S')
        new_dir_name = f'run_{run_datetime}'
        output_dir = os.path.join(output_dir, new_dir_name)
        os.mkdir(output_dir)
    
    # Error statistics #######################################################
    
    raw_metrics(y_true, y_pred, output_dir, save_outputs=save_outputs, prefix='val')
    if train_y_true is not None and train_y_pred is not None and train_dates_arr is not None:
        raw_metrics(train_y_true, train_y_pred, output_dir, save_outputs=save_outputs, prefix='train')
    
    # Scatter plot of Predicted vs True Load #################################
        
    plot_scatter(y_true, y_pred, output_dir, save_outputs=save_outputs, prefix='val')
    if train_y_true is not None and train_y_pred is not None and train_dates_arr is not None:
        plot_scatter(train_y_true, train_y_pred, output_dir, save_outputs=save_outputs, prefix='train')
    
    # Timeseries Plot (automatically only plots first two weeks of data) ######   
        
    plot_timeseries(y_true, y_pred, dates_arr, output_dir, save_outputs=save_outputs, prefix='val')
    if train_y_true is not None and train_y_pred is not None and train_dates_arr is not None:
        plot_timeseries(train_y_true, train_y_pred, train_dates_arr, output_dir, save_outputs=save_outputs, prefix='train')
        
    # Save current run's hyperparameters to JSON file ########################
    if save_outputs:
        filename = 'hyperparameters.json'
        with open(os.path.join(output_dir,filename), 'w') as f:
            json.dump(hyperparams, f, indent=4)
            
    # Save arrays of predicted and true load value to CSV ####################
    save_CSV(y_true, y_pred, dates_arr, output_dir, save_outputs=save_outputs, prefix='val')
    if train_y_true is not None and train_y_pred is not None and train_dates_arr is not None:
        save_CSV(train_y_true, train_y_pred, train_dates_arr, output_dir, save_outputs=save_outputs, prefix='train')    
    
    return



def raw_metrics(y_true, y_pred, output_dir, save_outputs=True, prefix='val'):
    '''
    Various useful regression metrics, the following link is a helpful reference
    https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    '''
            
    print('------------------------------------------------------------------')
    print(f'{prefix} raw metrics')
    print('------------------------------------------------------------------')
    
    MSE = metrics.mean_squared_error(y_true, y_pred)
    RMSE = metrics.mean_squared_error(y_true, y_pred, squared=False)
    
    Mean_abs_error = metrics.mean_absolute_error(y_true, y_pred)
    MAPE = (sum(abs(y_true-y_pred) / y_true) / len(y_true))[0]
    Median_abs_error = metrics.median_absolute_error(y_true, y_pred)
    
    Max_error = metrics.max_error(y_true, y_pred)
    R2_score = metrics.r2_score(y_true, y_pred)
    
    metrics_str = ''
    metrics_str += f'Mean Squared Error: {MSE:.2f}\n'
    metrics_str += f'Root Mean Squared Error: {RMSE:.2f}\n'
    metrics_str += f'Mean Absolute Error: {Mean_abs_error:.2f}\n'
    metrics_str += f'Mean Absolute Percentage Error (MAPE): {MAPE:.4f}\n'
    metrics_str += f'Median Absolute Error: {Median_abs_error:.2f}\n'
    metrics_str += f'Max Error: {Max_error:.2f}\n'
    metrics_str += f'R2 Score: {R2_score:.3f}\n'
    
    print(metrics_str)
    
    if save_outputs:
        filename = f'{prefix}_error_stats.txt'
        with open(os.path.join(output_dir,filename), 'w') as f:
            f.write(metrics_str)
    
    return

def plot_scatter(y_true, y_pred, output_dir, save_outputs=True, prefix='val'):
    '''
    Scatter plot of predicted vs true load, ideal is a line of slope 1
    '''
    
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Load (MW)')
    plt.ylabel('Predicted Load (MW)')
    plt.title('Predicted vs True Load')
    
    if save_outputs:
        filename = f'{prefix}_scatter.png'
        plt.savefig(os.path.join(output_dir,filename))
        
    return

def plot_timeseries(y_true, y_pred, dates_arr, output_dir, save_outputs=True, prefix='val'):
    '''
    Timeseries plot of predicted/true load vs datetime to see how well the 
    prediction values match the true curve
    '''
    
    dates_arr_plt = to_dt(dates_arr)
    
    plt.figure()
    
    start_ind = 0
    end_ind = 336 # First two weeks
    end_ind = min(end_ind, len(dates_arr_plt))
    
    plt.plot_date(dates_arr_plt[start_ind:end_ind], y_true[start_ind:end_ind], 'b:', label='True')
    plt.plot_date(dates_arr_plt[start_ind:end_ind], y_pred[start_ind:end_ind], 'g:', label='Predicted')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Load (MW)')
    plt.title('2-Week Predicted vs True Load')
    plt.legend()

    if save_outputs:
        filename = f'{prefix}_timeseries.png'
        plt.savefig(os.path.join(output_dir,filename))    
        
    return

def save_CSV(y_true, y_pred, dates_arr, output_dir, save_outputs=True, prefix='val'):
    '''
    Save CSV file of predicted and true load
    '''
    
    if save_outputs:
        dates_arr = np.reshape(dates_arr, (len(dates_arr),1))
        df_out = np.hstack([dates_arr, y_true, y_pred])
        
        header = ['date_hour', 'True Load', 'Predicted Load']
        filename = f'{prefix}_pred_array.csv'
        output_filepath = os.path.join(output_dir, filename)
        pd.DataFrame(df_out).to_csv(output_filepath, header=header, index=None)
    
    return

def get_datetime(dt_str):
    '''
    Helper function to get a datetime object from a date_hour string
    '''
    
    dt_obj = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    
    return dt_obj

def to_dt(dt_arr):
    '''
    Helper function to convert array of date_hour timestrings to datetime objects
    '''
    
    # If already a list of datetime objects, just return the list
    if isinstance(dt_arr[0], datetime.datetime):
        return dt_arr
    
    # Other wise convert from strings to datetime objects
    N = len(dt_arr)
    
    return_arr = [''] * N
    
    j = 0
    for i in range(N):
        return_arr[j] = get_datetime(dt_arr[i])
        j+=1
    
    return return_arr

def main():
    
    # Path to data directory contaning CSVs 
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'interim')
    data_dir = os.path.abspath(data_dir)
    
    # For now, just run on one city at a time
    region = 'ercot'
    city = 'houston'
    
    # Get design matrix and labels
    from base_model_sklearn import get_data
    X,y,X_test,y_test,dates_range_test = get_data(data_dir, region, city)
    
    # Split into train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Train model and predict 
    mdl = linear_model.LinearRegression()
    mdl.fit(X_train, y_train)
    
    ##########################################################################
    
    # Will need to update our LSTM architecture so that we input all of our
    # hyperparameters in a dictionary format
    hyperparams = {'window': 24,
                   'memory_layer_units': 10,
                   'num_hidden_layers': 2,
                   'training_epochs': 30
                   }
    
    # Need 3 array inputs: true load, predicted load, and array of date_hour 
    # strings for the range of prediction
    y_true = y_test
    y_pred = mdl.predict(X_test)
    dates_arr = dates_range_test
    
    from copy import deepcopy
    
    # For optional train inputs (this example isn't real training data!!!!)
    train_y_true = deepcopy(y_true[500:])
    train_y_pred = deepcopy(y_pred[500:])
    train_dates_arr = deepcopy(to_dt(dates_arr[500:]))
    
    # Calculate metrics
    calc_metrics(y_true, y_pred, dates_arr, hyperparams, save_outputs=True, 
                 train_y_true=train_y_true, train_y_pred=train_y_pred, train_dates_arr=train_dates_arr)
    
    return

if __name__ == "__main__":
    main()
    