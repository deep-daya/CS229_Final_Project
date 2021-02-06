#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:26:59 2020

@author: joshuageiser
"""

import os
import pandas as pd
import numpy as np
import datetime


def get_io_filepaths():
    '''
    Returns a list containing tuples of input filepaths (for raw CSV data) and
    desired output filepaths for cleaned dataset. 

    Returns
    -------
    io_filepaths : List
        List containing tuples of form (input_filepath, output_filepath). The
        input dataset will be read in from input_filepath and the output data
        will be written to output_filepath

    '''
    
    io_filepaths = []
    
    mapping = {'caiso': 'la',
               'ercot': 'houston', 
               'isone': 'boston', 
               'nyiso': 'nyc', 
               'pjm'  : 'chicago', 
               'spp'  : 'kck'}
    
    # For output data
    data_output_dir = os.path.join(os.getcwd(), '..', '..', 'data', 'interim')
    data_output_dir = os.path.abspath(data_output_dir)
    
    # Generate a list containing tuples of input/output filepaths
    for region in mapping.keys():
        city = mapping[region]
        
        input_file = os.path.join(data_output_dir, f'{region}_{city}_second_pass.csv')
        output_file = os.path.join(data_output_dir, f'{region}_{city}_third_pass.csv')
        
        io_filepaths.append((input_file, output_file))
    
    
    return io_filepaths

def hr_str_to_int(hour):
    return int(hour.split(':')[0])
    

def remove_cols(df_in):
    
    cols_to_remove = [x for x in df_in.columns if '_next_' in x or '_load' in x]
    cols_to_remove.append('date')
    df_in = df_in.drop(cols_to_remove, axis=1)
    
    cols_to_remove = [x for x in df_in.columns if 'max_24_hour' in x or 'min_24_hour' in x or 'mean_24_hour' in x]
    df_in = df_in.drop(cols_to_remove, axis=1)
    
    return df_in

def update_holiday_col(df_in):
    '''
    Since there was a weird bug in the feature engineering that caused only 
    midnight of a holiday to be set to 1. This function ensures that all hours
    on a holiday are set to 1.
    '''
    
    holiday_indices = df_in.index[df_in['holiday'] == 1].tolist()
    
    holiday_dates = [df_in['date'][index] for index in holiday_indices]
        
    for holiday_date in holiday_dates:
        curr_holiday_indices = df_in.index[df_in['date'] == holiday_date].tolist()
        for curr_holiday_index in curr_holiday_indices:
            df_in.loc[curr_holiday_index, 'holiday'] = 1
            #df_in['holiday'][curr_holiday_index] = 1
    
    return df_in

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

def add_week_of_year(df_in):
    '''
    Adds week_of_year column to dataset. NOTE that this uses the ISO definition
    for week 1 where week 1 is the week with the first Thursday in it. This may 
    give some unexpected results near the beginning and end of a year. For 
    example, 2020-01-01 falls in ISO week 52 of ISO year 1999. 
    '''
    
    date_times = np.array(df_in['date_hour'])
    
    date_times = to_dt(date_times)
    
    week_nums = [x.isocalendar()[1] for x in date_times]
    
    df_in['week_of_year'] = week_nums
    
    return df_in

def add_sin_cos_terms(df_in, col_name):
    
    time_arr = np.array(df_in[col_name])
    
    # Make zero indexed if not already
    if min(time_arr) == 1:
        time_arr -= 1
    
    max_val = max(time_arr)
    
    df_in[f'sin_{col_name}'] = np.sin(2 * np.pi * time_arr / max_val)
    df_in[f'cos_{col_name}'] = np.cos(2 * np.pi * time_arr / max_val)
    
    return df_in

def parse_third_pass(input_filepath, output_filepath):
    '''
    Parses through raw CSV dataset and writes it into a more usable CSV format.

    Parameters
    ----------
    input_filepath : String
        Filepath to input CSV dataset.
    output_filepath : String
        Filepath that output CSV data will be written to.

    Returns
    -------
    None.

    '''
    
    print(f'Parsing {os.path.basename(input_filepath)}')
    
    # Read in input weather datafile using pandas
    df_in = pd.read_csv(input_filepath)
    
    # Change hour column from string to int
    df_in['hour'] = df_in['hour'].apply(hr_str_to_int)
    
    # Update holiday column so that all hours in a holiday contain 1
    df_in = update_holiday_col(df_in)
    
    # Add week of year column 
    df_in = add_week_of_year(df_in)
    
    # Add sin/cos terms for hour column and week of year column
    df_in = add_sin_cos_terms(df_in, 'week_of_year')
    df_in = add_sin_cos_terms(df_in, 'hour')
    df_in = add_sin_cos_terms(df_in, 'weekday')
    
    # Remove unneeded features
    df_in = remove_cols(df_in)
    
    # Get the first row that contains non-null values for all columns
    # Uses the cum_avg_7_day_load column as reference
    test_list = df_in['cum_avg_7_day_relh'].isnull()
    start_index = [ind for ind,val in enumerate(test_list) if not val][0]
    
    # Rearrange column order
    cols = df_in.columns.tolist()
    new_cols = [cols[0], cols[2], cols[1]] + cols[3:]
    df_in = df_in[new_cols]
    
    # Start at first row with data in all columns
    df_out = df_in[start_index:]
    
    # Write file to output    
    pd.DataFrame(df_out).to_csv(output_filepath, index=None)
    
    return

def main():
    
    # Get a list containing tuples of input/output filepaths
    io_filepaths = get_io_filepaths()
    
    #io_filepaths = [io_filepaths[1]] # Just for Houston
    
    # Iterate through each region/city and parsing datafile
    for input_filepath,output_filepath in io_filepaths:
        parse_third_pass(input_filepath, output_filepath)
    

    return

if __name__ == "__main__":
    main()