#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:09:03 2020

    Initial cleaning/parsing of dataset

@author: joshuageiser
"""

import os
import pandas as pd
import numpy as np


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
    
    # For input data
    data_release_dir = os.path.join(os.getcwd(), '..', '..', 'data', 'COVID-EMDA', 'data_release')
    data_release_dir = os.path.abspath(data_release_dir)
    
    # For output data
    data_output_dir = os.path.join(os.getcwd(), '..', '..', 'data', 'interim')
    data_output_dir = os.path.abspath(data_output_dir)
    
    # Generate a list containing tuples of input/output filepaths
    for region in mapping.keys():
        city = mapping[region]
        
        input_file = os.path.join(data_release_dir, region, f'{region}_{city}_weather.csv')
        output_file = os.path.join(data_output_dir, f'{region}_{city}_first_pass.csv')
        
        io_filepaths.append((input_file, output_file))
    
    
    return io_filepaths

def parse_data(input_filepath, output_filepath):
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
    
    n_hours = 24 # hours in a day
    
    # Headers and thus number of columns
    header = ['date', 'hour', 'date_time', 'dwpc', 'relh', 'sped', 'tmpc', 'load']
    n_cols = len(header)
    
    # Read in input weather datafile using pandas
    df_in = pd.read_csv(input_filepath)
    
    # Also read in input load datafile using pandas
    df_load = pd.read_csv(input_filepath.replace('_weather.csv', '_load.csv'))
    
    # Array of length 24 denoting strings for each hour (i.e. '0:00')
    hours_arr = np.array(list(df_in.keys()[2:]))
    
    # Initialize output dataframe
    df_out = np.array([]).reshape(0,n_cols)
    
    # Loop through every four columns as a time
    for i in range(0, len(df_in), 4):
        
        # Chunk of data is 4 rows at a time
        chunk = df_in[i:i+4]
        
        # Ensure that all date values are the same for current chunk
        assert(len(set(chunk['date'])) == 1), f"Dates don't match up at index: {i}"
        date_curr = chunk['date'][i]
        
        # Array of length 24 with current date as each entry (i.e. '9/28/20')
        dates_arr = np.full(n_hours, date_curr)
        
        # Array of length 24 with datetime for each entry
        date_time_arr = [date_curr+' '+hour for hour in hours_arr]
        
        # Array of length 4x24 containing weather data for each hour on date
        chunk_arr = np.array(chunk[hours_arr])
        
        # Array of length 24x1 with load data for specified date
        # Try-except block is only needed for caiso_la data as there is some
        # missing data in that load dataset
        try:
            i_load = df_load.index[df_load['date'] == date_curr][0]
            load_data = np.array(df_load[hours_arr].loc[i_load])
            load_data = np.nan_to_num(load_data)
        except IndexError:
            load_data = np.zeros((n_hours))
        
        # Array of length 24x6 with current chunk data
        full_chunk = np.vstack((dates_arr,hours_arr,date_time_arr, chunk_arr, load_data)).T
        
        # Add current chunk to output dataframe
        df_out = np.vstack((df_out, full_chunk))
    
    
    # Write file to output    
    pd.DataFrame(df_out).to_csv(output_filepath, header=header, index=None)
    
    return

def main():
    
    # Get a list containing tuples of input/output filepaths
    io_filepaths = get_io_filepaths()
    
    # Iterate through each region/city and parsing datafile
    for input_filepath,output_filepath in io_filepaths:
        parse_data(input_filepath, output_filepath)
    

    return

if __name__ == "__main__":
    main()