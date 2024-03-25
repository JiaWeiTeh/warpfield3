#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:30:04 2023

@author: Jia Wei Teh

This script handles reading files.
"""

import os
import re
import csv
import numpy as np
from astropy.io import fits

def read_data(input_file_str, x_axis_str, y_axis_str):

    # Check if the directry to file is given.
    if os.path.isfile(input_file_str):
        path2file = input_file_str
    else:
        # by default, the script looks for *_evolution*.
        _defaultfile = r'evolution'
        # if it does not exist:
        # a) Perhaps its a folder path
        if os.path.isdir(input_file_str):
            path2folder = input_file_str
        # b) Perhaps its a folder name
        else:
            path2folder = os.path.join('./outputs/', input_file_str)
        # 
        try:
            filename = '\w' + _defaultfile + '\.\w'
            for files in os.listdir(path2folder):
                if re.search(filename, files):
                    path2file = os.path.join(path2folder,files)
                    break
        except:
            raise Exception(f'{input_file_str} is not a valid path or folder name.')
            
    # grab extension
    _, file_extension = os.path.splitext(path2file)
    
    if file_extension in ['.dat', '.txt', '.csv']:
        # data
        csv_file = open(path2file)
        reader = csv.reader(csv_file, delimiter='\t')
        header = np.array(next(reader))
        data = np.array(list(reader))
        # check columns
        if x_axis_str not in header:
            raise Exception(f'\'{x_axis_str}\' is invalid. Column must be one of the following:\n{header}')
        if y_axis_str not in header:
            raise Exception(f'\'{y_axis_str}\' is invalid. Column must be one of the following:\n{header}')
            
        # x and y values
        x_array = np.array(data[:, np.where(header == x_axis_str)[0][0]]).astype(float)
        y_array = np.array(data[:, np.where(header == y_axis_str)[0][0]]).astype(float)
            
    
    elif file_extension == '.fits':
        # reads hdu
        hdu = fits.open(path2file)
        # grabs data
        data = hdu[1].data    
        # get header
        header_info = hdu[1].columns
        # check availability
        if x_axis_str not in header_info.names:
            raise Exception(f'\'{x_axis_str}\' is invalid. Column must be one of the following:\n{header_info.names}')
        if y_axis_str not in header_info.names:
            raise Exception(f'\'{y_axis_str}\' is invalid. Column must be one of the following:\n{header_info.names}')
        # get data
        x_array = np.array(data[x_axis_str])
        y_array = np.array(data[y_axis_str])

    return x_array, y_array







