#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:07:00 2023

@author: Jia Wei Teh


graph.py input 'x', 'y' savefile
"""



import os
import argparse
import numpy as np
import yaml
import pandas as pd
import sys
import csv
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, ScalarFormatter

# =============================================================================
# Read in parameter files
# =============================================================================
# parser
parser = argparse.ArgumentParser()
# Add option to read in file
parser.add_argument('model_name', help = 'name of model')
parser.add_argument('x_axis', default = None, help = 'x-axis of the plot')
parser.add_argument('y_axis', default = None, help = 'y-axis of the plot')
parser.add_argument('-s', '--savefig', default = None, help = 'filename to save to directory where .csv file is stored', default=argparse.SUPPRESS)

# grab argument
args = parser.parse_args()

# =============================================================================
# Read table
# =============================================================================
assert os.path.isdir('outputs/'+args.model_name+'/'), f"no such directory \'{args.model_name}\'"
assert os.path.exists(r'outputs/'+args.model_name+'/evo.csv'), f"unable to find \'evo.csv\' in \'{r'outputs/'+args.model_name+'/'}\'"

# paths
path2csv = r'outputs/'+args.model_name+'/evo.csv'
# data
csv_file = open(path2csv)
reader = csv.reader(csv_file, delimiter='\t')
header = np.array(next(reader))
data = np.array(list(reader))
# check columns
if args.x_axis not in header:
    raise Exception(f'\'{args.x_axis}\' is not a column name in {path2csv}')
if args.y_axis not in header:
    raise Exception(f'\'{args.y_axis}\' is not a column name in {path2csv}')

# =============================================================================
# Plot
# =============================================================================
# x and y values
x_array = np.array(data[:, np.where(header == args.x_axis)[0][0]]).astype(float)
y_array = np.array(data[:, np.where(header == args.y_axis)[0][0]]).astype(float)
# plot parameters
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=12)
fig, ax1 = plt.subplots(1, 1, figsize = (7,5), dpi = 100)

plt.plot(x_array, y_array, 'k-')

# asthetics
#-----------------------------------------------
plt.gca().tick_params(axis='both', which = 'major', direction = 'in',length = 5, width = 1, labelsize = 20)
plt.gca().tick_params(axis='both', which = 'minor', direction = 'in',length = 3, width =1)
plt.gca().yaxis.set_ticks_position('both')
plt.gca().xaxis.set_ticks_position('both')
plt.xlabel(args.x_axis, font = 'Times New Roman', size = 20) 
plt.ylabel(args.y_axis, font = 'Times New Roman', size = 20) 
# save?
if 'savefig' in args:
    plt.savefig('outputs/'+args.model_name+'/'+args.savefig)
    print(f'figure saved to {path2csv[:-7]}')

plt.show()






