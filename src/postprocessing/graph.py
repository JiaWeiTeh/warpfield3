#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:07:00 2023

@author: Jia Wei Teh

graph.py input 'x', 'y' savefile
"""

import argparse
import matplotlib.pyplot as plt
import read_data 
import cmasher as cmr
from astropy.io import fits
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, ScalarFormatter


# =============================================================================
# Read in parameter files
# =============================================================================
# parser
parser = argparse.ArgumentParser()
# Add option to read in file
parser.add_argument('file', help = 'accepts PATH or model name. When PATH is given, it searches accordingly. If a model name is provided, it looks for a matching folder.')
parser.add_argument('x_axis', default = None, help = 'x-axis of the plot')
parser.add_argument('y_axis', default = None, help = 'y-axis of the plot')
parser.add_argument('-s', '--savefig',
                    help = 'PATH for saving in the directory. If only the filename is given, the plot is saved in a folder with the same model name.', 
                    default=argparse.SUPPRESS)
# grab argument
args = parser.parse_args()

# =============================================================================
# Read table
# =============================================================================

x_array, y_array = read_data.read_data(args.file, args.x_axis, args.y_axis)

# =============================================================================
# Plot
# =============================================================================
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
# if 'savefig' in args:
#     plt.savefig('outputs/'+args.file+'/'+args.savefig)
#     print(f'figure saved to {path2csv[:-7]}')

plt.show()








