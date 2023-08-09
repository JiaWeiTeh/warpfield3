#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:27:38 2022

@author: Jia Wei Teh

This script contains the main file to run WARPFIELD.

In the main directory, type (as an example):
   python3 ./run.py param/example.param
"""

import os
import argparse
import yaml
from src.input_tools import read_param
# =============================================================================
# Read in parameter files
# =============================================================================
# parser
parser = argparse.ArgumentParser()
# Add option to read in file
parser.add_argument('path2param')
# grab argument
args = parser.parse_args()
# Get class and write summary file
params = read_param.read_param(args.path2param, write_summary = True)
print('Finished reading parameters.')

# from src.warpfield import main
## test
# With this dictionary, run the simulation.
# main.start_expansion()
# Done!
# print("Done!")



from src.input_tools import get_param
warpfield_params = get_param.get_param()

from src.output_tools import write_outputs

write_outputs.get_dir()