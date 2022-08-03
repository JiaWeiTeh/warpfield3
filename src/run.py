#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:27:38 2022

@author: Jia Wei Teh

This script contains the main file to run WARPFIELD.

In the main directory, type (as an example):
    python3 -m src.input_tools.run param/example.param
"""

import argparse
from src.input_tools import read_param
from src.warpfield import main

# =============================================================================
# Read in parameter files
# =============================================================================
# parser
parser = argparse.ArgumentParser()
# Add option to read in file
parser.add_argument('path2param')
# grab argument
args = parser.parse_args()
# Get dictionary and write summary file
params_dict = read_param.read_param(args.path2param, write_summary = True)
# With this dictionary, run the simulation.
main.expansion(params_dict)
# Done!