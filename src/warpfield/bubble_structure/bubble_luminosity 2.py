#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:43:02 2023

@author: Jia Wei Teh
"""




# libraries
import numpy as np
import os
import sys
import scipy.optimize
import scipy.integrate
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as c
from astropy.table import Table
#--
import src.warpfield.cooling.get_coolingFunction as get_coolingFunction
import src.warpfield.functions.operations as operations
# get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()




