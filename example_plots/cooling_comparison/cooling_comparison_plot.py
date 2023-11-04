#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:24:39 2023

@author: Jia Wei Teh
"""


import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np


# os.environ['PATH_TO_CONFIG'] = r'/Users/jwt/Documents/Code/warpfield3/outputs/example_pl/example_pl_config.yaml'



x4, y4 = np.load(r'/Users/jwt/Documents/Code/warpfield3/example_plots/cooling_comparison/cooling4.npy')
x3, y3 = np.load(r'/Users/jwt/Documents/Code/warpfield3/example_plots/cooling_comparison/cooling3.npy')


plt.figure(figsize = (7, 5), dpi = 200)
plt.plot(x4, y4, label ='wfld4', linewidth = 10, alpha = 0.4, c = 'c')
plt.plot(x3, y3, label ='wfld3', c = 'k')
plt.xlabel('logT')
plt.ylabel('logLambda')
plt.axvline(5.5)
plt.savefig('/Users/jwt/Documents/Code/warpfield3/example_plots/cooling_comparison/first_zone_cooling.png')
plt.legend()