#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 23:10:05 2023

@author: Jia Wei Teh

This script plots a simple net-cooling rate curve (dudt).
"""

import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np

import src.warpfield.cooling.net_coolingcurve as net_coolingcurve

# os.environ['PATH_TO_CONFIG'] = r'/Users/jwt/Documents/Code/warpfield3/outputs/example_pl/example_pl_config.yaml'


T_arr = np.logspace(4, 8, 100) * u.K

age = np.log10(0.00012057636642393612 * 1e6)
# ndens = np.log10(506663.2212419483)
ndens = np.log10(1972534.1634600703) 
# T = np.log10(294060.78931362595)
# phi = np.log10(1.5473355225629384e+16)
phi = np.log10(1.104236441405574e+17) 

dudt_arr = []
for T in T_arr:
    dudt_arr.append(net_coolingcurve.get_dudt(10**age * u.yr, 10**ndens/ u.cm**3, T, 10**phi/ u.cm**2 / u.s))



#%%

# dudt_arr = [i * (u.erg/u.pc**3/u.Myr) for i in dudt_arr]

dudt_arr = [i.value for i in dudt_arr]

plt.plot(T_arr.value, dudt_arr, 'k-')
plt.vlines(10**(5.5), min(dudt_arr), max(dudt_arr), linestyle = '--')
plt.xscale('log')
# plt.yscale('log')
plt.show()