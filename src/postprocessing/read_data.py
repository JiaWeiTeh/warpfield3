#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:30:04 2023

@author: Jia Wei Teh

This script plots the result in evo.dat
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, ScalarFormatter

path2fig = r'/Users/jwt/Documents/Code/Courseworks/random/warpfield_brooke/'
path2csv = r'/Users/jwt/Documents/Code/WARPFIELDv4/model_test_output/Z1.00/M6.00/n6079.0_nalpha0.00_nc3.78/'


def get_column(header):
    """ Returns column values given header string."""
    # only for visualisation. Values are already stored in {headers}
    column_names = np.array(['t',
                     'r',
                     'v',
                     'Eb',
                     'Tb',
                     'Pb',
                     'R1',
                     'n0',  
                     'nmax',
                     'fabs',
                     'fabs_i',
                     'fabs_n',
                     'Mshell',
                     'Mcloud',
                     'Mcluster',
                     'Qi',
                     'Lbol',
                     'Li',
                     'Ln',
                     'Lw',
                     'Fram'])
    # these parameters can be requested (as strings):
    # t: time in Myr
    # r: inner shell radius in pc
    # v: shell velocity in km/s
    # Eb: energy in bubble - not available in momentum 
    # Pb: pressure in bubble - not available in momentum 
    # Tb: temperature at certain scaled radius in bubble - not available in momentum 
    # R1: radius of inner shock front - not available in momentum 
    # alpha: dlog(r)/dlog(t) - not available in momentum 
    # beta: -dlog(Pb)/dlog(t) - not available in momentum 
    # delta dlog(Tb)/dlog(t) - not available in momentum 
    # phase: phase of expansion (see phase_lookup.py)
    # nmax: maximum density in shell
    # fabs: fraction of bolometric luminosity that is absorbed by the shell
    # fabs_i: fraction of ionizing luminosity that is absorbed by the shell
    # fabs_n: fraction of non-ionizing luminosity that is absorbed by the shell
    # dRs: thickness of shell in pc
    # Lcool: luminosity lost to cooling (sum of Lbb, Lbcz, and Lb3) - not available in momentum 
    # Lmech: mechanical luminosty of cluster(s)
    # Fgrav: total gravitational force (cluster and self-gravity) - only valid for momentum phase
    # Fwind: wind force imparted on shell - only valid for momentum phase
    # FSN: Supernova force imparted on shell - only valid for momentum phase
    # Fradp_dir: force by direct radiation pressure imparted on shell - only valid for momentum phase
    # Fradp_IR: dorce by indirect (scattered) radiation pressure imparted on shell - only valid for momentum phase
    # frag: fragmentation value (if equal to 1.0, shell fragments) - not available in momentum 
    # logMshell: log10 of shell mass (solar masses)
    # logMcloud: log10 of cloud mass (solar masses)
    # logMcluster: log10 of cluster mass (solar masses)
    # Lbb: cooling luminosity from inner bubble - not available in momentum 
    # Lbcz: cooling luminosity from conduction zone - not available in momentum 
    # Lb3: cooling luminosity from outmost edge of conduction zone - not available in momentum 


    if header not in column_names:
        raise Exception(f"Header not found. Valid inputs are {headers}")
    
    return np.array(data[:, np.where(column_names == header)[0][0]]).astype(float)
    

# =============================================================================
# Take sfe as parameter
# =============================================================================

# in terms of percentage
sfe_list = [10, 20, 30, 40, 50, 60, 70, 80, 85, 90]


# plot
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=12)
fig, ax1 = plt.subplots(1, 1, figsize = (7,5), dpi = 200)
# color
colors = cmr.take_cmap_colors('cmr.rainforest', len(sfe_list), cmap_range=(0.10, 0.90))


xarray = []
yarray = []

for ii, sfe in enumerate(sfe_list[::-1]):
    # get folder
    sfe_str = 'SFE' + str(sfe) + '.00/evo.dat'
    # open file
    csv_file = open(path2csv + sfe_str)
    reader = csv.reader(csv_file, delimiter='\t')
    # skip headers
    headers = next(reader)
    # data
    data = np.array(list(reader))
    
    x = get_column('t')
    y = get_column('r')
    
    if sfe == 85:
        print('here')
        print(max(y))
        print(x[np.where(y == max(y))])
        print(max(x))
        continue
        
    plt.plot(x, y, c = colors[ii])
    
    

sm = plt.cm.ScalarMappable(cmap=cmr.rainforest_r, norm=plt.Normalize(vmin=0, vmax=100))
cbar = plt.colorbar(sm, pad = 0.01)
cbar.ax.tick_params(labelsize=20) 
cbar.set_label(label = '$\\rm \epsilon_\star\ [\%]$', size=15)

# plt.yscale('log')
# plt.xscale('log')
plt.xlim(0, 12)
plt.ylim(0, 59)

#-----------------------------------------------
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(4))
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(4))
plt.gca().tick_params(axis='both', which = 'major', direction = 'in',length = 6, width = 1, labelsize = 20)
plt.gca().tick_params(axis='both', which = 'minor', direction = 'in',length = 4, width =1)
plt.gca().yaxis.set_ticks_position('both')
plt.gca().xaxis.set_ticks_position('both')
plt.gca().get_yaxis().set_major_formatter(ScalarFormatter())
plt.xlabel('$\\rm{t\ [Myr]}$', font = 'Times New Roman', size = 20) 
plt.ylabel('$\\rm Radius\ [pc]$', font = 'Times New Roman', size = 20) 
plt.text(.5, 54, '$\\rm M_{\\rm{cl}} = 10^6\ \\rm{M_\odot}$', size = 15) 
plt.text(.5, 50, '$\\bar{\\rho} = 150\ \\rm M_\odot/pc^3$', size = 15) 
# plt.colorbar()
# plt.legend()
plt.savefig(path2fig+'Warpfield.png')


# with open(path2csv) as csv_file:
#     # initialize
#     reader = csv.reader(csv_file, delimiter='\t')
#     # store headers
#     headers = next(reader)
#     # read data
#     data = np.array(list(reader))
    











