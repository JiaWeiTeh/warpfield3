#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:02:19 2023

@author: Jia Wei Teh


Note:
    Old code: warp_writedata.py and warp_nameparser.py
    
"""

import os
# get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()

def get_dir():
    """
    This function handles the creation of output files and directories.
    

    Returns
    -------
    warpfield_dir : str
        The path to output. This is the main directory where everything
        from a run is being dump into. 
    cloudy_dir : str
        The path to what will be input into cloudy (and output).
    output_filename : str
        The path to (and including) the output filename.

    In addition, this function also creates directory to /bubbles, /potential, /figures, 
    and /stellar_prop if desired.
    
    - /stellar_prop: contains stellar (e.g., starburst99) feedback parameters as a 
    function of time as were used by WARPFIELD.
    - /potential: stores file with gravitational potential.

    """
    
    # Old code: getmake_dir(), savedir(), create_inputfile(), check_outdir(), get_cloudypath(), get_fname()

    # =============================================================================
    # First, create folder where data will be stored.
    # =============================================================================
    # create folder where project data is stored if it does not exist
    # Old code: check_outdir()
    # if not os.path.isdir(basedir):
    #     os.makedirs(basedir)

    # =============================================================================
    # Then, create paths to output
    # =============================================================================

    # Old code: get_mypath(), modelprop_to_string() removed. 
    # mypath = warpfield_dir
    warpfield_dir = warpfield_params.out_dir
    print('here')
    # create folder where project data is stored if it does not exist
    # Old code: check_outdir()
    print(warpfield_dir)
    if not os.path.isdir(warpfield_dir):
        print(warpfield_dir)
        # os.makedirs(mypath)

    if warpfield_params.write_potential == True:
        potential_dir = os.path.join(warpfield_dir, "potential")
        os.makedirs(potential_dir)

    if warpfield_params.write_bubble == True:
        bubble_dir = os.path.join(warpfield_dir, "bubble")
        os.makedirs(bubble_dir)

    if warpfield_params.write_figures == True:
        figures_dir = os.path.join(warpfield_dir, "figures")
        os.makedirs(figures_dir)
  
    # This was write_SB99, but changed because SB99 will not be the only option. We
    # might have SLUG, CIGALE etc in the future.
    if warpfield_params.write_stellar_prop == True:
        figures_dir = os.path.join(warpfield_dir, "stellar_prop")
        os.makedirs(figures_dir)
        
    # Old: input is saved in input.dat, and output is saved in evo.dat.
    output_filename = os.path.join(warpfield_dir, warpfield_params.model_name+'_summary.txt')
    
    return print(warpfield_dir, output_filename)



#%%


# 2023-08-09 15:08:17.673730: start model 1 with Mcloud=1000000000.0 (9.0), SFE=0.01, n0=1000, g=15 and Z=1
# Initialization of Bonnor-Ebert spheres may take a few moments....
# Tsol =  451690.2638133162
# Test
# 1000 451690.2638133162 990000000.0
# Cloud radius in pc= 355.8658723191992
# nedge in 1/ccm= 66.66666667279057
# g after= 14.999999998622123
# navg= 167.04665361693927
# ('ODEpar:', {'gamma': 1.6666666666666667, 'Mcloud_au': 990000000.0, 'rhocore_au': 31.394226159698523, 'Rcore_au': 451690.2638133162, 'nalpha': -2, 'Mcluster_au': 10000000.0, 'Rcloud_au': 355.8658723191992, 'SFE': 0.01, 'nedge': 66.66666667279057, 'Rsh_max': 0.0, 't_dissolve': 1e+30, 'tSF_list': array([0.]), 'Mcluster_list': array([10000000.]), 'tStop': 50})
# getSB99_data: mass scaling f_mass = 10.000
# this is input to modelprop_to_string
# 170 1 0.01 1000000000.0 -2 1000
# this is output to modelprop_to_string
# n170.0
# Z1.00
# SFE1.00
# M9.00
# nalpha-2.00
# nc3.00
# this is mypath
# /Users/jwt/Documents/Code/WARPFIELDv4/model_test_output/Z1.00/M9.00/g15.00_nc3.00/SFE1.00
# this is input to modelprop_to_string
# 170 1 0.01 1000000000.0 -2 1000
# this is output to modelprop_to_string
# n170.0
# Z1.00
# SFE1.00
# M9.00
# nalpha-2.00
# nc3.00












