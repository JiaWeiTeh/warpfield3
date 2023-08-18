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

def init_dir():
    """
    This function handles the creation of output files and directories.
    
    Creates
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
    # Create paths to output
    # =============================================================================

    # Old code: get_mypath(), modelprop_to_string() removed. 
    # mypath = warpfield_dir
    warpfield_dir = warpfield_params.out_dir
    # create folder where project data is stored if it does not exist
    # Old code: check_outdir()
    if not os.path.isdir(warpfield_dir):
        os.makedirs(warpfield_dir)

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
    
    return output_filename







