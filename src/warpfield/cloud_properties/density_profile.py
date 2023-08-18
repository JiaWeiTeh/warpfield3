#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:37:53 2022

@author: Jia Wei Teh

This script includes function that calculates the density profile.
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
import sys
import scipy.integrate
#--
import src.warpfield.cloud_properties.bonnorEbert as bE
from src.input_tools import get_param
warpfield_params = get_param.get_param()

def get_density_profile(r_arr,
                         rCloud, 
                         ):
    """
    This function takes in a list of radius and evaluates the density profile
    based on the points given in the list. The output will depend on selected
    type of density profile describing the sphere.
    
    Watch out the units!

    Parameters
    ----------
    r_arr : list/array of radius of interest
        Radius at which we are interested in the density (Units: pc).
    rCloud : float
        Cloud radius. (Units: pc)

    Returns
    -------
    dens_arr : array of float
        NUMBER DENSITY profile for given radius profile. n(r). (Units: 1/cm^3)

    """
    
    # Note:
        # old code: f_dens and f_densBE().

    # array for easier operation
    if hasattr(r_arr, '__len__'):
        r_arr  = np.array(r_arr)
    else:
        r_arr  = np.array([r_arr])
        
    # =============================================================================
    # For a power-law profile
    # =============================================================================
    # redefine for clarity
    alpha = warpfield_params.dens_a_pL
    nAvg = warpfield_params.dens_navg_pL
    rCore = warpfield_params.rCore
    # Initialise with power-law
    # for different alphas:
    if alpha == 0:
        dens_arr = warpfield_params.nISM * r_arr ** alpha
        dens_arr[r_arr <= rCloud] = nAvg
    else:
        dens_arr = warpfield_params.nCore * (r_arr/rCore)**alpha
        dens_arr[r_arr <= warpfield_params.rCore] = warpfield_params.nCore
        dens_arr[r_arr > rCloud] = warpfield_params.nISM
    
    # return n(r)
    return dens_arr
        






