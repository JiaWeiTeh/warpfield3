#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:37:53 2022

@author: Jia Wei Teh

This script includes function that calculates the density profile.
"""

import numpy as np
import astropy.units as u
#--
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

    # =============================================================================
    # For a power-law profile
    # =============================================================================
    # redefine for clarity
    alpha = warpfield_params.dens_a_pL
    nAvg = warpfield_params.dens_navg_pL
    rCore = warpfield_params.rCore
    nISM = warpfield_params.nISM
    nCore = warpfield_params.nCore
    # make sure units are right for operations
    rCore = rCore.to(u.pc)
    rCloud = rCloud.to(u.pc)
    r_arr = r_arr.to(u.pc)
    
    # Initialise with power-law
    # for different alphas:
    if alpha == 0:
        dens_arr = nISM * r_arr ** alpha
        dens_arr[r_arr <= rCloud] = nAvg
    else:
        dens_arr = nCore * (r_arr/rCore)**alpha
        dens_arr[r_arr <= rCore] = nCore
        dens_arr[r_arr > rCloud] = nISM
    
    # return n(r)
    return dens_arr
        






