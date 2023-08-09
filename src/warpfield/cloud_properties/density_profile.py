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
                         density_specific_param, 
                         rCloud, 
                         mCloud,
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
    mCloud : float
        Mass of cloud (Units: solar mass).
    density_specific_param: float
        Available parameters = { rCore | T }
        - rCore : float
            Core radius (Units: pc). 
            This parameter is only invoked if profile_type == 'pL_prof'
        - T : float
            The temperature of the BE sphere. (Units: K). The default value is 1e5 K.
            This parameter is only invoked if profile_type == 'warpfield_params.dens_profile'
    Returns
    -------
    dens_arr : array of float
        NUMBER DENSITY profile for given radius profile. n(r). (Units: 1/cm^3)
    xi_arr: array of float
        Dimensionless radius xi(r). This will only be returned if 'bE_profile' 
        is selected.

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
    if warpfield_params.dens_profile == "pL_prof":
        # redefine for clarity
        alpha = warpfield_params.dens_a_pL
        rCore = density_specific_param
        # Initialise with power-law
        dens_arr = warpfield_params.nCore * (r_arr/rCore)**alpha
        dens_arr[r_arr <= warpfield_params.rCore] = warpfield_params.nCore
        dens_arr[r_arr > rCloud] = warpfield_params.nISM
        # return n(r)
        return dens_arr, []
        
    # =============================================================================
    # For a Bonnor-Ebert profile
    # =============================================================================
    elif warpfield_params.dens_profile == "bE_prof":
        # redefine for clarity
        T = density_specific_param
        # sound speed
        c_s = bE.get_bE_soundspeed(T, warpfield_params.mu_n, warpfield_params.gamma_adia)
        # initialise
        dens_arr = np.nan * r_arr
        # Convert number density to mass density
        rhoCore = warpfield_params.nCore * warpfield_params.mu_n
        # First convert all to SI units
        rCloud = rCloud * u.pc.to(u.m)
        r_arr = r_arr * u.pc.to(u.m)
        rhoCore = rhoCore * (u.g/u.cm**3).to(u.kg/u.m**3)
        # dimensionless radius array
        xi_arr = np.sqrt(4 * np.pi * c.G.value * rhoCore / c_s**2) * r_arr
        # initial values (boundary conditions)
        # effectively 0.s
        y0 = [1e-12, 1e-12]
        # solve Lane-Emden equation
        # print(xi_arr)
        psi, omega = zip(*scipy.integrate.odeint(bE.laneEmden, y0, xi_arr))
        # store into array
        psi = np.array(psi)
        dens_arr = warpfield_params.nCore * np.exp(-psi)
        # density outside sphere
        dens_arr[r_arr > rCloud] = warpfield_params.nISM
        # return in n(r) cgs
        # print("here is dens_arr, xi_arr")
        # print(dens_arr, xi_arr)
        # sys.exit()
        return dens_arr, xi_arr


# Uncomment to check out plot
#%%
# import matplotlib.pyplot as plt


# rCloud = 355
# rCore = 10
# mu_n = 2.1287915392418182e-24
# gamma = 5/3
# mCloud =  1e9
# r_arr = np.linspace(1e-4, 500, 201)
# nCore = 1000
# nISM = 10


# dens_arr, xi_arr = get_density_profile(r_arr,
#                           rCore, 
#                           rCloud, 
#                           mCloud,
#                           warpfield_params
#                           )

# fig = plt.subplots(1, 1, figsize = (7, 5), dpi = 200)
# plt.plot(xi_arr, dens_arr, 
#           label = 'Density',
#           )
# plt.xlabel('$\\xi(pc)$')
# plt.ylabel('density')
# plt.yscale('log')
# plt.legend()
# print(dens_arr[-5:])
# print(xi_arr[-5:])


