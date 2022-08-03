#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:37:53 2022

@author: Jia Wei Teh

This script includes function that calculates the density profile.
"""

import numpy as np
import src.warpfield.bonnorEbert as bE
import astropy.units as u
import astropy.constants as c
import scipy.integrate
    
def get_density_profile(r_arr,
                         rCore, rCloud, 
                         nCore, nISM,
                         mCloud,
                         mu_n, gamma,
                         profile_type = "bE_prof", 
                         alpha = -2, g = 14.1,
                         T = 1e5,
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
    rCore : float
        Core radius. (Units: pc)
    rCloud : float
        Cloud radius. (Units: pc)
    nCore : float
        Core number density. (Units: 1/cm^3)
    nISM : float
        ISM number density. (Units: 1/cm^3)
    mCloud : float
        Mass of cloud (Units: solar mass).
    mu_n : float
        Mean mass per nucleus (Units: cgs, i.e., g)
    gamma: float
        Adiabatic index of gas.
    profile_type : string { bE_prof | pL_prof }
        The type of density profile. Currently supports either a power-law 
        distribution, or a Bonnor-Ebert sphere. The default is "bE_prof".
    alpha : float
        The exponent, if power-law profile is chosen for `profile_type`. 
        The default is -2.
    g : float
        The ratio given as g = rho_core/rho_edge. The default is 14.1.
        This will only be considered if `bE_profile` is selected.
    T : float
        The temperature of the BE sphere. (Units: K). The default value is 1e5 K.
        This will only be considered if `bE_profile` is selected. (Units: K)

    Returns
    -------
    dens_arr : array of float
        NUMBER DENSITY profile for given radius profile. n(r). (Units: 1/cm^3)
    xi_arr: array of float
        Dimensionless radius xi(r). This will only be returned if 'bE_profile' 
        is selected.

    """

    # array for easier operation
    r_arr  = np.array(r_arr)
        
    # =============================================================================
    # For a power-law profile
    # =============================================================================
    if profile_type == "pL_prof":
        # Initialise with power-law
        dens_arr = nCore * (r_arr/rCore)**alpha
        dens_arr[r_arr <= rCore] = nCore
        dens_arr[r_arr > rCloud] = nISM
        # return n(r)
        return dens_arr, []
        
    # =============================================================================
    # For a Bonnor-Ebert profile
    # =============================================================================
    elif profile_type == "bE_prof":
        # sound speed
        c_s = bE.get_bE_soundspeed(T, mu_n, gamma)
        # initialise
        dens_arr = np.nan * r_arr
        # Convert number density to mass density
        rhoCore = nCore * mu_n
        # First convert all to SI units
        rCloud = rCloud * u.pc.to(u.m)
        r_arr = r_arr * u.pc.to(u.m)
        rhoCore = rhoCore * (u.g/u.cm**3).to(u.kg/u.m**3)
        # dimensionless radius array
        xi_arr = np.sqrt(4 * np.pi * c.G.value * rhoCore / c_s**2) * r_arr
        # initial values (boundary conditions)
        # effectively 0.
        y0 = [1e-12, 1e-12]
        # solve Lane-Emden equation
        psi, omega = zip(*scipy.integrate.odeint(bE.laneEmden, y0, xi_arr))
        # store into array
        psi = np.array(psi)
        dens_arr = nCore * np.exp(-psi)
        # density outside sphere
        dens_arr[r_arr > rCloud] = nISM
        # return in n(r) cgs
        return dens_arr, xi_arr


# # Uncomment to check out plot
# import matplotlib.pyplot as plt

# rCloud = 355
# rCore = 10
# mu_n = 2.1287915392418182e-24
# gamma = 5/3
# mCloud =  1e9
# r_arr = np.linspace(1e-4, 500, 201)
# nCore = 1000
# nISM = 10


# dens_arr, xi_arr = get_density_profile(
#                           r_arr,
#                           rCore, rCloud, 
#                           nCore, nISM,
#                           mCloud,
#                           mu_n, gamma,
#                           profile_type = "bE_prof", 
#                           # profile_type = "pL_prof", 
#                           alpha = -2, g = 14.1,
#                           T = 451690.2638133162, #1e6
#                           )

# fig = plt.subplots(1, 1, figsize = (7, 5), dpi = 200)
# plt.plot(xi_arr, dens_arr, 
#           label = 'Density',
#           )
# plt.xlabel('$\\xi(pc)$')
# plt.ylabel('density')
# plt.legend()

