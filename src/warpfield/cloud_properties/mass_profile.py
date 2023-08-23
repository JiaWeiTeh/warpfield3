#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:37:58 2022

@author: Jia Wei Teh

This script contains function which computes the mass profile of cloud.
"""

import astropy.units as u
import numpy as np
import sys 

from src.input_tools import get_param
warpfield_params = get_param.get_param()

def get_mass_profile(r_arr,
                         rCloud, 
                         mCloud,
                         return_mdot,
                         **kwargs
                         ):
    """
    This function takes in basic properties of cloud and calculates the 
    radial profile of swept-up mass and time-derivative mass of the shell.
    
    Watch out the units!

    Parameters
    ----------
    r_arr [pc]: list/array of radius of interest
        Radius at which we are interested in the density.
    rCloud [pc]: outer radius of the cloud
        Outer radius of the cloud.
    mCloud [Msol]: float
        Mass of cloud.
    return_rdot: boolean { True | False }
        Whether or not to also compute the time-derivative of radius.
        If True, then further specify an array of velocity.
        - **kwargs -> rdot_arr: None or an array of float
            Time-derivative of radius (i.e., shell velocity). (dr/dt)

    Returns
    -------
    mGas [Msol]: array of float
        The mass profile. 
    mGasdot: array of float. Only returned if return_mdot == True.
        The time-derivative mass profile dM/dt. 

    """
    # convert to np.array
    # array for easier operation
    if hasattr(r_arr, '__len__'):
        r_arr  = np.array(r_arr)
    else:
        r_arr = np.array([r_arr])
    
    # retrieve values
    alpha = warpfield_params.dens_a_pL
    rCore = warpfield_params.rCore
    # Setting up values for mass density (from number density) and 
    # change from g/cm3 to pc/Msol for further computation
    rhoCore = warpfield_params.nCore * warpfield_params.mu_n * (u.g/u.cm**3).to(u.M_sun/u.pc**3)
    rhoAvg = warpfield_params.dens_navg_pL * warpfield_params.mu_n * (u.g/u.cm**3).to(u.M_sun/u.pc**3)
    rhoISM = warpfield_params.nISM * warpfield_params.mu_n * (u.g/u.cm**3).to(u.M_sun/u.pc**3)
    
    # initialise arrays
    mGas = r_arr * np.nan
    mGasdot = r_arr * np.nan

    # ----
    # Case 1: The density profile is homogeneous, i.e., alpha = 0
    if alpha == 0:
        # sphere
        mGas =  4 / 3 * np.pi * r_arr**3 * rhoAvg
        # outer region
        mGas[r_arr > rCloud] =  mCloud + 4. / 3. * np.pi * rhoISM * (r_arr[r_arr > rCloud]**3 - rCloud**3)
        
        # if computing mdot is desired
        if return_mdot:
            try:
                # try to retrieve velocity array
                rdot_arr = kwargs.pop('rdot_arr')
                mGasdot = rdot_arr
                mGasdot[r_arr <= rCloud] = 4 * np.pi * rhoAvg * r_arr[r_arr <= rCloud]**2 * rdot_arr[r_arr <= rCloud]
                mGasdot[r_arr > rCloud] = 4 * np.pi * rhoISM * r_arr[r_arr > rCloud]**2 * rdot_arr[r_arr > rCloud]
                # return value
                return mGas, mGasdot
            except: 
                raise Exception('Velocity array expected.')
        else:
            return mGas
        
    # ----
    # Case 2: The density profile has power-law profile (alpha)
    else:
        # input values into mass array
        # inner sphere
        mGas[r_arr <= rCore] = 4 / 3 * np.pi * r_arr[r_arr <= rCore]**3 * rhoCore
        # composite region, see Eq25 in WARPFIELD 2.0 (Rahååner et al 2018)
        # assume rho_cl \propto rho (r/rCore)**alpha
        mGas[r_arr > rCore] = 4. * np.pi * rhoCore * (
                       rCore**3/3. +\
                      (r_arr[r_arr > rCore]**(3.+alpha) - rCore**(3.+alpha))/((3.+alpha)*rCore**alpha)
                      )
        # outer sphere
        mGas[r_arr > rCloud] = mCloud + 4. / 3. * np.pi * rhoISM * (r_arr[r_arr > rCloud]**3 - rCloud**3)
        
        # return dM/dt.
        if return_mdot:
            try:
                rdot_arr = kwargs.pop('rdot_arr')
            except: 
                raise Exception('Velocity array expected.')
            rdot_arr = np.array(rdot_arr)
            # input values into mass array
            # dm/dt, see above for expressions of m.
            mGasdot[r_arr <= rCore] = 4 * np.pi * rhoCore * r_arr[r_arr <= rCore]**2 * rdot_arr[r_arr <= rCore]
            mGasdot[r_arr > rCore] = 4 * np.pi * rhoCore * (r_arr[r_arr > rCore]**(2+alpha) / rCore**alpha) * rdot_arr[r_arr > rCore]
            mGasdot[r_arr > rCloud] = 4 * np.pi * rhoISM * r_arr[r_arr > rCloud]**2 * rdot_arr[r_arr > rCloud]
            return mGas, mGasdot
        else:
            return mGas
        
