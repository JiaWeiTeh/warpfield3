#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 15:08:23 2023

@author: Jia Wei Teh

This is a master script which craetes a NET cooling rate (dudt) curve containing both CIE and non-CIE
conditions.

old code: coolnoeq.cool_interp_master()
"""
import scipy.interpolate
import numpy as np
import astropy.units as u

import src.warpfield.cooling.CIE.read_coolingcurve as CIE
# get_Lambda
import src.warpfield.cooling.non_CIE.read_cloudy as non_CIE
from src.output_tools.terminal_prints import cprint as cpr


def get_dudt(age, ndens, T, phi):
    """
    

    Parameters
    ----------
    age [Myr]: TYPE
        DESCRIPTION.
    ndens : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    dudt is erg /cm3 /s, because cooling is in units of [erg cm3 / s]
    TYPE
        DESCRIPTION.

    """
    
    # These value should not be logged!
    # double checking units
    ndens = ndens.to(1/u.cm**3)
    phi = phi.to(1/u.cm**2/u.s)
    
    # New idea, since non-CIE curve is only up to 10^5.5K, which is exactly
    # what our threshold is, we create an if/else function that returns 
    # Lambda(T)_CIE if above, and Lambda(n,T,phi) if below. 
    # If between max of nonCIE and min of CIE, take interpolation between two Lambda values.
    
    # import values from two cooling curves
    cooling_nonCIE, heating_nonCIE = non_CIE.get_coolingStructure(age)
    Lambda_CIE, logT_CIE, logLambda_CIE, _ = CIE.get_Lambda(T)
    
    # we take the cutoff at 10e5.5 K. 
    # These are all in log-space. 
    # cutoff at which temperature above switches to CIE file:
    nonCIE_Tcutoff = max(cooling_nonCIE.temp[cooling_nonCIE.temp.value <= 5.5])
    # cutoff at which temperature below switches to non-CIE file:
    CIE_Tcutoff = min(logT_CIE[logT_CIE.value > 5.5])
    # output
    # print(f'{cpr.WARN}Taking net-cooling curve from non-CIE condition at T <= {nonCIE_Tcutoff}K and CIE condition at T >= {CIE_Tcutoff}K.{cpr.END}')
    # if nonCIE_Tcutoff != CIE_Tcutoff:
        # print(f'{cpr.WARN}Net cooling for temperature values in-between will be interpolated{cpr.END}.')

    # if temperature is lower than the non-CIE temperature, use non-CIE
    if np.log10(T.value) <= nonCIE_Tcutoff.value and np.log10(T.value) >= min(cooling_nonCIE.temp).value:
        # print(f'{cpr.WARN}Entering non-CIE regime...{cpr.END}')
        # All this does here is to interpolate for values of Lambda based on
        # T, dens and phi.
        
        # netcooling grid
        netcooling = cooling_nonCIE.datacube - heating_nonCIE.datacube
        # create interpolation function
        f_dudt = scipy.interpolate.RegularGridInterpolator((cooling_nonCIE.ndens, cooling_nonCIE.temp, cooling_nonCIE.phi), netcooling)
        # get net cooling rate
        # remember that these have to be logged!
        dudt = f_dudt([np.log10(ndens.value), np.log10(T.value), np.log10(phi.value)])[0] * u.erg / u.cm**3 / u.s
        # return in negative sign for convension (since the rate of change is negative due to net cooling)
        return -1 * dudt
        
    # if temperature is higher than the CIE curve, use CIE.
    elif np.log10(T.value) >= CIE_Tcutoff.value:
        # print(f'{cpr.WARN}Entering CIE regime...{cpr.END}')
        # get CIE cooling rate
        dudt = ndens**2 * Lambda_CIE
        return -1 * dudt.to(u.erg / u.cm**3 / u.s)
        
    # if temperature is between, do interpolation
    elif (np.log10(T.value) > nonCIE_Tcutoff.value) and (np.log10(T.value) < CIE_Tcutoff.value):
        # print(f'{cpr.WARN}Entering interpolation regime...{cpr.END}')
        # =============================================================================
        # This part is just for non-CIE, and slight-modification from above
        # Get the maximum point of non-CIE. 
        # =============================================================================
        # netcooling grid
        netcooling = cooling_nonCIE.datacube - heating_nonCIE.datacube
        # create interpolation function
        f_dudt = scipy.interpolate.RegularGridInterpolator((cooling_nonCIE.ndens, cooling_nonCIE.temp, cooling_nonCIE.phi), netcooling)
        # get net cooling rate
        dudt_nonCIE = f_dudt([np.log10(ndens.value), nonCIE_Tcutoff.value, np.log10(phi.value)])[0] * u.erg / u.cm**3 / u.s
        
        # =============================================================================
        # This part is just for CIE
        # =============================================================================
    
        # # get CIE cooling rate
        Lambda, _, _= CIE.get_Lambda(10**CIE_Tcutoff.value * u.K)
        dudt_CIE = (ndens**2 * Lambda).to(u.erg / u.cm**3 / u.s)
        
        # =============================================================================
        # Do interpolation now
        # =============================================================================
        
        # print(np.log10(T), [nonCIE_Tcutoff, CIE_Tcutoff],[dudt_nonCIE, dudt_CIE])
        dudt = np.interp(np.log10(T.value), [nonCIE_Tcutoff.value, CIE_Tcutoff.value],[dudt_nonCIE.value, dudt_CIE.value])
    
        return -1 * dudt * u.erg / u.cm**3 / u.s
    
    
    # if temperature is lower than the available non-CIE curve, error (or better, provide some interpolation in the future?)
    else:
        raise Exception('Temperature not understood. Cooling curve and dudt cannot be computed.')
        
    
    
