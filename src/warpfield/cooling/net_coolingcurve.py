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

import src.warpfield.cooling.CIE.read_coolingcurve as CIE
# get_Lambda
import src.warpfield.cooling.non_CIE.read_cloudy as non_CIE
from src.warpfield.functions.terminal_prints import cprint as cpr

def get_dudt(age, ndens, T, phi):
    
    # These value should not be logged!
    
    # New idea, since non-CIE curve is only up to 10^5.5K, which is exactly
    # what our threshold is, we create an if/else function that returns 
    # Lambda(T)_CIE if above, and Lambda(n,T,phi) if below. 
    # If between max of nonCIE and min of CIE, take interpolation between two Lambda values.
    
    # import values from two cooling curves
    cooling_nonCIE, heating_nonCIE = non_CIE.get_coolingStructure(age)
    Lambda_CIE, logT_CIE, logLambda_CIE = CIE.get_Lambda(T)
    
    # we take the cutoff at 10e5.5 K. 
    # cutoff at which temperature above switches to CIE file:
    nonCIE_Tcutoff = max(cooling_nonCIE.temp[cooling_nonCIE.temp <= 5.5])
    # cutoff at which temperature below switches to non-CIE file:
    CIE_Tcutoff = min(logT_CIE[logT_CIE > 5.5])
    # output
    # print(f'{cpr.WARN}Taking net-cooling curve from non-CIE condition at T <= {nonCIE_Tcutoff}K and CIE condition at T >= {CIE_Tcutoff}K.{cpr.END}')
    # if nonCIE_Tcutoff != CIE_Tcutoff:
        # print(f'{cpr.WARN}Net cooling for temperature values in-between will be interpolated{cpr.END}.')

    # if temperature is lower than the non-CIE temperature, use non-CIE
    if np.log10(T) <= nonCIE_Tcutoff and np.log10(T) >= min(cooling_nonCIE.temp):
        # print(f'{cpr.WARN}Entering non-CIE regime...{cpr.END}')
        # All this does here is to interpolate for values of Lambda based on
        # T, dens and phi.
        
        # netcooling grid
        netcooling = cooling_nonCIE.datacube - heating_nonCIE.datacube
        # create interpolation function
        f_dudt = scipy.interpolate.RegularGridInterpolator((cooling_nonCIE.ndens, cooling_nonCIE.temp, cooling_nonCIE.phi), netcooling)
        # get net cooling rate
        # remember that these have to be logged!
        dudt = f_dudt([np.log10(ndens), np.log10(T), np.log10(phi)])[0]
        # return in negative sign for convension (since the rate of change is negative due to net cooling)
        return -1 * dudt
        
    # if temperature is higher than the CIE curve, use CIE.
    elif np.log10(T) >= CIE_Tcutoff:
        # print(f'{cpr.WARN}Entering CIE regime...{cpr.END}')
        # get CIE cooling rate
        dudt = ndens**2 * Lambda_CIE
        return -1 * dudt        
        
    # if temperature is between, do interpolation
    elif (np.log10(T) > nonCIE_Tcutoff) and (np.log10(T) < CIE_Tcutoff):
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
        dudt_nonCIE = f_dudt([np.log10(ndens), nonCIE_Tcutoff, np.log10(phi)])[0]
        
        # =============================================================================
        # This part is just for CIE
        # =============================================================================
    
        # # get CIE cooling rate
        Lambda, _, _= CIE.get_Lambda(10**CIE_Tcutoff)
        dudt_CIE = ndens**2 * Lambda 
        
        # =============================================================================
        # Do interpolation now
        # =============================================================================
        
        # print(np.log10(T), [nonCIE_Tcutoff, CIE_Tcutoff],[dudt_nonCIE, dudt_CIE])
        dudt = np.interp(np.log10(T), [nonCIE_Tcutoff, CIE_Tcutoff],[dudt_nonCIE, dudt_CIE])
    
        return -1 * dudt
    
    
    # if temperature is lower than the available non-CIE curve, error (or better, provide some interpolation in the future?)
    else:
        raise Exception('Temperature not understood. Cooling curve and dudt cannot be computed.')
        
    
    
    



    
    # # THese if/else cases seem to be for T range for when to/not to use CIE cooling curves?
    
    # if (np.log10(point["T"]) > log_T_intermax) or (np.log10(point["T"]) < log_T_intermin):
    #     Lambda = get_coolingFunction(point["T"], metallicity)
    #     dudt = -1. * (point["n"]) ** 2 *  Lambda / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))

    # elif (np.log10(point["T"]) >= log_T_noeqmax):
    #     dudt1 = -1. * (point["n"]) ** 2 * get_coolingFunction(point["T"], metallicity) / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))
    #     dudt0 = -1. * Interp3_dudt({"n": point["n"], "T": point["T"], "Phi": point["Phi"]}, Cool_Struc) / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))
    #     dudt = linear(np.log10(point["T"]), [log_T_noeqmax, log_T_intermax], [dudt0, dudt1])

    # elif (np.log10(point["T"]) <= log_T_noeqmin):
    #     dudt0 = -1. * (point["n"]) ** 2 * get_coolingFunction(point["T"], metallicity) / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))
    #     dudt1 = -1. * Interp3_dudt({"n": point["n"], "T": point["T"], "Phi": point["Phi"]}, Cool_Struc) / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))
    #     dudt = linear(np.log10(point["T"]), [log_T_intermin, log_T_noeqmin], [dudt0, dudt1])

    # else:
    #     dudt = -1. * Interp3_dudt({"n": point["n"], "T": point["T"], "Phi": point["Phi"]}, Cool_Struc) / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))

    # return dudt






