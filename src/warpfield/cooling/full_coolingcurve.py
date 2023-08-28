#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 15:08:23 2023

@author: Jia Wei Teh

This is a master script which craetes a cooling curve containing both CIE and non-CIE
conditions.

old code: coolnoeq.cool_interp_master()
"""

import numpy as np
import src.warpfield.cooling.CIE as CIE
# get_Lambda
import src.warpfield.cooling.non_CIE as non_CIE



def master_coolingcurve(age, ndens, T, phi):
    
    # New idea, since non-CIE curve is only up to 10^5.5K, which is exactly
    # what our threshold is, we create an if/else function that returns 
    # Lambda(T)_CIE if above, and Lambda(n,T,phi) if below. 
    # If between max of nonCIE and min of CIE, take interpolation between two Lambda values.
    
    # import values from two cooling curves
    cooling_grid_nonCIE = non_CIE.read_cloudy.get_coolingStructure(age)
    Lambda, logT, logLambda = CIE.get_Lambda(T)

    # if temperature is lower than the non-CIE age, use non-CIE
    if np.log10(T) < max(cooling_grid_nonCIE['log_n']) :
        
        
        
        
        pass
        
        
        
    # if temperature is higher than the CIE curve, use CIE.
    elif np.log10(T) > min():
        
        
        
    # if temperature is between, do interpolation
    elif np.log10(T) > max(cooling_grid_nonCIE['log_n']) and np.log10(T) < min():
        
        Lambda, logT, logLambda = CIE.get_Lambda(T)
        
        
    # if temperature is lower than the available non-CIE curve, error (or better, provide some interpolation in the future?)
    else:
        print('Temperature not understood. Cooling curve and dudt cannot be computed.')
        
    
    
    
    return



    
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






