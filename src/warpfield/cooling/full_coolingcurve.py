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

import src.warpfield.cooling.CIE as CIE
# get_Lambda
import src.warpfield.cooling.non_CIE as non_CIE



def get_dudt(age, ndens, T, phi):
    
    # New idea, since non-CIE curve is only up to 10^5.5K, which is exactly
    # what our threshold is, we create an if/else function that returns 
    # Lambda(T)_CIE if above, and Lambda(n,T,phi) if below. 
    # If between max of nonCIE and min of CIE, take interpolation between two Lambda values.
    
    # import values from two cooling curves
    cooling_grid_nonCIE = non_CIE.read_cloudy.get_coolingStructure(age)
    Lambda_CIE, logT_CIE, logLambda_CIE = CIE.get_Lambda(T)

    # if temperature is lower than the non-CIE age, use non-CIE
    if np.log10(T) < max(cooling_grid_nonCIE['log_n']) :
        
        # All this does here is to interpolate for values of Lambda based on
        # T, dens and phi.
        
        # log-space
        log_ndens = np.log10(ndens)
        log_T = np.log10(T)
        log_phi = np.log10(phi)
        
        # non-CIE tables
        log_ndens_table = cooling_grid_nonCIE['log_n']
        log_T_table = cooling_grid_nonCIE['log_T']
        log_phi_table = cooling_grid_nonCIE['log_phi']
        
        # find the spacings in array
        d_log_ndens = np.round(np.diff(log_ndens_table), decimals = 3)[0]
        d_log_T = np.round(np.diff(log_T_table), decimals = 3)[0]
        d_log_phi = np.round(np.diff(log_phi_table), decimals = 3)[0]
        
        # find the indices of endpoints
        ndens_ii = int((log_ndens - min(log_ndens_table))/d_log_ndens)
        T_ii = int((log_T - min(log_T_table))/d_log_T)
        phi_ii = int((log_phi - min(log_phi_table))/d_log_phi)
        
        # define datastructure from netcooling
        dens0 = 10 ** log_ndens_table[ndens_ii]
        dens1 = 10 ** log_ndens_table[ndens_ii + 1]
        dens_arr = np.linspace(dens0, dens1, 2)

        T0 = 10 ** log_T_table[T_ii]
        T1 = 10 ** log_T_table[T_ii + 1]
        T_arr = np.linspace(T0, T1, 2)
        
        phi0 = 10 ** log_phi_table[phi_ii]
        phi1 = 10 ** log_phi_table[phi_ii + 1]
        phi_arr = np.linspace(phi0, phi1, 2)
        
        # netcooling grid
        netcooling = cooling_grid_nonCIE['netcooling']
        data = netcooling[ndens_ii:ndens_ii+2, T_ii:T_ii+2, phi_ii:phi_ii+2]
        
        # create interpolation function
        f_dudt = scipy.interpolate.RegularGridInterpolator((dens_arr, T_arr, phi_arr), data)
        
        # get net cooling rate
        dudt = f_dudt([ndens, T, phi])
        # return in negative sign for convension (since the rate of change is negative due to net cooling)
        return -1 * dudt
        
    # if temperature is higher than the CIE curve, use CIE.
    elif np.log10(T) > min(logT_CIE):
        
        # get CIE cooling rate
        dudt = ndens**2 * Lambda_CIE
        return -1 * dudt        
        
    # if temperature is between, do interpolation
    elif np.log10(T) >= max(cooling_grid_nonCIE['log_n']) and np.log10(T) <= min(logT_CIE):
        
        not correct, have to use endpoints instead because T is not same in both cases. 
        
        
        # =============================================================================
        # This part is just for non-CIE, and slight-modification from above
        # Get the maximum point of non-CIE. 
        # =============================================================================
        
        # non-CIE tables
        log_ndens_table = cooling_grid_nonCIE['log_n']
        log_T_table = cooling_grid_nonCIE['log_T']
        log_phi_table = cooling_grid_nonCIE['log_phi']
        
        # log-space
        log_ndens = np.log10(ndens)
        log_T = np.log10(T)
        log_phi = np.log10(phi)
        
        # CIE tables
        log_ndens_table = cooling_grid_nonCIE['log_n']
        log_T_table = cooling_grid_nonCIE['log_T']
        log_phi_table = cooling_grid_nonCIE['log_phi']
        
        # find the spacings in array
        d_log_ndens = np.round(np.diff(log_ndens_table), decimals = 3)[0]
        d_log_T = np.round(np.diff(log_T_table), decimals = 3)[0]
        d_log_phi = np.round(np.diff(log_phi_table), decimals = 3)[0]
        
        # find the indices of endpoints
        ndens_ii = int((log_ndens - min(log_ndens_table))/d_log_ndens)
        T_ii = int((max(log_T_table) - min(log_T_table))/d_log_T)
        phi_ii = int((log_phi - min(log_phi_table))/d_log_phi)
        
        # define datastructure from netcooling
        dens0 = 10 ** log_ndens_table[ndens_ii]
        dens1 = 10 ** log_ndens_table[ndens_ii + 1]
        dens_arr = np.linspace(dens0, dens1, 2)

        T0 = 10 ** log_T_table[T_ii]
        T1 = 10 ** log_T_table[T_ii]
        T_arr = np.linspace(T0, T1, 2)
        
        phi0 = 10 ** log_phi_table[phi_ii]
        phi1 = 10 ** log_phi_table[phi_ii + 1]
        phi_arr = np.linspace(phi0, phi1, 2)
        
        # netcooling grid
        netcooling = cooling_grid_nonCIE['netcooling']
        data = netcooling[ndens_ii:ndens_ii+2, T_ii:T_ii+2, phi_ii:phi_ii+2]
        
        # create interpolation function
        f_dudt = scipy.interpolate.RegularGridInterpolator((dens_arr, T_arr, phi_arr), data)
        
        # get net cooling rate
        dudt_nonCIE = -1 * f_dudt([ndens, T, phi])
        
        # # =============================================================================
        # # This part is just for CIE, and copy-paste from above
        # # =============================================================================
    
        # # get CIE cooling rate
        # dudt_CIE = -1 * ndens**2 * Lambda 
    
    
        # =============================================================================
        # Now find interpolation and return
        # =============================================================================
        
        
        
        
        
        np.interp(x, xp, fp)
    
    
    
    
    
    
    
    
    
    
    
    
    # if temperature is lower than the available non-CIE curve, error (or better, provide some interpolation in the future?)
    else:
        print('Temperature not understood. Cooling curve and dudt cannot be computed.')
        
    
    
    



    
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






