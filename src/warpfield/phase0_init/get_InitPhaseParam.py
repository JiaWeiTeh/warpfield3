#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:44:00 2022

@author: Jia Wei Teh

This script contains a function that computes the initial values for the
energy-driven phase (from a short free-streaming phase).
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
#--
from src.input_tools import get_param
warpfield_params = get_param.get_param()

def get_y0(tSF, SB99f):
    """
    Obtain initial values for the energy driven phase.

    Parameters
    ----------
    tSF : float
        time of last star formation event (or - if no SF ocurred - time of last recollapse).
    SB99f : func
        starburst99 interpolation functions.

    Returns
    -------
    t0 : TYPE
        DESCRIPTION.
    y0 : array
        An array of initial values.
        r0: initial separation
        v0: velocity
        E0: energy
        T0: temperature
        
    """
    # Note:
        # old code: get_startvalues.get_y0()
        
    Lw_evo0 = SB99f['fLw_cgs'](tSF)
    pdot_evo0 = SB99f['fpdot_cgs'](tSF)
    
    # print(Lw_evo0, pdot_evo0)
    # 1.2217996601648809e+41 6.683439175686136e+32

    # mass loss rate from winds and SNe (cgs)
    Mdot0_cgs = pdot_evo0**2/(2.*Lw_evo0) 
    # terminal velocity from winds and SNe (cgs)
    vterminal0_cgs = 2.*Lw_evo0/pdot_evo0 

    rhoa =  warpfield_params.nCore * warpfield_params.mu_n
    dt_phase0 = np.sqrt(3. * Mdot0_cgs / (4. * np.pi * rhoa * vterminal0_cgs ** 3.)) / u.Myr.to(u.s)  # duration of inital free-streaming phase (Myr)
    # start time for Weaver phase (Myr)
    t0 = tSF + dt_phase0  
    # initial separation (pc)
    r0 = (vterminal0_cgs / (( u.km/u.s).to(u.cm/u.s))) * dt_phase0  
    # initial velocity (km/s)
    v0 = vterminal0_cgs / (( u.km/u.s).to(u.cm/u.s)) 
    E0 = 5. / 11. * Lw_evo0*(u.Myr.to(u.s)/((c.M_sun.cgs.value)*(( u.km/u.s).to(u.cm/u.s))**2)) * dt_phase0 
    # see Weaver+77, eq. (37)
    T0 = 1.51e6 * (Lw_evo0/1e36)**(8./35.) * warpfield_params.nCore**(2./35.) * (t0-tSF)**(-6./35.) * (1.-warpfield_params.xi_Tb)**0.4 

    # print(Mdot0_cgs, vterminal0_cgs, rhoa, dt_phase0)
    # 1.8279739580656053e+24 365620043.2285518 2.128791539241818e-21 6.489708190469503e-05

    y0 = [r0, v0, E0, T0]

    return t0, y0    











