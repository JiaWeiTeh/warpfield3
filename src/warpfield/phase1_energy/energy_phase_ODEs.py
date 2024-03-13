#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:32:47 2022

@author: Jia Wei Teh
"""
# libraries
import numpy as np
import src.warpfield.cloud_properties.mass_profile as mass_profile
import src.warpfield.cloud_properties.density_profile as density_profile
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams
import astropy.units as u
import astropy.constants as c
import scipy.optimize
import sys 

# get parameters
from src.input_tools import get_param
warpfield_params = get_param.get_param()


def get_ODE_Edot(y, t, params):
    """
    This ODE solver solves for y (see below). This is being used in run_energy_phase(), after 
    bubble and shell structure is calculated. This is a general ODE, with more specific ones
    for other stages in other scripts. 

    old code: fE_gen()

    Parameters
    ----------
    y contains:
        - r (R2), shell radius [cm].
        - v (v2), shell velocity [cm/s].
        - E (Eb), bubble energy [erg]
    t : time [s]

    params : see run_energy_phase() for more description of params. Here unit does not matter, because
            astropy will take care of it. Units do matter for y and t, because scipy strips them off. 
            In case the naming does not makes sense:
        
        - Lw: mechanical luminosity
        - pdot_wind: momentum rate 
        - L_bubble: luminosity loss to cooling (see get_bubbleproperties() in bubble_luminosity.py)
        - FRAD: radiation pressure coupled to the shell, i.e. Lbol/c * fabs (calculate fabs from shell structure)
        - tFRAG: time takes to fragmentation
        - tSCR: sound crossing time

    Returns
    -------
    time derivatives of the ODE: 
        - drdt [cm/s]
        - dvdt [cm/s/s]
        - dEdt [erg/s]

    """
    
    # unpack current values of y (r, rdot, E)
    rShell, vShell, E_bubble = y 
    
    # units
    rShell *= u.cm
    vShell *= u.cm/u.s
    E_bubble *= u.erg
    t *= u.s
    
    # unpack parameters
    L_wind, pdot_wind, mCloud, rCore, mCluster, L_bubble,\
        FRAD, FABSi,\
        rCloud,\
        tSF, tFRAG, tSCR, CS  = params  
            
    # velocity from luminosity and change of momentum
    v_wind = (2.*L_wind/pdot_wind).to(u.cm/u.s)
    # calculate shell mass and time derivative of shell mass
    mShell, mShell_dot = mass_profile.get_mass_profile(rShell, rCloud, mCloud, return_mdot = True, rdot_arr = vShell)
    
    def get_press_ion(r, rcloud):
        """
        calculates pressure from photoionized part of cloud at radius r
        :return: pressure of ionized gas outside shell
        """
        # old code: ODE.calc_ionpress()
        
        # n_r: total number density of particles (H+, He++, electrons)
        n_r = density_profile.get_density_profile(r, rcloud)
        P_ion = n_r * c.k_B * warpfield_params.t_ion
        return P_ion

    # calc inward pressure from photoionized gas outside the shell 
    # (is zero if no ionizing radiation escapes the shell)
    if FABSi < 1.0:
        press_HII = get_press_ion(rShell, rCloud)
    else:
        press_HII = 0.0
    
    # gravity correction (self-gravity and gravity between shell and star cluster)
    # if you don't want gravity, set .inc_grav to zero
    F_grav = (c.G * mShell / rShell**2 * (mCluster + mShell/2)  * warpfield_params.inc_grav).to(u.g*u.cm/u.s**2)
    
    # get pressure from energy
    # calculate radius of inner discontinuity (inner radius of bubble)
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 0.0, rShell.to(u.cm).value, args=([L_wind.to(u.erg/u.s).value, E_bubble.to(u.erg).value, v_wind.to(u.cm/u.s).value, rShell.to(u.cm).value])) * u.cm
    
    # the following if-clause needs to be rethought. for now, this prevents negative energies at very early times
    # IDEA: move R1 gradually outwards
    dt_switchon = 1e-3 * u.Myr # gradually switch on things during this time period
    tmin = dt_switchon
    if (t.to(u.yr).value > (tmin + tSF).to(u.yr).value):
        # equation of state
        press_bubble = get_bubbleParams.bubble_E2P(E_bubble, rShell, R1)
    elif (t.to(u.yr).value <= (tmin + tSF).to(u.yr).value):
        R1_tmp = (t-tSF)/tmin * R1
        press_bubble = get_bubbleParams.bubble_E2P(E_bubble, rShell, R1_tmp)[0]
    #else: #case pure momentum driving
    #    # ram pressure from winds
    #    press_bubble = state_eq.Pram(r,LW,VW)
    
    def calc_coveringf(t,tFRAG,ts):
        """
        estimate covering fraction cf (assume that after fragmentation, during 1 sound crossing time cf goes from 1 to 0)
        if the shell covers the whole sphere: cf = 1
        if there is no shell: cf = 0
        
        Note: I think, since we set tFRAG ultra high, that means cf will almost always be 1. 
        """
        cfmin = 0.4
        # simple slope
        cf = 1. - ((t - tFRAG) / ts)**1.
        cf[cf>1.0] = 1.0
        cf[cf<cfmin] = cfmin
        # return
        return cf

    # calculate covering fraction
    cf = calc_coveringf(np.array([t.value])*u.s,tFRAG,tSCR)
    
    # transform to float if necessary
    if hasattr(cf, "__len__"): 
        cf = cf[0] 
    # leaked luminosity
    if cf < 1:
        L_leak = (1. - cf)  * 4. * np.pi * rShell ** 2 * press_bubble * CS / (warpfield_params.gamma_adia - 1.)
    else:
        L_leak = 0 * u.erg / u.s
        
    # time derivatives￼￼
    rd = vShell
    vd = (4.*np.pi*rShell**2.*(press_bubble-press_HII) - mShell_dot * vShell - F_grav + FRAD)/mShell
    # factor cf for debugging
    Ed = (L_wind - L_bubble) - (4.*np.pi*rShell**2.*press_bubble) * vShell - L_leak 

    # list of dy/dt=f functions
    derivs = [rd.to(u.cm/u.s).value, vd.to(u.cm/u.s**2).value, Ed.to(u.erg/u.s).value]
    # return
    return derivs





