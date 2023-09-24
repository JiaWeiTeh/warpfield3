#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:32:47 2022

@author: Jia Wei Teh
"""
# libraries
import numpy as np
import src.warpfield.cloud_properties.mass_profile as mass_profile
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
    general energy-driven phase including stellar winds, gravity, power law density profiles, cooling, radiation pressure/
    This set of ODEs solve for
    :param y: [r,v,E]: shell radius (R2), shell velocity (v2), bubble energy (Eb)
    :param t: time (since the ODE is autonomous, t does not appear. The ODE solver still expects it though)
    :param params: (see below)
    :return: time derivative of y, i.e. [rd, vd, Ed]
    # parameters:
    # LW : mechanical luminosity
    # GAM : adiabatic index
    # M0T : core mass
    # RHOA : core density
    # RCORE : core radius
    # A_EXP : exponent of density profile
    # LB: luminosity lost to cooling (calculate from bubble structure)
    # FRAD: radiation pressure coupled to the shell, i.e. Lbol/c * fabs (calculate fabs from shell structure)
    """
    
    # Note: 
    # old code: fE_gen()
    rShell, vShell, E_bubble = y  # unpack current values of y (r, rdot, E)
    L_wind, pdot_wind, _, mCloud, _, rCore, _, mCluster, L_bubble, FRAD, FABSi,\
        rCloud, _,\
            tSF, tFRAG, tSCR, CS, _  = params  # unpack parameters
            
    v_wind = 2.*L_wind/pdot_wind
    
    # calculate shell mass and time derivative of shell mass
    

    # print('\n\nHere are the values that are fed into mass_profile')
    # print(r,  density_specific_param,\
    #                                 RCLOUD, MCLOUD,\
    #                                     v)

    mShell, mShell_dot = mass_profile.get_mass_profile(rShell, rCloud, mCloud, return_mdot = True, rdot_arr = vShell)

    
    def calc_ionpress(r, rcore, rcloud):
        """
        calculates pressure from photoionized part of cloud at radius r
        by default assume units (Msun, Myr, pc) but in order to use cgs, just change mykboltz
        :param r: radius
        :param rcore: core radius (only important if density slope is used)
        :param rcloud: cloud radius (outside of rcloud, density slope)
        :param alpha: exponent of density slope: rho = rhoa*(r/rcore)**alpha, alpha is usually zero or negative
        :param rhoa: core density
        :param mykboltz: by default assume astro units (Myr, Msun, pc)
        :return: pressure of ionized gas outside shell
        """
        # old code: ODE.calc_ionpress()
        
        rhoa = warpfield_params.nCore * warpfield_params.mu_n

        if r < rcore:
            rho_r = rhoa
        elif ((r >= rcore) and (r < rcloud)):
            rho_r = rhoa * (r/rcore)**warpfield_params.dens_a_pL
        else:
            rho_r = warpfield_params.nISM * warpfield_params.mu_n 
        # n_r: total number density of particles (H+, He++, electrons)
        n_r = rho_r/warpfield_params.mu_p
        # boltzmann constant in astronomical units
        P_ion = n_r * c.k_B * warpfield_params.t_ion
        # return
        return P_ion

    # calc inward pressure from photoionized gas outside the shell (is zero if no ionizing radiation escapes the shell)
    if FABSi < 1.0:
        press_HII = calc_ionpress(rShell, rCore, rCloud)
    else:
        press_HII = 0.0
        
    
    # gravity correction (self-gravity and gravity between shell and star cluster)
    # if you don't want gravity, set .inc_grav to zero
    F_grav = c.G * mShell / rShell**2 * (mCluster + mShell/2)  * warpfield_params.inc_grav
    
    
    # get pressure from energy
    # radius of inner discontinuity
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 0.0, rShell, args=([L_wind, E_bubble, v_wind, rShell]))
    

    # the following if-clause needs to be rethought. for now, this prevents negative energies at very early times
    # IDEA: move R1 gradually outwards
    dt_switchon = 1e-3 # gradually switch on things during this time period
    tmin = dt_switchon
    if (t > tmin + tSF):
        # equation of state
        press_bubble = get_bubbleParams.bubble_E2P(E_bubble, rShell, R1)
    elif (t <= tmin + tSF):
        R1_tmp = (t-tSF)/tmin * R1
        press_bubble = get_bubbleParams.bubble_E2P(E_bubble, rShell, R1_tmp)
    #else: #case pure momentum driving
    #    # ram pressure from winds
    #    press_bubble = state_eq.Pram(r,LW,VW)

    def calc_coveringf(t,tFRAG,ts):
        """
        estimate covering fraction cf (assume that after fragmentation, during 1 sound crossing time cf goes from 1 to 0)
        if the shell covers the whole sphere: cf = 1
        if there is no shell: cf = 0
        
        Note: since we set tFRAG ultra high, that means cf will almost always be 1. 
        """
        cfmin = 0.4
        cf = 1. - ((t - tFRAG) / ts)**1.
        cf[cf>1.0] = 1.0
        cf[cf<cfmin] = cfmin
        # return
        return cf

    # calculate covering fraction
    cf = calc_coveringf(np.array([t]),tFRAG,tSCR)
    # transform to float if necessary
    if hasattr(cf, "__len__"): cf = cf[0] 
    # leaked luminosity
    if cf < 1:
        L_leak = (1. - cf)  * 4. * np.pi * rShell ** 2 * press_bubble * CS / (warpfield_params.gamma_adia - 1.)
    else:
        L_leak = 0
    # time derivatives￼￼
    rd = vShell
    vd = (4.*np.pi*rShell**2.*(press_bubble-press_HII) - mShell_dot * vShell - F_grav + FRAD)/mShell
    # factor cf for debugging
    Ed = (L_wind - L_bubble) - (4.*np.pi*rShell**2.*press_bubble) * vShell - L_leak 
    # list of dy/dt=f functions
    derivs = [rd, vd, Ed]    
    # return
    return derivs
