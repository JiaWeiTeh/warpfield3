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


def get_ODE_Edot(y, t, params):
    """
    general energy-driven phase including stellar winds, gravity, power law density profiles, cooling, radiation pressure
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
    r, v, E = y  # unpack current values of y (r, rdot, E)
    LW, PWDOT, GAM, MCLOUD, RHOA, RCORE, A_EXP, MSTAR, LB, FRAD, FABSi,\
        RCLOUD, density_specific_param, warpfield_params,\
            tSF, tFRAG, tSCR, CS, SFE  = params  # unpack parameters
    VW = 2.*LW/PWDOT
    
    # calculate shell mass and time derivative of shell mass
    

    # print('\n\nHere are the values that are fed into mass_profile')
    # print(r,  density_specific_param,\
    #                                 RCLOUD, MCLOUD,\
    #                                     v)

    Msh, Msh_dot = mass_profile.get_mass_profile(r, density_specific_param, RCLOUD, MCLOUD, warpfield_params, rdot_arr = v, return_rdot = True)
    # sys.exit()
    # print("We are now in energy_phase_ODEs.get_ODE_Edot() to check for the values of Msh")
    # print('Msh',Msh)
    # print('Msh_dot',Msh_dot)
    
    # print("r, density_specific_param, RCLOUD, MCLOUD,", r, density_specific_param, RCLOUD, MCLOUD)
    # print("Msh, Msh_dot", Msh, Msh_dot)
    
    def calc_ionpress(r, rcore, rcloud, alpha, rhoa):
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
        
        if r < rcore:
            rho_r = rhoa
        elif ((r >= rcore) and (r < rcloud)):
            rho_r = rhoa * (r/rcore)**alpha
        else:
            rho_r = warpfield_params.nISM * warpfield_params.mu_n * (u.g/u.cm**3).to(u.M_sun/u.pc**3)
        # n_r: total number density of particles (H+, He++, electrons)
        n_r = rho_r/(warpfield_params.mu_p/c.M_sun.cgs.value) 
        # boltzmann constant in astronomical units
        kboltz_au = c.k_B.cgs.value * u.g.to(u.Msun) * u.cm.to(u.pc)**2 / u.s.to(u.Myr)**2
        P_ion = n_r * kboltz_au * warpfield_params.t_ion
        # return
        return P_ion

    # calc inward pressure from photoionized gas outside the shell (is zero if no ionizing radiation escapes the shell)
    if FABSi < 1.0:
        PHII = calc_ionpress(r, RCORE, RCLOUD, A_EXP, RHOA)
    else:
        PHII = 0.0
        
    
    # gravity correction (self-gravity and gravity between shell and star cluster)
    GRAV = c.G.to(u.pc**3/u.M_sun/u.Myr**2).value * warpfield_params.inc_grav  # if you don't want gravity, set to zero
    Fgrav = GRAV*Msh/r**2 * (MSTAR + Msh/2.)
    
    
    # get pressure from energy
    # radius of inner discontinuity
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 0.0, r, args=([LW, E, VW, r]))
    

    # the following if-clause needs to be rethought. for now, this prevents negative energies at very early times
    # IDEA: move R1 gradually outwards
    dt_switchon = 1e-3 # gradually switch on things during this time period
    tmin = dt_switchon
    if (t > tmin + tSF):
        # equation of state
        Pb = get_bubbleParams.bubble_E2P(E,r,R1,GAM)
    elif (t <= tmin + tSF):
        R1_tmp = (t-tSF)/tmin * R1
        Pb = get_bubbleParams.bubble_E2P(E, r, R1_tmp,GAM)
    #else: #case pure momentum driving
    #    # ram pressure from winds
    #    Pb = state_eq.Pram(r,LW,VW)

    def calc_coveringf(t,tFRAG,ts):
        """
        estimate covering fraction cf (assume that after fragmentation, during 1 sound crossing time cf goes from 1 to 0)
        if the shell covers the whole sphere: cf = 1
        if there is no shell: cf = 0
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
        L_leak = (1. - cf)  * 4. * np.pi * r ** 2 * Pb * CS / (GAM - 1.)
    else:
        L_leak = 0
    # time derivatives￼￼
    rd = v
    vd = (4.*np.pi*r**2.*(Pb-PHII) - Msh_dot*v - Fgrav + FRAD)/Msh
    # factor cf for debugging
    Ed = (LW - LB) - (4.*np.pi*r**2.*Pb) * v - L_leak 
    # list of dy/dt=f functions
    derivs = [rd, vd, Ed]    
    # return
    return derivs
