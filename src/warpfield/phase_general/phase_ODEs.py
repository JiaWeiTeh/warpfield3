#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:28:20 2023

@author: Jia Wei Teh

A general ODE script that is used across phases.
"""


import numpy as np
import os
import scipy.optimize
import astropy.constants as c
import astropy.units as u
#--
import src.warpfield.cloud_properties.mass_profile as mass_profile
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams
import src.warpfield.shell_structure.shell_structure as shell_structure

from src.input_tools import get_param
warpfield_params = get_param.get_param()



def fE_tot_part1(t, y, 
                 ODEpar, SB99f, 
                 Eb0=1, cfs=False, cf_reconstruct=1):
    
    # Note:
        # old code: ODE_tot_aux.fE_tot_part1

    # unpack current values of y (r, rdot, E, T)
    r, v, E, T = y  

    # unpack 'ODEpar' parameters
    GAM  = ODEpar['gamma']
    MCLOUD = ODEpar['Mcloud_au']
    RHOA = ODEpar['rhocore_au']
    RCORE = ODEpar['Rcore_au']
    A_EXP = ODEpar['nalpha']
    MSTAR = ODEpar['Mcluster_au']
    RCLOUD = ODEpar['Rcloud_au']
    tSF = 0.0

    # sanity check: energy should not be negative!
    #if E < 0.0:
    #    sys.exit("Energy is negative in ODEs.py")

    # get current feedback parameters from interpolating SB99
    
    # unit conversions
    L_cgs = u.g.to(u.Msun) * u.cm.to(u.pc)**2/u.s.to(u.Myr)**3
    Myr = u.Myr.to(u.s)
    Msun = c.M_sun.cgs.value
    kms = ( u.km/u.s).to(u.cm/u.s)
    clight_au = c.c.to(u.pc/u.Myr).value
    Grav_au = c.G.to(u.pc**3/u.M_sun/u.Myr**2).value 

    LW = SB99f['fLw_cgs'](t) / L_cgs
    PWDOT = SB99f['fpdot_cgs'](t) * Myr / (Msun * kms)
    LBOL = SB99f['fLbol_cgs'](t) /L_cgs
    VW = 2.*LW/PWDOT

    # check max extent of shell radius ever (important for recollapse, since the shell mass will be kept constant)
    # We only want to set this parameter when the shell has reached its maximum extent and is about to collapse
    # (reason: solver might overshoot and 'check out' a state with a large radius, only to then realize that it should take smaller time steps)
    # This is hard to do since we can define an event (velocity == 0) but I don't know how to set this value only in the case when the event occurs
    # workaround: only set this value when the velocity is close to 0. (and via an event make sure that solver gets close to v==0)
    if (v <= 0.0): ODEpar['Rsh_max'] = max(r,ODEpar['Rsh_max'])

    # calculate shell mass and time derivative of shell mass
    # as radius take largest extent of shell during this expansion
    # during recollapse, the shell mass will remain constant

    # use this radius to calculate shell mass
    this_r = max(ODEpar['Rsh_max'], r) # since ODEpar['Rsh_max'] is only updated when shell turns around, r can be larger than ODEpar['Rsh_max']

    density_specific_param = ODEpar['density_specific_param']
    
    Msh, _ = mass_profile.get_mass_profile(this_r, density_specific_param, 
                                                 RCLOUD, MCLOUD, 
                                                 rdot_arr = v, return_rdot = True)
    if (r < ODEpar['Rsh_max']):
        Msh_dot = 0
    else:
        _, Msh_dot = mass_profile.get_mass_profile(this_r, density_specific_param, 
                                             RCLOUD, MCLOUD, 
                                             rdot_arr = v, return_rdot = True)
    # should we add[0] here?
            
    
    def coverfrac(E,E0,cfe):
        if int(os.environ["Coverfrac?"])==1:
            if (1-cfe)*(E/E0)+cfe < cfe:    # just to be safe, that 'overshooting' is not happening. 
                return cfe
            else:
                return (1-cfe)*(E/E0)+cfe
        else:
            return 1
    
    
    if cfs == True:
        cf = coverfrac(E,Eb0,warpfield_params.frag_cf_end)
        try:
            tcf,cfv=np.loadtxt(ODEpar['mypath'] +"/FragmentationDetails/Coverfrac"+str(len(ODEpar['Mcluster_list']))+".txt", skiprows=1, delimiter='\t', usecols=(0,1), unpack=True)
            tcf=np.append(tcf, t)
            cfv=np.append(cfv, cf)
            np.savetxt(ODEpar['mypath'] +"/FragmentationDetails/Coverfrac"+str(len(ODEpar['Mcluster_list']))+".txt", np.c_[tcf,cfv],delimiter='\t',header='Time'+'\t'+'Coverfraction (=1 for t<Time[0])')
        except:
            pass
    elif cfs=='recon':     ##### coverfraction has to be considered again in the reconstruction of the data. 
        cf = cf_reconstruct
    else:
        cf=1
        

    # gravity correction (self-gravity and gravity between shell and star cluster)
    GRAV = Grav_au * warpfield_params.inc_grav  # if you don't want gravity, set Weav_grav to 0 in myconfig.py
    Fgrav = GRAV*Msh/r**2 * (MSTAR + Msh/2.)
    dt_switchon = 1e-3
    
    # get pressure from energy
    # radius of inner discontinuity
    if E > 1.1*warpfield_params.phase_Emin: # if energy is larger than some very small energy
        R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 0.0, r, args=([LW, E, VW, r]))
        # the following if-clause needs to be rethought. for now, this prevents negative energies at very early times
        # IDEA: move R1 gradually outwards
        tmin = dt_switchon
        if (t > tmin + tSF):
            # equation of state
            Pb = get_bubbleParams.bubble_E2P(E,r,R1,gamma=GAM)
        elif (t <= tmin + tSF):
            R1_tmp = (t-tSF)/tmin * R1
            Pb = get_bubbleParams.bubble_E2P(E, r, R1_tmp, gamma=GAM)
    else: # energy is very small: case of pure momentum driving
        R1 = r # there is no bubble --> inner bubble radius R1 equals shell radius r
        # ram pressure from winds
        Pb = get_bubbleParams.pRam(r,LW,VW)

    # calculate simplified shell structure (warpfield-internal shell structure, not cloudy)
    # warning: Minterior is not set, calculation of Potential will be wrong!
    age1e7_str = ('{:0=5.7f}e+07'.format(t / 10.))
    fname = "shell_" + age1e7_str + ".dat"
    filename_shell = ODEpar['mypath'] + '/shellstruct/' + fname
    [fabs_i, fabs_n, fabs, fion_dust, ionsh, dRs, n0, nmax, rhodr, n0_cloudy, r_Phi_sh, Phi_grav_r0s, f_grav_sh] =\
        shell_structure.shell_structure(r, Pb, 0, SB99f['fLn_cgs'](t), SB99f['fLi_cgs'](t), SB99f['fQi_cgs'](t), Msh, cf)
    FRAD = fabs * LBOL/clight_au

    # set dissolution time
    # the earliest time when the shell dissolved and remained dissolved afterwards marks the dissolution time
    if (nmax < warpfield_params.stop_n_diss and t < ODEpar['t_dissolve']): ODEpar['t_dissolve'] = t
    # as soon as the shell is dense enough (i.e. not dissolved), the dissolution time is set to an arbitrary large number
    if (nmax > warpfield_params.stop_n_diss and t > ODEpar['t_dissolve']): ODEpar['t_dissolve'] = 1e30


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
    if fabs_i < 1.0:
        PHII = calc_ionpress(r, RCORE, RCLOUD, A_EXP, RHOA)
    else:
        PHII = 0.0
        
    #print('cf_in_ODE',cf)

    ########################## ODEs: acceleration ###############################
    vd = (cf*4. * np.pi * r ** 2. * (Pb - PHII) - Msh_dot * v - Fgrav + cf*FRAD) / Msh
    ##########################################################################################

    result = {'vd': vd, 'Msh': Msh, 'fabs_i': fabs_i, 'fabs_n': fabs_n, 'fabs': fabs, 'Pb': Pb, 'R1': R1, 'n0':n0, 'n0_cloudy':n0_cloudy, 'nmax':nmax}

    return result
