#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:28:20 2023

@author: Jia Wei Teh

"""


import numpy as np
import os
import scipy.optimize
import astropy.constants as c
import astropy.units as u
#--
import src.warpfield.cloud_properties.mass_profile as mass_profile
import src.warpfield.cloud_properties.density_profile as density_profile
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams
import src.warpfield.shell_structure.shell_structure as shell_structure

from src.input_tools import get_param
warpfield_params = get_param.get_param()



'''
Actually, the meaning of this function is that this calculates
part 1 of the fE_tot that is in phase_energy1. Meaning, 
that this only returns vd (importantly), since the whole fE_tot should
return vd, rd, Ed, and Td.

Interestingly, this function is also being used in warp_reconstruct() in warp_writedata.py, 
to calculate values and to 


'''



def fE_tot_part1(t, y, 
                 ODEpar, SB99f, 
                 Eb0=1, cfs=False, 
                 cf_reconstruct=1):
    
    
    # Note:
        # old code: ODE_tot_aux.fE_tot_part1



    # unpack current values of y (r, rdot, E, T)
    rShell, vShell, Ebubble, TBubble = y  

    # unpack 'ODEpar' parameters
    mCloud = ODEpar['mCloud']
    mCluster = warpfield_params.mCluster
    rCloud = ODEpar['rCloud']
    tSF = 0.0 * u.Myr


    # Interpolate SB99 to get feedback parameters
    # mechanical luminosity at time t (erg)
    L_wind = SB99f['fLw_cgs'](t) * u.erg / u.s
    # momentum of stellar winds at time t (cgs)
    pdot_wind = SB99f['fpdot_cgs'](t) * u.g * u.cm / u.s**2
    # other luminosities
    Lbol = SB99f['fLbol_cgs'](t) * u.erg / u.s
    Ln = SB99f['fLn_cgs'](t) * u.erg / u.s
    Li = SB99f['fLi_cgs'](t) * u.erg / u.s
    Qi = SB99f['fQi_cgs'](t) * u.erg / u.s
    
    # velocity from luminosity and change of momentum
    v_wind = (2.*L_wind/pdot_wind).to(u.cm/u.s)    
    
    # TODO!! check in the old version, which script updates [Rsh_max] and add it in.
    # It might have been removed cause I originally thought it was a useless additon.


    # check max extent of shell radius ever (important for recollapse, since the shell mass will be kept constant)
    # We only want to set this parameter when the shell has reached its maximum extent and is about to collapse
    # (reason: solver might overshoot and 'check out' a state with a large radius, only to then realize that it should take smaller time steps)
    # This is hard to do since we can define an event (velocity == 0) but I don't know how to set this value only in the case when the event occurs
    # workaround: only set this value when the velocity is close to 0. (and via an event make sure that solver gets close to v==0)
    if vShell.value <= 0.0:
        ODEpar['Rsh_max'] = max(rShell, ODEpar['Rsh_max'])

    # =============================================================================
    # Shell mass, where radius = maximum extent of shell.
    # =============================================================================

    # If there is a collapse event, ODEpar['Rsh_max'] could be smaller than r. 
    # this is because ODEpar['Rsh_max'] is only updated after the completion of event.
    # straightforward solution: take max(ODEpar['Rsh_max'], r) to ensure we are using the right shell radius.
    mShell, mShell_dot = mass_profile.get_mass_profile(max(ODEpar['Rsh_max'], rShell),
                                                       rCloud, mCloud, return_mdot = True, rdot_arr = vShell)
    
    # However this is not so straightforward. During recollapse we have to make sure that the 
    # shell mass stays constant, i.e., Msh_dot = 0.
    # Define recollapse event such that the radius is smaller than the maximum radius from previous step (ODEpar['Rsh_max']).
    if (rShell < ODEpar['Rsh_max']):
        mShell_dot = 0 * u.M_sun
        
    # ----
    # To future: this section should be used. However, im a lil too scared to change this cause it might break everything
    # and then shi hits the fan and all hell break loose.
    
    
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
    # cf = calc_coveringf(np.array([t.value])*u.s,tFRAG,tSCR)


    #----
    
    # If frag_cf is enabled, what is the final cover fraction at the end
    # of the fragmentation process?

    def coverfrac(E,E0,cfe):
        if int(os.environ["Coverfrac?"])==1:
            if (1-cfe)*(E/E0)+cfe < cfe:    # just to be safe, that 'overshooting' is not happening. 
                return cfe
            else:
                return (1-cfe)*(E/E0)+cfe
        else:
            return 1
    
    
    if cfs == True:
        cf = coverfrac(Ebubble,Eb0,warpfield_params.frag_cf_end)
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
        
    #----
        
        
    # gravity correction (self-gravity and gravity between shell and star cluster)
    # if you don't want gravity, set .inc_grav to zero
    F_grav = (c.G * mShell / rShell**2 * (mCluster + mShell/2)  * warpfield_params.inc_grav).to(u.g*u.cm/u.s**2)
  
    # get pressure from energy. 
    if Ebubble > 1.1 * warpfield_params.phase_Emin:
        # calculate radius of inner discontinuity (inner radius of bubble)
        R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 0.0, rShell.to(u.cm).value,\
                                   args=([L_wind.to(u.erg/u.s).value, Ebubble.to(u.erg).value, v_wind.to(u.cm/u.s).value, rShell.to(u.cm).value])) * u.cm
        # the following if-clause needs to be rethought. for now, this prevents negative energies at very early times
        # IDEA: move R1 gradually outwards
        dt_switchon = 1e-3 * u.Myr # gradually switch on things during this time period
        tmin = dt_switchon
        if (t.to(u.yr).value > (tmin + tSF).to(u.yr).value):
            # equation of state
            press_bubble = get_bubbleParams.bubble_E2P(Ebubble, rShell, R1)
        elif (t.to(u.yr).value <= (tmin + tSF).to(u.yr).value):
            R1_tmp = (t-tSF)/tmin * R1
            press_bubble = get_bubbleParams.bubble_E2P(Ebubble, rShell, R1_tmp)[0]
    else: # energy is very small: case of pure momentum driving
        R1 = rShell # there is no bubble --> inner bubble radius R1 equals shell radius r
        # ram pressure from winds
        press_bubble = get_bubbleParams.pRam(rShell, L_wind, v_wind)
            
        
    # =============================================================================
    # Shell structure
    # =============================================================================
    
    # calculate simplified shell structure (warpfield-internal shell structure, not cloudy)
    # We are setting mBubble = 0 here, since we are not interested in the potential. This can skip some calculations.
    shell_prop = shell_structure.shell_structure(rShell, press_bubble, 
                                        0, 
                                        Ln, Li, Qi,
                                        mShell,
                                        f_cover = 1,
                                        )
    # clarity
    f_absorbed_ion, f_absorbed_neu, f_absorbed, _, _, shellThickness, nShellInner, nShell_max, _, _, _, _ = shell_prop

    # radiation pressure coupled to the shell
    fRad = f_absorbed * Lbol / c.c

    # set dissolution time
    # the earliest time when the shell dissolved and remained dissolved afterwards marks the dissolution time
    if (nShell_max < warpfield_params.stop_n_diss and t < ODEpar['t_dissolve']):
        ODEpar['t_dissolve'] = t
    # as soon as the shell is dense enough (i.e. not dissolved), the dissolution time is set to an arbitrary large number
    if (nShell_max > warpfield_params.stop_n_diss and t > ODEpar['t_dissolve']):
        ODEpar['t_dissolve'] = 1e30

    
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
    

    # calculate inward pressure from photoionized gas outside the shell 
    # (is zero if no ionizing radiation escapes the shell)
    if f_absorbed_ion < 1.0:
        press_HII = get_press_ion(rShell, rCloud)
    else:
        press_HII = 0.0
        
    # =============================================================================
    # calculate the ODE part: Acceleration 
    # =============================================================================
    vd = (cf * 4 * np.pi * rShell**2 * (press_bubble - press_HII)\
            - mShell_dot * vShell\
                - F_grav + cf * fRad) / mShell
    

    # TODO: update these dictionary names in the future.
    # return vd (main result of this function), and additional parameters for recording the evolution.
    evolution_data = {'Msh': mShell, 'fabs_i': f_absorbed_ion, 'fabs_n': f_absorbed_neu,
              'fabs': f_absorbed, 'Pb': press_bubble, 'R1': R1, 'n0':nShellInner, 'nmax':nShell_max}

    return vd.to(u.km/u.s), evolution_data
