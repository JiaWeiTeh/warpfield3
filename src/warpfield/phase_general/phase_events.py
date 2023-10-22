#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:13:10 2023

@author: Jia Wei Teh

This script contains functions which help determing the current stage
of the evolution. Used in different phase functions (phase1, 2, etc...)

Old code: my_events.py
"""


import numpy as np
from numpy import diff
import time
import astropy.constants as c
import astropy.units as u
import os
#--
import src.warpfield.phase_general.phase_ODEs as phase_ODEs

from src.input_tools import get_param
warpfield_params = get_param.get_param()

#%%


def check_outdir(dirstring):
    """Check if output directory exists. If not, create it."""
    if not os.path.isdir(dirstring):
        os.makedirs(dirstring)

    return 0

def event_density_gradient(t,y,ODE_params):
    
    r, v, E, T = y
    coll_counter=len(ODE_params['Mcluster_list'])-1
    r0,n0=np.loadtxt(ODE_params['mypath'] +'/dlaw'+ str(coll_counter) + '.txt', skiprows=1, unpack=True)
    r_d=10**r0*u.cm.to(u.pc) #r in pc
    n_d=10**n0 # in 1/ccm
    
    def find_rdens(r,n):
        ''' Finds radius where density profile is steeper than r**(-2) 
        r = x-coordinate of dens profile
        n = y-coordinate of dens profile
        '''
        # Note:
            # old code: aux.find_rdens()
            
        n=np.log10(n) #take logarithm 
        r=np.log10(r)
        
        dr = np.mean(diff(r)[5:10])
    
        fdot=diff(n)/dr #derivative
        
        index =  np.where(np.sqrt(fdot**2) > 2) #find index
        
        return 10**r[index[0][0]] #return first r where density profile is steeper

    r_dens_grad= find_rdens(r_d,n_d)
    residual = r - r_dens_grad
    
    if residual >= 0:
        print('density profile steeper than r**(-2): switching to momentum driven phase')
        
        tcf=[0]
        cfv=[1]
        
        check=os.path.join(ODE_params['mypath'], "FragmentationDetails")
        check_outdir(check)
        
        os.environ["Coverfrac?"] = str(1)
        
        np.savetxt(ODE_params['mypath'] +"/FragmentationDetails/Coverfrac"+str(len(ODE_params['Mcluster_list']))+".txt", np.c_[tcf,cfv],delimiter='\t',header='Time'+'\t'+'Coverfraction (=1 for t<Time[0])')
        

    return residual


def event_cool_switch(t,y,ODE_params):
    residual =1
    Lcool=np.log10(float(os.environ["Lcool_event"]))
    Lmech=np.log10(float(os.environ["Lgain_event"]))
    
    '''
    try:
        Lc,Lm=np.loadtxt(ODE_params['mypath'] +"/FragmentationDetails/LcoolLmech.txt", skiprows=1, delimiter='\t', unpack=True)
    except:
        Lc=Lm=np.array([])
        check=os.path.join(ODE_params['mypath'], "FragmentationDetails")
        check_outdir(check)
    
    Lc=np.append(Lc, Lcool)
    Lm=np.append(Lm, Lmech)
    
    np.savetxt(ODE_params['mypath'] +"/FragmentationDetails/LcoolLmech.txt", np.c_[Lc,Lm],delimiter='\t',header='Time'+'\t'+'Coverfraction (=1 for t<Time[0])')
    '''
    
    if Lmech-Lcool < 0.05:
        residual =0
        print('Lcool~Lmech: switching to momentum driven phase')
    
    return residual 
    
    

def event_grav_frag(t, y, ODE_params, SB99f):
    """
    event of shell fragmentation due to gravitational collapse (compare McCray & Kafatos 1987, eq. 14)
    :param t: time (float)
    :param: y: (list)
    :param ODE_params: additional parameters
    :return: float, when 0: event!
    """

    Grav_au = c.G.to(u.pc**3/u.M_sun/u.Myr**2).value 
    
    r, v, E, T = y
    
    try:
        tgrav,rgrav,resgrav,csgrav=np.loadtxt(ODE_params['mypath'] +"/FragmentationDetails/GRAV.txt", skiprows=1, delimiter='\t', usecols=(0,1,2,3), unpack=True)
    except:
        tgrav=rgrav=resgrav=csgrav=np.array([])
        check=os.path.join(ODE_params['mypath'], "FragmentationDetails")
        check_outdir(check)

    part1_dict = phase_ODEs.fE_tot_part1(t, y, ODE_params, SB99f)
    fabs_i = part1_dict['fabs_i']
    Msh = part1_dict['Msh']

    # if some ionizing radiation escapes some part of the shell is neutral
    # take the lowest temperature of all temperatures in the shell
    if fabs_i < 0.999:  # why not fabs_i < 1.0? machine precision safety
        Tsh = warpfield_params.t_neu  # 100 K or so
    else:
        Tsh = warpfield_params.t_ion  # 1e4 K or so
        
    # sound speed in shell (if a part is neutral, take the lower sound speed)
    
    if Tsh > 1e3:
        mu = warpfield_params.mu_p
    else:
        mu = warpfield_params.mu_n
        
    cs = np.sqrt(warpfield_params.gamma_adia * c.k_B.cgs.value * T / mu )

    frag_value = warpfield_params.frag_grav_coeff * Grav_au * 3. * Msh / (4. * np.pi * v * cs * r) # (compare McCray & Kafatos 1987, eq. 14)

    # prevent fragmentation at small radii
    Rmin = 0.1 * ODE_params['Rcloud_au']
    if r < Rmin:
        frag_value -= (1e5 * np.abs(r - Rmin)) ** 2

    residual = frag_value - 1.0 # fragmentation occurs when frag_value reaches 1.0 (or higher)
    
    if residual >= 0:
        try:
            ttime,tfrag,tend=np.loadtxt(ODE_params['mypath'] +"/FragmentationDetails/GRAVtime.txt", skiprows=1, delimiter='\t', usecols=(0,1,2), unpack=True)
        except:
            ttime=tfrag=tend=np.array([])
            
        if warpfield_params.frag_enable_timescale:   
            time_value=(4*cs*(r**2))/(Msh*Grav_au)
        else:
            time_value=0    
            
        ttime=np.append(ttime, t)
        tfrag=np.append(tfrag, time_value)
        
        tendc=ttime[0]+time_value
        
        tend=np.append(tend, tendc)
        
        tcf=[0]
        cfv=[1]
        
        np.savetxt(ODE_params['mypath'] +"/FragmentationDetails/GRAVtime.txt", np.c_[ttime,tfrag,tend],delimiter='\t',header='realTime'+'\t'+'fragTimescale'+'\t'+'endTime')
        np.savetxt(ODE_params['mypath'] +"/FragmentationDetails/Coverfrac"+str(len(ODE_params['Mcluster_list']))+".txt", np.c_[tcf,cfv],delimiter='\t',header='Time'+'\t'+'Coverfraction (=1 for t<Time[0])')
        
        if t < tend[len(tend)-1]:
            residual = -1
            
    tgrav=np.append(tgrav, t)
    rgrav=np.append(rgrav, r)
    resgrav=np.append(resgrav, residual)
    csgrav=np.append(csgrav, cs)
    
    np.savetxt(ODE_params['mypath'] +"/FragmentationDetails/GRAV.txt", np.c_[tgrav,rgrav,resgrav,csgrav],delimiter='\t',header='fragtime'+'\t'+'fragradius'+'\t'+'residual'+'\t'+'soundspeed')
    
    print("Grav_coll:", residual)
    
    return residual

def event_inhom_frag(t, y, ODE_params):
    """
    event of shell fragmentation due to inhomogeneities (when whole cloud has been swept up)
    :param t: time (float)
    :param: y: (list)
    :param ODE_params:
    :return:
    """
    r, v, E, T = y

    residual = r - ODE_params['Rcloud_au'] # fragmentation occurs when shell radius equals cloud radius

    return residual

def event_EnergyZero(t, y):
    """
    event of "energy is 0" (switch to momentum driving)
    :param t: time (float)
    :param: y: (list)
    :return: residual (float)
    """

    r, v, E, T = y

    Eb_min = warpfield_params.phase_Emin

    residual = E - Eb_min # let's not go below zero, so stop slightly above zero

    #print "residual in EnergyZero", residual

    return residual

def event_Radius1000(t, y):
    """
    event of "Shell Radius is 1000 pc" (stop)
    :param t: time (float)
    :param: y: (list)
    :return: residual (float)
    """

    r, v, E, T = y

    residual = r - warpfield_params.stop_r
    # use with: event_Radius1000.direction = -1.0
    # (event only triggered when rStop is approached from above)

    return residual

def event_Radius1(t, y):
    """
    event of "Shell Radius is 1 pc" (stop)
    :param t: time (float)
    :param: y: (list)
    :return: residual (float)
    """

    r, v, E, T = y

    # if velocity is positive, this is not a collapse, and even if the radius is 1 pc, we do not want to stop!
    #if v >= 0.:
    #    add_term = 100. + v # some large number to prevent the residual from becoming 0.
    #else:
    #    add_term = 0.
    #residual = (r - 1.) + add_term
    residual = r - warpfield_params.r_coll

    return residual

def event_StopTime(t, y, tStop):
    # End of simulation time.
    residual = t - tStop

    return residual


def event_RTinstab(t, y, ODE_params, SB99f):
    """
    Event of Rayleigh-Taylor instabilities (i.e. acceleration is positive)
    :param t: time (float)
    :param y: (list)
    :param ODE_params: additional parameters
    :return: float, when 0: event!
    :return:
    """
    r, v, E, T = y
    
    try:
        tgrav,rgrav,resgrav,dRswrite=np.loadtxt(ODE_params['mypath'] +"/FragmentationDetails/RT.txt", skiprows=1, delimiter='\t', usecols=(0,1,2,3), unpack=True)
    except:
        tgrav=rgrav=resgrav=dRswrite=np.array([])
        check=os.path.join(ODE_params['mypath'], "FragmentationDetails")
        check_outdir(check)
        

    part1_dict = phase_ODEs.fE_tot_part1(t, y, ODE_params, SB99f)

    # acceleration
    acc = part1_dict['vd']

    # prevent fragmentation at small radii
    Rmin = 0.1 * ODE_params['Rcloud_au']
    if r < Rmin:
        acc -= (1e5 * np.abs(r - Rmin)) ** 2
        
        
        
    if r < ODE_params['Rcloud_au']:
        acc = -5

    #elif acc > 0:
     #   if r > 0.98 * ODE_params['Rcloud_au'] and r < 10* ODE_params['Rcloud_au']:
      #      acc=-acc

    residual = acc # - 0.0
    # use with: event_RTinstab.direction = 1.0
    # (event only triggered when acc==0 is approached from below)

    
    #print('RT.............t=',t,'**************r=',r,'************RESIDUAL=',residual)
    
    
    
    dRs = float(os.environ["ShTh"])
    
    if residual >= 0:
        try:
            ttime,tfrag,tend=np.loadtxt(ODE_params['mypath'] +"/FragmentationDetails/RTtime.txt", skiprows=1, delimiter='\t', usecols=(0,1,2), unpack=True)
        except:
            ttime=tfrag=tend=np.array([])
            
        if warpfield_params.frag_enable_timescale:   
            time_value=np.sqrt(dRs/(2*np.pi*acc))
        else:
            time_value=0

            
        ttime=np.append(ttime, t)
        tfrag=np.append(tfrag, time_value)
        
        tendc=ttime[0]+time_value
        
        tend=np.append(tend, tendc)
        
        tcf=[0]
        cfv=[1]
        
        #print('nr.clusters=',len(ODE_params['Mcluster_list']))
        
        np.savetxt(ODE_params['mypath'] +"/FragmentationDetails/RTtime.txt", np.c_[ttime,tfrag,tend],delimiter='\t',header='realTime'+'\t'+'fragTimescale'+'\t'+'endTime')
        np.savetxt(ODE_params['mypath'] +"/FragmentationDetails/Coverfrac"+str(len(ODE_params['Mcluster_list']))+".txt", np.c_[tcf,cfv],delimiter='\t',header='Time'+'\t'+'Coverfraction (=1 for t<Time[0])')
        if t < tend[-1]:
            residual = -1

    tgrav=np.append(tgrav, t)
    rgrav=np.append(rgrav, r)
    resgrav=np.append(resgrav, residual)
    dRswrite=np.append(dRswrite, dRs)
    
    np.savetxt(ODE_params['mypath'] +"/FragmentationDetails/RT.txt", np.c_[tgrav,rgrav,resgrav,dRswrite],delimiter='\t',header='fragtime'+'\t'+'fragradius'+'\t'+'residual'+'\t'+'shellthickness')
    
    print(["RT:", residual], verbose=1)

    return residual


def event_vel0(t,y):
    """
    Event of velocity changes from positive to negative (start of a collapse) or negative to positive (start of expansion after a bit collapsing)
    When velocity changes from positive to negative, do not terminate integration but keep shell mass constant
    :param t: time (float)
    :param y: (list)
    :return:
    """

    r, v, E, T = y

    residual = v # -0.0

    return residual

def event_dissolution(t,y, ODE_params):
    """
    Event of shell dissolution final (that is, a certain time span since shell dissolution has passed)
    :param t: time (float)
    :param y: (list)
    :param ODE_params: additional parameters
    :return: float, when 0: event!
    :return:
    """

    r, v, E, T = y

    residual = (t - ODE_params['t_dissolve']) - warpfield_params.stop_t_diss

    return residual





