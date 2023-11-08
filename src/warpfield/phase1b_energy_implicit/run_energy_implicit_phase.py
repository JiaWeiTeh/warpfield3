#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:12:37 2023

@author: Jia Wei Teh
"""

import scipy.optimize
import numpy as np
import astropy.units as u
import sys
#--
import src.warpfield.phase_general.phase_events as phase_events
import src.warpfield.phase_general.phase_ODEs as phase_ODEs
import src.warpfield.phase1b_energy_implicit.find_root_betadelta as find_root_betadelta


# get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()

#relative tolerance of stiffest ODE solver (energy phase),decrease to 1e-4 if you have crashes --> slower but more stable
rtol=1e-3 






'''

Here is what I think is happening in this script. This script has a ODE solver,
much like the one in energy_phase_ODEs.get_ODE.Edot().

However, this one works like this: in run_phase_energy(), it has several event
functions that sets stopping condition. These will be used to run scipy on fE_tot, 
which 1) outputs the ODE derivatives, and 2) includes wrapper for 
i) fE_tot_part1() to calculate vd, rd, and ii) Edot_Tdot() to find Ed, Td. 
'''





def run_phase_energy(params, ODEpar, SB99f,rtole = rtol):
    """
    solves fE_tot
    :param params:
    :param ODEpar:
    :param SB99f:
    :return:
    """

    params['temp_counter'] = 0
    params['xtolbstrux'] = 1e-6
    params['tfrag'] = 1e99
    params['shell_frag'] = False
    params['L_leak'] = 0.
    params['v2'] = params['alpha'] * params['R2'] / (params["t_now"])
    params['beta_guess'] = params['beta']
    params['delta_guess'] = params['delta']
    params['t_last_coolupdate'] = -1e30 # arbitrary large negative number

    tmin = params["t_now"]
    tmax = ODEpar['tStop']
    y0 = [params["R2"], params["v2"], params["Eb"], params["T0"]]
    
    # stop integration when:
    #--- 1) Stopping time reached.
    def _stop(t, y, tStop):
        return t - tStop
    
    event_stop = lambda t, y: _stop(t, y, tmax)
    event_stop.terminal = True 
    
    #--- 2) Small radius reached during recollapse. 
    def _smallRadius(t, y):
        r, v, _, _ = y
        # happens only during recollapse, i.e. v < 0.
        if v < 0:
            return r - warpfield_params.r_coll
        # otherwise just return an arbritrary number that doesn't mean anything, as long as it is not zero. 
        else:
            return 100
    event_smallRadius = lambda t, y: _smallRadius(t, y)
    event_smallRadius.terminal = True
    event_smallRadius.direction = -1.0 
    
    #--- 3) Large radius reached during expansion.
    def _largeRadius(t, y):
        r, _, _, _ = y
        return r - warpfield_params.stop_r
    
    event_largeRadius = lambda t, y: _largeRadius(t, y)
    event_largeRadius.terminal = True 
    
    #--- 4) When velocity crosses from + to - (re-collapse) or the other way (re-expand)
    def _velocity(t, y):
        _, v, _, _ = y
        return v

    event_velocity = lambda t, y: _velocity(t, y)
    # do not switch off, but smaller time steps
    event_velocity.terminal = False
    event_velocity.direction = -1.0 

    #--- 5)




    event_fun4 = lambda t, y: phase_events.event_grav_frag(t,y,ODEpar,SB99f); event_fun4.terminal = True # graviational fragmentation


    # list of events which will cause the solver to terminate (see my_events.py to see what they are)
    event_fun5 = lambda t, y: phase_events.event_RTinstab(t, y, ODEpar, SB99f); event_fun5.terminal = True; event_fun5.direction = 1.0 # Rayleigh-Taylor instability
    event_fun6 = lambda t, y: phase_events.event_inhom_frag(t, y, ODEpar); event_fun6.terminal = True # fragmentation due to density inhomogeneities
    event_fun8 = lambda t, y: phase_events.event_dissolution(t, y, ODEpar); event_fun8.terminal = True # shell dissolves into ambient ISM
    
    event_fun9 = lambda t, y: phase_events.event_density_gradient(t, y, ODEpar); event_fun9.terminal = True; # radius reached where density profile is stepper than r**(-2)
    event_fun10= lambda t, y: phase_events.event_cool_switch(t,y, ODEpar); event_fun10.terminal = True; # switch to momentum driving if Lcool ~ Lmech

    event_fun_list = [event_fun1, event_fun2, event_fun3, event_fun7, event_fun8,event_fun9,event_fun10]
    if warpfield_params.frag_grav:
        event_fun_list.append(event_fun4)
    if warpfield_params.frag_RTinstab:
        event_fun_list.append(event_fun5)
    if warpfield_params.frag_densInhom:
        event_fun_list.append(event_fun6)
        
        
    
        
    #atole=[10**(-3),10**(-3),10**(-1),10**(-1)]
    
    #rtole=10**(-4)

    # call ODE solver
    print('call ode energy')
    
    
    # define ODE which shall be solved
    ODE_fun = lambda t, y: fE_tot(t, y, params, ODEpar, SB99f)
    
    psoln = scipy.integrate.solve_ivp(ODE_fun, [tmin, tmax], y0, method='LSODA', events=event_fun_list, min_step=10**(-7),rtol=rtole) #atol=atole
    
    #print('info_ODE',info)
    
    #psoln = scipy.integrate.solve_ivp(ODE_fun, [tmin, tmax], y0, method='LSODA', events=event_fun_list)

    # call fE_tot with end values to get cs_avg in params dict (this is dirty, better: write a routine to return cs_avg)
    # params['cs_avg'] will be overwritten just by calling fE_tot
    fE_tot(psoln.t[-1], [psoln.y[0][-1], psoln.y[1][-1], psoln.y[2][-1], psoln.y[3][-1]], params, ODEpar, SB99f)

    return psoln, params


def fE_tot(t, y, params, ODEpar, SB99f):
    
    
    
    
    
    
    
    """
    general energy-driven phase including stellar winds, gravity, power law density profiles, cooling, radiation pressure
    :param y: [r,v,E]: shell radius (R2), shell velocity (v2), bubble energy (Eb)
    :param t: time (since the ODE is autonomous, t does not appear. The ODE solver still expects it though)
    :param params: (see below)
    :return: time derivative of y, i.e. [rd, vd, Ed, Td]
    # parameters:
    # LW : mechanical luminosity
    # GAM : adiabatic index
    # M0T : core mass
    # RHOA : core density
    # RCORE : core radius
    # A_EXP : exponent of density profile
    # LB: luminosity lost to cooling (calculate from bubble structure)
    # FRAD: radiation pressure coupled to the shell, i.e. Lbol/c * fabs (calculate fabs from shell structure)
    # PHASE: current phase (core, gradient, ambient, collapse)
    """

    # where is the counterpart? I feel like I have seen this before.
    # ans: 


    r, v, E, T = y  # unpack current values of y (r, rdot, E, T)


    # sanity check: energy should not be negative!

    # update the cooling curve once in a while 
    if (np.abs(t - params['t_last_coolupdate'])) > 0.1:
        # get time-dependent cooling structure
        print("Updating cooling curve ...")
        Cool_Struc = read_opiate.get_Cool_dat_timedep(warpfield_params.metallicity, t * 1e6, indiv_CH=True)
        onlycoolfunc, onlyheatfunc = read_opiate.create_onlycoolheat(warpfield_params.metallicity, t * 1e6)
        Cool_Struc['Cfunc'] = onlycoolfunc
        Cool_Struc['Hfunc'] = onlyheatfunc
        params['t_last_coolupdate'] = t
        params['Cool_Struc'] = Cool_Struc
    Cool_Struc = params['Cool_Struc'] # no need to pass on as extra argument ---> change!!!

    # get current feedback parameters from interpolating SB99
    LW = SB99f['fLw_cgs'](t) / c.L_cgs
    PWDOT = SB99f['fpdot_cgs'](t) * c.Myr / (c.Msun * c.kms)
    LBOL = SB99f['fLbol_cgs'](t) /c.L_cgs
    VW = 2.*LW/PWDOT
    QI = SB99f['fQi_cgs'](t)*c.Myr

    # overwrite paramters used in bubble structure
    params["alpha"] = t/r * v
    params["R2"] = r
    params["v2"] = v
    params["Eb"] = E
    params["T0"] = T
    params["t_now"] = t
    params["Qi"] = QI
    params['vw'] = VW
    params['Lw'] = LW
    params['pwdot'] = PWDOT
    tplus = t+1e-9
    tminus = t-1e-9
    params['pwdot_dot'] = ((SB99f['fpdot_cgs'](tplus) - SB99f['fpdot_cgs'](tminus))* c.Myr / (c.Msun * c.kms)) / (tplus-tminus)
    params['L_leak'] = 0. # legacy (remove when certain that not needed)
    params['temp_counter'] += 1
    
    def trunc_auto(f, n):
        
        # Automatically truncates/pads a float f to n decimal places without rounding
        def truncate(f, n):
            s = '{}'.format(f)
            if 'e' in s or 'E' in s:
                return '{0:.{1}f}'.format(f, n)
            i, p, d = s.partition('.')
            return '.'.join([i, (d+'0'*n)[:n]])
        try:
            if f < 0:
                f*=-1
                m=True
                log=int(np.log10(f))-1
            else:
                log=int(np.log10(f))
                m=False
            f=f*10**(-log)
            trunc=float(truncate(f, n))
            
            if m==True:
                res=float(str(-1*trunc)+'e'+str(log))
            else:
                res=float(str(trunc)+'e'+str(log))
        except:
            res=f
        return res   


    for ii in params:
        params[ii] = trunc_auto(params[ii], 4)

    ########################## ODEs: acceleration and velocity ###############################
    vd, _ = phase_ODEs.fE_tot_part1(t, y, ODEpar, SB99f)
    rd = v # velocity
    ##########################################################################################

    beta_guess = params["beta_guess"]
    delta_guess = params["delta_guess"]

    # TODO: add verbosity
    print("params:", {ii:params[ii] for ii in params if ii!='Cool_Struc'}, "ODEpar['Rsh_max']:", ODEpar['Rsh_max'])
    # print("elapsed real time (s):", time.time() - i.start_time)

    rootf_bd_res = find_root_betadelta.rootfinder_bd_wrap(beta_guess, delta_guess, params, Cool_Struc,
                                                              xtol=1e-5, verbose=0)  # xtol=1e-5
    beta = rootf_bd_res['beta']
    delta = rootf_bd_res['delta']
    residual = rootf_bd_res['residual']
    dMdt_factor = rootf_bd_res['dMdt_factor']
    Tavg = rootf_bd_res['Tavg']
    
    def get_soundspeed(T):
        # old code: aux.sound_speed()
        """
        This function computes the isothermal soundspeed, c_s, given temperature
        T and mean molecular weight mu.
    
        Parameters
        ----------
        T : float (Units: K)
            Temperature of the gas.
        mu_n : float (Units: g) 
            Mean molecular weight of the gas. Watch out for the units.
        gamma: float 
            Adiabatic index of gas. 
    
        Returns
        -------
        The isothermal soundspeed c_s (Units: m/s)
    
        """
        # return
        mu_n = warpfield_params.mu_n * u.g.to(u.kg)
        return np.sqrt(warpfield_params.gamma_adia * c.k_B.value * T / mu_n )

    cs_avg = get_soundspeed(Tavg)

    # TODO: add verbosity
    print("time:", t, " rootfinder result (beta, delta):", beta, delta, "residual:", residual, 'Tavg', Tavg, 'cs', cs_avg)

    ############################## ODEs: energy and temperature ##############################
    
    
    # TODO
    #--- add this in ---
    
    
    
    def asdasd():
        
        pass
    
    # def Edot_Tdot(beta, delta, params, verbose=0):
    # """
    # convert beta and delta to dE/dt and dT/dt
    # :param beta:
    # :param delta:
    # :param params:
    # :param verbose:
    # :return:
        
    #     This function should be moved to phase_energy.py
    # """

    # R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 1e-3 * params["R2"], params["R2"],args=([params["Lw"], params["Eb"], params["vw"], params["R2"]]), xtol=1e-18)
    # Pb = get_bubbleParams.bubble_E2P(params["Eb"], params["R2"], R1)
    # Edot = get_bubbleParams.beta2Edot(Pb, R1, beta, params)
    # Tdot = get_bubbleParams.delta2dTdt(params["t_now"],params["T0"],delta)

    # return Edot, Tdot
    
    
    
    
    
    #--
    
    # convert beta and delta to energy and temperature
    Ed, Td = find_root_betadelta.Edot_Tdot(beta, delta, params, verbose=0)
    ##########################################################################################

    params["delta"] = delta  # Do I need this?
    params["beta"] = beta  # Do I need this?

    params["beta_guess"] = beta
    params["delta_guess"] = delta
    params['dMdt_factor'] = dMdt_factor
    params['cs_avg'] = cs_avg # dirty way to get cs_avg at the end of phase_energy

    derivs = [rd, vd, Ed, Td]  # list of dy/dt=f functions
    return derivs





