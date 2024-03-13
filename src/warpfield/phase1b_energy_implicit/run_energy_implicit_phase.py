#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:12:37 2023

@author: Jia Wei Teh

technically not 'implicit', it's just that (i think) some of the conditions 
satisfies that the bubble moves into phase two (momentum driven), 
but not completely yet. Thus this script is basically the same as run_energy_phase.py in phase1 energy,
but like with additional events and constrains to better understand the transition period.
BUT, not quite the transition period too, because that is being handled in phase1c_transition.


Also, I think the important point here is to more precisely find the values of beta and delta,
which was kinda omitted and neglected in the previous run_energy_phase (possibly due to low
                                                                        impact during early phases.).

"""

import scipy.optimize
import numpy as np
import astropy.units as u
import astropy.constants as c
import sys
import os
#--
import src.warpfield.phase_general.phase_events as phase_events
import src.warpfield.phase_general.phase_ODEs as phase_ODEs
import src.warpfield.phase1b_energy_implicit.find_root_betadelta as find_root_betadelta
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams
import src.warpfield.phase1b_energy_implicit.get_betadelta as get_betadelta


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







def run_phase_energy(params, ODEpar, SB99f, rtole = rtol):
    """
    solves fE_tot
    :param params:
    :param ODEpar:
    :param SB99f:
    :return:
    """

    '''
    params is from energy_phase (Dw), whereas ODEpar is from main.py including mcloud etc.
    TODO: perhaps make this more readable and merge?
    '''



    # IMPORTANT QUESTION: SHOULD WE MERGE PARAMS AND ODEPAR TOGETHER?


    # TODO:!! add shell thickness so that we won't have to call os.environ[shTh].
    # idea on how to do it: go to shell_structure(), and then make sure the dR variable is 
    # being tracked. Then, make sure it is bring brought along all the way to phase1a_result, 
    # then add it into the dictionary ODEpar or params.

    
    # create variables.
    # redefine values
    
    # In the old code, the values are directly called: e,g, a = params['b']*c.
    # Here, we first declare ALL values from the dictionary instead, then call
    # the declared values via the assigned variables. e.g., b = params['b']; a = b * c.
    # TODO: the only problam is that I am not sure if the use of dictionary calling
    # is intentional, i.e., due to the fact that dictionaries will update the values
    # themselves if fed into a loop. 
    
    #----
    
    #---- ODEpar contains the following:
    # mCloud = ODEpar['mCloud']
    # rCore = ODEpar['rCore']
    # rhoCore = ODEpar['rhoCore']
    # mCluster = ODEpar['mCluster']
    # rCloud = ODEpar['rCloud']
    # i think stop_t is from warpfield_params, since in the old code it says i.tStop.
    # tStop = ODEpar['tStop']
    # t_dissolve = ODEpar['t_dissolve']
    # Rsh_max = ODEpar['Rsh_max']
    mCluster_list = ODEpar['mCluster_list']
    ODEpar['collapse_counter'] = len(mCluster_list)
    # dR_shell = OPDEpar['dR_shell']  #shell thickness
    
    #---- params contains the following:
    t_now = params['t_now']
    R2 = params['R2']
    T0 = params['T0']
    beta = params['beta']
    delta = params['delta']
    alpha = params['alpha']
    Eb = params['Eb']
    dMdt_factor = params['dMdt_factor']
    params['shell_frag'] = False
    
    #----additional params
    # these are the redefinition in run_phase_energy()
    # params['xtolbstrux'] = 1e-6
    params['t_frag'] = 1e99 * u.yr
    v = (alpha * R2 / t_now).to(u.km/u.s)
    params['v'] = v
    params['beta_guess'] = beta
    params['delta_guess'] = delta
    params['L_leak'] = 0 * u.erg/u.s
    # track the number of times ODE_equations(), the ODE solver equation, is being looped.
    params['phase1b_loop_counter'] = 0
    
    # this is only used here and is used for cooling curve slow-increment
    params['time_since_last_cooling_update'] = -1e30 * u.yr  # arbitrary large negative number


    # ---- Note
    # Actually, I think dictionary params are absolutely necessary. I think it is one 
    # of the best way to solve the problem, which is that in odeint() or any 
    # other solver, parameters are updated independently to the t, y stuffs, 
    # but also needs to be passed onto the new loop. E.g., in each loop of ODE_equations, 
    # one needs to update e.g., dMdt_factor and beta_guess; these can only be 
    # updated either using a list (getting list[-1] every call), or a dictionary. 
    # A dictionary will change its contents if it is being changed in a loop, 
    # regardless of whether or not the function/loop produces an output. 
    
    
    # # -- I comment out this part for now, because I thinkt his isnt actually needed. 
    # # TOOD: fix this with try/except
    # # path to saved output
    # path2RT = os.path.join(warpfield_params.out_dir, 'Fragmentation/RayleighTaylor.txt')
    # # read fragmentation data
    # # TOOD: fix this with try/except
    # # try: 
    # t_RT, r_RT, residual_RT, dR_RT = np.loadtxt(path2RT)
    # # -- end of comment
    
    
    
    
    #-- theoretical minimum and maximum of this phase
    tmin = params['t_now']
    tmax = ODEpar['tStop']
    
    # initial conditions for the ODE equation.
    y0 = [R2.to(u.pc).value, v.to(u.km/u.s).value, Eb.to(u.erg).value, T0.to(u.K).value]

    
    # stop integration (.terminal = True) when:
    #--- 1) Stopping time reached.
    def _stop(t, y, tStop):
        return t - tStop.to(u.Myr).value
    
    event_stop = lambda t, y: _stop(t, y, tmax)
    event_stop.terminal = True 
    
    #--- 2) Small radius reached during recollapse. 
    def _smallRadius(t, y):
        r, v, _, _ = y
        # happens only during recollapse, i.e. v < 0.
        if v < 0:
            return r - warpfield_params.r_coll.to(u.pc).value
        # otherwise just return an arbritrary number that doesn't mean anything, as long as it is not zero. 
        else:
            return 100
        
    event_smallRadius = lambda t, y: _smallRadius(t, y)
    event_smallRadius.terminal = True
    event_smallRadius.direction = -1.0 
    
    #--- 3) Large radius reached during expansion.
    def _largeRadius(t, y):
        r, _, _, _ = y
        return r - warpfield_params.stop_r.to(u.pc).value
    
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

    #--- 5) FRAGMENTATIONL: Inhomogeneities, i.e. when all cloud has been swept up.
    def _dens_inhom(t, y, rCloud):
        r, _, _, _ = y
        return r - rCloud.to(u.pc).value

    event_dens_inhom = lambda t, y: _dens_inhom(t, y, ODEpar['rCloud'])
    event_dens_inhom.terminal = True

    #--- 6) Dissolution of shell into the ISM
    def _dissolution(t, y, time_params):
        t_dissolve, stop_t_diss = time_params
        return (t - t_dissolve.to(u.Myr).value) - stop_t_diss.to(u.Myr).value

    event_dissolution = lambda t, y: _dissolution(t, y, [ODEpar['t_dissolve'], warpfield_params.stop_t_diss])
    event_dissolution.terminal = True


    #--- 7) When the density profile is steeper than r^(-2), switch to momentum driven.
    def _density_gradient(t, y, params):
        
        # unpacking
        r, _, _, _ = y
        path2output, collapse_counter = warpfield_params
        
        # original density profile in log pc and log 1/cm3
        full_path = os.path.join(path2output, 'init_density_profile_col'+ str(collapse_counter) + '.csv')
        r0, n0 = np.loadtxt(full_path, skiprows = 1, unpack = True)
        
        dr = np.mean(np.diff(r0)[5:10])
    
        fdot = np.diff(n0)/dr #derivative
        
        # TODO: I don't understand this part from the old code. What happens
        # then if there aren't any? what if np.where returns [], and [0][0] will just crash?
        # What happens also if the density profile is homogeneous? Perhaps we should add a logic gate?
        index = np.where(np.sqrt(fdot**2) > 2)[0][0] #find index
        
        #return first r where density profile is steeper
        r_dens_grad = 10**r0[index]
    
        residual = r - r_dens_grad
        
        if residual >= 0:
            print('density profile steeper than r**(-2): switching to momentum driven phase')
            # TODO: originally here also includes [Coverfrac] and [FragmentationDetails] stuffs, but we didn't care. Remember to add them back!
        
        return residual
        
    # TODO: add collapse_counter in ODEparams in main.py.
    dens_params = [warpfield_params.out_dir, ODEpar['collapse_counter']]
    
    event_density_gradient = lambda t, y: _density_gradient(t, y, dens_params)
    # radius reached where density profile is stepper than r**(-2)
    event_density_gradient.terminal = True; 
     
    
    #--- 8) FRAGMENTATION: Rayleigh - Taylor instability (i.e., positive acceleration)
    def _RT_instability(t, y, params):
        # unpack
        r, v, E, T = y
        ODE_params, SB99f = params
        # TODO: add parameters for 'get_vdot()' in ODE_params 
        _, _ = ODE_params
        
        # path to saved output
        path2RTprop = os.path.join(warpfield_params.out_dir, 'Fragmentation/RayleighTaylor/prop.txt')
        path2RTtime = os.path.join(warpfield_params.out_dir, 'Fragmentation/RayleighTaylor/time.txt')
        # read fragmentation data
        # TOOD: fix this with try/except
        try: 
            t_RT, r_RT, residual_RT, dRshell_RT = np.loadtxt(path2RTprop,
                                                             skiprows = 1,
                                                             unpack = True)
        except:
            # if file not found, initialise
            t_RT = r_RT = residual_RT = dRshell_RT = np.array([])
        
        # acceleration
        vd, _ = phase_ODEs.get_vdot(t, y, ODE_params, SB99f)
        
        # prevent fragmentation happening at small radius (~0.1 rCloud)
        if r < 0.1 * ODE_params['rCloud'].to(u.pc).value:
            vd -= (1e5 * np.abs(r - 0.1 * ODE_params['rCloud'].to(u.pc).value))**2
        
        if r < ODE_params['rCloud'].to(u.pc).value:
            # an arbitrary number when radius is still small
            vd = -5

        # TODO: add also a function to save cover fraction details
        if vd >= 0:
            try:
                t_frag_now, t_frag_timescale, t_frag_end = np.loadtxt(path2RTtime,
                                                                      skiprows = 1, unpack = True)
            except:
                t_frag_now, t_frag_timescale, t_frag_end = np.array([])
            
            # if timescale is enabled, calculate fragmentation time
            if warpfield_params.frag_enable_timescale:
                timescale = np.sqrt(ODE_params['dR_shell'] / (2 * np.pi * vd))
            # otherwise, instatiate collapse
            else:
                timescale = 0

            # update
            t_frag_now = np.append(t_frag_now, t)
            t_frag_timescale = np.append(t_frag_now, timescale)
            t_frag_end = np.append(t_frag_end, t_frag_now[0] + timescale)
            
            # TODO: units?
            time_data = np.c_[t_frag_now, t_frag_timescale, t_frag_end]
            
            np.savetxt(path2RTtime, time_data,
                       header = 'Time (now), Fragmentation timescale, Time at end of fragmentation'
                       )
            
            if t < t_frag_end[-1]:
                vd = -1
            
        # TODO: units?
        t_RT = np.append(t_RT, t)
        r_RT = np.append(r_RT, r)
        residual_RT = np.append(residual_RT, vd)
        dRshell_RT = np.append(dRshell_RT, dRshell)
            

        prop_data = np.c_[t_RT, r_RT, residual_RT, dRshell_RT]        
        # TODO: add units
        np.savetxt(path2RTprop, prop_data, #do we need delimiter?
                   header = 'Fragmentation time, Fragmentation radius, Residual, Shell Thickness')

        
        return vd
    
    # TODO: dRshell is shTh from os.enrivon?
    event_RT_instability = lambda t, y: _RT_instability(t, y, ODEpar, SB99f)
    event_RT_instability.terminal = True
    event_RT_instability.direction = 1.0 
    
    #--- 9) FRAGMENTATION: Gravitatinoal fragmentation
    def _grav_fragmentation(t, y, params):
        
        # unpack
        r, v, E, T = y
        ODE_params, SB99f = params
        # TODO: add parameters for 'get_vdot()' in ODE_params 
        _, _ = ODE_params
        
        # path to saved output
        path2GRAVprop = os.path.join(warpfield_params.out_dir, 'Fragmentation/GravitationalFragmentation/prop.txt')
        path2GRAVtime = os.path.join(warpfield_params.out_dir, 'Fragmentation/GravitationalFragmentation/time.txt')
        # read fragmentation data
        # TOOD: fix this with try/except
        try: 
            t_grav, r_grav, residual_grav, cs_grav = np.loadtxt(path2GRAVprop,
                                                             skiprows = 1,
                                                             unpack = True)
        except:
            # if file not found, initialise
            t_grav = r_grav = residual_grav = cs_grav = np.array([])        
        
        # acceleration
        vd, ODEdata = phase_ODEs.get_vdot(t, y, ODE_params, SB99f) 
        fabs_i = ODEdata['fabs_i']
        mShell = ODEdata['Msh']
        
        # Now, if the shell neutral? If it is, take low temperature. 
        # Check shell neutrality from escaped radiation.
        if fabs_i < 0.999:
            Tshell = warpfield_params.t_neu
            mu = warpfield_params.mu_n
        else:
            Tshell = warpfield_params.t_ion
            mu = warpfield_params.mu_p
        # sound speed
        cs = np.sqrt(warpfield_params.gamma_adia * c.k_B * T / mu)
        # frag parameter (e.g., compare McCray & Kafatos 1987, eq. 14)
        frag_param = warpfield_params.frag_grav_coeff * c.G * 3 * mShell / (4 * np.pi * v * cs * r)
        
        # prevent fragmentation happening at small radius (~0.1 rCloud)
        if r < 0.1 * ODE_params['rCloud'].to(u.pc).value:
            frag_param -= (1e5 * np.abs(r - 0.1 * ODE_params['rCloud'].to(u.pc).value))**2
        
        # define occurence of fragmentation at frag_param 1.0 or higher.
        residual = frag_param - 1.0   
        

        # TODO: add also a function to save cover fraction details
        if residual >= 0:
            try:
                t_frag_now, t_frag_timescale, t_frag_end = np.loadtxt(path2GRAVtime,
                                                                      skiprows = 1, unpack = True)
            except:
                t_frag_now = t_frag_timescale = t_frag_end = np.array([])
            
            # if timescale is enabled, calculate fragmentation time
            if warpfield_params.frag_enable_timescale:
                timescale = 4 * cs * r**2 / (mShell * c.G)
            # otherwise, instatiate collapse
            else:
                timescale = 0

            # update
            t_frag_now = np.append(t_frag_now, t)
            t_frag_timescale = np.append(t_frag_now, timescale)
            t_frag_end = np.append(t_frag_end, t_frag_now[0] + timescale)
            
            # TODO: units?
            time_data = np.c_[t_frag_now, t_frag_timescale, t_frag_end]
            
            np.savetxt(path2GRAVtime, time_data,
                       header = 'Time (now), Fragmentation timescale, Time at end of fragmentation'
                       )
            
            if t < t_frag_end[-1]:
                residual = -1
            
        # TODO: units?
        t_grav = np.append(t_grav, t)
        r_grav = np.append(r_grav, r)
        residual_grav = np.append(residual_grav, residual)
        cs_grav = np.append(cs_grav, cs)        
        
        prop_data = np.c_[t_grav, r_grav, residual_grav, cs_grav]        
        # TODO: add units
        np.savetxt(path2GRAVprop, prop_data, #do we need delimiter?
                   header = 'Fragmentation time, Fragmentation radius, Residual, Sound Speed')


        return

    grav_params = [ODEpar, SB99f]
    event_grav_fragmentation = lambda t, y: _grav_fragmentation(t, y, grav_params)
    event_grav_fragmentation.terminal = True
    
    #--- 10) Cooling switch. When Lcool ~ Lmech, switch to momentum-driven phase.
    def _cooling_switch():
        return
    
    # this one has Lcool=np.log10(float(os.environ["Lcool_event"])) which is defined in zeroODE34, like, this is crazy.
    # need to solve the circular import etc and make sure all functions are not badly entangled.
    event_cooling_switch = lambda t, y: _cooling_switch(t,y, ODEpar)
    event_cooling_switch.terminal = True 

    # Now, we gather all events into a list.
    ODE_event_list = [event_stop, 
                      event_smallRadius,
                      event_largeRadius,
                      event_velocity,
                      event_dissolution,
                      ]
    
    # should we include these instabilities?
    # TODO: remember to enable fragmentation so that one can check if these functions are correctly implemented.
    # --fragmentation related
    if warpfield_params.frag_enabled:
        if warpfield_params.frag_grav:
            ODE_event_list.append(event_grav_fragmentation)
        if warpfield_params.frag_RTinstab:
            ODE_event_list.append(event_RT_instability)
        if warpfield_params.frag_densInhom:
            ODE_event_list.append(event_dens_inhom)  
    # --others
    if warpfield_params.dens_a_pL != 0:
        ODE_event_list.append(event_density_gradient)    

    
    # define ODE which shall be solved
    ODE_solve = lambda t, y: ODE_equations(t, y, params, ODEpar, SB99f)
    
    psoln = scipy.integrate.solve_ivp(ODE_solve, [tmin.to(u.Myr).value, tmax.to(u.Myr).value], y0,
                                      method='LSODA', events = ODE_event_list,
                                      min_step=10**(-7), rtol=rtole) #atol=atole
    
    # call ODE_equations with end values to get cs_avg in params dict (this is dirty, better: write a routine to return cs_avg)
    # params['cs_avg'] will be overwritten just by calling fE_tot
    # TODO: check also what parameters are being overwritten by fE_tot.
    # TODO: wait, shouldn't cs_avg already be calculated? as in solution[-1] was the last
    # loop anyway?
    ODE_equations(psoln.t[-1], 
                  [psoln.y[0][-1], psoln.y[1][-1], psoln.y[2][-1], psoln.y[3][-1]],
                  params, ODEpar, SB99f)

    # old way of return
    # return psoln, params

    # new way of return
    return [psoln, params]







def ODE_equations(t, y, params, ODEpar, SB99f):
    
    # old code: fE_tot()
    
    
    # --- These are R2, v2, Eb and T0.
    r, v, E, T = y
    
    # --- add units for future calculation
    t *= u.Myr
    r *= u.pc
    v *= u.km/u.s
    E *= u.erg
    T *= u.K
    
    # --- feedback parameters required to find beta/delta etc
    # Interpolate SB99 to get feedback parameters
    # mechanical luminosity at time t (erg)
    L_wind = SB99f['fLw_cgs'](t) * u.erg / u.s
    # momentum of stellar winds at time t (cgs)
    pdot_wind = SB99f['fpdot_cgs'](t) * u.g * u.cm / u.s**2
    # get the slope via mini interpolation for some dt. 
    pdotdot_wind = (SB99f['fpdot_cgs'](t + 1e-9 * u.Myr) - SB99f['fpdot_cgs'](t - 1e-9 * u.Myr))/(1e-9+1e-9)
    # and then add units
    pdotdot_wind *= u.g * u.cm / u.s**3
    # other luminosities
    Qi = SB99f['fQi_cgs'](t) / u.s
    # velocity from luminosity and change of momentum
    v_wind = (2.*L_wind/pdot_wind).to(u.cm/u.s)    
    # ---  
    
    #-- updating values in the loop
    # params['alpha'] = (t / r * v).decompose().value
    # params['R2'] = r
    # params['v2'] = v
    # params['Eb'] = E
    # params['T0'] = T
    # params['t_now'] = t
    # params['Qi'] = Qi
    # params['vw'] = v_wind
    # params['pwdot'] = pdot_wind
    # params['pwdot_dot'] = pdotdot_wind
    # params['phase1b_loop_counter'] += 1
    
    # # legacy (remove when certain that not needed)
    # params['L_leak'] 
    
    

    # Thinking if this is needed. The thing is, Cool_struc is now written
    # to run independently in the bubble_luminosity code. With this the run time can be shorter.
    #  >>>>    # update the cooling curve once in a while 
                # if (np.abs(t - params['t_last_coolupdate'])) > 0.1:
                #     # get time-dependent cooling structure
                #     print("Updating cooling curve ...")
    
    # TODO: think of a way to implement this.
    
    
    # question: if it is all being rewritten, what is the point of params?
    # answer: some params[] are being updated in the loop, thus it is important to recalculate. 
    
    
    
    
    # perhaps we should redo this here?
    
    
    
    
    # take note that params is used in rootfinder_bd_wrap, and 
    # they take the following:
        # beta, delta, Edot_residual_guess, T_residual_guess,\
        #             L_wind, L_leak, Eb, Qi,\
        #             v_wind, v2, R2, T0,\
        #             alpha, t_now,\
        #             pwdot, pwdotdot,\
        #             dMdt_factor = wrapper_params
                        
    
    
        
    # =============================================================================
    # Part 1: find acceleration and velocity
    # =============================================================================
    
    # ODEpar seems to need: mCloud, rCloud, Rsh_max, Mcluster_list, t_dissolve
    
    
    vd, _ = phase_ODEs.get_vdot(t, y, ODEpar, SB99f)
    rd = v
        
        
    # =============================================================================
    # Part 2: find beta, delta and convert them to dEdt and dTdt
    # =============================================================================
        
    # [r, v, E, T] from y.
    # t from current time in function loop
    
    # TODO: why is this extracted from params[] and not instead the newly found ones?
    # import data from params
    beta = params['beta']
    delta = params['delta']
    L_leak = params['L_leak']
    alpha = params['alpha']
    dMdt_factor = params['dMdt_factor']
    
    # L_leak seems to be zero everytime according to old code.
    
    
    # these two values are unitless because they are normalised. see line 494 in get_betadelta.get_Edot_Tdot_residual_fast().
    # Actually this is not true now. We stopped using unitless normalisation.
    Edot_residual_guess, T_residual_guess = 1 * u.erg / u.s, 1 * u.K
    
    # TODO: now I am trying to match the dictionaries with the old one.
    # bd_params = { 
        
    #     't_now': t,
    #     'R2': r,
    #     'Eb': E,
    #     'T0': T,
    #     'beta': beta,
    #     'delta': delta,
    #     'alpha': alpha,
    #     'dMdt_factor': dMdt_factor,
    #     'pwdot': pdot_wind,
    #     'pwdot_dot': pdotdot_wind,
    #     'Qi': Qi,
    #     'vw': v_wind,
    #     'Lw': L_wind,
        
        
        
    #     }
    
    
    # --- old version dictionary
      # 'xtolbstrux': 1e-06, 'tfrag': 1e+99, 'shell_frag': False,
      # 'L_leak': 0.0, 'v2': 106.0, 'beta_guess': 0.83,
      # 'delta_guess': -0.17851, 't_last_coolupdate': 0.003002,
      # 'Cool_Struc': {
    
      #   'Cfunc': <scipy.interpolate.interpnd.LinearNDInterpolator object at 0x7fe060cb5df0>,
      #   'Hfunc': <scipy.interpolate.interpnd.LinearNDInterpolator object at 0x7fe060cb5e50>},
      #   'Qi': 1.6993e+65, 'vw': 3810.2, 'Lw': 2016500000.0,
         
    # --- 
     
    
    
    # --- my version without dictionary. I should give up on this? Maybe not.
    bd_params = [   beta, delta, #unitless
                    Edot_residual_guess, T_residual_guess, #normalised, so unitless
                    # these are in cgs units
                    L_wind.value, L_leak.value, E.value, Qi.value,
                    # v_wind is in cm.s here, v2, R2 in km/s and pc.
                    v_wind.value, v.value, r.value, T.value,
                    # t_now in Myr, alpha is unitless
                    alpha, t.value,
                    # these are in cgs (see run_implicit)
                    pdot_wind.value, pdotdot_wind.value,
                    dMdt_factor
                    ]
    # ---
    print('\n\n\nentering get_beta_delta_wrapper')
    beta, delta = get_betadelta.get_beta_delta_wrapper(params["beta_guess"], params["delta_guess"], bd_params)
    print('\n\n\nexiting get_beta_delta_wrapper')
    sys.exit()
    
    #------ convert them to dEdt and dTdt.
    
    
    def get_EdotTdot(t, beta, delta,
                  pwdot, pwdotdot,
                  rShell, vShell, E_bubble, T,
                  v_wind, L_wind
                  ):
        # convert beta and delta to dE/dt and dT/dt.
        
        # old code: Edot_Tdot()
        
        # R1 (inner bubble radius; point of discontinuity)
        R1 = (scipy.optimize.brentq(get_bubbleParams.get_r1, 
                               1e-3 * rShell.to(u.cm).value, rShell.to(u.cm).value, 
                               args=([L_wind.to(u.erg/u.s).value, 
                                      E_bubble.to(u.erg).value, 
                                      v_wind.to(u.cm/u.s).value, 
                                      rShell.to(u.cm).value
                                      ])) * u.cm)\
                                .to(u.pc)#back to pc
                                
        # bubble pressure
        pBubble = get_bubbleParams.bubble_E2P(E_bubble, rShell, R1)
        
        # get new beta value
        Edot = get_bubbleParams.beta2Edot(pBubble, E_bubble, beta, 
                                          t, pwdot, pwdotdot,
                                          R1, rShell, vShell,
                                          )
        # get dTdt
        Tdot = get_bubbleParams.delta2dTdt(t, T, delta)
        
        return Edot, Tdot

    
    Ed, Td = get_EdotTdot(t, beta, delta,
                  pdot_wind, pdotdot_wind,
                  r, v, E, T,
                  v_wind, L_wind)
    
    
        
    
    return [rd, vd, Ed, Td]






