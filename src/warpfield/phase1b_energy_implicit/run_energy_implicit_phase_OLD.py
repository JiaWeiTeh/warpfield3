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
#--
import src.warpfield.phase_general.phase_events as phase_events
import src.warpfield.phase_general.phase_ODEs as phase_ODEs
import src.warpfield.phase1b_energy_implicit.find_root_betadelta as find_root_betadelta
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams


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







def run_phase_energy(params, ODEpar, SB99f, rCloud, rtole = rtol):
    """
    solves fE_tot
    :param params:
    :param ODEpar:
    :param SB99f:
    :return:
    """

    '''
    params is from energy_phase (Dw), whereas ODEpar is from main.py including mcloud etc.
    
    params include these: 
    params = {'t_now': Dw['t_end'], 'R2': Dw['r_end'], 'Eb': Dw['E_end'], 'T0': Dw['Tb'][-1], 'beta': Dw['beta'][-1],
              'delta': Dw['delta'][-1], 'alpha': Dw['alpha'][-1], 'dMdt_factor': Dw['dMdt_factor_end']}
    params['alpha'] = Dw['v_end'] * (params["t_now"]) / params['R2'] # this is not quite consistent (because we are only using rough guesses for beta and delta) but using a wrong value here means also using the wrong velocity
    params['temp_counter'] = 0
    params['mypath'] = mypath
    
    
    TODO: perhaps make this more readable and merge?
    '''



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
    
    
    t_dissolve = ODEpar['t_dissolve']
    
    # stop integration (.terminal = True) when:
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

    #--- 5) Inhomogeneities, i.e. when all cloud has been swept up.
    def _dens_inhom(t, y, rCloud):
        r, _, _, _ = y
        return r - rCloud

    event_dens_inhom = lambda t, y: _dens_inhom(t, y, rCloud)
    event_dens_inhom.terminal = True

    #--- 6) Dissolution of shell into the ISM
    def _dissolution(t, y, time_params):
        t_dissolve, stop_t_diss = time_params
        return (t - t_dissolve) - stop_t_diss

    event_dissolution = lambda t, y: _dissolution(t, y, [t_dissolve, warpfield_params.stop_t_diss])
    event_dissolution.terminal = True


    #--- 7) 








            
            # def event_density_gradient(t,y,ODE_params):
            #     r, v, E, T = y
            #     coll_counter=len(ODE_params['Mcluster_list'])-1
            #     r0,n0=np.loadtxt(ODE_params['mypath'] +'/dlaw'+ str(coll_counter) + '.txt', skiprows=1, unpack=True)
            #     r_d=10**r0*u.cm.to(u.pc) #r in pc
            #     n_d=10**n0 # in 1/ccm
                
            #     def find_rdens(r,n):
            #         ''' Finds radius where density profile is steeper than r**(-2) 
            #         r = x-coordinate of dens profile
            #         n = y-coordinate of dens profile
            #         '''
            #         # Note:
            #             # old code: aux.find_rdens()
                        
            #         n=np.log10(n) #take logarithm 
            #         r=np.log10(r)
                    
            #         dr = np.mean(diff(r)[5:10])
                
            #         fdot=diff(n)/dr #derivative
                    
            #         index =  np.where(np.sqrt(fdot**2) > 2) #find index
                    
            #         return 10**r[index[0][0]] #return first r where density profile is steeper
            
            #     r_dens_grad= find_rdens(r_d,n_d)
            #     residual = r - r_dens_grad
                
            #     if residual >= 0:
            #         print('density profile steeper than r**(-2): switching to momentum driven phase')
                    
            #         tcf=[0]
            #         cfv=[1]
                    
            #         check=os.path.join(ODE_params['mypath'], "FragmentationDetails")
            #         check_outdir(check)
                    
            #         os.environ["Coverfrac?"] = str(1)
                    
            #         np.savetxt(ODE_params['mypath'] +"/FragmentationDetails/Coverfrac"+str(len(ODE_params['Mcluster_list']))+".txt", np.c_[tcf,cfv],delimiter='\t',header='Time'+'\t'+'Coverfraction (=1 for t<Time[0])')
                    
            
            #     return residual




    event_fun9 = lambda t, y: phase_events.event_density_gradient(t, y, ODEpar); event_fun9.terminal = True; # radius reached where density profile is stepper than r**(-2)



    event_fun4 = lambda t, y: phase_events.event_grav_frag(t,y,ODEpar,SB99f); event_fun4.terminal = True # graviational fragmentation


    # list of events which will cause the solver to terminate (see my_events.py to see what they are)
    event_fun5 = lambda t, y: phase_events.event_RTinstab(t, y, ODEpar, SB99f); event_fun5.terminal = True; event_fun5.direction = 1.0 # Rayleigh-Taylor instability
    
    
    
    
    # this one has Lcool=np.log10(float(os.environ["Lcool_event"])) which is defined in zeroODE34, like, this is crazy.
    # need to solve the circular import etc and make sure all functions are not badly entangled.
    event_fun10= lambda t, y: phase_events.event_cool_switch(t,y, ODEpar); event_fun10.terminal = True; # switch to momentum driving if Lcool ~ Lmech




    # importnat to exclude non-interested events
    event_fun_list = [event_fun1, event_fun2, event_fun3, event_fun7, event_fun8,event_fun9,event_fun10]
    if warpfield_params.frag_grav:
        event_fun_list.append(event_fun4)
    if warpfield_params.frag_RTinstab:
        event_fun_list.append(event_fun5)
    if warpfield_params.frag_densInhom:
        event_fun_list.append(event_fun6)
        
        
    # call ODE solver
    print('call ode energy')
    
    
    # define ODE which shall be solved
    ODE_fun = lambda t, y: fE_tot(t, y, params, ODEpar, SB99f)
    
    psoln = scipy.integrate.solve_ivp(ODE_fun, [tmin, tmax], y0, method='LSODA', events=event_fun_list, min_step=10**(-7),rtol=rtole) #atol=atole
    
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
    
    
    
    params is from energy_phase (Dw), whereas ODEpar is from main.py including mcloud etc.
    
    params include these: 
    params = {'t_now': Dw['t_end'], 'R2': Dw['r_end'], 'Eb': Dw['E_end'], 'T0': Dw['Tb'][-1], 'beta': Dw['beta'][-1],
              'delta': Dw['delta'][-1], 'alpha': Dw['alpha'][-1], 'dMdt_factor': Dw['dMdt_factor_end']}
    params['alpha'] = Dw['v_end'] * (params["t_now"]) / params['R2'] # this is not quite consistent (because we are only using rough guesses for beta and delta) but using a wrong value here means also using the wrong velocity
    params['temp_counter'] = 0
    params['mypath'] = mypath
    
    -- in run_phase_energy()
    
    params['temp_counter'] = 0
    params['xtolbstrux'] = 1e-6
    params['tfrag'] = 1e99
    params['shell_frag'] = False
    params['L_leak'] = 0.
    params['v2'] = params['alpha'] * params['R2'] / (params["t_now"])
    params['beta_guess'] = params['beta']
    params['delta_guess'] = params['delta']
    params['t_last_coolupdate'] = -1e30 # arbitrary large negative number
    
    """

    # where is the counterpart? I feel like I have seen this before.


    r, v, E, T = y  # unpack current values of y (r, rdot, E, T)


    # sanity check: energy should not be negative!

    # t_last_coolupdate = - 1e30
    
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



    #---
    # Interpolate SB99 to get feedback parameters
    # mechanical luminosity at time t (erg)
    L_wind = SB99f['fLw_cgs'](t) * u.erg / u.s
    # momentum of stellar winds at time t (cgs)
    pdot_wind = SB99f['fpdot_cgs'](t) * u.g * u.cm / u.s**2
    # other luminosities
    Lbol = SB99f['fLbol_cgs'](t) * u.erg / u.s
    Qi = SB99f['fQi_cgs'](t) * u.erg / u.s
    
    # velocity from luminosity and change of momentum
    v_wind = (2.*L_wind/pdot_wind).to(u.cm/u.s)    
    #---
    
    
    
    
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
    
    
    
    # get the slope via mini interpolation for some dt. 
    pwdotdot = (SB99f['fpdot_cgs'](t + 1e-9 * u.Myr) - SB99f['fpdot_cgs'](t - 1e-9 * u.Myr))/(1e-9+1e-9)
    
    
    
    
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




    # now try to work on this
    rootf_bd_res = find_root_betadelta.rootfinder_bd_wrap(beta_guess, delta_guess, params, Cool_Struc,
                                                              xtol=1e-5, verbose=0)  # xtol=1e-5
    
    # 
    
    
    beta = rootf_bd_res['beta']
    delta = rootf_bd_res['delta']
    residual = rootf_bd_res['residual']
    dMdt_factor = rootf_bd_res['dMdt_factor']
    Tavg = rootf_bd_res['Tavg']
    

    # TODO: isn't this 1e4?
    if Tavg.value > 1e3:
        cs_avg = np.sqrt(warpfield_params.gamma_adia * c.k_B * T /  warpfield_params.mu_p )
    else:
        cs_avg = np.sqrt(warpfield_params.gamma_adia * c.k_B * T /  warpfield_params.mu_n )


    # TODO: add verbosity
    print("time:", t, " rootfinder result (beta, delta):", beta, delta, "residual:", residual, 'Tavg', Tavg, 'cs', cs_avg)

    ############################## ODEs: energy and temperature ##############################
    
    
    
    
    
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
                  pwdot, pwdotdot,
                  r, v, E, T,
                  v_wind, L_wind)
    
    ##########################################################################################

    params["delta"] = delta  # Do I need this?
    params["beta"] = beta  # Do I need this?

    params["beta_guess"] = beta
    params["delta_guess"] = delta
    params['dMdt_factor'] = dMdt_factor
    params['cs_avg'] = cs_avg # dirty way to get cs_avg at the end of phase_energy

    derivs = [rd, vd, Ed, Td]  # list of dy/dt=f functions
    return derivs





