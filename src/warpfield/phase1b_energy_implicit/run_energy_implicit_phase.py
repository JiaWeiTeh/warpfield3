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
import src.warpfield.cooling.read_opiate as read_opiate
import src.warpfield.phase1b_energy_implicit.find_root_betadelta as find_root_betadelta


# get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()

#relative tolerance of stiffest ODE solver (energy phase),decrease to 1e-4 if you have crashes --> slower but more stable
rtol=1e-3 


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

    # define ODE which shall be solved
    ODE_fun = lambda t, y: fE_tot(t, y, params, ODEpar, SB99f)

    # list of events which will cause the solver to terminate (see my_events.py to see what they are)
    event_fun1 = lambda t, y: phase_events.event_StopTime(t, y, ODEpar['tStop']); event_fun1.terminal = True # End time reached
    event_fun2 = lambda t, y: phase_events.event_Radius1(t, y); event_fun2.terminal = True; event_fun2.direction = -1.0 # Small radius (rcoll) reached and collapsing
    event_fun3 = lambda t, y: phase_events.event_Radius1000(t, y); event_fun3.terminal = True # Large radius (rstop) reached
    event_fun4 = lambda t, y: phase_events.event_grav_frag(t,y,ODEpar,SB99f); event_fun4.terminal = True # graviational fragmentation
    event_fun5 = lambda t, y: phase_events.event_RTinstab(t, y, ODEpar, SB99f); event_fun5.terminal = True; event_fun5.direction = 1.0 # Rayleigh-Taylor instability
    event_fun6 = lambda t, y: phase_events.event_inhom_frag(t, y, ODEpar); event_fun6.terminal = True # fragmentation due to density inhomogeneities
    event_fun7 = lambda t, y: phase_events.event_vel0(t, y); event_fun7.terminal = False; event_fun7.direction = -1.0 # velocity switches from + to -, i.e. onset of collapse (no termination but small time steps)
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

    r, v, E, T = y  # unpack current values of y (r, rdot, E, T)

    #start_time = time.time()

    # sanity check: energy should not be negative!
    if E < 0.0:
        sys.exit("Energy is negative in ODEs.py")

    #end = time.time()
    #print(end - start_time, '######')

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
    part1_dict = phase_ODEs.fE_tot_part1(t, y, ODEpar, SB99f)
    vd = part1_dict['vd'] # acceleration
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


#params= {'y0': [-0.014501362193203233, 0.0010269386629510654], 'verbose': 0, 'R2': 71.41447228233649, 'vw': 2317.3857284642586, 'T0': 3333616.5343592316, 'Lw': 19326348107.83083, 'beta': 1.403895313722844, 'dMdt_factor': 3.098840695731767, 'delta': -0.18683748074949244, 'Qi': 1.9407983718388162e+66, 'alpha': 0.54793390461244773, 'x0': [1.4695022708395618, -0.22908828431917563], 'delta_guess': -0.18683748074949244, 'v2': 13.851913627803652, 'Eb': 10899465469.846792, 'temp_counter': 515, 't_now': 2.8249100950900554, 'beta_guess': 1.403895313722844}
#params = {'vw_dot': -139.05973816984618, 'verbose': 0, 'R2': 76.643063451976246, 'alpha': 0.52505779894047266, 'delta': 1.7807461777943547, 'v2': 12.479379138498963, 'T0': 3682066.6614771187, 'Eb': 13577422196.932648, 'Lw': 34552697654.783737, 'beta': -2.4688130653215916, 'temp_counter': 660, 'dMdt_factor': 3.098840695731767, 'vw': 2072.5403838934139, 'Qi': 1.9746696043917723e+66, 'y0': [0.056824731946308042, 0.003507493745658907], 'x0': [-2.4688130653215916, 1.7807461777943547], 'delta_guess': 1.7807461777943547, 't_now': 3.22468271486381, 'beta_guess': -2.4688130653215916, 'Lw_dot': 4796960930.4576063}
#params = {'vw_dot': -138.25808549506755, 'verbose': 0, 'R2': 76.815751258826992, 'alpha': 0.52518430416022577, 'delta': 1.7807461777943547, 'v2': 12.456938441735947, 'T0': 3710359.153428053, 'Eb': 13823801886.693964, 'Lw': 34619221900.442177, 'beta': -2.4688130653215916, 'temp_counter': 661, 'dMdt_factor': 3.098840695731767, 'vw': 2070.5364667044619, 'Qi': 1.9797469432295774e+66, 'y0': [0.089477076049725282, 0.0073531155219712799], 'x0': [-2.4688130653215916, 1.7807461777943547], 'delta_guess': 1.7807461777943547, 't_now': 3.2385507130907913, 'beta_guess': -2.4688130653215916, 'Lw_dot': 4796960930.4576063}
#params = {'verbose': 0, 'R2': 75.796144075511322, 'alpha': 0.52599324413454196, 'delta': 1.8184745577788339, 'v2': 12.627909762265309, 'T0': 3544673.1508749486, 'Eb': 12497770202.625795, 'Lw': 31208640760.426952, 'beta': -2.1193139891637749, 'temp_counter': 639, 'dMdt_factor': 3.098840695731767, 'vw': 2092.9728078696353, 'Qi': 1.9621299431911798e+66, 'y0': [-0.0013079636814639886, 0.00017068661822885583], 'x0': [-2.1141859778044267, 1.8186173109498853], 'delta_guess': 1.8184745577788339, 't_now': 3.157154308649052, 'beta_guess': -2.1193139891637749}
#params = {'y0': [0.0014412285478598286, 0.00019391337032868196], 'verbose': 0, 'R2': 71.881764525211992, 'vw': 2293.8031087326267, 'T0': 3330387.940031637, 'Lw': 19280971391.193981, 'beta': 1.3422470326532401, 'dMdt_factor': 3.098840695731767, 'delta': -0.16037824824039973, 'Qi': 1.9271940782382083e+66, 'alpha': 0.54551050031313431, 'x0': [1.3429512897773228, -0.15626435319179768], 'delta_guess': -0.16037824824039973, 'v2': 13.715460271911166, 'Eb': 10955400977.881248, 'temp_counter': 518, 't_now': 2.8589822399066533, 'beta_guess': 1.3422470326532401}
#params = {'xtolbstrux': 1e-06, 'verbose': 0, 'R2': 83.990676008166176, 'alpha': 0.59134057156273556, 'delta': 0.23455464102966958, 'v2': 13.033695095558945, 'T0': 4117884.9899935406, 'Eb': 23751019247.372181, 'Lw': 48573082058.893364, 'beta': -1.2734813494394182, 'temp_counter': 746, 'pwdot_dot': -33382627.877265703, 'dMdt_factor': 5.0717365709593603, 'vw': 2973.9198146843692, 'Qi': 1.7746661407535445e+66, 'y0': [0.039672342815200455, -0.00097634148811060319], 'x0': [-1.4284777065066112, 0.35510757737106646], 'delta_guess': 0.23455464102966958, 't_now': 3.810668731504462, 'beta_guess': -1.2734813494394182, 'pwdot': 32666033.441152863}
#params = {'y0': [0.0019895664871130026, -0.0012951081559506178], 'vw': 3597.4528490428524, 'verbose': 0, 'R2': 98.730352411275518, 'pwdot_dot': -8356217.9095687931, 'xtolbstrux': 1e-06, 'beta': 3.3600249357392946, 'T0': 3594882.1102034235, 'Eb': 27788618490.504257, 'Lw': 34778231857.611816, 'v2': 15.640792062791457, 'temp_counter': 885, 'dMdt_factor': 0.49768748738024776, 'delta': -1.3162659143942776, 'Qi': 9.5197789692970684e+65, 'alpha': 0.76291514822219897, 'x0': [3.1892937250907063, -1.1786291372529094], 'delta_guess': -1.3162659143942776, 't_now': 4.815797124690826, 'beta_guess': 3.3600249357392946, 'pwdot': 19334920.187690578}
#params = {'y0': [0.013522569035624791, -0.0018930136704551209], 'verbose': 0, 'R2': 106.53866968925023, 'xtolbstrux': 1e-06, 'T0': 6528248.1006543254, 'Lw': 33235008043.672428, 'beta': 8.2714497096918969, 'dMdt_factor': 0.087226276052106477, 'delta': -6.0047669481648507, 'Qi': 6.9437869960850624e+65, 'alpha': 0.76335489476573049, 'x0': [8.1152316749720175, -5.849399033977539], 'delta_guess': -6.0047669481648507, 'vw': 4033.407237905898, 'pwdot': 16479867.309866628, 'pwdot_dot': -1969681.3317389158, 'v2': 15.283727988676521, 'Eb': 24304930575.900406, 'temp_counter': 969, 't_now': 5.321137293818126, 'beta_guess': 8.2714497096918969}
#params = {'y0': [0.0075069861994037663, -0.00078619435076594595], 'verbose': 0, 'R2': 42.409186632368879, 'xtolbstrux': 1e-06, 'T0': 8773818.8619661443, 'Lw': 22972229030.462837, 'beta': 0.7269893727706036, 'dMdt_factor': 2.1591821446649067, 'delta': -0.10672741656501269, 'Qi': 1.8196845188363419e+66, 'alpha': 0.5321733555749345, 'x0': [0.71588923199186516, -0.10641133769322317], 'delta_guess': -0.10672741656501269, 'vw': 3446.1844506869816, 'pwdot': 13331978.806812517, 'pwdot_dot': 3040192.7842912367, 'v2': 20.478115769115963, 'Eb': 6082531163.6190405, 'temp_counter': 351, 't_now': 1.1021052626037433, 'beta_guess': 0.7269893727706036}
#params = {'y0': [0.0037645949748241017, 0.00025589201390689541], 'verbose': 0, 'R2': 89.721170080394216, 'xtolbstrux': 1e-06, 'T0': 8621469.2895822171, 'Lw': 40846529543.432907, 'beta': 1.4113428380769864, 'dMdt_factor': 2.6859422094072403, 'delta': -0.78130374512574507, 'Qi': 1.3638344250101388e+66, 'alpha': 0.67933519401118203, 'x0': [1.3987431719962664, -0.75894385155903132], 'delta_guess': -0.78130374512574507, 'vw': 3268.0478392375089, 'pwdot': 24997510.166780848, 'pwdot_dot': -8702428.739861276, 'v2': 14.427754123374765,  'Eb': 27811582842.787846, 'temp_counter': 812, 't_now': 4.2245486000296495, 'beta_guess': 1.4113428380769864}
#params = {'y0': [-0.0013016640352302919, -0.00029710325152944268], 'verbose': 0, 'R2': 106.76976513042426, 'xtolbstrux': 1e-06, 'T0': 6403643.9377199756, 'Lw': 33235008043.672428, 'beta': 9.0201300908554725, 'dMdt_factor': 0.077752103511832565, 'delta': -6.7487274171634999, 'Qi': 6.8845962564198965e+65, 'alpha': 0.76198273655410487, 'x0': [8.9966216614040224, -6.7255090445859107], 'delta_guess': -6.7487274171634999, 'vw': 4040.7207972109613, 'pwdot': 16450039.342788706, 'pwdot_dot': -1969681.3317389158, 'v2': 15.245958800076451, 'Eb': 23837237167.115314, 'temp_counter': 973, 't_now': 5.3362808388877285, 'beta_guess': 9.0201300908554725}
#params = {'y0': [-0.00048556210169586661, 6.6110021875087464e-05], 'verbose': 0, 'R2': 45.23143707160456, 'xtolbstrux': 1e-06, 'T0': 8667174.4899557065, 'Lw': 23403179089.840622, 'beta': 0.73101003052543667, 'dMdt_factor': 2.1400361502832124, 'delta': -0.10314946099413259, 'Qi': 1.8506614444204328e+66, 'alpha': 0.53781183990572146, 'x0': [0.7318518741545752, -0.10304961090137868], 'delta_guess': -0.10314946099413259, 'vw': 3395.6126887526066, 'pwdot': 13784363.079664355, 'pwdot_dot': 3505869.8380179787, 'v2': 19.563265796298321, 'Eb': 6753897041.4320469, 'temp_counter': 354, 't_now': 1.2434530433902486, 'beta_guess': 0.73101003052543667} # shortly before fragmentation
#params = {'y0': [0.0052484615688093801, -2.1823713130893917e-06], 'verbose': 0, 'R2': 2.5701214306146456, 'xtolbstrux': 1e-06, 'T0': 17890406.831923556, 'Lw': 20164886794.262321, 'beta': 0.83454348386182775, 'dMdt_factor': 3.3458891366989483, 'delta': -0.23240325442412141, 'Qi': 1.6994584609226494e+66, 'alpha': 0.58943168110324184, 'x0': [0.83450769255352775, -0.23237674301529296], 'delta_guess': -0.23240325442412141, 'vw': 3810.2196569216962, 'pwdot': 10584632.178688448, 'pwdot_dot': 0.0, 'v2': 195.77757781438393, 'Eb': 62744610.157777384, 'temp_counter': 40, 't_now': 0.007737918777005923, 'beta_guess': 0.83454348386182775}
#params = {'y0': [4.8621386008658503, 0.016112902526489135], 'verbose': 0, 'R2': 2.522837096049054, 'xtolbstrux': 1e-06, 'T0': 18097032.235924549, 'Lw': 20164886794.262321, 'beta': 0.85563950441516556, 'dMdt_factor': 3.4243249806188159, 'delta': -0.27203804054206931, 'Qi': 1.6994584609226494e+66, 'alpha': 0.59111566904138357, 'x0': [0.83, -0.13], 'delta_guess': -0.27203804054206931, 'vw': 3810.2196569216962, 'pwdot': 10584632.178688448, 'pwdot_dot': 0.0, 'v2': 198.94268835986321, 'Eb': 60866738.387646228, 'temp_counter': 2, 't_now': 0.007496071105744271, 'beta_guess': 0.85563950441516556}
#params = {'R2': 2.4765607555888711, 'xtolbstrux': 1e-06, 'T0': 18171371.296796411, 'Lw': 20164886794.262321, 'beta': 0.850667548283903, 'dMdt_factor': 3.2059994914363994, 'delta': -0.16663474473008064, 'Qi': 1.6994584609226494e+66, 'alpha': 0.59377786703394153, 'delta_guess': -0.13, 'vw': 3810.2196569216962, 'pwdot': 10584632.178688448, 'pwdot_dot': 0.0, 'v2': 202.34644016802042, 'Eb': 59000964.471560813, 'temp_counter': 1, 't_now': 0.007267372540937511, 'beta_guess': 0.83}
#params = {'dMdt_factor': 3.2157681136498444, 'beta_guess':0.85, 'delta_guess':-0.16, 'beta': 0.85070457979890945, 't_now': 0.005892590924965079, 'Lw': 20164886794.262321, 'temp_counter': 0, 'R2': 2.195077717684434, 'T0': 18627262.584549118, 'delta': -0.16239811123695733, 'alpha': 0.59378359250617363, 'Eb': 47623832.773869388}



