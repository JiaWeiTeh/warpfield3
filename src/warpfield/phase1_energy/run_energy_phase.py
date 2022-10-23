#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 23:16:58 2022

@author: Jia Wei Teh

"""
# libraries
import numpy as np
import astropy.units as u
import astropy.constants as c
import scipy.interpolate
import sys
import scipy.optimize
#--
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams
import src.warpfield.bubble_structure.bubble_structure as bubble_structure
import src.warpfield.shell_structure.shell_structure as shell_structure
import src.warpfield.cloud_properties.mass_profile as mass_profile
from src.warpfield.cooling.get_coolingFunction import find_nearest_lower

def run_energy(t0, y0, #r0, v0, E0, T0
        rCloud, mCloud, sfe, mCluster, nEdge, rCore, 
        tcoll, coll_counter,
        density_specific_param,
        warpfield_params,
        Cool_Struc,
        shell_dissolved,
        stellar_outputs, # old code: SB99_data
        t_shelldiss,
        
        
        # TODO: make it so that the function does not depend on these constants,
        # but rather simple call them by invoking the param class (i.e., params.rCloud).
        
        ):
    
    
    # the energy-driven phase
    # winds hit the shell --> reverse shock --> thermalization
    # shell is driven mostly by the high thermal pressure by the shocked ISM, also (though weaker) by the radiation pressure, at late times also SNe

    print("############################################################")
    print("           Entering phase I (energy-driven)...")
    print("############################################################")
    

    mypath = warpfield_params.out_dir

    # first stopping time (will be incremented at beginning of while loop)
    # start time t0 will be incremented at end of while loop
    tStop_i = t0
    # get data from stellar evolution code output
    # unit of t_evo is Myr, the other units are cgs
    t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo = stellar_outputs 

    # interpolation functions for SB99 values
    fQi_evo = scipy.interpolate.interp1d(t_evo, Qi_evo, kind = 'linear')
    fLi_evo = scipy.interpolate.interp1d(t_evo, Li_evo, kind = 'linear')
    fLn_evo = scipy.interpolate.interp1d(t_evo, Ln_evo, kind = 'linear')
    fLbol_evo = scipy.interpolate.interp1d(t_evo, Lbol_evo, kind = 'linear')
    fLw_evo = scipy.interpolate.interp1d(t_evo, Lw_evo, kind = 'linear')
    fpdot_evo = scipy.interpolate.interp1d(t_evo, pdot_evo, kind = 'linear')

    # identify potentially problematic time steps (where feedback parameters change by a large percentage)
    # To do this, check slope
    dLwdt = (Lw_evo[1:] - Lw_evo[0:-1])/(t_evo[1:] - t_evo[0:-1])
    dLwdt = np.concatenate([dLwdt,[0.]])
    # absolute value of timestep, which we care
    abs_dLwdt = np.abs(dLwdt)
    # problematic time (which needs small timesteps) 
    # is defined by relative change in mechanical luminosity being more than 
    # 300% per Myr (if number on RHS is 3.0)
    t_problem = t_evo[abs_dLwdt / Lw_evo > 3.] 
    

    # initial mechanical luminosity in astro units
    Lw0 = fLw_evo(t0) * (u.g.to(u.Msun) * u.cm.to(u.pc)**2/u.s.to(u.Myr)**3)
    # initial momentum of stellar winds in astro units
    pdot0 = fpdot_evo(t0) * (u.g.to(u.Msun) * u.cm.to(u.km) / u.s.to(u.Myr))
    # initial terminal wind velocity
    vterminal0 = 2. * Lw0 / pdot0 

    # initial values (radius, velocity, energy, temperature)
    if len(y0) == 3:
        r0, v0, E0 = y0
        # initial temperature of bubble
        # see Weaver+77, eq. (37)
        T0 = 1.51e6 * (Lw_evo[0]/1e36)**(8./35.) * warpfield_params.Core**(2./35.) *\
            (t0-tcoll[coll_counter])**(-6./35.) * (1.- warpfield_params.xi_Tb)**0.4 
    elif len(y0) == 4:
        r0, v0, E0, T0 = y0

    # initial radius of inner discontinuity
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 1e-3 * r0, r0, 
                               args=([Lw0, E0, vterminal0, r0]),maxiter=1000) 
    
    # initial energy derivative
    Ebd0 = 0. 
    E0m1 = 0.9*E0
    t0m1 = 0.9*t0
    r0m1 = 0.9*r0
    
    # calculate swept mass depending on density profile.
    # watch out units
    Msh0 = mass_profile.get_mass_profile(r0, density_specific_param, rCloud, mCloud, warpfield_params)
    
    # The initial bubble pressure
    P0 = get_bubbleParams.bubble_E2P(E0, R1, r0, warpfield_params.gamma_adia)

    # initialize r and derivatives
    tSweaver = []; rSweaver = []; vSweaver = []; ESweaver = []
    fabsweaver = []; fabs_i_weaver = []; fabs_n_weaver = []; ionshweaver = []; Mshell_weaver = []
    FSgrav_weaver = []; FSwind_weaver = []; FSradp_weaver = []; FSsne_weaver = []; FSIR_weaver = []; dRs_weaver = []; nmax_weaver = []
    n0weaver = []; n0_cloudyweaver = []; logMcluster_weaver = []; logMcloud_weaver = []; phase_weaver = []; R1weaver = []; Ebweaver = []; Pbweaver = []
    Lbweaver = []; Lwweaver = []; Tbweaver = []; alphaweaver = []; betaweaver = []; deltaweaver = []
    Lbbweaver = []; Lbczweaver = []; Lb3weaver = []; fragweaver = [];

    # check how long to stay in Weaver phase
    if warpfield_params.adiabaticOnlyInCore: # old code: adiabatic_core_only
        rfinal = rCore
    else:
        rfinal = rCloud
 
    # Now, we define some start values
    alpha = 0.6 # 3/5
    beta = 0.8 # 4/5
    # taking more negative value seems to be better for massive clusters (start further out of equilibrium?)
    if fLw_evo(t0) < 1e40: 
        delta = -6./35. 
        # -6./35.=-0.171 
    else: delta = -6./35. # -0.5
    # TOASK: Double confirm all these parameters
    temp_counter = 0
    dt_L = 1e-4
    frag_value = 0.
    t_frag = 1e99
    tscr = 1e99
    dMdt_factor = 1.646
    mom_phase = False
    first_frag = True
    Lw_old = Lw0
    delta_old = delta
    cf0 = 1.0
    frag_value0 = 0.
    dfragdt = 0.
    was_close_to_frag = False
    frag_stop = 1.0
    fit_len = 13
    dt_real = 1e-4
    Lres0 = 1.0
    
    
    dt_Emax = 0.04166
    dt_Emin = 1e-5
    dt_Estart = 1e-4
    
    tfinal = warpfield_params.stop_t
    fit_len_max = 13
    fit_len_min = 7
    lum_error = 0.005
    lum_error2 = 0.005
    delta_error = 0.03
    dt_switchon = 0.001

    t_10list = np.array([])
    r_10list = np.array([])
    P_10list = np.array([])
    T_10list = np.array([])

    continueWeaver = True
    
    while all([r0 < rfinal, (tfinal - t0) > dt_Emin, continueWeaver]):

        # calculate bubble structure and shell structure?
        # r0 > 0.1, Do I need only temp_counter > 0?
        structure_switch = (temp_counter > 0) 
        if structure_switch is not True:
            dt_Emin = dt_Emin
        else:
            dt_Emin = dt_Estart

        fit_len_interp = fit_len_max + (np.log(dt_real) - np.log(dt_Emin)) * (fit_len_min - fit_len_max) / (np.log(dt_Emax) - np.log(dt_Emin))
        # used for delta, and in the beginning also for alpha, beta (min 3, max 10 or so)
        fit_len = int(np.round(np.min([fit_len_max, fit_len + 1, np.max([fit_len_min, fit_len_interp])])))  
        
        
        def del_append(mylist, value, maxlen = 10):
            """
            appends value to end of list and removes first element of list if list would become 
            longer than a given maximum length.
            """
            mylist = np.append(mylist, value)
            while (len(mylist) > maxlen):
                mylist = np.delete(mylist, 0)
            return  mylist

        # time list and temperature list (coarse grid) for calculation of delta
        t_10list = del_append(t_10list, t0-tcoll[coll_counter], maxlen=fit_len)
        r_10list = del_append(r_10list, r0, maxlen=fit_len)
        P_10list = del_append(P_10list, P0, maxlen=fit_len)
        if temp_counter >= 1:
            # is shifted 1 time step
            T_10list = del_append(T_10list, T0, maxlen=fit_len - 1)  

        # for early time (1st Myr) use a small cloudy dt
        if (t0 - tcoll[coll_counter]) < 1:
            my_cloudy_dt = 0.1 # 0.1 Myr
        else:
            my_cloudy_dt = 0.5 # 0.5 Myr
            
        t_cloudy = np.ceil(t0/my_cloudy_dt) * my_cloudy_dt
        # set tmax for this time step according to cloudy dump time
        tmax = tfinal
        # calculate time step
        # in the very beginning (and around first SNe) decrease tCheck because fabs varys a lot
        # increment stopping time

        ii_dLwdt = find_nearest_lower(t_evo,t0)
        if abs_dLwdt[ii_dLwdt] > 0.:
            dt_Lw = 0.94 * lum_error * Lw_old /abs_dLwdt[ii_dLwdt] / (u.g.to(u.Msun) * u.cm.to(u.pc)**2/u.s.to(u.Myr)**3)
        else:
            dt_Lw = dt_Emax

        # make sure, that only 1 cloudy output is written per time step (if necessary, decrease time step)
        # TODO
        # if (i.write_cloudy is True):
        #     tStop_i = np.min([tmax,t0+np.min([dt_L,my_cloudy_dt,dt_Emax,dt_Lw])]) # this is the stop time for this time step
        # else:
        tStop_i = np.min([tmax,t0+np.min([dt_L,dt_Emax,dt_Lw])])

        # find nearest problematic neighbor of current stop time
        t_problem_hi = t_problem[t_problem > (t0 + 0.001*dt_Emin)]
        t_problem_nnhi = t_problem_hi[0] # the next problematic time
        tStop_i = np.min([tStop_i, (t_problem_nnhi - float(fit_len_max+1)*dt_Emin)])
        # ensure that time step becomes not too small or negative
        if tStop_i < (t0 + dt_Emin):
            tStop_i = t0 + dt_Emin

        if ((frag_value > frag_stop) and (frag_value < 1.0)):
            tStop_i = t0 + dt_Emin

        dt_real = tStop_i - t0

        if (temp_counter == 2. * fit_len): dt_real = dt_Emin

        # correct (?) calculation of beta
        beta_temp = -t0 / P0 * 1. / (2. * np.pi) * (Ebd0 / (r0 ** 3 - R1 ** 3) - 3. * E0 * v0 * r0 ** 2 / (r0 ** 3 - R1 ** 3) ** 2.)
        alpha_temp = t0 / r0 * v0

        # before this time step is accepted, check how the mechanical luminosity would change
        # if it would change too much, reduce time step
        while True:

            t_inc = dt_real/1000.
            # time vector for this time step
            # make sure to include tStop_i here, i.e. go one small dt farther
            t_temp = np.arange(t0, tStop_i + t_inc, t_inc)  

            # insert a value close to t[-1]
            val = t_temp[-1] - 9.9147e-6
            if val > t_temp[0]:
                idx_val = np.searchsorted(t_temp, val)
                t = np.insert(t_temp,idx_val,val)
            else:
                idx_val = 0
                t = t_temp
            # midpoint time, evaluate feedback parameters here
            thalf = 0.5 * (t[0]+t[-1])

            # get SB99 values
            # TODO: add option for manual input
            # if i.SB99 == True:
            #     if (thalf > t_evo[-1]):
            #         print("Warning: End of SB99 file reached")
            #         sys.exit("Stop Code: Feedback parameters not provided")

            # convert cgs to astro units (Myr, Msun, pc)
            Lw = fLw_evo(thalf)  *(u.g.to(u.Msun) * u.cm.to(u.pc)**2/u.s.to(u.Myr)**3)
            Lw_temp = fLw_evo(tStop_i)  *(u.g.to(u.Msun) * u.cm.to(u.pc)**2/u.s.to(u.Myr)**3)
            Lbol = fLbol_evo(thalf)  *(u.g.to(u.Msun) * u.cm.to(u.pc)**2/u.s.to(u.Myr)**3)
            pdot= fpdot_evo(thalf)         * (u.g.to(u.Msun) * u.cm.to(u.km) / u.s.to(u.Myr))
            vterminal = 2. * Lw / pdot

            # if mechanical luminosity would change too much, run through this loop again with reduced time step
            # in that case, the following condition is not fulfilled

            if (((abs(Lw_temp - Lw_old) < lum_error*Lw_old) and (abs(Lw - Lw_old) < lum_error*Lw_old)) or (dt_real < 2*dt_Emin)):
                # if mechanical luminosity does not change much, check how much delta would change

                if temp_counter < 1e30*fit_len: # 2.*fit_len
                    # TODO uncomment for more info
                    # aux.printl("entering old bubble_wrap...", verbose=1)
                    structure_bubble = (structure_switch and not mom_phase)
                    bubble_wrap_struc = {'structure_switch': structure_bubble, 'alpha': alpha, 'beta': beta, 'delta': delta,
                                         'Lres0': Lres0, 't_10list': t_10list, 'r_10list': r_10list, 'P_10list': P_10list,
                                         'T_10list': T_10list, 'Lw': Lw, 'vterminal': vterminal, 'r0': r0,
                                         't0': t0 - tcoll[coll_counter], 'E0': E0,
                                         'T0': T0, 'dt_L': dt_real, 'temp_counter': temp_counter, 'dMdt_factor': dMdt_factor,
                                         'Qi': fQi_evo(thalf)*u.Myr.to(u.s), 'mypath': mypath}

                    # it's important to use the time since the last restarting expansion, not the time since the start of the simulation
                    # calculate bubble structure
                    bubbleFailed, Lb, T0, alpha, beta, delta, dt_L, Lb_b, Lb_cz, Lb3, dMdt_factor, Tavg, Mbubble, r_Phi_b, Phi_grav_r0b, f_grav_b = bubble_structure.get_bubbleStructure(bubble_wrap_struc, Cool_Struc, warpfield_params, fit_len=fit_len, fit_len_short=fit_len)
                
                else:
                    print("entering delta_new_root...")
                    alpha = alpha_temp
                    beta = beta_temp
                    # temporary ###########
                    param1 = {'alpha': alpha, 'beta': beta, 'Eb': E0, 'R2': r0, 't_now': t0, 'Lw': Lw, 'vw': vterminal, 'dMdt_factor': dMdt_factor, 'Qi': fQi_evo(thalf) * c.Myr, 'mypath': mypath}
                    param0 = {'alpha': alpha, 'beta': beta, 'Eb': E0m1, 'R2': r0m1, 't_now': t0m1, 'Lw': Lw, 'vw': vterminal, 'dMdt_factor': dMdt_factor, 'Qi': fQi_evo(thalf) * c.Myr, 'mypath': mypath}
                    dzero_params = [param0, param1, Cool_Struc]
                    delta, bubbleFailed = get_bubbleParams.get_delta_new(delta, dzero_params)
                    param1["delta"] = delta
                    Lb, T_rgoal, Lb_b, Lb_cz, Lb3, dMdt_factor, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav = bubble_structure.calc_Lb(param1, Cool_Struc, temp_counter)
                    # new time step
                    Lres_temp = Lw - Lb
                    fac = np.max([np.min([lum_error2 / (np.abs(Lres_temp - Lres0) / Lw), 1.42]), 0.1])
                    dt_L = fac * dt_L  # 3 per cent change
                    param1['dt_L'] = dt_L
                    param1['T0'] = T_rgoal
                    param1['temp_counter'] = temp_counter
                    param1['Lres0'] = Lw - Lb

                # average sound speed in bubble
                cs_avg = np.sqrt(2.*c.gamma*c.kboltz_au*Tavg*c.Msun/c.mp)

                # in the very early energy-phase, it is not important to get it right
                # only leave loop if bubble_structure did not throw errors (as measured by bubbleFailed) and delta did not change a lot
                # OR if time step is already close to the lower limit
                if (((bubbleFailed == False) and ((temp_counter < 2*fit_len) or (abs(delta-delta_old) < delta_error))) or (dt_real < 2*dt_Emin)):
                    break # everything is good, don't change time step any more, leave "while(True)"-loop

            # not everything is good --> reduce time step in next iteration of loop
            dt_real = 0.5*dt_real
            tStop_i = t0 + dt_real

        if bubbleFailed:
            print("Could not figure out the correct delta. Worrisome!")

        # gradually switch on cooling (i.e. reduce the amount of cooling at very early times)
        if (t0-tcoll[coll_counter] < dt_switchon):
            reduce_factor = np.max([0.,np.min([1.,(t0-tcoll[coll_counter])/dt_switchon])])
            Lb *= reduce_factor # scale and make sure it is between 0 and 1
            Lb_b *= reduce_factor
            Lb_cz *= reduce_factor
            Lb3 *= reduce_factor



        ################ CLOUDY #############################################################################################
        # TODO
        # make_cloudy_model = (i.write_cloudy and t0 <= t_cloudy and tStop_i >= t_cloudy)
        #####################################################################################################################

        if structure_switch:
            # shell structure routine wants cgs units
            np.seterr(all='warn')
            fabs_i, fabs_n, fabs, fion_dust, ionsh, dRs, n0, nmax, rhodr, n0_cloudy, r_Phi_sh, Phi_grav_r0s, f_grav_sh = shell_structure.shell_structure(r0, P0, Mbubble * c.M_sun.cgs.value,
                                                                                                                                                          fLn_evo(thalf), fLi_evo(thalf), fQi_evo(thalf),
                                                                                                                                                          Msh0, sigma_dust, 1, warpfield_params)
            r_Phi = np.concatenate([r_Phi_b, r_Phi_sh])
            Phi_grav_r0 = Phi_grav_r0b + Phi_grav_r0s # this is the integral from r0 to rsh_out over r*rho(r); the integral part from rsh_out to very far out will be done in write_pot_to_file
            f_grav = np.concatenate([f_grav_b, f_grav_sh])
            
            # TODO
            # if i.write_potential is True:
            #     aux.write_pot_to_file(mypath, t0, r_Phi, Phi_grav_r0, f_grav, rcloud_au, rcore_au, nalpha, Mcloud_au, Mcluster_au, SFE)

        else:
            # If it is very early, no shell has yet been swept. Set some default values!
            fabs = 0.0
            fabs_i = 0.0
            fabs_n = 0.0
            ionsh = False
            dRs = 0.0
            nmax = 1e5
            n0 = warpfield_params.nCore
            n0_cloudy = n0
            Lb = 0.
        if shell_dissolved == False and nmax < warpfield_params.nISM:
            shell_dissolved = True
            t_shdis = t0

        # now solve eq. of motion
        # this part of the routine assumes astro-units

        # wind luminosity
        LW = Lw
        # wind pdot
        PWDOT = pdot
        # adiamatic constant
        GAM = warpfield_params.gamma_adia
        # mass density of ambient medium
        RHOA = i.rhoa_au
        RCORE = rcore_au
        A_EXP = nalpha
        MSTAR = Mcluster_au
        LB = Lb # astro units!!
        FRAD = fabs * Lbol/c.clight_au # astro units
        CS = cs_avg

        # inside core or inside density slope?
        if (r0 < rcore_au or not i.density_gradient):
            phase0 = ph.weaver
        else:
            phase0 = ph.Egrad


        # Bundle initial conditions for ODE solver
        y0 = [r0, v0, E0]

        # bundle parameters for ODE solver
        params = [LW, PWDOT, GAM, Mcloud_au, RHOA, RCORE, A_EXP, MSTAR, LB, FRAD, fabs_i, rcloud_au, phase0, tcoll[coll_counter], t_frag, tscr, CS, SFE]
        # aux.printl(("params", params), verbose=1)

        # call ODE solver
        try:
            psoln = scipy.integrate.odeint(ODEs.fE_gen, y0, t, args=(params,))
        except:
            sys.exit("EoM")

        # get r, rdot and rdotdot
        r = psoln[:,0]
        rd = psoln[:, 1]
        Eb = psoln[:, 2]
        if i.dens_profile == "powerlaw":
            #print('##############*************************#############r=',r)
            Msh = mass_profile.calc_mass(r, rcore_au, rcloud_au, i.rhoa_au, i.rho_intercl_au, nalpha, Mcloud_au)
            #print('##############*************************#############MshzuR=',Msh)
        
        elif i.dens_profile == "BonnorEbert":
            T_BE=rcore_au
            #print('##############*************************#############r=',r)
            Msh = mass_profile.calc_mass_BE(r, i.rhoa_au, T_BE, i.rho_intercl_au, rcloud_au, Mcloud_au)
            #print('##############*************************#############MshzuR=',Msh)

        """
        ################ CLOUDY #############################################################################################
        # if this is the active branch store data in cloudy input file
        if make_cloudy_model:
            #print("write cloudy model at t=", t_cloudy)
            jj_cloudy = find_nearest_higher(t, t_cloudy)
            r_cloudy = r[jj_cloudy]
            v_cloudy = rd[jj_cloudy]
            Msh_cloudy = Msh[jj_cloudy]
            E_cloudy = Eb[jj_cloudy]
            Lw_cloudy = fLw_evo(t_cloudy)         /c.L_cgs
            pdot_cloudy= fpdot_evo(t_cloudy)         * c.Myr / (c.Msun * c.kms)
            vterminal_cloudy = 2. * Lw_cloudy / pdot_cloudy
            R1_cloudy = scipy.optimize.brentq(bubble_structure.R1_zero, 1e-3 * r0, r0, args=([Lw_cloudy, E_cloudy, vterminal_cloudy, r_cloudy]))
            P_cloudy = state_eq.PfromE(E_cloudy, r_cloudy, R1_cloudy)
            [n0_temp, n0_cloudy] = n_from_press(P_cloudy, i.Ti, B_cloudy=i.B_cloudy)
            create_model(cloudypath, SFE, Mcloud_au, i.namb, i.Zism, n0_cloudy, r_cloudy, v_cloudy, Msh_cloudy,
                         np.log10(fLbol_evo(t_cloudy)), t_cloudy, rcloud_au, nedge,
                         SB99model=i.SB99cloudy_file, shell=i.cloudy_stopmass_shell, turb=i.cloudy_turb,
                         coll_counter=coll_counter, Tarr=Tarr, Larr=Larr, Li=Li*c.L_cgs, Qi=fQi_evo(thalf), Mcluster=Mcluster_au, phase=phase0)
        #####################################################################################################################
        """

        # check whether shell fragments or similar

        # if a certain radius has been exceeded, stop branch
        if (r[-1] > rfinal or (r0 < rcore_au and r[-1] >= rcore_au and i.density_gradient)):
            if r[-1] > rfinal:
                continue_branch = False
                include_list = r<rfinal
            else:
                continue_branch = True
                rtmp = r[r>=rcore_au][0] # find first radius larger than core radius
                include_list = r <= rtmp
            t = t[include_list]
            r = r[include_list]
            rd = rd[include_list]
            Eb = Eb[include_list]
            Msh = Msh[include_list]


        # calculate fragmentation time
        if fabs_i < 0.999:
            Tsh = i.Tn  # 100 K or so
        else:
            Tsh = i.Ti  # 1e4 K or so
        cs = aux.sound_speed(Tsh, unit="kms")  # sound speed in shell (if a part is neutral, take the lower sound speed)
        frag_list = i.frag_c * c.Grav_au * 3. * Msh / (4. * np.pi * rd * cs * r) # compare McCray and Kafatos 1987
        frag_value = frag_list[-1] # fragmentation occurs when this number is larger than 1.

        # frag value can jump from positive value directly to negative value (if velocity becomes negative) if time resolution is too coarse
        # however, before the velocity becomes negative, it would become 0 first. At v=0, fragmentation always occurs
        if frag_value < 0.:
            frag_value = 2.0
        # another way to estimate when fragmentation occurs: Raylor-Taylor instabilities, see Baumgartner 2013, eq. (48)
        # not implemented yet

        if (was_close_to_frag is False):
            frag_stop = 1.0 - float(fit_len)*dt_Emin*dfragdt # if close to fragmentation: take small time steps
            if (frag_value >= frag_stop):
                aux.printl("close to fragmentation", verbose=1)
                ii_fragstop = aux.find_nearest_higher(frag_list, frag_stop)
                if (ii_fragstop == 0): ii_fragstop = 1
                t = t[:ii_fragstop]
                r = r[:ii_fragstop]
                rd = rd[:ii_fragstop]
                Eb = Eb[:ii_fragstop]
                Msh = Msh[:ii_fragstop]
                frag_list = frag_list[:ii_fragstop]
                frag_value = frag_list[-1]
                dt_L = dt_Emin # reduce time step
                was_close_to_frag = True


        if ((frag_value > 1.0) and (first_frag is True)):
            #print frag_value
            # fragmentation occurs
            aux.printl("shell fragments", verbose=1)
            ii_frag = aux.find_nearest_higher(frag_list, 1.0)  # index when fragmentation starts #debugging
            if (ii_frag == 0): ii_frag = 1
            if frag_list[ii_frag] < 1.0:
                print(ii_frag, frag_list[0], frag_list[ii_frag], frag_list[-1])
                sys.exit("Fragmentation value does not match criterion!")
            t_frag = t[ii_frag]  # time when shell fragmentation starts
            tscr = r[ii_frag]/cs_avg # sound crossing time
            # if shell has just fragemented we need to delete the solution at times later than t_frag, since for those it was assumed that the shell had not fragmented
            t = t[:ii_frag]
            r = r[:ii_frag]
            rd = rd[:ii_frag]
            Eb = Eb[:ii_frag]
            Msh = Msh[:ii_frag]
            frag_list = frag_list[:ii_frag]
            frag_value = frag_list[-1]
            if i.output_verbosity >= 1: print(t_frag)
            first_frag = False
            # OPTION 1 for switching to mom-driving: if i.immediate_leak is set to True, we will immeadiately enter the momentum-driven phase now
            if i.immediate_leak is True:
                mom_phase = True
                continue_branch = False
                Eb[-1] = 0.

        # OPTION 2 for switching to mom-driving: if i.immediate_leak is set to False, when covering fraction drops below 50%, switch to momentum driving
        # (THIS APPROACH IS NOT STABLE YET)
        cf = ODEs.calc_coveringf(t, t_frag, tscr)
        if cf[-1] < 0.5:
            ii_cov50 = aux.find_nearest_lower(cf, 0.5)
            if (ii_cov50 == 0): ii_cov50 = 1
            if i.output_verbosity >= 1: print(cf[ii_cov50])
            t = t[:ii_cov50]
            r = r[:ii_cov50]
            rd = rd[:ii_cov50]
            Eb = Eb[:ii_cov50]
            Msh = Msh[:ii_cov50]
            mom_phase = True
            continue_branch = False
            Eb[-1] = 0.

        # store data

        # fine spacing
        rweaver_end = r[-1]
        vweaver_end = rd[-1]
        Eweaver_end = Eb[-1]
        tweaver_end = t[-1]

        # coarse spacing
        tSweaver = np.concatenate([tSweaver, [t0]])
        rSweaver = np.concatenate([rSweaver, [r0]])
        vSweaver = np.concatenate([vSweaver, [v0]])
        ESweaver = np.concatenate([ESweaver, [E0]])
        fabsweaver = np.concatenate([fabsweaver, [fabs]])
        fabs_i_weaver = np.concatenate([fabs_i_weaver, [fabs_i]])
        fabs_n_weaver = np.concatenate([fabs_n_weaver, [fabs_n]])
        ionshweaver = np.concatenate([ionshweaver, [ionsh]])
        Mshell_weaver = np.concatenate([Mshell_weaver, [Msh0]])
        dRs_weaver = np.concatenate([dRs_weaver, [dRs]])
        n0weaver = np.concatenate([n0weaver, [n0]])
        n0_cloudyweaver = np.concatenate([n0_cloudyweaver, [n0_cloudy]])
        FSgrav_weaver = np.concatenate([FSgrav_weaver, [1e-30]])
        FSwind_weaver = np.concatenate([FSwind_weaver, [1e-30]])
        FSradp_weaver = np.concatenate([FSradp_weaver, [1e-30]])
        FSsne_weaver = np.concatenate([FSsne_weaver, [1e-30]])
        FSIR_weaver = np.concatenate([FSIR_weaver, [1e-30]])
        nmax_weaver = np.concatenate([nmax_weaver, [nmax]])
        logMcluster_weaver = np.concatenate([logMcluster_weaver, [np.log10(Mcluster_au)]])
        logMcloud_weaver = np.concatenate([logMcloud_weaver, [np.log10(Mcloud_au)]])
        phase_weaver = np.concatenate([phase_weaver, [phase0]]) # set by hand (dangerous)
        R1weaver = np.concatenate([R1weaver, [R1]])
        Ebweaver = np.concatenate([Ebweaver, [E0]])
        Pbweaver = np.concatenate([Pbweaver, [P0]])
        Lwweaver = np.concatenate([Lwweaver, [Lw]])
        Lbweaver = np.concatenate([Lbweaver, [Lb]])
        Tbweaver = np.concatenate([Tbweaver, [T0]])
        alphaweaver = np.concatenate([alphaweaver, [alpha]])
        betaweaver = np.concatenate([betaweaver, [beta]])
        deltaweaver = np.concatenate([deltaweaver, [delta]])
        fragweaver = np.concatenate([fragweaver, [frag_value]])

        Lbbweaver = np.concatenate([Lbbweaver, [Lb_b]])
        Lbczweaver = np.concatenate([Lbczweaver, [Lb_cz]])
        Lb3weaver = np.concatenate([Lb3weaver, [Lb3]])

        Data_w = {'t':tSweaver, 'r':rSweaver, 'v':vSweaver, 'E':ESweaver, 't_end': tweaver_end, 'r_end':rweaver_end, 'v_end':vweaver_end, 'E_end':Eweaver_end,
                  'fabs':fabsweaver, 'fabs_n': fabs_n_weaver, 'Fgrav':FSgrav_weaver, 'Fwind':FSwind_weaver, 'Fradp_dir':FSradp_weaver, 'FSN':FSsne_weaver, 'Fradp_IR':FSIR_weaver,
                  'fabs_i':fabs_i_weaver, 'dRs':dRs_weaver, 'logMshell':np.log10(Mshell_weaver), 'nmax':nmax_weaver, 'logMcluster':logMcluster_weaver, 'logMcloud':logMcloud_weaver,
                  'phase': phase_weaver, 'R1':R1weaver, 'Eb':Ebweaver, 'Pb':Pbweaver, 'Lmech':Lwweaver, 'Lcool':Lbweaver, 'Tb':Tbweaver,
                  'alpha':alphaweaver, 'beta':betaweaver,'delta':deltaweaver, 'Lbb':Lbbweaver, 'Lbcz':Lbczweaver, 'Lb3':Lb3weaver, 'frag':fragweaver, 'dMdt_factor_end': dMdt_factor}

        Flux_Phi = fQi_evo(thalf)/(4.*np.pi*(r0*c.pc)**2)
        Pb_cgs = P0 /(c.pc*c.Myr**2/(c.Msun))
        min_Pb = np.min([1e-16*Flux_Phi**0.667, 1e-21*Flux_Phi])
        max_Pb = np.max([1e-16*Flux_Phi, 2e-11*Flux_Phi**0.667])

        #print '%.4e' % Flux_Phi,  '%.4e' % min_Pb, '%.4e' % Pb_cgs, '%.4e' % max_Pb
        if ((Pb_cgs < min_Pb) or (Pb_cgs > max_Pb)):
            print("Warning: might not have necessary cooling values tabulated. Let's see...")
            print('%.4e' % Flux_Phi, '%.4e' % min_Pb, '%.4e' % Pb_cgs, '%.4e' % max_Pb)

        # get new initial values
        t0 = t[-1]
        r0 = r[-1] # shell radius
        v0 = rd[-1] # shell velocity
        E0 = Eb[-1] # bubble energy
        Lw_new = (fLw_evo(t0) *(u.g.to(u.Msun) * u.cm.to(u.pc)**2/u.s.to(u.Myr)**3))
        vterminal0 = 2.*Lw_new / (fpdot_evo(t0) * c.Myr / c.Msun / c.kms)
        if (not mom_phase):
            R1 = scipy.optimize.brentq(bubble_structure.R1_zero, 1e-3*r0, r0, args=([Lw_new, E0, vterminal0, r0]))
            P0 = state_eq.PfromE(E0, r0, R1) # bubble pressure
        else:
            R1 = r0
            P0 = state_eq.Pram(r0,Lw_new,vterminal0)
        Msh0 = Msh[-1] # shell mass
        Lres0 = LW - LB
        Lw_old = Lw
        delta_old = delta
        cf0 = cf[-1]
        dfragdt = (frag_value - frag_value0)/(t[-1]-t[0])
        frag_value0 = frag_value

        if idx_val >= len(t): idx_val = -2

        t0m1 = t[idx_val]
        r0m1 = r[idx_val]
        E0m1 = Eb[idx_val]
        Ebd0 = (Eb[-1] - Eb[idx_val]) / (t[-1] - t[idx_val])  # last time derivative of Eb


        temp_counter += 1


    return Data_w,shell_dissolved, t_shdis
