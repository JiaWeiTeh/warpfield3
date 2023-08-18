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
import src.warpfield.phase1_energy.energy_phase_ODEs as energy_phase_ODEs
import src.warpfield.functions.terminal_prints as terminal_prints
from src.warpfield.functions.operations import find_nearest_lower, find_nearest_higher

from src.input_tools import get_param
warpfield_params = get_param.get_param()

def run_energy(t0, y0, #r0, v0, E0, T0
        rCloud, 
        mCloud, 
        mCluster, 
        nEdge, 
        rCore, #this is modified, not necessary the one in warpfield_params
        sigma_dust,
        tcoll, coll_counter,
        Cool_Struc,
        shell_dissolved, t_shelldiss,
        stellar_outputs, # old code: SB99_data
        # TODO: change tfinal to depend on warpfield_param
        # not only tfinal, but all others too.
        tfinal = 50,
        Tarr = [], Larr = [], 
        
    # Note:
    # old code: Weaver_phase()
        
# """Test: In this function, rCloud and mCloud is called assuming xx units."""
        
        # TODO: Check these units. Is rCloud in pc? cm? AU?
        
        # TODO: make it so that the function does not depend on these constants,
        # but rather simple call them by invoking the param class (i.e., params.rCloud).
        
      ):
    
    # TODO: remember double check with old files to make sure
    # that the cloudy business are taken care of. This is becaus
    # in the original file, write_cloudy is set to False. 
    # But we have to be prepared for if people wanted to check out
    # write_cloudy = True.
    
    # 
    # print(t0, y0,
    #     rCloud, 
    #     mCloud, 
    #     mCluster, 
    #     nEdge, 
    #     rCore, #this is modified, not necessary the one in warpfield_params
    #     sigma_dust,
    #     tcoll, coll_counter,
    #     density_specific_param)
    
    # print('\n\n\n')
    
    
    # the energy-driven phase
    # winds hit the shell --> reverse shock --> thermalization
    # shell is driven mostly by the high thermal pressure by the shocked ISM, also (though weaker) by the radiation pressure, at late times also SNe

    # =============================================================================
    # Now, we begin Energy-driven calculations (Phase 1)
    # =============================================================================
    # header
    terminal_prints.phase1()

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
    
    # print('checkpoint1')
    # print(tStop_i,"\n\n", dLwdt,"\n\n", abs_dLwdt,"\n\n",\
    #       t_problem,"\n\n", Lw0,"\n\n", pdot0,"\n\n", vterminal0)
    # sys.exit()

    # initial values (radius, velocity, energy, temperature)
    r0, v0, E0, T0 = y0

    # initial radius of inner discontinuity
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 1e-3 * r0, r0, 
                               args=([Lw0, E0, vterminal0, r0]),maxiter=1000) 
    
    # initial energy derivative
    Ebd0 = 0. 
    E0m1 = 0.9*E0
    t0m1 = 0.9*t0
    r0m1 = 0.9*r0
    
    # print('checkpoint2')
    # print(fLi_evo, fQi_evo, T0,\
    #       fLn_evo, fLbol_evo, fLw_evo, fpdot_evo, \
    #       R1, E0m1, t0m1, r0m1)
    # sys.exit()
 
    
    
    # calculate swept mass depending on density profile.
    # watch out units
    Msh0, _ = mass_profile.get_mass_profile(r0, density_specific_param, rCloud, mCloud)
    # The initial bubble pressure
    P0 = get_bubbleParams.bubble_E2P(E0, r0, R1, warpfield_params.gamma_adia)

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
        
        
    # print('checkpoint3')
    # print("\n\n\nfirst Msh calculation params\n\n\n", r0, density_specific_param, rCloud, mCloud)
    # print(Msh0, P0, rCore, rfinal)
    # sys.exit()
 
    
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
    
    
    print('active branch check:', r0,t0, rfinal, tfinal)
    print(all([r0 < rfinal, (tfinal - t0) > dt_Emin, continueWeaver]))
    sys.exit()
    
    while all([r0 < rfinal, (tfinal - t0) > dt_Emin, continueWeaver]):

        # calculate bubble structure and shell structure?
        # r0 > 0.1, Do I need only temp_counter > 0?
        structure_switch = (temp_counter > 0) 
        if structure_switch == True:
            dt_Emin = dt_Emin
        else:
            dt_Emin = dt_Estart

        # print("here",fit_len_max,dt_real,dt_Emin,fit_len_min ,dt_Emin, dt_Emax)

        fit_len_interp = fit_len_max + (np.log(dt_real) - np.log(dt_Emin)) * (fit_len_min - fit_len_max) / (np.log(dt_Emax) - np.log(dt_Emin))
        # used for delta, and in the beginning also for alpha, beta (min 3, max 10 or so)
        fit_len = int(np.round(np.min([fit_len_max, fit_len + 1, np.max([fit_len_min, fit_len_interp])])))  
        
        # print("fit_len_interp", fit_len_interp, fit_len)
        
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
            
            
        # print('cp4')    
        # print(structure_switch, temp_counter,dt_Emin, fit_len_interp, fit_len,\
        #       t_10list,r_10list,P_10list, T_10list, my_cloudy_dt)
        # sys.exit()
        
# False 0 0.0001 13.0 13 [6.50681839e-05] [0.23790232] [1.97308704e+08] [] 0.1

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
        # if (i.write_cloudy == True):
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
            
            
        # print('cp5')    
        # print(t_cloudy, tmax,ii_dLwdt, abs_dLwdt, dt_Lw,\
        #       tStop_i,t_problem_hi,t_problem_nnhi, tStop_i, frag_value,\
        #           frag_stop, dt_real, alpha_temp, beta_temp)
        # sys.exit()
        
        
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
            Lw_temp = fLw_evo(tStop_i)  *(u.g.to(u.Msun) * u.cm.to(u.pc)**2/u
                                          .s.to(u.Myr)**3)
            Lbol = fLbol_evo(thalf)  *(u.g.to(u.Msun) * u.cm.to(u.pc)**2/u.s.to(u.Myr)**3)
            pdot= fpdot_evo(thalf) * (u.g.to(u.Msun) * u.cm.to(u.km) / u.s.to(u.Myr))
            vterminal = 2. * Lw / pdot
            
            
            # print('cp6')
            # print(t_inc, t_temp, val, thalf, Lw, Lw_temp,  Lbol, pdot,vterminal)
            # sys.exit()
            

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
                    
                    # bubble_wrap_struc is correct
                    # it's important to use the time since the last restarting expansion, not the time since the start of the simulation
                    # calculate bubble structure
                    bubbleFailed, Lb, T0, alpha, beta,\
                        delta, dt_L, Lb_b, Lb_cz, Lb3, dMdt_factor,\
                            Tavg, Mbubble, r_Phi_b, Phi_grav_r0b,\
                                f_grav_b = bubble_structure.get_bubbleStructure(bubble_wrap_struc, Cool_Struc, fit_len=fit_len, fit_len_short=fit_len)
                else:
                    print("entering delta_new_root...")
                    alpha = alpha_temp
                    beta = beta_temp
                    # temporary ###########
                    param1 = {'alpha': alpha, 'beta': beta, 'Eb': E0, 'R2': r0, 't_now': t0, 'Lw': Lw, 'vw': vterminal, 'dMdt_factor': dMdt_factor, 'Qi': fQi_evo(thalf) * u.Myr.to(u.s), 'mypath': mypath}
                    param0 = {'alpha': alpha, 'beta': beta, 'Eb': E0m1, 'R2': r0m1, 't_now': t0m1, 'Lw': Lw, 'vw': vterminal, 'dMdt_factor': dMdt_factor, 'Qi': fQi_evo(thalf) * u.Myr.to(u.s), 'mypath': mypath}
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
                k_B = c.k_B.cgs.value * u.g.to(u.Msun) * u.cm.to(u.pc)**2 / u.s.to(u.Myr)**2
                cs_avg = np.sqrt(2.*warpfield_params.gamma_adia*\
                                 k_B*Tavg*\
                                     c.M_sun.cgs.value/c.m_p.cgs.value)

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

        
        ##############################################################################################################


        ##############################################################################################################



        # These are the initial conditions fed into shell_structure.
        print('These are the initial conditions fed into shell_structure.')
        print(r0 * c.pc.cgs.value, P0, Mbubble * c.M_sun.cgs.value,\
                                        fLn_evo(thalf), fLi_evo(thalf), fQi_evo(thalf),\
                                        Msh0, sigma_dust, 1,)
        # 0.23790232199299727 -197308705.1874583 nan 
        # 1.515015429411944e+43 1.9364219639465924e+43 5.395106225151267e+53 
        # [1.77403693] 1.5e-21 1
        # -6.8348315115384816e+19 10.0
        
        # This is to test. Delete after test
        # Actually DO NOT DELET
        # TODO
        test_vals = shell_structure.shell_structure(r0 * c.pc.cgs.value, P0, Mbubble * c.M_sun.cgs.value,
                                        fLn_evo(thalf), fLi_evo(thalf), fQi_evo(thalf),
                                        Msh0, 1)
        
        print('\n\nhere is to test the shell_structure function')
        print("\nf_absorbed_ion, f_absorbed_neu, f_absorbed, f_ionised_dust, is_fullyIonised, shellThickness, nShell_max, tau_kappa_IR, grav_r, grav_phi, grav_force_m\n\n",test_vals)
        sys.exit('stop at test_vals')

        ################ CLOUDY #############################################################################################
        # TODO
        # make_cloudy_model = (i.write_cloudy and t0 <= t_cloudy and tStop_i >= t_cloudy)
        #####################################################################################################################

        if structure_switch:
            # shell structure routine wants cgs units
            np.seterr(all='warn')
            fabs_i, fabs_n, fabs, fion_dust, ionsh, dRs, n0, nmax, rhodr, n0_cloudy, r_Phi_sh, Phi_grav_r0s, f_grav_sh = shell_structure.shell_structure(r0* c.pc.cgs.value,
                                                                                                                                                         P0, Mbubble * c.M_sun.cgs.value,
                                                                                                                                                          fLn_evo(thalf), fLi_evo(thalf), fQi_evo(thalf),
                                                                                                                                                          Msh0, 1)
            # # TODO
            # r_Phi = np.concatenate([r_Phi_b, r_Phi_sh])
            # # this is the integral from r0 to rsh_out over r*rho(r); the integral part from rsh_out to very far out will be done in write_pot_to_file
            # Phi_grav_r0 = Phi_grav_r0b + Phi_grav_r0s 
            # f_grav = np.concatenate([f_grav_b, f_grav_sh])
            
            # # if i.write_potential == True:
            # #     aux.write_pot_to_file(mypath, t0, r_Phi, Phi_grav_r0, f_grav, rcloud_au, rcore_au, nalpha, Mcloud_au, Mcluster_au, SFE)

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
        
        RHOA = warpfield_params.nCore * warpfield_params.mu_n * (u.g/u.cm**3).to(u.M_sun/u.pc**3)
        RCORE = rCore
        A_EXP = warpfield_params.dens_a_pL
        MSTAR = mCluster
        LB = Lb # astro units!!
        FRAD = fabs * Lbol/c.c.to(u.pc/u.Myr).value # astro units
        CS = cs_avg

        # inside core or inside density slope?
        if (r0 < rCore or (warpfield_params.dens_a_pL == 0)):
            phase0 = 1
        else:
            phase0 = 1.1 

        # TODO: include outputs.
        # if i.output_verbosity >= 1:
        #     print('%d' % temp_counter, '%.1f' % phase0, '%.4e' % t0, "%.2e" % r0, "%.4e" % v0, "%.7e" % E0, "%.7e" % R1, "%.3f" % fabs, "%.4f" % fabs_i, "%.4f" % (dRs / r0), "%.2e" % Msh0, "%.4f" % tau_IR, "%.2e" % nmax, "%.3e" % (Lw*c.L_cgs), "%.3e" % (Lb*c.L_cgs), "%.3e" % (T0), "%.3e" % (Tavg), "%.2e" % (cs_avg), "%.2f" % (cf0), "%.4f" % (alpha), "%.4f" % (beta), "%.4f" % (delta), "%.1e" % (frag_value), "%.3e" % (dMdt_factor), "%.3e" % (dt_Lw), "%.3e" % (dt_real), "%d" % (float(fit_len)), "%.3e" %(t_problem_nnhi), "%.3e" %(time.time()-start_time))

        # Bundle initial conditions for ODE solver
        # print(r0)
        # print(v0)
        # print(E0)
        # sys.exit("stop here before using y0 to check")
        y0 = [r0, v0, E0]
        

        # bundle parameters for ODE solver
        params = [LW, PWDOT, GAM, mCloud, RHOA, RCORE, A_EXP,
                  MSTAR, LB, FRAD, fabs_i, rCloud, 
                  density_specific_param, warpfield_params,
                  tcoll[coll_counter], t_frag, tscr, CS, warpfield_params.sfe]

        # print('\n\n\n')
        # print("Here we check for the values of r, since this is why the Msh returns terrible value.")
        # print("y0", y0) # THe y values are right for Msh
        # print('rCloud', rCloud)
        # print('\n\n\n')
        # print("params", params)
        # print('\n\n\n')
        # print('t',t)
        # call ODE solver
        try:
            psoln = scipy.integrate.odeint(energy_phase_ODEs.get_ODE_Edot, y0, t, args=(params,))
        except:
            sys.exit("ODE solver not working in run_energy_phase")
        # get r, rdot and rdotdot
        r = psoln[:,0]
        rd = psoln[:, 1]
        Eb = psoln[:, 2]
        # print('\n\n\n')
        # print("We are now in run_energy.py and in run_energy(). Here are the values for\
        #       the r, rdot and edot after solving energy_phase_ODEs.get_ODE_Edot. This\
        #           might be helpful for checking why we are getting errors for Msh in line 497\
        #               with increasing values.")
        # print('\n\n\n')
        # print('length is', len(r))
        # print('\n\n\n')
        # print('psol', r, rd, Eb)
        # print('\n\n\n')
        # print('print(r, density_specific_param, rCloud, mCloud)')
        # print('\n\n\n')
        # print(r, density_specific_param, rCloud, mCloud)
        # print('\n\n\n')
        
        # get mass
        Msh, _ = mass_profile.get_mass_profile(r, density_specific_param, rCloud, mCloud)

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
            pdot_cloudy= fpdot_evo(t_cloudy)         * u.Myr.to(u.s) / (c.M_sun.cgs.value * 1e5)
            vterminal_cloudy = 2. * Lw_cloudy / pdot_cloudy
            R1_cloudy = scipy.optimize.brentq(bubble_structure.R1_zero, 1e-3 * r0, r0, args=([Lw_cloudy, E_cloudy, vterminal_cloudy, r_cloudy]))
            P_cloudy = state_eq.PfromE(E_cloudy, r_cloudy, R1_cloudy)
            [n0_temp, n0_cloudy] = n_from_press(P_cloudy, i.Ti, B_cloudy=i.B_cloudy)
            create_model(cloudypath, SFE, Mcloud_au, i.namb, i.Zism, n0_cloudy, r_cloudy, v_cloudy, Msh_cloudy,
                         np.log10(fLbol_evo(t_cloudy)), t_cloudy, rcloud_au, nedge,
                         warpfield_params,
                         SB99model=i.SB99cloudy_file, shell=i.cloudy_stopmass_shell, turb=i.cloudy_turb,
                         coll_counter=coll_counter, Tarr=Tarr, Larr=Larr, Li=Li*c.L_cgs, Qi=fQi_evo(thalf), Mcluster=Mcluster_au, phase=phase0)
        #####################################################################################################################
        """
        
        # check whether shell fragments or similar

        # if a certain radius has been exceeded, stop branch
        if (r[-1] > rfinal or (r0 < rCore and r[-1] >= rCore and (warpfield_params.dens_a_pL != 0))):
            if r[-1] > rfinal:
                continueWeaver = False
                include_list = r<rfinal
            else:
                continueWeaver = True
                rtmp = r[r>= rCore ][0] # find first radius larger than core radius
                include_list = r <= rtmp
            t = t[include_list]
            r = r[include_list]
            rd = rd[include_list]
            Eb = Eb[include_list]
            Msh = Msh[include_list]
            print('Radius has been exceeded...')


        # calculate fragmentation time
        if fabs_i < 0.999:
            Tsh = warpfield_params.t_neu  # 100 K or so
        else:
            Tsh = warpfield_params.t_ion  # 1e4 K or so
        # sound speed in shell (if a part is neutral, take the lower sound speed)
        def get_cs(T):
            # get sound speed in kms
            if T > 1e3:
                mu = warpfield_params.mu_p
            else:
                mu = warpfield_params.mu_n
            # in km/s
            cs = np.sqrt(warpfield_params.gamma_adia *c.k_B.cgs.value * T / mu) * 1e-5
            return cs
        
        cs = get_cs(Tsh)
        frag_list = warpfield_params.frag_grav_coeff * c.G.to(u.pc**3/u.M_sun/u.Myr**2).value * 3. * Msh / (4. * np.pi * rd * cs * r) # compare McCray and Kafatos 1987
        frag_value = frag_list[-1] # fragmentation occurs when this number is larger than 1.

        # print('Here are the values after Msh')
        # print()
        # sys.exit()
        
        
        
        # frag value can jump from positive value directly to negative value (if velocity becomes negative) if time resolution is too coarse
        # however, before the velocity becomes negative, it would become 0 first. At v=0, fragmentation always occurs
        if frag_value < 0.:
            frag_value = 2.0
        # another way to estimate when fragmentation occurs: Raylor-Taylor instabilities, see Baumgartner 2013, eq. (48)
        # not implemented yet

        if (was_close_to_frag == False):
            frag_stop = 1.0 - float(fit_len)*dt_Emin*dfragdt # if close to fragmentation: take small time steps
            if (frag_value >= frag_stop):
                # TODO
                # aux.printl("close to fragmentation", verbose=1)
                ii_fragstop = find_nearest_higher(frag_list, frag_stop)
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


        if ((frag_value > 1.0) and (first_frag == True)):
            #print frag_value
            # fragmentation occurs
            # TODO
            # aux.printl("shell fragments", verbose=1)
            ii_frag = find_nearest_higher(frag_list, 1.0)  # index when fragmentation starts #debugging
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
            # if i.output_verbosity >= 1: print(t_frag)
            first_frag = False
            # OPTION 1 for switching to mom-driving: if i.immediate_leak is set to True, we will immeadiately enter the momentum-driven phase now
            if warpfield_params.immediate_leak == True:
                mom_phase = True
                continueWeaver = False
                Eb[-1] = 0.

        # OPTION 2 for switching to mom-driving: if i.immediate_leak is set to False, when covering fraction drops below 50%, switch to momentum driving
        # (THIS APPROACH IS NOT STABLE YET)
        
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
    
        cf = calc_coveringf(t, t_frag, tscr)
        if cf[-1] < 0.5:
            ii_cov50 = find_nearest_lower(cf, 0.5)
            if (ii_cov50 == 0): ii_cov50 = 1
            # if i.output_verbosity >= 1: print(cf[ii_cov50])
            t = t[:ii_cov50]
            r = r[:ii_cov50]
            rd = rd[:ii_cov50]
            Eb = Eb[:ii_cov50]
            Msh = Msh[:ii_cov50]    
            mom_phase = True
            continueWeaver = False
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
        logMcluster_weaver = np.concatenate([logMcluster_weaver, [np.log10(mCluster)]])
        logMcloud_weaver = np.concatenate([logMcloud_weaver, [np.log10(mCloud)]])
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

        Flux_Phi = fQi_evo(thalf)/(4.*np.pi*(r0*c.pc.cgs.value)**2)
        Pb_cgs = P0 /(c.pc.cgs.value*u.Myr.to(u.s)**2/(c.M_sun.cgs.value))
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
        vterminal0 = 2.*Lw_new / (fpdot_evo(t0) * u.Myr.to(u.s) / c.M_sun.cgs.value / 1e5)
        if (not mom_phase):
            R1 = scipy.optimize.brentq(bubble_structure.R1_zero, 1e-3*r0, r0, args=([Lw_new, E0, vterminal0, r0]))
            P0 = get_bubbleParams.bubble_E2P(E0, r0, R1) # bubble pressure
        else: 
            R1 = r0
            P0 = get_bubbleParams.pRam(r0,Lw_new,vterminal0)
        Msh0 = Msh[-1] # shell mass
        Lres0 = LW - LB
        Lw_old = Lw
        delta_old = delta
        dfragdt = (frag_value - frag_value0)/(t[-1]-t[0])
        frag_value0 = frag_value

        if idx_val >= len(t): idx_val = -2

        t0m1 = t[idx_val]
        r0m1 = r[idx_val]
        E0m1 = Eb[idx_val]
        Ebd0 = (Eb[-1] - Eb[idx_val]) / (t[-1] - t[idx_val])  # last time derivative of Eb

        temp_counter += 1

    return Data_w,shell_dissolved, t_shdis


#%%

# import numpy as np

# # test runs

# Cool_Struc = np.load('/Users/jwt/Documents/Code/warpfield3/outputs/cool.npy', allow_pickle = True).item()
# stellar_outputs = np.load('/Users/jwt/Documents/Code/warpfield3/outputs/SB99_data.npy', allow_pickle = True)

# warpfield_params = {'model_name': 'example', 
#                    'out_dir': 'def_dir', 
#                    'verbose': 1.0, 
#                    'output_format': 'ASCII', 
#                    'rand_input': 0.0, 
#                    'log_mCloud': 9.0, 
#                    'mCloud_beforeSF': 1.0, 
#                    'sfe': 0.01, 
#                    'nCore': 1000.0, 
#                    'rCore': 0.099, 
#                    'metallicity': 1.0, 
#                    'stochastic_sampling': 0.0, 
#                    'n_trials': 1.0, 
#                    'rand_log_mCloud': ['5', ' 7.47'], 
#                    'rand_sfe': ['0.01', ' 0.10'], 
#                    'rand_n_cloud': ['100.', ' 1000.'], 
#                    'rand_metallicity': ['0.15', ' 1'], 
#                    'mult_exp': 0.0, 
#                    'r_coll': 1.0, 
#                    'mult_SF': 1.0, 
#                    'sfe_tff': 0.01, 
#                    'imf': 'kroupa.imf', 
#                    'stellar_tracks': 'geneva', 
#                     'dens_profile': 'bE_prof', 
#                    # 'dens_profile': 'pL_prof', 
#                    'dens_g_bE': 14.1, 
#                    'dens_a_pL': -2.0, 
#                    'dens_navg_pL': 170.0, 
#                    'frag_enabled': 0.0, 
#                    'frag_r_min': 0.1, 
#                    'frag_grav': 0.0, 
#                    'frag_grav_coeff': 0.67, 
#                    'frag_RTinstab': 0.0, 
#                    'frag_densInhom': 0.0, 
#                    'frag_cf': 1.0, 
#                    'frag_enable_timescale': 1.0, 
#                    'stop_n_diss': 1.0, 
#                    'stop_t_diss': 1.0, 
#                    'stop_r': 5050.0, 
#                    'stop_v': -10000.0,
#                    'stop_t': 15.05, 
#                    'stop_t_unit': 'Myr', 
#                    'adiabaticOnlyInCore': False,
#                    'immediate_leak': True,
#                    'write_main': 1.0, 
#                    'write_stellar_prop': 0.0, 
#                    'write_bubble': 0.0, 
#                    'write_bubble_CLOUDY': 0.0, 
#                    'write_shell': 0.0, 
#                    'xi_Tb': 0.99,
#                    'inc_grav': 1.0, 
#                    'f_Mcold_W': 0.0, 
#                    'f_Mcold_SN': 0.0, 
#                    'v_SN': 1000000000.0, 
#                    'sigma0': 1.5e-21, 
#                    'z_nodust': 0.05, 
#                    'mu_n': 2.1287915392418182e-24, 
#                    'mu_p': 1.0181176926808696e-24, 
#                    't_ion': 10000.0, 
#                    't_neu': 100.0, 
#                    # 'nISM': 0.1, 
#                    'nISM': 10, 
#                    'kappa_IR': 4.0, 
#                    'gamma_adia': 1.6666666666666667, 
#                    'thermcoeff_wind': 1.0, 
#                    'thermcoeff_SN': 1.0,
#                    'alpha_B': 2.59e-13,
#                    'gamma_mag': 1.3333333333333333,
#                    'log_BMW': -4.3125,
#                    'log_nMW': 2.065,
#                    'c_therm': 1.2e-6,
#                    }
    
# class Dict2Class(object):
#     # set object attribute
#     def __init__(self, dictionary):
#         for k, v in dictionary.items():
#             setattr(self, k, v)
            
# # initialise the class
# warpfield_params = Dict2Class(warpfield_params)

# #%%

# t0 = 6.506818386985495e-05
# y0 =  [0.23790232199299727, 3656.200432285518, 5722974.028981317, 67741779.55773313]
# shell_dissolved = False 
# t_shelldiss = 1e+99 
# mCluster = 10000000.0 

# rCore = 451690.2638133162 
# rCloud =  355.8658723191992
# nEdge = 66.66666667279057 
# mCloud = 990000000.0
# # though, note that with our own values, 
# # (464995.4640005363, 351.35326946660103, 70.92198590822773, 990000000.0)
# # is the supposed value for 
# # rCore, rCloud, nEdge, mCloud_afterSF

# coll_counter = 0 
# tcoll = [0.0] 
# Tarr = [] 
# Larr = [] 
# tfinal = 0.003065068183869855
# sigma_dust = warpfield_params.sigma0
# density_specific_param = rCore
    
# aa = run_energy(t0, y0, #r0, v0, E0, T0
#         rCloud, 
#         mCloud, 
#         mCluster, 
#         nEdge, 
#         rCore, #this is modified, not necessary the one in warpfield_params
#         sigma_dust,
#         tcoll, coll_counter,
#         density_specific_param,
#         warpfield_params,
#         Cool_Struc,
#         shell_dissolved,
#         stellar_outputs, # old code: SB99_data
#         t_shelldiss,
#         Tarr, Larr, tfinal)




# # print(t0, y0, rcloud_au, SB99_data, Mcloud_au, SFE, mypath, 
# #                  cloudypath, shell_dissolved, t_shdis, rcore_au, 
# #                  nalpha, Mcluster_au, nedge, coll_counter, tcoll, Tarr, Larr, tfinal)









