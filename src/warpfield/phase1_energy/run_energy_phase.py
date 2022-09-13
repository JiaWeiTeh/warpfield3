#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 23:16:58 2022

@author: Jia Wei Teh

"""

import src.warpfield.bubble_structure.get_bubbleProp as get_bubbleProp

import src.warpfield.cloud_properties.mass_profile as mass_profile

def run_energy(t0, r0, v0, E0, T0,
        rCloud, mCloud, sfe, mCluster, nEdge, rCore, 
        params,
        is_shell_dissolved,
        t_shelldiss,
        
        
        
        # TODO: make it so that the function does not depend on these constants,
        # but rather simple call them by invoking the param class (i.e., params.rCloud).
        
        ):
    
    
    
    
    
    
    # =============================================================================
    # calculate swept mass depending on density profile.
    # =============================================================================
    mShell = mass_profile.get_mass_profile(r_arr,
                         density_specific_param, 
                         rCloud, 
                         mCloud,
                         params,
                         return_rdot = False,
                         )
        
    
    
    
    
    # The initial bubble pressure
    get_bubbleProp.bubble_E2P(bubble_E, r1, r2, params.gamma_adia)
    
    
    
    return



#%%

    # the energy-driven phase
    # winds hit the shell --> reverse shock --> thermalization
    # shell is driven mostly by the high thermal pressure by the shocked ISM, also (though weaker) by the radiation pressure, at late times also SNe

    aux.printl("############################################################")
    aux.printl("entering phase I (energy-driven)...")
    aux.printl("############################################################")

    start_time = time.time()

    continue_branch = True
    # reset = True
    # Mcore_au = 4.*np.pi/3.*rcore_au**3.*i.rhoa_au

    # first stopping time (will be incremented at beginning of while loop)
    tStop_i = t0

    # start time t0 will be incremented at end of while loop

    # initial time steps
    # tInc_tmp = i.tInc_small
    # tCheck_tmp = i.tCheck_small


    t_10list = np.array([])
    r_10list = np.array([])
    P_10list = np.array([])
    T_10list = np.array([])

    # get Starburst99 data
    t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo = SB99_data # unit of t_evo is Myr, the other units are cgs

    # interpolation functions for SB99 values
    fQi_evo = interp1d(t_evo, Qi_evo, kind = 'linear')
    fLi_evo = interp1d(t_evo, Li_evo, kind = 'linear')
    fLn_evo = interp1d(t_evo, Ln_evo, kind = 'linear')
    fLbol_evo = interp1d(t_evo, Lbol_evo, kind = 'linear')
    fLw_evo = interp1d(t_evo, Lw_evo, kind = 'linear')
    fpdot_evo = interp1d(t_evo, pdot_evo, kind = 'linear')
    # fpdot_SNe_evo = interp1d(t_evo, pdot_SNe_evo, kind='linear')


    # identify potentially problematic time steps (where feedback parameters change by a large percentage)
    dLwdt = aux.my_differentiate(t_evo, Lw_evo)
    abs_dLwdt = np.abs(dLwdt)
    t_problem = t_evo[abs_dLwdt / Lw_evo > 3.] # problematic time (which needs small timesteps) 
    # is defined by relative change in mechanical luminosity being more than 
    # 300% per Myr (if number on RHS is 3.0)
    
    #import matplotlib.pyplot as plt
    #plt.semilogy(t_evo,(np.abs(dLwdt))/Lw_evo)
    #plt.semilogy(t_evo,(Lw_evo)/Lw_evo[0])
    #plt.semilogy(t_problem,1.*t_problem/t_problem,'x')
    #plt.show()

    Lw0 = fLw_evo(t0)        /c.L_cgs # initial mechanical luminosity in astro units
    pdot0 = fpdot_evo(t0) * c.Myr / (c.Msun * c.kms) # initial momentum of stellar winds in astro units
    vterminal0 = 2. * Lw0/pdot0 # initial terminal wind velocity


    # initial values (radius, velocity, energy, temperature)
    if len(y0) == 3:
        r0, v0, E0 = y0
        # initial temperature of bubble
        T0 = 1.51e6 * (Lw_evo[0]/1e36)**(8./35.) * i.namb**(2./35.) * (t0-tcoll[coll_counter])**(-6./35.) * (1.-i.r_Tb)**0.4 # see Weaver+77, eq. (37)
    elif len(y0) == 4:
        r0, v0, E0, T0 = y0
    else:
        sys.exit("y0 has wrong length in phase_solver2.py")

    R1 = scipy.optimize.brentq(bubble_structure.R1_zero, 1e-3 * r0, r0, args=([Lw0, E0, vterminal0, r0]),maxiter=1000) # initial radius of inner discontinuity
    Ebd0 = 0. # initial energy derivative
    E0m1 = 0.9*E0
    t0m1 = 0.9*t0
    r0m1 = 0.9*r0
    print(t0m1)

    # calculate swept mass (depends on used density profile)
    if i.dens_profile == "powerlaw":
        #print('##############*************************#############r0=',r0)
        Msh0 = mass_profile.calc_mass(np.array([r0]),rcore_au,rcloud_au,i.rhoa_au,i.rho_intercl_au, i.nalpha, Mcloud_au)[0] # initial mass of shell (swept up from cloud)
        #print('##############*************************#############Msh0=',Msh0)
    elif i.dens_profile == "BonnorEbert":
        T_BE=rcore_au
        #print('##############*************************#############r0=',r0)
        Msh0 = mass_profile.calc_mass_BE(r0, i.rhoa_au, T_BE, i.rho_intercl_au, rcloud_au, Mcloud_au)
        #print('##############*************************#############MshzuR0=',Msh0)
    
    # initial bubble pressure
    P0 = state_eq.PfromE(E0,r0, R1)

    Lres0 = 1.0

    # initialize r and derivatives
    # rweaver = []; vweaver = []; Eweaver = []; tweaver = []
    tSweaver = []; rSweaver = []; vSweaver = []; ESweaver = []
    fabsweaver = []; fabs_i_weaver = []; fabs_n_weaver = []; ionshweaver = []; Mshell_weaver = []
    FSgrav_weaver = []; FSwind_weaver = []; FSradp_weaver = []; FSsne_weaver = []; FSIR_weaver = []; dRs_weaver = []; nmax_weaver = []
    n0weaver = []; n0_cloudyweaver = []; logMcluster_weaver = []; logMcloud_weaver = []; phase_weaver = []; R1weaver = []; Ebweaver = []; Pbweaver = []
    Lbweaver = []; Lwweaver = []; Tbweaver = []; alphaweaver = []; betaweaver = []; deltaweaver = []
    Lbbweaver = []; Lbczweaver = []; Lb3weaver = []; fragweaver = [];

    # check how long to stay in Weaver phase
    if i.adiabatic_core_only:
        rfinal = rcore_au
    else:
        rfinal = rcloud_au
    #rfinal = 100.*rfinal # factor 100 for DEBUGGING

    # only stay in Weaver phase while in core (if density gradient with core)
    #if i.density_gradient:
    #    rfinal = np.min([rcore_au, rcloud_au]) # only stay until shell breaks out off core (or cloud edge if no density gradient)
    #else:
    #    rfinal = rcloud_au

    # start values
    alpha = 0.6 # 3/5
    beta = 0.8 # 4/5
    if fLw_evo(t0) < 1e40: delta = -6./35. # -6./35.=-0.171 # taking more negative value seems to be better for massive clusters (start further out of equilibrium?)
    else: delta = -6./35. # -0.5
    # TOASK: Double confirm all these parameters
    temp_counter = 0
    dt_L = i.dt_Estart
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
    fit_len = i.fit_len_max
    dt_real = i.dt_Estart

    aux.printl("i, phase, t,        Rs,      v,       Eb,      R1,   fabs, fabsi, rel dR, Msh,    tau_IR, nmax,    Lw,     Lb,      Tb,         Tavg,      cs_avg, cf,  alpha, beta, delta, frag, dMdt_f,   dt_Lw,   dt_real, fit, t_prob_nn, realtime", verbose=1)



                                            t0, y0, ODEpar['Rcloud_au'], SB99_data,
                                                                ODEpar['Mcloud_au'], ODEpar['SFE'], mypath,
                                                                cloudypath, shell_dissolved, t_shdis,
                                                                ODEpar['Rcore_au'], i.nalpha, ODEpar['Mcluster_au'],
                                                                ODEpar['nedge'], Cool_Struc, tcoll=tcoll,
                                                                coll_counter=ii_coll, tfinal=tfinal
                                                                
Weaver_phase(t0, y0, rcloud_au, SB99_data, Mcloud_au, SFE, mypath, 
                 cloudypath, shell_dissolved, t_shdis, rcore_au, 
                 nalpha, Mcluster_au, nedge, Cool_Struc, 
                 coll_counter=0, tcoll=[0.0], Tarr=[], Larr=[], tfinal = i.tStop):