#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:18:45 2022

@author: Jia Wei Teh
"""

# libraries
import sys
import numpy as np
import time
import astropy.units as u
import astropy.constants as c 
import scipy.interpolate
#--
from src.warpfield.cloud_properties import mass_profile
from src.warpfield.bubble_structure import get_bubbleParams
import src.warpfield.shell_structure.shell_structure as shell_structure
import src.warpfield.shell_structure.get_shellParams as get_shellParams
import src.warpfield.phase0_init.set_phase as set_phase
import src.warpfield.phase2_momentum.momentum_phase_ODEs as momentum_phase_ODEs
import src.warpfield.functions.operations as operations


#%%

def run_momentum(t0, y0, 
                 SB99_data,
                 mCloud,
                 mypath, cloudypath,
                 shell_dissolved,
                 t_shelldiss,
                 mCluster,
                 rCloud,
                 rCore,
                 density_specific_param,
                 nEdge,
                 warpfield_params,
                 coll_counter = 0, tcoll=[0.0], Tarr=[], Larr=[]):
    
    
    # Notes:
        # old code: Mom_phase(). The wrapper Mom_check_and_start() is not used.
    
    print("############################################################")
    print("entering phase II (momentum-driven)... t = " + str(t0) + " Myr")
    print("############################################################")

    # TODO: watch out the parameters. Since some are deleted, that means
    # the input might be messed up. 


    # TODO: remember a problem here, which is that rCore is overwritten as the
    # T temperature for Bonnor-Ebert spheres. This will cause huge difference
    # in values and results if one were to compare between rCore and T. 
    
    # This has to be fixed!!!
    
    start_time = time.time()
    
    tSmom = [];
    rSmom = [];
    vSmom = [];
    fabsmom = [];
    ionshmom = [];
    Mshell_mom = []
    FSgrav_mom = [];
    FSwind_mom = [];
    FSradp_mom = [];
    FSsne_mom = [];
    FSIR_mom = [];
    fabs_i_mom = [];
    fabs_n_mom = [];
    dRs_mom = [];
    nmax_mom = [];
    logMcloud_mom = [];
    logMcluster_mom = []
    phase_mom = []
    Lwmom = []

    n0mom = []
    n0_cloudymom = []
    
    
    # shell is not yet dissolved
    first_dissolve = False

    # first stopping time (will be incremented at beginning of while loop)
    tStop_i = t0

    # start time t0 will be incremented at end of while loop
    tInc = 5e-4 #output time step for ODEs
    tCheck = .04166 # this is the max main time step (used to calculated shell structure etc.) in Myr
    
    # reset time step
    tInc_tmp = tInc
    tCheck_tmp = tCheck
    
    # unravel parameters
    r0, v0 = y0

    # =============================================================================
    # Here we calculate the mass profile
    # =============================================================================
    Msh0, _ = mass_profile.get_mass_profile(r0, 
                                         density_specific_param, 
                                         rCloud, mCloud,  
                                         warpfield_params)

    # unpack SB99 data
    t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo = SB99_data

    # interpolation functions for SB99 values
    fQi_evo = scipy.interpolate.interp1d(t_evo, Qi_evo, kind = 'linear')
    fLi_evo = scipy.interpolate.interp1d(t_evo, Li_evo, kind = 'linear')
    fLn_evo = scipy.interpolate.interp1d(t_evo, Ln_evo, kind = 'linear')
    fLbol_evo = scipy.interpolate.interp1d(t_evo, Lbol_evo, kind = 'linear')
    fLw_evo = scipy.interpolate.interp1d(t_evo, Lw_evo, kind = 'linear')
    fpdot_evo = scipy.interpolate.interp1d(t_evo, pdot_evo, kind = 'linear')
    fpdot_SNe_evo = scipy.interpolate.interp1d(t_evo, pdot_SNe_evo, kind='linear')

    # TODO tStop -> stop_t
    # vstop -> stop_v
    # rcoll -> r_coll
    
    # cloudy time steps (only relevant for warpversion 1.0)
    small_cloudy_dt = 0.1 # (Myr), before a time of cloudy_t_switch has passed since the last SF event, use this small dt for writing cloudy output
    cloudy_dt = 0.5 # (Myr) after more than cloudy_t_switch has passed since last SF event, use this bigger dt
    cloudy_t_switch = 1.0 # (Myr) at this time after the last SF event, cloudy_dt is udes instead of small_cloudy_dt

    
    while tStop_i < warpfield_params.stop_t and v0 > warpfield_params.stop_v and\
        (r0 > warpfield_params.r_coll or v0 > -1.0) and r0 < warpfield_params.rstop:
        # increment stopping time (start time has already been increemented at end of loop)
        # for early time (1st Myr) use a small cloudy dt

        if (t0 - tcoll[coll_counter]) < cloudy_t_switch:
            my_cloudy_dt = small_cloudy_dt
        else:
            my_cloudy_dt = cloudy_dt
        t_cloudy = np.ceil(t0/my_cloudy_dt)*my_cloudy_dt

        # [removed]
        tmax = warpfield_params.stop_t
        # calculate time step
        tStop_i, tInc_tmp, tCheck_tmp = momentum_phase_ODEs.update_dt(r0, v0, tInc_tmp, t0, tCheck_tmp, tmax= tmax)

        dt_real = tStop_i - t0
        # TODO: write_cloudy
        # make sure, that only 1 cloudy output is written per time step (if necessary, decrease time step)
        # if ((i.write_cloudy is True) and (dt_real > my_cloudy_dt)):
        #     dt_real = my_cloudy_dt
        #     tStop_i = t0 + dt_real
            
        t_inc = dt_real/1000.

        # time vector for this time step
        t = np.arange(t0, tStop_i + t_inc, t_inc) # make sure to include tStop_i here, i.e. go one small dt farther
        #t = np.arange(t0, tStop_i+tInc_tmp, tInc_tmp)

        # get initial values
        y0 = [r0, v0]

        # midpoint time, evaluate feedback parameters here
        thalf = 0.5 * (t[0]+t[-1])

        # get SB99 values
        # TODO: add option for manual input
        # if i.SB99 == True:
        #     if (thalf > t_evo[-1]):
        #         print("Warning: End of SB99 file reached")
        #         sys.exit("Stop Code: Feedback parameters not provided")

        # convert cgs to astro units (Myr, Msun, pc)
        Lw = fLw_evo(thalf) *(u.g.to(u.Msun) * u.cm.to(u.pc)**2/u.s.to(u.Myr)**3)
        Lbol = fLbol_evo(thalf) *(u.g.to(u.Msun) * u.cm.to(u.pc)**2/u.s.to(u.Myr)**3)
        pdot= fpdot_evo(thalf) * (u.g.to(u.Msun) * u.cm.to(u.km) / u.s.to(u.Myr))
        pdot_SNe = fpdot_SNe_evo(thalf) * (u.g.to(u.Msun) * u.cm.to(u.km) / u.s.to(u.Myr))

        # vterminal_evo = 2. * fLw_evo(thalf) / fpdot_evo(thalf) # cgs units
        vterminal = 2. * Lw / pdot

        ##############################################################################################################

        ################ CLOUDY #############################################################################################
        make_cloudy_model = (i.write_cloudy and t0 <= t_cloudy and tStop_i >= t_cloudy)
        #####################################################################################################################
        
        # get ram pressure.
        P0 = get_bubbleParams.pRam(r0, Lw, vterminal)

        # calculate shell structure
        figtitle_shell = "shellstruct_1e" + str(round(np.log10(mCloud),ndigits=2)) + "(SB99)_sfe=" + str(
            round(100. * warpfield_params.sfe)) + "_n=" + str(int(i.namb)) + "_Z=" + str(warpfield_params.metallicity) + "_P3_t=" + str(
            round(t0, ndigits=2)) + ".png"
        figpath_shell = mypath + '/shellstruct/' + figtitle_shell
        print('*******cf in phase_solver2 2nd eq')
        
        [fabs_i, fabs_n, fabs, fion_dust, ionsh, dRs, n0, nmax, rhodr, n0_cloudy, r_Phi, Phi_grav_r0, f_grav] = shell_structure.shell_structure(r0, P0, 0, fLn_evo(thalf), fLi_evo(thalf), fQi_evo(thalf), Msh0, 1,)
        
        tau_IR = rhodr * warpfield_params.kappa_IR

        # TODO write these
        # if i.write_potential is True:
            # aux.write_pot_to_file(mypath, t0, r_Phi, Phi_grav_r0, f_grav, rCloud, rCore, warpfield_params.dens_a_pL, mCloud, mCluster, warpfield_params.sfe)

        # if i.output_verbosity >= 1:
        #     print('%.4f' % t0, "%.2f" % r0, "%.2f" % v0, "%.3f" % fabs, "%.4f" % fabs_i, "%.4f" % (dRs / r0), "%.2e" % Msh0, "%.2e" % tau_IR, "%.2e" % nmax, "%.3e" %(time.time()-start_time))

        if shell_dissolved == False and nmax < i.ndiss:
            shell_dissolved = True
            t_shdis = t0

        if (tStop_i > 3.0 and nmax < 1.0):
            # plot the shell structure, if density is very low to see whether something goes wrong #    
            shell_ploton = True
        else:
            shell_ploton = False

        # What phase is the expansion in? see phase_lookup
        # -1.0: collapse,
        # 2.1: expansion core (or uniform cloud)
        # 2.2: density gradient
        # 3.0: ambient medium
        phase0 = set_phase.set_phase(r0, rCore, rCloud, t0, v0, dens_grad=i.density_gradient)[0]

        LBOL_ABS = Lbol * fabs
        PW = pdot
        MSTAR = mCluster

        # case no collapse
        if phase0 > 0.:
            # temporary mass from which mass of the shell can be easily calculated on the fly in the ODE solver
            # Msh = M0T + 4.*pi/3. * rho_intercl_au * R**3 (see eq. in sec. phase III in paper)

            A_EXP = warpfield_params.dens_a_pL
            M0 = Msh0  # get mass from time step before (if shell had collapsed a bit before but is expanding now, this is important)
            RCORE = rCore
            RHOA = warpfield_params.nCore * warpfield_params.mu_n * (u.g/u.cm**3).to(u.M_sun/u.pc**3)
            # warpfield_params.nCore * warpfield_params.mu_n * (u.g/u.cm**3).to(u.M_sun/u.pc**3)
            # Bundle parameters for ODE solver
            # TODO: make this shorter and dependent on warpfield_params
            params = [M0, RHOA, LBOL_ABS, tau_IR, PW, MSTAR, RCORE, A_EXP, rCloud, mCloud, warpfield_params.sfe, warpfield_params]

            psoln = scipy.integrate.odeint(momentum_phase_ODEs.get_momentum_ODEs, y0, t, args=(params,))


        # case collapse
        elif phase0 == set_phase.collapse:
            # now solve eq. of motion for a shell of constant mass (assume that all the ambient material has been swept-up)
            # this part of the routine assumes astro-units
            M0 = Msh0

            # Bundle parameters for ODE solver
            params = [M0, LBOL_ABS, tau_IR, PW, MSTAR]

            psoln = scipy.integrate.odeint(momentum_phase_ODEs.get_collapse_ODEs, y0, t, args=(params,))

        # get r, rdot and rdotdot
        r = psoln[:, 0]
        rd = psoln[:, 1]

        # check whether phase switch occures
        # if so, only take result up to phase switch
        phase1 = set_phase.set_phase(r[-1], rCore, rCloud, tStop_i,  rd[-1], dens_grad=i.density_gradient)[0]
        if phase0 != phase1:
            phase = set_phase.set_phase(r, rCore, rCloud, t0,  rd, dens_grad=i.density_gradient)
            in_phase0 = (r <= r[phase != phase0][0]) # find index of phase switch
            r = r[in_phase0]
            rd = rd[in_phase0]
            t = t[in_phase0]

        # only change mass if not collapsing
        if phase0 > 0.:
            Msh, _ = mass_profile.get_mass_profile(r, 
                                     density_specific_param, 
                                     rCloud, mCloud,  
                                     warpfield_params)
        # if collapse
        elif phase0 == set_phase.collapse: 
            Msh = Msh0 * r / r

        ################## CLOUDY ###########################################################################################
        if make_cloudy_model:
            jj_cloudy = operations.find_nearest_higher(t, t_cloudy)
            r_cloudy = r[jj_cloudy]
            v_cloudy = rd[jj_cloudy]
            Msh_cloudy = Msh[jj_cloudy]
            # if density high enough or cloud not fully swept, make normal shell (+static) cloudy model
            if ((nmax > warpfield_params.stop_n_diss) or (max(np.concatenate([rSmom,r])) < rCloud)):
                # print("write cloudy model at t=", t_cloudy)
                Lw_cloudy = fLw_evo(t_cloudy) / ((u.s.to(u.Myr)**3/u.g.to(u.Msun)/u.cm.to(u.pc)**2))
                pdot_cloudy = fpdot_evo(t_cloudy) * u.Myr.to(u.s) / (c.M_sun.cgs.value * ( u.km/u.s).to(u.cm/u.s))
                vterminal_cloudy = 2. * Lw_cloudy / pdot_cloudy
                P_cloudy = get_bubbleParams.pRam(r_cloudy, Lw_cloudy, vterminal_cloudy)
                nShell0, nShell0_cloud = get_shellParams.get_nShell0(P_cloudy, warpfield_params.t_ion, warpfield_params)

                __cloudy__.create_model(cloudypath, warpfield_params.sfe, mCloud, i.namb, warpfield_params.metallicity, n0_cloudy, r_cloudy, v_cloudy, Msh_cloudy,
                             np.log10(fLbol_evo(t_cloudy)), t_cloudy, rCloud, nedge,
                             warpfield_params,
                             SB99model=i.SB99cloudy_file, shell=i.cloudy_stopmass_shell, turb=i.cloudy_turb,
                             coll_counter=coll_counter, Tarr=Tarr, Larr=Larr, Li=fLi_evo(thalf) , Qi=fQi_evo(thalf),
                             pdot_tot=fpdot_evo(thalf), Lw_tot=fLw_evo(thalf), Mcluster=mCluster, phase=phase0)
            else:
                # shell = False
                __cloudy__.create_model(cloudypath, warpfield_params.sfe, mCloud, np.nan, warpfield_params.metallicity, np.nan, r_cloudy, v_cloudy, Msh_cloudy,
                             np.log10(fLbol_evo(t_cloudy)), t_cloudy, rCloud, nedge,
                             warpfield_params,
                             SB99model=i.SB99cloudy_file, shell=False, turb=i.cloudy_turb,
                             coll_counter=coll_counter, Tarr=Tarr, Larr=Larr, Li=fLi_evo(thalf) , Qi=fQi_evo(thalf),
                             pdot_tot=fpdot_evo(thalf), Lw_tot=fLw_evo(thalf), Mcluster=mCluster, phase=set_phase.dissolve)
                #printl("Due to a very low shell density, I am not writing any cloudy input files, nmax = {}".format(nmax), verbose=1)
        #####################################################################################################################


        rmom_end = r[-1]
        vmom_end = rd[-1]
        tmom_end = t[-1]

        tSmom = np.concatenate([tSmom, [t0]])
        rSmom = np.concatenate([rSmom, [r0]])
        vSmom = np.concatenate([vSmom, [v0]])
        fabsmom = np.concatenate([fabsmom, [fabs]]) # TO DO: if there was a phase switch I actually need to do an interpolation for quantities marked with X
        fabs_i_mom = np.concatenate([fabs_i_mom, [fabs_i]]) # X
        fabs_n_mom = np.concatenate([fabs_n_mom, [fabs_n]])  # X
        dRs_mom = np.concatenate([dRs_mom, [dRs]]) # X
        ionshmom = np.concatenate([ionshmom, [ionsh]]) # X
        Mshell_mom = np.concatenate([Mshell_mom, [Msh0]])
        phase_mom = np.concatenate([phase_mom, [phase0]])

        n0mom = np.concatenate([n0mom, [n0]])
        n0_cloudymom = np.concatenate([n0_cloudymom, [n0_cloudy]])

        FSgrav_mom = np.concatenate([FSgrav_mom, [c.Grav_au * Msh0 / r0 ** 2 * (mCluster + Msh0 / 2.)]])
        FSwind_mom = np.concatenate([FSwind_mom, [pdot - pdot_SNe]])
        FSradp_mom = np.concatenate([FSradp_mom, [fabs * Lbol / (c.c.to(u.pc/u.Myr).value)]]) #  X
        FSsne_mom = np.concatenate([FSsne_mom, [pdot_SNe]])
        FSIR_mom = np.concatenate([FSIR_mom, [fabs * Lbol / (c.c.to(u.pc/u.Myr).value) * tau_IR]]) # X
        nmax_mom = np.concatenate([nmax_mom, [nmax]]) # X
        logMcluster_mom = np.concatenate([logMcluster_mom, [np.log10(mCluster)]])
        logMcloud_mom = np.concatenate([logMcloud_mom, [np.log10(mCloud)]])
        Lwmom = np.concatenate([Lwmom, [Lw]])

        # increment stopping time: new start time is old stopping time
        #if phase0 == phase1:
        #    t0 = tStop_i
        #else: t0 = t[-1]
        t0 = t[-1]

        # if abs(t0-5.0)<0.1: shell_ploton = True
        # else: shell_ploton = False

        r0 = r[-1]
        v0 = rd[-1]
        Msh0 = Msh[-1]
        dt_min_diss = 2

        if (nmax < warpfield_params.stop_n_diss):
            # is this the first time the shell density drops very low? If so, set first_dissolve to True and mark the time
            if first_dissolve is False:
                first_dissolve = True
                tdis = tStop_i
            # If this is not the first time, i.e. the density has been low for some time, check how long ago the density dropped low. If long enough in the past (e.g. 1 c.Myr), stop simulation
            # minimum time span during which the max density of shell has 
            # to fall below ndiss, in order to warrant stoppage of simulation (Myr).
            elif (tStop_i - tdis) >= dt_min_diss:
                print("Shell has dissolved! The given threshold for dissolution is %.2f, but a maximum denstiy of %.2f is reached."%(warpfield_params.stop_n_diss, nmax))
                break
        # if the density is high or the simulation is young, first_dissolve is reset to False
        # for break, continous low density is necessary
        else:
            first_dissolve = False

    Data_m = {'t': tSmom, 'r': rSmom, 'v': vSmom, 't_end': tmom_end, 'r_end': rmom_end, 'v_end': vmom_end,
              'fabs': fabsmom, 'Fgrav': FSgrav_mom, 'Fwind': FSwind_mom,
              'Fradp_dir': FSradp_mom, 'FSN': FSsne_mom, 'Fradp_IR': FSIR_mom,
              'fabs_i': fabs_i_mom, 'fabs_n': fabs_n_mom, 'dRs': dRs_mom, 'nmax': nmax_mom,
              'logMcluster': logMcluster_mom, 'logMcloud': logMcloud_mom, 'logMshell': np.log10(Mshell_mom),
              'phase': phase_mom, 'Lmech': Lwmom}

    
    return


