#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:46:31 2022

@author: Jia Wei Teh

This script contains a wrapper that initialises the expansion of 
shell.
"""

# libraries
import numpy as np
import datetime
import sys
import yaml
import os

#--
from src.warpfield.phase0_init import (get_InitCloudProp, get_InitBubStruc,
                                        get_InitCloudyDens, get_InitPhaseParam,
                                        set_phase)
from src.warpfield.sb99 import read_SB99
from src.warpfield.phase1_energy import run_energy_phase

# get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()


def start_expansion():
    """
    This wrapper takes in the parameters and feed them into smaller
    functions.

    Parameters
    ----------
    params : Object
        An object describing WARPFIELD parameters.

    Returns
    -------
    None.

    """
    # Note:
        # old code: expansion_main()
    
# TODO: make sure there is no confusion between mCloud (before and after)
    # =============================================================================
    # Step 0: Preliminary stuffs.
    # =============================================================================
    
    # Record timestamp
    startdatetime = datetime.datetime.now()
    # print some useful information
    print("Warpfield is running now with the following parameters:")
    print(f'Name: {warpfield_params.model_name}')
    print(f'log_mCloud: {warpfield_params.log_mCloud}')
    print(f'sfe: {warpfield_params.sfe}')
    print(f'metallicity: {warpfield_params.metallicity}')
    print(f'density profile: {warpfield_params.dens_profile}')
    sys.exit()
    
    # TODO: This shouldn't be required - because there should be an operation at the very
    # beginning that deals with this.
    
    # However, we still need to provide function that creasts '_evo' etc suffix files. 
    
    # # prepare directories and write some basic files (like the file cotaining the input parameters)
    # mypath, cloudypath, outdata_file, figure_file = warp_writedata.getmake_dir(i.basedir, i.navg, i.Zism, SFE,
    #                                                                            Mcloud_au_INPUT, SB99_data)
    
    # output path
    # params_dict['out_dir']+params_dict['model_name']+'_summary.txt'
    # e.g.,  mypath = r'./outputs/default/'
    mypath = warpfield_params.out_dir
    
    # General setup
    # time where SF event happens
    tSF = 0.
    # number of re-collapses (0 if only 1 cluster present)
    ii_coll = 0
    # time array
    t = np.array([tSF])
    # radius array
    r = np.array([1.e-7])
    # shell velocity array
    v = np.array([3000.])
    # energy of bubble/shell array
    E = np.array([1.])
    # temperature array
    T = np.array([100.])
    
    # =============================================================================
    # A: Initialising cloud properties. 
    # =============================================================================
    
    # Step 1: Obtain initial cloud properties
    # note now that the parameter mCloud here is the cloud mass AFTER star formation.
    rCore, bE_T, rCloud, nEdge, mCloud, mCluster = get_InitCloudProp.get_InitCloudProp(warpfield_params)
    
    # Now, set up density parameter
    if warpfield_params.dens_profile == "bE_prof":
        density_specific_param = bE_T
    elif warpfield_params.dens_profile == "pL_prof":
        density_specific_param = rCore
        
    # Turn into dictionary
    ODEpar = {'mCloud': mCloud,
              'nEdge': nEdge,
              'rCloud': rCloud,
              'rCore': rCore,
              'mCluster': mCluster,
              't_dissolve': 1e30,
              'Rsh_max': 0.,
              }
    
    # Additional parameters
    # set the maximum shell radius achieved during this expansion to 0. (i.e. the shell has not started to expand yet)
    # set dissolution time to arbitrary high number (i.e. the shell has not yet dissolved)
    ODEpar['Rsh_max'] = 0.
    ODEpar['tSF_list'] = np.array([tSF])
    ODEpar['Mcluster_list'] = np.array([ODEpar['Mcluster_au']])
    ODEpar['tStop'] = warpfield_params.stop_t
    ODEpar['mypath'] = mypath 
    
    if warpfield_params.dens_profile == 'bE_prof':
        ODEpar['density_specific_param'] = bE_T
    elif warpfield_params.dens_profile == 'pL_prof':
        ODEpar['density_specific_param'] = rCore
        
    #  ODEpar['density_specific_param'] added.
    #  ODEpar['gamma'] = myc.gamma removed.
    #  ODEpar['Mcloud_au'] = Mcloud_au renamed to ODEpar['mCloud'] = mCloud
    #  ODEpar['Rcloud_au'] = rcloud_au renamed to ODEpar['rCloud'] = rCloud
    #  ODEpar['Rcore_au'] = Rcore_au renamed to ODEpar['rCore'] = rCore
    #  ODEpar['Mcluster_au'] = Mcluster_au renamed to ODEpar['mCluster'] = mCluster
    #  ODEpar['nalpha'] = i.nalpha removed.
    #  ODEpar['SFE'] = SFE removed.
    #  ODEpar['rhocore_au'] = i.rhoa_au removed, because i.rhoa_au is rhoCore, and it can be calculated from scratch.
    
    # TODO: check if the dictionaries have values. Ex: Mcluster_au does not exist
    # anymore, so they should not apper; the interface will not say it is wrong!
    
    # Step 2: Obtain parameters from Starburst99
    # Scaling factor for cluster masses. Though this might only be accurate for
    # high mass clusters (~>1e5) in which the IMF is fully sampled.
    factor_feedback = ODEpar['mCluster'] / warpfield_params.SB99_mass
    # Get SB99 data. This function returns data and interpolation functions.
    SB99_data, SB99f = read_SB99.read_SB99( warpfield_params.metallicity,
                                                  rotation = warpfield_params.SB99_rotation, 
                                                  f_mass = factor_feedback, BHcutoff = warpfield_params.SB99_BHCUT)
    # if tSF != 0.: we would actually need to shift the feedback parameters by tSF
    
    # create density law for cloudy
    get_InitCloudyDens.get_InitCloudyDens(mypath, density_specific_param,
                                          ODEpar['rCloud'], ODEpar['mCloud'], 
                                          coll_counter = ii_coll)
    
    # get initial bubble structure and path to where the file is saved.
    get_InitBubStruc.get_InitBubStruc(warpfield_params.mCloud, warpfield_params.sfe, warpfield_params['out_dir'])


    ########### Simulate the Evolution (until end time reached, cloud dissolves, or re-collapse happens) ###############


    # =============================================================================
    # Begin WARPFIELD simulation.
    # =============================================================================


    # MAIN WARPFIELD CODE
    t1, r1, v1, E1, T1 = run_expansion(ODEpar, SB99_data, SB99f, mypath, cloudypath)
    t = np.append(t, t1)
    r = np.append(r, r1)
    v = np.append(v, v1)
    E = np.append(E, E1)
    T = np.append(T, T1)
 
    
    
    # write data (make new file) and cloudy data
    # (this must be done after the ODE has been solved on the whole interval between 0 and tcollapse (or tdissolve) because the solver is implicit)
    warp_writedata.warp_reconstruct(t1, [r1,v1,E1,T1], ODEpar, SB99f, ii_coll, cloudypath, outdata_file, data_write=i.write_data, cloudy_write=i.write_cloudy, append=False)








    ########### STEP 2: In case of recollapse, prepare next expansion ##########################

    # did the expansion stop because a recollapse occured? If yes, start next expansion
    while set_phase.check_simulation_status(t[-1], r[-1], v[-1], warpfield_params) == set_phase.coll:

        ii_coll += 1

        # run expansion_next
        t1, r1, v1, E1, T1, ODEpar, SB99_data, SB99f = expansion_next(t[-1], ODEpar, SB99_data, SB99f, mypath, cloudypath, ii_coll)
        t = np.append(t, t1 + t[-1])
        r = np.append(r, r1)
        v = np.append(v, v1)
        E = np.append(E, E1)
        T = np.append(T, T1)

        # write data (append to old file) and cloudy data
        # TOASK: WHY RERUN RECONSTRUCT?
        # how to call cf in reconstruct?
        warp_writedata.warp_reconstruct(t1, [r1,v1,E1,T1], ODEpar, SB99f, ii_coll, cloudypath, outdata_file, data_write=i.write_data, cloudy_write=i.write_cloudy, append=True)

    # write success message to file
    with open(success_file, "w") as text_file:
        text_file.write("Stopped because...")
        if abs(t[-1]-i.tStop) < 1e-3: text_file.write("end time reached")
        elif (abs(r[-1]-i.rcoll) < 1e-3 and v[-1] < 0.): text_file.write("recollapse")
        else: text_file.write("unknown")

    return 0
    
    
    #%%
    
def run_expansion(ODEpar, SB99_data, SB99f, mypath, cloudypath):
    """
    Model evolution of the cloud (both energy- and momentum-phase) until next recollapse or (if no re-collapse) until end of simulation
    :param ODEpar:
    :param SB99_data:
    :param SB99f:
    :param mypath:
    :param cloudypath:
    :return:
    """

    print('here we enter run_expansion.')
    ii_coll = 0
    tcoll = [0.]
    tStop = ODEpar['tStop']

    ######## STEP A: energy-phase (explicit) ###########
    # t0 = start time for Weaver phase
    # y0 = [r0, v0, E0, T0]
    # r0 = initial separation (pc)
    # v0 = initial velocity (km/s)
    t0, y0 = get_InitPhaseParam.get_y0(0., SB99f, warpfield_params)
    # print(t0)
    # 6.506818386985495e-05
    # print(y0)
    # [0.23790232199299727, 3656.200432285518, 5722974.028981317, 67741779.55773313]
    # sys.exit('stop')
    Cool_Struc = get_startvalues.get_firstCoolStruc(i.Zism, t0 * 1e6, warpfield_params)

    shell_dissolved = False
    t_shdis = 1e99

    tfinal = t0 + 30. * i.dt_Estart
    #print "tfinal:", tfinal

    [Dw, shell_dissolved, t_shdis] = run_energy_phase.run_energy(t0, y0, ODEpar['Rcloud_au'], SB99_data,
                                                                ODEpar['Mcloud_au'], ODEpar['SFE'], mypath,
                                                                cloudypath, shell_dissolved, t_shdis,
                                                                ODEpar['Rcore_au'], i.nalpha, ODEpar['Mcluster_au'],
                                                                ODEpar['nedge'], Cool_Struc, tcoll=tcoll,
                                                                coll_counter=ii_coll, tfinal=tfinal)

    ######## STEP B: energy-phase (implicit) ################
    # if (aux.check_continue(Dw['t_end'], Dw['r_end'], Dw['v_end']) == ph.cont):

    params = {'t_now': Dw['t_end'], 'R2': Dw['r_end'], 'Eb': Dw['E_end'], 'T0': Dw['Tb'][-1], 'beta': Dw['beta'][-1],
              'delta': Dw['delta'][-1], 'alpha': Dw['alpha'][-1], 'dMdt_factor': Dw['dMdt_factor_end']}
    params['alpha'] = Dw['v_end'] * (params["t_now"]) / params['R2'] # this is not quite consistent (because we are only using rough guesses for beta and delta) but using a wrong value here means also using the wrong velocity
    params['temp_counter'] = 0
    params['mypath'] = mypath

    
    
    tfeed=np.linspace(5e-03, 40,num=6000)
    Qifee=SB99f['fQi_cgs'](tfeed)
    Lifee=SB99f['fLi_cgs'](tfeed)
    Lnfee=SB99f['fLn_cgs'](tfeed)
    Lbolfee=SB99f['fLbol_cgs'](tfeed)
    Lwfee=SB99f['fLw_cgs'](tfeed)
    pdotfee=SB99f['fpdot_cgs'](tfeed)
    pdot_SNefee=SB99f['fpdot_SNe_cgs'](tfeed)

    
    psoln_energy, params = phase_energy.run_phase_energy(params, ODEpar, SB99f)

    t = psoln_energy.t
    r = psoln_energy.y[0]
    v = psoln_energy.y[1]
    E = psoln_energy.y[2]
    T = psoln_energy.y[3]

    ######### STEP C: transition phase ########################
    if (aux.check_continue(t[-1], r[-1], v[-1], tStop) == ph.cont):


        t0 = psoln_energy.t[-1]
        y0 = [psoln_energy.y[0][-1], psoln_energy.y[1][-1], psoln_energy.y[2][-1], psoln_energy.y[3][-1]]
        cs = params['cs_avg']

        psoln_transition = phase_transition.run_phase_transition(t0, y0, cs, ODEpar, SB99f)
        #print psoln_transition

        t = np.append(t, psoln_transition.t)
        r = np.append(r, psoln_transition.y[0])
        v = np.append(v, psoln_transition.y[1])
        E = np.append(E, psoln_transition.y[2])
        T = np.append(T, psoln_transition.y[3])

        ######### STEP D: momentum phase ##############################
        if (aux.check_continue(t[-1], r[-1], v[-1], tStop) == ph.cont):
            t0 = psoln_transition.t[-1]
            y0 = [psoln_transition.y[0][-1], psoln_transition.y[1][-1], psoln_transition.y[2][-1],
                  psoln_transition.y[3][-1]]

            psoln_momentum = phase_momentum.run_fE_momentum(t0, y0, ODEpar, SB99f)
            #print psoln_momentum

            t = np.append(t, psoln_momentum.t)
            r = np.append(r, psoln_momentum.y[0]);
            v = np.append(v, psoln_momentum.y[1]);
            E = np.append(E, psoln_momentum.y[2]);
            T = np.append(T, psoln_momentum.y[3]);

    return t, r, v, E, T



