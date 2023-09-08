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
from warpfield.phase0_init import (get_InitCloudProp, get_InitBubStruc,
                                        get_InitCloudyDens, get_InitPhaseParam)
from src.warpfield.phase_general import set_phase
from src.warpfield.sb99 import read_SB99
from src.warpfield.phase1_energy import run_energy_phase
from src.warpfield.phase1b_energy_implicit import run_energy_implicit_phase
from src.warpfield.phase1c_transition import run_transition_phase
from src.warpfield.phase2_momentum import run_momentum_phase
from src.warpfield.cloudy import __cloudy__
from src.warpfield.cooling import read_opiate
import src.warpfield.functions.terminal_prints as terminal_prints
import src.output_tools.write_outputs as write_outputs


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
    
    # from src.warpfield.cooling.non_CIE import read_cloudy
    # from src.warpfield.cooling.net_coolingcurve import get_dudt
    # os.environ['PATH_TO_CONFIG'] = r'/Users/jwt/Documents/Code/warpfield3/outputs/example_pl/example_pl_config.yaml'

    # test cases
    # cooling_data, heating_data = read_cloudy.get_coolingStructure(1e6)
    # cooling_data, heating_data = read_cloudy.get_coolingStructure(1.5e6)
    # print(cooling_data.temp)
    # print(cooling_data.interp([-1, 3.6, 2]))
    # print(heating_data.ndens)
    # print(heating_data.interp([-1, 3.6, 2]))    
    # print(logT)
    # print(logLambda)
    # 
    # ndens = np.log10(506663.2212419483)
    # T = 5.51
    # # T = np.log10(294060.78931362595)
    # phi = np.log10(1.5473355225629384e+16)
    # dudt  = get_dudt(1e6, 10**ndens, 10**T, 10**phi)
    # print(dudt)
    
    # sys.exit()
    
    # Note:
        # old code: expansion_main()
    
    # TODO: make sure there is no confusion between mCloud (before and after)
    
    # =============================================================================
    # Step 0: Preliminary stuffs.
    # =============================================================================
    
    terminal_prints.phase0()
    
    # Record timestamp
    startdatetime = datetime.datetime.now()
    
    # TODO: This shouldn't be required - because there should be an operation at the very
    # beginning that deals with this.
    
    # However, we still need to provide function that creasts '_evo' etc suffix files. 
    
    # prepare directories and write some basic files (like the file cotaining the input parameters)
    output_filename = write_outputs.init_dir()
    
    # output path
    # params_dict['out_dir']+params_dict['model_name']+'_summary.txt'
    # This prints '/Users/jwt/Documents/Code/warpfield3/outputs/example_run/'
    path2output = warpfield_params.out_dir
    
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
    rCloud, nEdge = get_InitCloudProp.get_InitCloudProp()
    
    # -----
    # Turn into dictionary
    ODEpar = {'mCloud': warpfield_params.mCloud,
              'nEdge': nEdge,
              'rCloud': rCloud,
              'rCore': warpfield_params.rCore,
              'mCluster': warpfield_params.mCluster,
              't_dissolve': 1e30,
              'Rsh_max': 0.,
              }
    
    # Additional parameters
    # set the maximum shell radius achieved during this expansion to 0. (i.e. the shell has not started to expand yet)
    # set dissolution time to arbitrary high number (i.e. the shell has not yet dissolved)
    ODEpar['Rsh_max'] = 0.
    ODEpar['tSF_list'] = np.array([tSF])
    ODEpar['Mcluster_list'] = np.array([ODEpar['mCluster']])
    ODEpar['tStop'] = warpfield_params.stop_t
    ODEpar['mypath'] = path2output 
    
    #  ODEpar['density_specific_param'] added.
    #  ODEpar['gamma'] = myc.gamma removed.
    #  ODEpar['Mcloud_au'] = Mcloud_au renamed to ODEpar['mCloud'] = mCloud
    #  ODEpar['Rcloud_au'] = rcloud_au renamed to ODEpar['rCloud'] = rCloud
    #  ODEpar['Rcore_au'] = Rcore_au renamed to ODEpar['rCore'] = rCore
    #  ODEpar['Mcluster_au'] = Mcluster_au renamed to ODEpar['mCluster'] = mCluster
    #  ODEpar['nalpha'] = i.nalpha removed.
    #  ODEpar['SFE'] = SFE removed.
    #  ODEpar['rhocore_au'] = i.rhoa_au removed, because i.rhoa_au is rhoCore, and it can be calculated from scratch.
    # -----
    
    # TODO: check if the dictionaries have values. Ex: Mcluster_au does not exist
    # anymore, so they should not apper; the interface will not say it is wrong!
    # Step 2: Obtain parameters from Starburst99
    # Scaling factor for cluster masses. Though this might only be accurate for
    # high mass clusters (~>1e5) in which the IMF is fully sampled.
    factor_feedback = ODEpar['mCluster'] / warpfield_params.SB99_mass
    # Get SB99 data and interpolation functions.
    SB99_data = read_SB99.read_SB99(f_mass = factor_feedback)
    SB99f = read_SB99.get_interpolation(SB99_data)
    # if tSF != 0.: we would actually need to shift the feedback parameters by tSF
    
    # create density law for cloudy
    get_InitCloudyDens.get_InitCloudyDens(path2output,
                                          ODEpar['rCloud'], ODEpar['mCloud'], 
                                          coll_counter = ii_coll)
    
    # get initial bubble structure and path to where the file is saved.
    get_InitBubStruc.get_InitBubStruc()

    # =============================================================================
    # Begin WARPFIELD simulation.
    # Simulate the Evolution (until end time reached, cloud dissolves, 
    # or re-collapse)
    # =============================================================================

    # MAIN WARPFIELD CODE
    t1, r1, v1, E1, T1 = run_expansion(ODEpar, SB99_data, SB99f)
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
        t1, r1, v1, E1, T1, ODEpar, SB99_data, SB99f = expansion_next(t[-1], ODEpar, SB99_data, SB99f, path2output, cloudypath, ii_coll)
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
    
def run_expansion(ODEpar, SB99_data, SB99f):
    """
    Model evolution of the cloud (both energy- and momentum-phase) until next recollapse or (if no re-collapse) until end of simulation
    :param ODEpar:
    :param SB99_data:
    :param SB99f:
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
    # E0 = initial energy (erg/s)
    # T0 = initial temperature (K)
    t0, y0 = get_InitPhaseParam.get_y0(0., SB99f)
    
    # print(t0)
    # 6.506818386985495e-05
    # print(y0)
    # [0.23790232199299727, 3656.200432285518, 5722974.028981317, 67741779.55773313]
    # sys.exit('stop')
    
    # get cooling function
    Cool_Struc = read_opiate.get_coolingStructure(t0 * 1e6)
    
    shell_dissolved = False
    t_shdis = 1e99

    dt_Estart = 0.0001
    tfinal = t0 + 30. * dt_Estart
    #print "tfinal:", tfinal

    [Dw, shell_dissolved, t_shdis] = run_energy_phase.run_energy(t0, y0, 
                                                                 ODEpar['rCloud'],
                                                                 ODEpar['mCloud'],
                                                                 ODEpar['mCluster'],
                                                                 ODEpar['nEdge'],
                                                                 ODEpar['rCore'],
                                                                 warpfield_params.sigma_d,
                                                                 tcoll, ii_coll,
                                                                 Cool_Struc,
                                                                 shell_dissolved, t_shdis,
                                                                 SB99_data, SB99f,
                                                                 tfinal,
                                                                 )
                                                                 
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

    
    psoln_energy, params = run_energy_implicit_phase.run_phase_energy(params, ODEpar, SB99f)

    t = psoln_energy.t
    r = psoln_energy.y[0]
    v = psoln_energy.y[1]
    E = psoln_energy.y[2]
    T = psoln_energy.y[3]

    ######### STEP C: transition phase ########################
    if (set_phase.check_simulation_status(t[-1], r[-1], v[-1], tStop) == set_phase.cont):


        t0 = psoln_energy.t[-1]
        y0 = [psoln_energy.y[0][-1], psoln_energy.y[1][-1], psoln_energy.y[2][-1], psoln_energy.y[3][-1]]
        cs = params['cs_avg']

        psoln_transition = run_transition_phase.run_phase_transition(t0, y0, cs, ODEpar, SB99f)
        #print psoln_transition

        t = np.append(t, psoln_transition.t)
        r = np.append(r, psoln_transition.y[0])
        v = np.append(v, psoln_transition.y[1])
        E = np.append(E, psoln_transition.y[2])
        T = np.append(T, psoln_transition.y[3])

        ######### STEP D: momentum phase ##############################
        if (set_phase.check_simulation_status(t[-1], r[-1], v[-1], tStop) == set_phase.cont):
            t0 = psoln_transition.t[-1]
            y0 = [psoln_transition.y[0][-1], psoln_transition.y[1][-1], psoln_transition.y[2][-1],
                  psoln_transition.y[3][-1]]

            psoln_momentum = run_momentum_phase.run_fE_momentum(t0, y0, ODEpar, SB99f)
            #print psoln_momentum

            t = np.append(t, psoln_momentum.t)
            r = np.append(r, psoln_momentum.y[0]);
            v = np.append(v, psoln_momentum.y[1]);
            E = np.append(E, psoln_momentum.y[2]);
            T = np.append(T, psoln_momentum.y[3]);

    return t, r, v, E, T




def expansion_next(tStart, ODEpar, SB99_data_old, SB99f_old, mypath, cloudypath, ii_coll):

    print("Preparing new expansion...")
    
    ODEpar['tSF_list'] = np.append(ODEpar['tSF_list'], tStart) # append time of this SF event
    dtSF = ODEpar['tSF_list'][-1] - ODEpar['tSF_list'][-2] # time difference between this star burst and the previous
    ODEpar['tStop'] = i.tStop - tStart

    # TODO: make verbose
    print('list of collapses:', ODEpar['tSF_list'])

    # get new cloud/cluster properties and overwrite those keys in old dictionary
    CloudProp2 = get_startvalues.make_new_cluster(ODEpar['Mcloud_au'], ODEpar['SFE'], ODEpar['tSF_list'], ii_coll)
    for key in CloudProp2:
        ODEpar[key] = CloudProp2[key]

    # create density law for cloudy
    get_InitCloudyDens.get_InitCloudyDens(mypath, i.namb, i.nalpha, ODEpar['Rcore_au'], ODEpar['Rcloud_au'], ODEpar['Mcluster_au'], ODEpar['SFE'], coll_counter=ii_coll)

    ODEpar['Mcluster_list'] = np.append(ODEpar['Mcluster_list'], CloudProp2['Mcluster_au']) # contains list of masses of indivdual clusters

    # new method to get feedback
    SB99_data, SB99f = read_SB99.full_sum(ODEpar['tSF_list'], ODEpar['Mcluster_list'], i.Zism, rotation=i.rotation, BHcutoff=i.BHcutoff, return_format='array')

    # get new feedback parameters
    #factor_feedback = ODEpar['Mcluster_au'] / i.SB99_mass
    #SB99_data2 = getSB99_data.load_stellar_tracks(i.Zism, rotation=i.rotation, f_mass=factor_feedback,BHcutoff=i.BHcutoff, return_format="dict") # feedback of only 2nd cluster
    #SB99_data = getSB99_data.sum_SB99(SB99f_old, SB99_data2, dtSF, return_format = 'array') # feedback of summed cluster --> use this
    #SB99f = getSB99_data.make_interpfunc(SB99_data) # make interpolation functions for summed cluster (allowed range of t: 0 to (99.9-tStart))

    if i.write_SB99 == True:
        SB99_data_write = getSB99_data.SB99_conc(SB99_data_old, getSB99_data.time_shift(SB99_data, tStart))
        warp_writedata.write_warpSB99(SB99_data_write, mypath)  # create file containing SB99 feedback info

    # for the purpose of gravity it is important that we pass on the summed mass of all clusters
    # we can do this as soon as all all other things (like getting the correct feedback) have been finished
    ODEpar['Mcluster_au'] = np.sum(ODEpar['Mcluster_list'])

    print("ODEpar:", ODEpar)

    t, r, v, E, T = run_expansion(ODEpar, SB99_data, SB99f, mypath, cloudypath)

    return t, r, v, E, T, ODEpar, SB99_data, SB99f

