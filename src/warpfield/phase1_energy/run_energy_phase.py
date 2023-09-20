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
from src.warpfield.cooling.non_CIE import read_cloudy
import src.warpfield.bubble_structure.bubble_luminosity as bubble_luminosity

from src.input_tools import get_param
warpfield_params = get_param.get_param()

def run_energy(t0, y0, #r0, v0, E0, T0
               ODEpar,
                tcoll, coll_counter,
                shell_dissolved, t_shelldiss,
                stellar_outputs, # old code: SB99_data
                SB99f,
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
    
    # the energy-driven phase
    # winds hit the shell --> reverse shock --> thermalization
    # shell is driven mostly by the high thermal pressure by the shocked ISM, also (though weaker) by the radiation pressure, at late times also SNe

    # -----------
    # Describing the free-expanding phase
    # We consider first region (c) of swept-up interstellar
    # gas, whose outer boundary, at R2, is a shock separating
    # it from the ambient interstellar gas (d), and whose
    # inner boundary, at Rc, is a contact discontinuity
    # separating it from the shocked stellar wind (b). The
    # structure of this region can be described by a similarity
    # solution (Avedisova 1972). Our calculation parallels
    # the theory of the adiabatic blast wave given by Taylor
    # (1950); the only substantive difference in the case at
    # hand is that the energy is fed into the system at a
    # constant rate instead of in an initial blast.
    # -----------

    # get cooling cube
    cooling_data, heating_data = read_cloudy.get_coolingStructure(t0)

    # =============================================================================
    # Now, we begin Energy-driven calculations (Phase 1)
    # =============================================================================
    # header
    terminal_prints.phase1()
    
    mypath = warpfield_params.out_dir

    # -----------
    # Step1: Obtain initial values
    # -----------
        
    # first stopping time (will be incremented at beginning of while loop)
    # start time t0 will be incremented at end of while loop
    tStop_i = t0
    # get data from stellar evolution code output
    # unit of t_evo is Myr, the other units are cgs
    # See read_SB99.read_SB99 for documentation.
    t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo = stellar_outputs 
    
    # Question: isnt this already handelled in main.py in read_SB99?
    # Also, why is this linear instead?
    # interpolation functions for SB99 values
    fQi_evo = scipy.interpolate.interp1d(t_evo, Qi_evo, kind = 'linear')
    fLi_evo = scipy.interpolate.interp1d(t_evo, Li_evo, kind = 'linear')
    fLn_evo = scipy.interpolate.interp1d(t_evo, Ln_evo, kind = 'linear')
    fLbol_evo = scipy.interpolate.interp1d(t_evo, Lbol_evo, kind = 'linear')
    fLw_evo = scipy.interpolate.interp1d(t_evo, Lw_evo, kind = 'linear')
    fpdot_evo = scipy.interpolate.interp1d(t_evo, pdot_evo, kind = 'linear')

    # mechanical luminosity at time t0 (erg)
    Lw0 = fLw_evo(t0) * u.erg / u.s
    # momentum of stellar winds at time t0 (cgs)
    pdot0 = fpdot_evo(t0) * u.g * u.cm / u.s**2
    # terminal wind velocity at time t0 (km/s)
    vterminal0 = (2. * Lw0 / pdot0).to(u.km/u.s)
    
    # bubble parameter values at time t0 (radius, velocity, energy, temperature)
    # r0 (pc), v0 = vterminal0 (km/s), E0 (erg), T0 (K)
    r0, v0, E0, T0 = y0
    
    # Some cloud properties 
    rCloud = ODEpar['rCloud']
    mCloud = ODEpar['mCloud']
    
    # print('\n\ncheckpoint1')
    # print('Lw0, pdot0, vterminal0')
    # print(Lw0.to(u.M_sun * u.pc**2 / u.Myr**3), pdot0.to(u.M_sun * u.pc / u.Myr**2), vterminal0)
    # sys.exit()

    # -----------
    # Solve equation for inner radius of the inner shock.
    # -----------
    
    # print('\n\nvalues to solve r1')
    # print(                       r0,  Lw0, 
    #                                   E0, 
    #                                   vterminal0, 
    #                                   r0)
                                      
    # initial radius of inner discontinuity [pc]
    R1 = (scipy.optimize.brentq(get_bubbleParams.get_r1, 
                               1e-3 * r0.to(u.cm).value, r0.to(u.cm).value, 
                               args=([Lw0.to(u.erg/u.s).value, 
                                      E0.to(u.erg).value, 
                                      vterminal0.to(u.cm/u.s).value, 
                                      r0.to(u.cm).value
                                      ])) * u.cm)\
                                .to(u.pc)#back to pc
    print(f'Inner discontinuity: {R1}.')
    
    # initial energy derivative
    # Question: why?
    Ebd0 = 0. 
    E0m1 = 0.9*E0
    t0m1 = 0.9*t0
    r0m1 = 0.9*r0
    
    # print('\n\ncheckpoint2')
    # print(R1, E0m1, t0m1, r0m1)
    # sys.exit()
 
    # -----------
    # Solve equation for mass and pressure within bubble (r0)
    # -----------
    
    # The initial mass [Msol]
    Msh0 = mass_profile.get_mass_profile(r0, rCloud, mCloud, return_mdot = False)
    # The initial pressure [cgs - g/cm/s2, or dyn/cm2]
    P0 = get_bubbleParams.bubble_E2P(E0, r0, R1)
    # How long to stay in Weaver phase? Until what radius?
    if warpfield_params.density_gradient == True:
        rfinal = np.min([rCloud.cgs.value, warpfield_params.rCore.cgs.value])
    else:
        rfinal = rCloud
    
    print(f'Initial bubble mass: {Msh0}')
    print(f'Initial bubble pressure: {P0.to(u.M_sun/u.pc/u.Myr**2)}')
    
    
    # print('checkpoint3')
    # print(E0, Msh0, P0, rfinal)
    # 3.596718555609108e+49 erg 0.5953786133541732 solMass 0.0012556410178208998 g / (cm s2) 90.91228527839561 pc
    
    
    # Calculate bubble structure
    # preliminary - to be tested
    # This should be something in bubble_structure.bubble_wrap(), which is being called in phase_solver2. 
    
    
    alpha = 0.6
    beta = 0.8
    delta = -0.17142857142857143
    Eb = 11858019.317814864 * u.M_sun * u.pc**2 / u.Myr**2
    # Why is this 0.2?
    # change back? in old code R2 = r0.
    R2 = 90.0207551764992493 * u.pc
    R2 = r0
    print('r0', r0)
    # 
    t_now = 0.00012057636642393612 * u.Myr
    Lw =  201648867747.70163 * u.M_sun * u.pc**2 / u.Myr**3
    vw = 3810.2196532385897 * u.km / u.s
    dMdt_factor = 1.646
    Qi = 1.6994584609226492e+67 / u.Myr
    v0 = 0.0 * u.km / u.s 
    T_goal = 3e4 * u.K
    r_inner = R1
    rgoal = 0.18186796588493245 * u.pc


    # {'v0': 0.0, 'cons': {'a': 4976.099527584466, 'b': 5213.056647945631,
    #                      'c': 6.2274324244100785e+25, 'd': 380571798.5188472,
    #                      'e': 3080.442564695146, 't_now': 0.00012057636642393612,
    #                      'Qi': 1.6994584609226492e+67}, 'rgoal': 0.18186796588493245,
    #  'Tgoal': 30000.0, 'R2': 0.20207551764992493, 'R_small': 0.14876975625376893,
    #  'press': 380571798.5188472}

    bubble_prop = bubble_luminosity.get_bubbleproperties(t_now, T_goal, rgoal,
                                                         r_inner, R2,
                                                         Qi, alpha, beta, delta,
                                                         Lw, Eb, vw, v0,
                                                         )
    
    
    
    
    
    
    # Then, turn on cooling gradually. I.e., reduce the amount of cooling at very early times. 
    
    
    
    
    
    # Calculate shell structure.
    # preliminary - to be tested
    shell_prop = shell.shell_structure()
    
    
    
    
    
    # Calculate bubble mass
    bubbble_mass = mass_profile.calc_mass()
    
    
    
    
    # Get new values for next loop.
    
    
    return
    
 










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









