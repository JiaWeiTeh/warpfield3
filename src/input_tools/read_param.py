#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:33:31 2022

@author: Jia Wei Teh

This script contains a function that reads in parameter file and passes it to WARPFIELD. 
The function will also create a summary.txt file in the output directory.

"""

from pathlib import Path
import random # for random numbers
import numpy as np
      
def read_param(path2file, write_summary = True):    
    """
    This function takes in the path to .param file, and returns a dictionary of parameters.
    Additionally, this function filters out non-useful parameters, then write them into 
    a .txt summary file in the output directory.

    Parameters
    ----------
    path2file : str
        Path to the .param file.
    write_summary: boolean
        Whether or not to write a summary .txt file.

    Returns
    -------
    params_dict : dict
        Dictionary of parameters.

    """
    # =============================================================================
    # Create list from input
    # =============================================================================
    with open(path2file, 'r') as f:
        # grab all parameters (non-comments and non-newlines)
        params_input_list = [line.strip() for line in f\
                        if line.strip() and not line.strip().startswith("#")]

    
    # =============================================================================
    # Record inputs
    # =============================================================================
    # Routine for initialising parameters
    # First, initialise dictionary with default values. Includes all parameter
    params_dict = {'model_name': ['example'], 
                   'out_dir': ['def_dir'], 
                   'verbose': ['1'], 
                   'output_format': ['ASCII'], 
                   'rand_input': ['0'], 
                   'log_mCloud': ['6.0'], 
                   'mCloud_beforeSF': ['1'], 
                   'sfe': ['0.01'], 
                   'n_cloud': ['1000'], 
                   'metallicity': ['0.15'], 
                   'stochastic_sampling': ['0'], 
                   'n_trials': ['1'], 
                   'rand_log_mCloud': ['5', ' 7.47'], 
                   'rand_sfe': ['0.01', '0.10'], 
                   'rand_n_cloud': ['100.', ' 1000.'], 
                   'rand_metallicity': ['0.15', ' 1'], 
                   'mult_exp': ['0'], 
                   'r_coll': ['1.0'], 
                   'mult_SF': ['1'], 
                   'sfe_tff': ['0.01'], 
                   'imf': ['kroupa.imf'], 
                   'stellar_tracks': ['geneva'], 
                   'dens_cloud': ['1000'], 
                   'dens_profile': ['bE_prof'], 
                   'dens_g_bE': ['14.1'], 
                   'dens_a_pL': ['-2'], 
                   'dens_rcore': ['0.099'], 
                   'frag_enabled': ['0'], 
                   'frag_r_min': ['0.1'], 
                   'frag_grav': ['0'], 
                   'frag_grav_coeff': ['0.67'], 
                   'frag_RTinstab': ['0'], 
                   'frag_densInhom': ['0'], 
                   'frag_cf': ['1'], 
                   'frag_enable_timescale': ['1'], 
                   'stop_n_diss': ['1'], 
                   'stop_t_diss': ['1.0'], 
                   'stop_r': ['1e3'], 
                   'stop_t': ['15.05'], 
                   'stop_t_unit': ['Myr'], 
                   'write_main': ['1'], 
                   'write_stellar_prop': ['0'], 
                   'write_bubble': ['0'], 
                   'inc_grav': ['1'], 
                   'f_Mcold_W': ['0.0'], 
                   'f_Mcold_SN': ['0.0'], 
                   'v_SN': ['1e9'], 
                   'sigma0': ['1.5e-21'], 
                   'z_nodust': ['0.05'], 
                   'u_n': ['2.1287915392418182e-24'], 
                   'u_p': ['1.0181176926808696e-24'], 
                   't_ion': ['1e4'], 
                   't_neu': ['100'], 
                   'n_ISM': ['0.1'], 
                   'kappa_IR': ['4'], 
                   'thermcoeff_wind': ['1.0'], 
                   'thermcoeff_SN': ['1.0']
                   }
    
    # =============================================================================
    # Check if parameters given in .param file makes sense
    # =============================================================================
    
    # TODO. E.g., if metalicity is >0, dens profile str is correct, etc. 
    
    
    
    
    
    # Then, for parameters specified in .param file, update dictionary and use the
    # specified values instead of the default.
    for pairs in params_input_list:
        param, value = pairs.split(maxsplit = 1)
        value = value.split(',')
        params_dict[param] = value
    
    # =============================================================================
    # Here we deal with conditional parameters.
    # =============================================================================
    
    # Check if random input is desired
    if params_dict['rand_input'][0] == '1':
        # if yes, read limits. Note: even if the user mixed up max/min values, random will deal with that.
        minM, maxM = params_dict['rand_log_mCloud']
        minSFE, maxSFE = params_dict['rand_sfe']
        # take random input from uniform distribution in log space
        # also round to three decimal places
        params_dict['log_mCloud'] = str(round(
                                        random.uniform(
                                            float(minM), float(maxM)
                                            ),
                                        3
                                        )
                                    ).split() # split() to record as list.
        params_dict['sfe'] = str(round(
                                        random.uniform(
                                            float(minSFE), float(maxSFE)
                                            ),
                                        3
                                        )
                                    ).split()
        params_dict['metallicity'] = np.random.choice(
                                        ['0.15', '1'] #TODO add distribution once metallicity is done
                                        ).split()
        params_dict['n_cloud'] = np.random.choice(
                                        ['100', '1000'] #TODO add distribution once metallicity is done
                                        ).split()
    elif params_dict['rand_input'][0] == '0':
        params_dict.pop('rand_log_mCloud')
        params_dict.pop('rand_sfe')
        params_dict.pop('rand_n_cloud')
        params_dict.pop('rand_metallicity')
        
    # =============================================================================
    # Store only meaningful parameters into the summary.txt file
    # =============================================================================
    # First, grab output directory
    if params_dict['out_dir'][0] == 'def_dir':
        # If user did not specify, the directory will be set as ./outputs/ 
        # check if directory exists; if not, create one.
        path2output = r'./outputs/'
        Path(path2output).mkdir(parents=True, exist_ok = True)
        params_dict['out_dir'] = path2output.split()
    else:
        # if instead given a path, then use that instead
        path2output = str(params_dict['out_dir'][0])
        Path(path2output).mkdir(parents=True, exist_ok = True)
        params_dict['out_dir'] = path2output.split()
    
    # Then, organise dictionary so that it does not include useless info
    # Remove fragmentation if frag_enabled == 0
    if params_dict['frag_enabled'][0] == '0':
        params_dict.pop('frag_r_min')
        params_dict.pop('frag_grav')
        params_dict.pop('frag_grav_coeff')
        params_dict.pop('frag_RTinstab')
        params_dict.pop('frag_densInhom')
        params_dict.pop('frag_cf')
        params_dict.pop('frag_enable_timescale')
    
    # Remove stochasticity related parameters
    if params_dict['stochastic_sampling'][0] == '0':
        params_dict.pop('n_trials')
    
    # Remove unrelated parameters depending on selected density profile
    if params_dict['dens_profile'][0] == 'bE_prof':
        params_dict.pop('dens_a_pL')
    elif params_dict['dens_profile'][0] == 'pL_prof':
        params_dict.pop('dens_g_bE')
    
    if write_summary:
        # Store into summary file as output.
        with open(path2output+params_dict['model_name'][0]+'_summary.txt', 'w') as f:
            # header
            f.writelines('\
        # =============================================================================\n\
        # Summary of \'%s\' run.\n\
        # =============================================================================\n\n\
        '%(params_dict['model_name'][0]))
            # body
            for key, val in params_dict.items():
                f.writelines(key+'    '+",".join(val)+'\n')
            # close
            f.close()
        
    return params_dict
