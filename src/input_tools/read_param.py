#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:33:31 2022

@author: Jia Wei Teh

This script contains a function that reads in parameter file and passes it to WARPFIELD. 
The function will also create a summary.txt file in the output directory.

"""


from datetime import datetime
from pathlib import Path
import random # for random numbers
import numpy as np
import src.input_tools.input_warnings as input_warnings 

def read_param(path2file, write_summary = True):    
    """
    This function takes in the path to .param file, and returns an object containing parameters.
    Additionally, this function filters out non-useful parameters, then writes
    useful parameters into a .txt summary file in the output directory.

    Parameters
    ----------
    path2file : str
        Path to the .param file.
    write_summary: boolean
        Whether or not to write a summary .txt file.

    Returns
    -------
    params : Object
        An object describing WARPFIELD parameters.
        Example: To extract value for `sfe`, simply invoke params.sfe

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
    params_dict = {'model_name': 'example', 
                   'out_dir': 'def_dir', 
                   'verbose': 1.0, 
                   'output_format': 'ASCII', 
                   'rand_input': 0.0, 
                   'log_mCloud': 6.0, 
                   'mCloud_beforeSF': 1.0, 
                   'sfe': 0.01, 
                   'nCore': 1000.0, 
                   'rCore': 0.099, 
                   'metallicity': 1.0, 
                   'stochastic_sampling': 0.0, 
                   'n_trials': 1.0, 
                   'rand_log_mCloud': ['5', ' 7.47'], 
                   'rand_sfe': ['0.01', ' 0.10'], 
                   'rand_n_cloud': ['100.', ' 1000.'], 
                   'rand_metallicity': ['0.15', ' 1'], 
                   'mult_exp': 0.0, 
                   'r_coll': 1.0, 
                   'mult_SF': 1.0, 
                   'sfe_tff': 0.01, 
                   'imf': 'kroupa.imf', 
                   'stellar_tracks': 'geneva', 
                   'dens_profile': 'bE_prof', 
                   'dens_g_bE': 14.1, 
                   'dens_a_pL': -2.0, 
                   'dens_navg_pL': 170.0, 
                   'frag_enabled': 0.0, 
                   'frag_r_min': 0.1, 
                   'frag_grav': 0.0, 
                   'frag_grav_coeff': 0.67, 
                   'frag_RTinstab': 0.0, 
                   'frag_densInhom': 0.0, 
                   'frag_cf': 1.0, 
                   'frag_enable_timescale': 1.0, 
                   'stop_n_diss': 1.0, 
                   'stop_t_diss': 1.0, 
                   'stop_r': 1000.0, 
                   'stop_t': 15.05, 
                   'stop_t_unit': 'Myr', 
                   'write_main': 1.0, 
                   'write_stellar_prop': 0.0, 
                   'write_bubble': 0.0, 
                   'write_bubble_CLOUDY': 0.0, 
                   'write_shell': 0.0, 
                   'xi_Tb': 0.99,
                   'inc_grav': 1.0, 
                   'f_Mcold_W': 0.0, 
                   'f_Mcold_SN': 0.0, 
                   'v_SN': 1000000000.0, 
                   'sigma0': 1.5e-21, 
                   'z_nodust': 0.05, 
                   'mu_n': 2.1287915392418182e-24, 
                   'mu_p': 1.0181176926808696e-24, 
                   't_ion': 10000.0, 
                   't_neu': 100.0, 
                   'nISM': 0.1, 
                   'kappa_IR': 4.0, 
                   'gamma_adia': 1.6666666666666667, 
                   'thermcoeff_wind': 1.0, 
                   'thermcoeff_SN': 1.0,
                   'alpha_B': 2.59e-13,
                   'gamma_mag': 1.3333333333333333,
                   'log_BMW': -4.3125,
                   'log_nMW': 2.065,
                   'c_therm': 1.2e-6,
                   }
    
    # =============================================================================
    # Check if parameters given in .param file makes sense
    # =============================================================================
    # First, for parameters specified in .param file, update dictionary and use the
    # specified values instead of the default.
    for pairs in params_input_list:
        param, value = pairs.split(maxsplit = 1)
        value = value.split(',')
        if len(value) == 1:
            # Convert to float if possible
            try:
                val = float(value[0])
                # However, if output is integer, write them as
                # integer instead (for OCD purposes)
                if int(val) == val:
                    params_dict[param] = int(val)
                    params_dict[param] = val
            # otherwise remain as string
            except:
                params_dict[param] = value[0]
        else:
            params_dict[param] = value 
    
    # give warning if parameter does not make sense
    input_warnings.input_warnings(params_dict)
            

    # =============================================================================
    # Here we deal with randomised parameters.
    # =============================================================================
    # Check if random input is desired
    if params_dict['rand_input'] == 1:
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
                                    )
        params_dict['sfe'] = str(round(
                                        random.uniform(
                                            float(minSFE), float(maxSFE)
                                            ),
                                        3
                                        )
                                    )
        params_dict['metallicity'] = np.random.choice(
                                        ['0.15', '1'] #TODO add distribution once metallicity is done
                                        )
        params_dict['n_cloud'] = np.random.choice(
                                        ['100', '1000'] #TODO add distribution once metallicity is done
                                        )
    elif params_dict['rand_input'] == 0:
        params_dict.pop('rand_log_mCloud')
        params_dict.pop('rand_sfe')
        params_dict.pop('rand_n_cloud')
        params_dict.pop('rand_metallicity')
        
    # =============================================================================
    # Store only useful parameters into the summary.txt file
    # =============================================================================
    # First, grab output directory
    if params_dict['out_dir'] == 'def_dir':
        # If user did not specify, the directory will be set as ./outputs/ 
        # check if directory exists; if not, create one.
        # TODO: Add smart system that adds 1, 2, 3 if repeated default to avoid overwrite.
        path2output = r'./outputs/default/'
        Path(path2output).mkdir(parents=True, exist_ok = True)
        params_dict['out_dir'] = path2output
    else:
        # if instead given a path, then use that instead
        path2output = str(params_dict['out_dir'])
        Path(path2output).mkdir(parents=True, exist_ok = True)
        params_dict['out_dir'] = path2output
    
    # Then, organise dictionary so that it does not include useless info
    # Remove fragmentation if frag_enabled == 0
    if params_dict['frag_enabled'] == 0:
        params_dict.pop('frag_r_min')
        params_dict.pop('frag_grav')
        params_dict.pop('frag_grav_coeff')
        params_dict.pop('frag_RTinstab')
        params_dict.pop('frag_densInhom')
        params_dict.pop('frag_cf')
        params_dict.pop('frag_enable_timescale')
    
    # Remove stochasticity related parameters
    if params_dict['stochastic_sampling'] == 0:
        params_dict.pop('n_trials')
    
    # Remove unrelated parameters depending on selected density profile
    if params_dict['dens_profile'] == 'bE_prof':
        params_dict.pop('dens_a_pL')
    elif params_dict['dens_profile'] == 'pL_prof':
        params_dict.pop('dens_g_bE')
        params_dict.pop('dens_navg_pL')
        
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    # Store into summary file as output.
    if write_summary:
        with open(path2output+params_dict['model_name']+'_summary.txt', 'w') as f:
            # header
            f.writelines('\
# =============================================================================\n\
# Summary of parameters in the \'%s\' run.\n\
# Created at %s.\n\
# =============================================================================\n\n\
'%(params_dict['model_name'], dt_string))
            # body
            for key, val in params_dict.items():
                f.writelines(key+'    '+"".join(str(val))+'\n')
            # close
            f.close()
   
    print(f"Summary file created and saved at {path2output}{params_dict['model_name']}{'_summary.txt'}")
        
    # =============================================================================
    # Define a class for parameters as the dictionary is rather large
    # =============================================================================
    class Dict2Class(object):
        # set object attribute
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
                
    # initialise the class
    params = Dict2Class(params_dict)
    # return
    return params


