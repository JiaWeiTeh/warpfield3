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
import sys
import numpy as np
import os
import yaml
import astropy.units as u
import astropy.constants as c 
# from src.input_tools import input_warnings 

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

    # TODO: double check all these parameters with terminal.
    # do not trust what is in parameter.py, as some of them may
    # be overwritten.
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
                   'is_mCloud_beforeSF': 1.0, 
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
                   'SB99_mass': 1e6,
                   'SB99_rotation': 1.0,
                   'SB99_BHCUT': 120.0,
                   'SB99_forcefile': 0.0,
                   'SB99_age_min': 500000.0,
                   'dens_profile': 'bE_prof', 
                   'dens_g_bE': 14.1, 
                   'dens_a_pL': 0, 
                   'dens_navg_pL': 100.0, 
                   'frag_enabled': 0.0, 
                   'frag_r_min': 0.1, 
                   'frag_grav': 0.0, 
                   'frag_grav_coeff': 0.67, 
                   'frag_RTinstab': 0.0, 
                   'frag_densInhom': 0.0, 
                   'frag_cf': 1.0, 
                   'frag_cf_end': 0.1,
                   'frag_enable_timescale': 1.0, 
                   'stop_n_diss': 1.0, 
                   'stop_t_diss': 1.0, 
                   'stop_r': 5050.0, 
                   'stop_v': -10000.0,
                   'stop_t': 15.05, 
                   'stop_t_unit': 'Myr', 
                   'adiabaticOnlyInCore': False,
                   'immediate_leak': True,
                   'phase_Emin': 1e-4,
                   'write_main': 1.0, 
                   'write_stellar_prop': 0.0, 
                   'write_bubble': 0.0, 
                   'write_bubble_CLOUDY': 0.0, 
                   'write_shell': 0.0, 
                   'write_figures': 0.0,
                   'write_potential': 0.0,
                   'path_cooling': 'def_dir',
                   'xi_Tb': 0.9,
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
                   'nISM': 10, 
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
    
    # TODO:
        # What do if randomised? Should show both randomised range, and randomised result. 
    
    
    try:
        for pairs in params_input_list:
            param, value = pairs.split(maxsplit = 1)
            # if there are weird behaviours in the file:
            if param not in params_dict:
                sys.exit(f'{param} is not a parameter.')
            value = value.split(',')
            # value is a list. That's why we needed [0], and can calculate its length.
            if len(value) == 1:
                # Convert to float if possible
                try:
                    val = float(value[0])
                    params_dict[param] = val
                # otherwise remain as string
                except:
                    params_dict[param] = value[0]
            # not done, but for now we save them as a list.
            else:
                params_dict[param] = value 
    except Exception:
        sys.exit("Error detected. Make sure to adhere to the rules when creating the .param file. There appears to be a formatting issue.") 
    
    # TODO
    # give warning if parameter does not make sense
    # input_warnings.input_warnings(params_dict)
            

    # =============================================================================
    # Here we deal with additional parameters that will be recorded in summary.txt.
    # For those that are not recorded, scroll down to the final section of this 
    # script.
    # =============================================================================
    # We have assumed the dust cross section scales linearly with metallicity. However,
    # below a certain metallicity, there is no dust
    if params_dict['metallicity'] >= params_dict['sigma0']:
        params_dict['sigma_d'] = params_dict['sigma0'] * params_dict['metallicity']
    else:
        params_dict['sigma_d'] = 0

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
        
        # TODO:
            
        # warnings.warn("Forcing WARPFIELD to use the following SB99 file: %s" % (force_file))
        # warnings.warn("WARNING: Make sure you still provided the correct metallicity and mass scaling")
        
        # # figure out the SB99 file for use in cloudy (cloudy will not interpolate between files, just pick the one that comes closest)
        # if force_SB99file == 0: # no specific cloudy file is forced, determine which file to use from BHcutoff, metallicity, ...
        #     if rotation == True: rot_string = "_rot_"
        #     else: rot_string = "_norot_"
        #     BH_string = "_BH" + str(int(BHcutoff))
        #     if abs(Zism-1.0) < abs(Zism-0.15): Z_string = "Z0014"
        #     else: Z_string = "Z0002"
        #     SB99cloudy_file = '1e6cluster'+rot_string+Z_string+BH_string
        # else:
        #     # if a specific file is forced, remove extension for cloudy
        #     print(("forcing specific starburst99 file: " + force_SB99file))
        #     idx = [pos for pos, char in enumerate(force_SB99file) if char == '.'] # find position of last '.' (beginning of file extension)
        #     SB99cloudy_file = force_SB99file[:idx[-1]] # remove extension

        # if SB99file != SB99cloudy_file + '.txt':
        #     sys.exit("SB99file != SB99cloudy_file in read_SB99.py!")
        #     print(("SB99file: " + SB99file))
        #     print(("SB99cloudy_file +.txt: " + SB99cloudy_file))

        
    # =============================================================================
    # Store only useful parameters into the summary.txt file
    # =============================================================================
    # First, grab directories
    # 1. Output directory:
    if params_dict['out_dir'] == 'def_dir':
        # If user did not specify, the directory will be set as ./outputs/ 
        # check if directory exists; if not, create one.
        # TODO: Add smart system that adds 1, 2, 3 if repeated default to avoid overwrite.
        path2output = os.path.join(os.getcwd(), 'outputs/'+params_dict['model_name']+'/')
        Path(path2output).mkdir(parents=True, exist_ok = True)
        params_dict['out_dir'] = path2output
    else:
        # if instead given a path, then use that instead
        path2output = str(params_dict['out_dir'])
        Path(path2output).mkdir(parents=True, exist_ok = True)
        params_dict['out_dir'] = path2output
 
    # 2. Cooling table directory:
    if params_dict['path_cooling'] == 'def_dir':
        # If user did not specify, the directory will be set as ./lib/cooling_tables/opiate/
        # check if directory exists; if not, create one.
        # TODO: Add smart system that adds 1, 2, 3 if repeated default to avoid overwrite.
        path2cooling = os.path.join(os.getcwd(), 'lib/cooling_tables/opiate/')
        Path(path2cooling).mkdir(parents=True, exist_ok = True)
        params_dict['path_cooling'] = path2cooling
    else:
        # if instead given a path, then use that instead
        path2cooling = str(params_dict['path_cooling'])
        Path(path2cooling).mkdir(parents=True, exist_ok = True)
        params_dict['path_cooling'] = path2cooling
        
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
        params_dict.pop('dens_navg_pL')
    elif params_dict['dens_profile'] == 'pL_prof':
        params_dict.pop('dens_g_bE')
        
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
   
    # =============================================================================
    # This section contains information which you do not want to appear in the 
    # final summary.txt file, but want it included. Mostly mini-conversions.
    # =============================================================================
            
    # Here we deal with tStop based on units. If the unit is Myr, then
    # tStop is tStop. If it is in free-frall time, then we need to change
    # to a proper unit of time. Add a parameter 'tff' that calculates the 
    # free-fall time. This may be used even if stop_t_unit is not in tff time, 
    # as it may be called if mult_SF = 2, where the second startburst is
    # characterised by the free-fall time.
    tff = np.sqrt(3. * np.pi / (32. * c.G.cgs.value * params_dict['dens_navg_pL'] * params_dict['mu_n'])) / u.Myr.to(u.s)
    params_dict['tff'] = float(tff)
    if params_dict['stop_t_unit'] == 'tff':
        params_dict['stop_t'] = params_dict['stop_t'] * params_dict['tff']
    # if params_dict['stop_t_unit'] == 'Myr', it is fine; Myr is also the 
    # unit in all other calculations. å
    
    # Here we include calculations for mCloud, for future ease.
    params_dict['mCloud'] = 10**params_dict['log_mCloud']
    
    # Here for the magnatic field related constants
    params_dict['BMW'] = 10**params_dict['log_BMW']
    params_dict['nMW'] = 10**params_dict['log_nMW']
    
    # Is there a density gradient?
    params_dict['density_gradient'] = float((params_dict['dens_profile'] == 'pL_prof') and (params_dict['dens_profile'] != 0))
    
    # =============================================================================
    # Save output to yaml. This contains parameters in which you do not whish
    # user to see in the output summary.txt.
    # =============================================================================
    # relative path to yaml
    path2yaml = r'./param/'
    # Write this into a file
    filename =  path2output + params_dict['model_name'] + '_config.yaml'
    with open(filename, 'w',) as file :
        # header
        file.writelines('\
# =============================================================================\n\
# Summary of parameters in the \'%s\' run.\n\
# Created at %s.\n\
# =============================================================================\n\n\
'%(params_dict['model_name'], dt_string))
        yaml.dump(params_dict, file, sort_keys=False) 
    
    # save path to object
    # TODO: delete this after warpfield is finished.
    # save file
    os.environ['PATH_TO_CONFIG'] = filename
    
    return params_dict











