#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:25:54 2022

@author: Jia Wei Teh

This script contains functions to read cooling curves in non-CIE environments. 
"""


import numpy as np
import sys
import os
from astropy.io import ascii
from scipy.interpolate import LinearNDInterpolator
import warnings
#--
import src.warpfield.functions.operations as operations
# Old file: coolnoeq.py

# # get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()





# cooling files take metallicity and age.




def read_opiate(age):
    
    
    
    # Grab path to cooling table
    path2cooling = warpfield_params.path_cooling
    
    # =============================================================================
    # Step1: Time-dependent cooling curve: figure out which time!
    # Goal of this part: return tables for:
    #   number density, temperature, photon number flux, 
    #   (only) cooling, (only) heating, net cooling.
    # =============================================================================
    # Time-dependent cooling curve files:
    # Availabel ages: 1e6, 2e6, 3e6, 4e6, 5e6, 1e7 yr. 
    # For given time (cluster age), find the nearest available age. 
    filename = get_filename(age)
    
    
    # if return only one file, no need interpolation. see get_filename()
    if len(filename) == 1:
        create_cooling_grid(filename)
    # if two files, then it means there is interpolation. This is the nearest higher/lower file
    else:
        # pseudocode
        age_lower, age_higher = filename
        # values from higher and lower ages
        up = create_cooling_grid(higher)
        down = create_cooling_grid(lower)
        # get interpolation value
        tru = get_interpolation(up, down)
    
        
# THEN



    #     np_onlycoolfilename = make_cooling_filename(Zism, age, 
    #                                                 basename=basename + "C", extension = ".npy", cool_folder = cool_folder)
    #     np_onlyheatfilename = make_cooling_filename(Zism, age, 
    #                                                 basename=basename + "H", extension=".npy", cool_folder=cool_folder)
    #     # do the numpy readable-files already exist?
    #     if (os.path.isfile(np_coolfilename) and os.path.isfile(np_onlycoolfilename)) and os.path.isfile(np_onlyheatfilename):
    #         ln_dat, lT_dat, lP_dat = get_opiate_gridstruc(Zism, age, 
    #                                                       basename="opiate_cooling", extension=".dat", cool_folder="cooling_tables")
    #         NetCool = np.load(np_coolfilename)
    #         Cool_dat = {"Netcool": NetCool, "log_n": ln_dat, "log_T": lT_dat, "log_Phi": lP_dat}
    #         Cool = np.load(make_cooling_filename(Zism, age, 
    #                                              basename=basename + "C", extension=".npy", cool_folder=cool_folder))
    #         Cool_dat["Cool"] = Cool
    #         Heat = np.load(make_cooling_filename(Zism, age, 
    #                                              basename=basename + "H", extension=".npy", cool_folder=cool_folder))
    #         Cool_dat["Heat"] = Heat
    #     else:
    #         Cool_dat = prep_coolingtable(Zism, age, 
    #                                      basename=basename, extension=extension, cool_folder=cool_folder, indiv_CH=indiv_CH)

    # return Cool_dat
   
        
    return 



def create_cooling_grid(filename):
    
    
    
    """
    This function will take filename and just return values?
    
    
    
    Here are the important columnes:
        n: ion number density [cm-3]
        T: temperature [T]
        phi: number flux of ionizing photons [cm-2s-1]
    
    
    """


    # =============================================================================
    # Step1: read in file, perform some basic operations
    # =============================================================================

    # read in the file 
    opiate_file = ascii.read(warpfield_params.path_cooling + filename)
    # read in the columns
    ndens = opiate_file['ndens']
    temp = opiate_file['temp']
    phi = opiate_file['phi']
    # these are derived quantities in CLOUDY output
    cooling = opiate_file['cool']
    heating = opiate_file['heat']
    # make sure signs in heating/cooling column are positive!
    if np.sign(heating[0]) == -1:
        heating = -1 * heating
        print(f'\033[1m\033[94mHeating values have negative signs in {filename}. They are now changed to positive.\033[0m')
    if np.sign(cooling[0]) == -1:
        cooling = -1 * cooling
        print(f'\033[1m\033[94mHeating values have negative signs in {filename}. They are now changed to positive.\033[0m')
    # now, we can calculate the net cooling
    netcooling = cooling - heating
    
    # =============================================================================
    # Step2: create cooling structure  
    # =============================================================================
    
    # full log values
    # set it to avoid duplication, sort it, then take log10, then round it
    # here is the function listed for readability. In reality this can easily be a one-liner. 
    def convert(x):
        x = np.array(list(set(x)))
        x = np.sort(x)
        x = np.log10(x)
        x = np.round(x, decimals = 3)
        return x
    log_ndens = convert(ndens)
    log_temp = convert(temp)
    log_phi = convert(phi)
    
    # sanity check: make sure that the values are constantly spaced in log-space. 
    # This should always be true, because this should be how it was defined in CLOUDY.
    if len(set(np.diff(log_ndens))) != 1 or len(set(np.diff(log_temp))) != 1 or len(set(np.diff(log_phi))) != 1:
        sys.exit('Structure of cooling table not recognised. Distance between grid points in log-space is not constant.')
    
    # return cooling data structure
    Cool_Struc = {"log_n": log_ndens, "log_T": log_temp, "log_Phi": log_phi}
    
    return Cool_Struc





def get_filename(age):

    # All filenames have the convention of opiate_cooling_[rotation]_Z[metallicity]_age[age].dat
    # Right now, only solar metallicity and rotation is considered. 
    try:
        # with rotation?
        if warpfield_params.SB99_rotation == True:
            rot_str = 'rot'
        else:
            rot_str = 'norot'
        # metallicity?
        if float(warpfield_params.metallicity) == 1.0:
            # solar, Z = 0.014
            Z_str = '1.00'        
        elif float(warpfield_params.metallicity) == 0.15:
            # 0.15 solar, Z = 0.002
            Z_str = '0.15'
        # age? TODO
        age_str = format(age, '.2e')
            
        # What are the available ages? If the given age is greater than the maximum or
        # is lower than the minimum, then use the max/min instead. Otherwise, do interpolation (in another function).
        # loop through the folder which contains all the data
        age_list = []
        for files in os.listdir(warpfield_params.path_cooling):
            # look for .dat
            if files[-4:] == '.dat':
                # look for the numbers after 'age'. 
                age_index_begins = files.find('age')
                # returns i.e. '1.00e+06'.
                age_list.append(float(files[age_index_begins+3:age_index_begins+3+8]))
        # array
        age_list = np.array(age_list)
        # for min/max age
        if age > max(age_list):
            age_str = format(max(age_list), '.2e')
            filename = 'opiate_cooling' + '_' + rot_str + '_' + 'Z' + Z_str + '_' + 'age' + age_str + '.dat'
            return filename
        elif age < min(age_list):
            age_str = format(min(age_list), '.2e')
            filename = 'opiate_cooling' + '_' + rot_str + '_' + 'Z' + Z_str + '_' + 'age' + age_str + '.dat'
            return filename
        else:
            # If age is between files, we find the nearest higher age and lower age neighbour, and do interpolation.
            # e.g., if age = 2.3, do interpolation from 2 and 3. 
            # This is o(n) time, but since the list is small this wouldn't matter much.
            # No sorting is needed, otherwise it'd have become nlogn. 
            higher_age = age_list[age_list > age].min()
            lower_age = age_list[age_list < age].max()
            # return both
            filename = [get_filename(lower_age), get_filename(higher_age)]
            return filename
    except:
        raise Exception("Opiate/CLOUDY file (non-CIE) for cooling curve not found. Make sure to double check parameters in the 'parameters for Starburst99 operations' and 'parameters for setting path' section.")
    

def get_interpolation():
    
    
    
    
    
    
    return

 
























