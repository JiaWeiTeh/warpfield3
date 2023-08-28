#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:25:54 2022

@author: Jia Wei Teh

This script contains functions to read cooling curves in non-CIE environments. 

Old code: coolnoeq.py
"""


import numpy as np
import sys
import os
from astropy.io import ascii
from scipy.interpolate import LinearNDInterpolator
import warnings
#--
import src.warpfield.functions.operations as operations

# # get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()


def get_coolingStructure(age):
    """
    Time-dependent cooling curve.
    See create_cooling_grid() for values contained in the variable `Cool_Struc`, and 
    what is available or should be contained in the cooling files. 
    
    It's called a structure because it is a grid that depends on three parameters.

    Parameters
    ----------
    age : float
        Current age.

    Returns
    -------
    Cool_Struc:
        In addition to what was already in (see create_cooling_grid()), this also inludes:
            log_cooling_interpolation:
            log_cooling_interpolation:

    """
    
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
        Cool_Struc = create_cooling_grid(age)
        cooling = Cool_Struc['cooling']
        heating = Cool_Struc['heating']
    # if two files, then it means there is interpolation. This is the nearest higher/lower file
    else:
        # pseudocode
        age_lower, age_higher = filename
        # values from higher and lower ages
        Cool_Struc = create_cooling_grid(age_lower)
        Cool_Struc_higher = create_cooling_grid(age_higher)
        # get values
        cooling_higherage = Cool_Struc_higher['cooling']
        heating_higherage = Cool_Struc_higher['heating']
        cooling_lowerage = Cool_Struc['cooling']
        heating_lowerage = Cool_Struc['heating']
        # create cooling and heating from dict if tney don't exist. 
        def simple_linear_interpolation(x, xList, yList):
            return yList[0] + (yList[1] - yList[0]) * (x - xList[0])/(xList[1] - xList[0])
        
        cooling = simple_linear_interpolation(age, [age_lower, age_higher], [cooling_lowerage, cooling_higherage])
        heating = simple_linear_interpolation(age, [age_lower, age_higher], [heating_lowerage, heating_higherage])
    
    # Create interpolation functions
    phase_space = np.transpose(np.vstack(Cool_Struc['ndens'], Cool_Struc['temp'], Cool_Struc['phi']))
    
    # remember that these are in log
    log_cooling_interpolation = LinearNDInterpolator(np.log10(phase_space), np.log10(cooling))
    log_heating_interpolation = LinearNDInterpolator(np.log10(phase_space), np.log10(heating))
    
    # record
    # old code: create_onlycoolheat(), Cool_Struc['Cfunc'] = onlycoolfunc, Cool_Struc['Hfunc'] = onlyheatfunc
    Cool_Struc['log_cooling_interpolation'] = log_cooling_interpolation
    Cool_Struc['log_heating_interpolation'] = log_heating_interpolation
    
    return Cool_Struc


def create_cooling_grid(filename):
    """
    This function will take filename and return useful variables.

    Parameters
    ----------
    filename : str
        Filename -> contains cooling table.

    Returns
    -------
    Cool_Struc: A dictionary which inclues:
        ndens: ion number density [cm-3] 
        T: temperature [T]
        phi: number flux of ionizing photons [cm-2s-1]
        cooling:
        heating:
        log_n: modified ndens; written in log10, sorted, and removed any duplicates.
        log_T: modified T; written in log10, sorted, and removed any duplicates. 
        log_Phi: modified phi; written in log10, sorted, and removed any duplicates.
        
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
    Cool_Struc = {"ndens": ndens, "temp": temp, "phi": phi,
                  "cooling": cooling, "heating": heating, "netcooling": netcooling,
                  "log_n": log_ndens, "log_T": log_temp, "log_phi": log_phi, 
                  }
                      
    
    
    return Cool_Struc



# Remember, read_opiatetable returns UNMODULATED data. some interpolations with this.
# whereas get_opiate_gridstruc returns modulated, logged data.


def get_filename(age):
    """
    This function creates the filename appropriate for curent run.

    Parameters
    ----------
    age : float
        Current time.

    Returns
    -------
    filename : str
        Filename corresponding to the parameters set.

    """

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
    













