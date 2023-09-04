#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:25:54 2022

@author: Jia Wei Teh

This script contains functions to read cooling curves in non-CIE environments. 

Old code: coolnoeq.py
"""


import numpy as np
import math 
import sys
import os
from astropy.io import ascii
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import warnings
#--
import src.warpfield.functions.operations as operations
from src.warpfield.functions.terminal_prints import cprint as cpr

# # get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()


def get_coolingStructure(age):
    """
    Time-dependent cooling curve.
    See create_cubes() for values contained in the variable `Cool_Struc`, and 
    what is available or should be contained in the cooling files. 
    
    It's called a structure because it is a grid that depends on three parameters.

    Parameters
    ----------
    age : float
        Current age.

    Returns
    -------
    Cool_Struc:

    """
    
    
    # =============================================================================
    # Step1: Time-dependent cooling curve: figure out which time!
    # Goal of this part: return tables for:
    #   number density, temperature, photon number flux, 
    #   (only) cooling, (only) heating, net cooling.
    # =============================================================================
    # Time-dependent cooling curve files:
    # Available ages: 1e6, 2e6, 3e6, 4e6, 5e6, 1e7 yr. 
    # For given time (cluster age), find the nearest available age. 
    filename = get_filename(age)
    
    
    # if return only one file, no need interpolation. see get_filename()
    if isinstance(filename, list) ==  False:
        log_ndens_arr, log_temp_arr, log_phi_arr, cool_cube, heat_cube = create_cubes(filename)
    
    # if two files, then it means there is interpolation. This is the nearest higher/lower file
    else:
        file_age_lower, file_age_higher = filename
        # values from higher and lower ages
        log_ndens_arr, log_temp_arr, log_phi_arr,\
            cool_cube_lower, heat_cube_lower = create_cubes(file_age_lower)
        _, _, _,\
            cool_cube_higher, heat_cube_higher = create_cubes(file_age_higher)    
        
        # create cooling and heating from dict if tney don't exist. 
        # get ages
        # returns i.e. '1.00e+06'.
        age_lower = float(get_fileage(file_age_lower))
        age_higher = float(get_fileage(file_age_higher))
            
        # Get cooling/heating values for fixed points/range for this particular age
        def cube_linear_interpolate(x, ages, cubes):
            # get values
            ages_low, ages_high = ages
            cubes_low, cubes_high = cubes
            return cubes_low + (x - ages_low) * (cubes_high - cubes_low)/(ages_high - ages_low)

        cool_cube = cube_linear_interpolate(age, [age_lower, age_higher], [cool_cube_lower, cool_cube_higher])
        heat_cube = cube_linear_interpolate(age, [age_lower, age_higher], [heat_cube_lower, heat_cube_higher])
    
    # Create interpolation functions
    # old code: create_onlycoolheat(), Cool_Struc['Cfunc'] = onlycoolfunc, Cool_Struc['Hfunc'] = onlyheatfunc
    cooling_interpolation = RegularGridInterpolator((log_ndens_arr, log_temp_arr, log_phi_arr), np.log10(cool_cube),
                                              method = 'linear')
    heating_interpolation = RegularGridInterpolator((log_ndens_arr, log_temp_arr, log_phi_arr), np.log10(heat_cube),
                                              method = 'linear')
    
    return cool_cube, heat_cube, cooling_interpolation, heating_interpolation



def create_cubes(filename):
    """
    This function will take filename and return cooling/heating in the form of cubes.

    Parameters
    ----------
    filename : str
        Filename -> contains cooling table.

    Returns
    -------
    These define the side of the cooling/heating cube.
        log_ndens_arr: [cm-3] 
            np.array of density ticks in log space. 
        log_temp_arr: [T]
            np.array of temperature ticks in log space. 
        log_phi_arr: [cm-2s-1]
            np.array of phi (number flux of ionizing photons) ticks in log space. 
            
     cool_cube:
         Stores cooling value for any [ndens, temp, phi] triple. Some are NaN, because
         they are not available in the cooling table (perhaps non-physical)
     heat_cube:
         Same as cool_cube, but for heating values. 
    
    """

    # =============================================================================
    # Step1: read in file, perform some basic operations
    # =============================================================================

    # read file
    opiate_file = ascii.read(warpfield_params.path_cooling_nonCIE + filename)
    
    # read in the columns
    ndens_data = opiate_file['ndens']
    temp_data = opiate_file['temp']
    phi_data = opiate_file['phi']
    # these are derived quantities in CLOUDY output
    cooling_data = opiate_file['cool']
    heating_data = opiate_file['heat']
    # make sure signs in heating/cooling column are positive!
    if np.sign(heating_data[0]) == -1:
        heating_data = -1 * heating_data
        print(f'{cpr.WARN}Heating values have negative signs in {filename}. They are now changed to positive.{cpr.END}')
    if np.sign(cooling_data[0]) == -1:
        cooling_data = -1 * cooling_data
        print(f'{cpr.WARN}Cooling values have negative signs in {filename}. They are now changed to positive.{cpr.END}')
    
    # =============================================================================
    # Step2: create cubes
    # =============================================================================
    
    def create_limits(array):
        # This function creates the lines for cuboid, for future interpolation.
        # here is the function listed for readability. In reality this can easily be a one-liner.    
        array = np.array(list(set(array)))
        # sort array
        array = np.sort(array)
        # log array, because the original was created in log space. 
        array = np.log10(array)
        # round, because it makes things easier.
        array = np.round(array, decimals = 3)
        return array
    
    # create lines for cube sides
    log_ndens_arr = create_limits(ndens_data)
    log_temp_arr = create_limits(temp_data)
    log_phi_arr = create_limits(phi_data)
    
    # -----
    # A) Cooling cube
    # create rows of data
    cool_table = np.transpose(np.vstack([ndens_data, temp_data, phi_data, cooling_data]))
    # go from ndens, then T, then phi.
    cool_cube = np.empty((len(log_ndens_arr), len(log_temp_arr),len(log_phi_arr)))
    # size = (31, 21, 22), meaning 31 slices of (21x22) arrays
    cool_cube[:] = np.nan
    
    # fill in cooling cube
    for (ndens_val, temp_val, phi_val, cooling_val) in cool_table:
        # find which index these belong to
        ndens_index = np.where(log_ndens_arr == np.round(np.log10(ndens_val), decimals = 5))[0][0]
        temp_index = np.where(log_temp_arr == np.round(np.log10(temp_val), decimals = 5))[0][0]
        phi_index = np.where(log_phi_arr == np.round(np.log10(phi_val), decimals = 5))[0][0]
        # record into the cube
        cool_cube[ndens_index, temp_index, phi_index] = cooling_val
        
    # -----
    # B) Heating cube
    # create rows of data
    heat_table = np.transpose(np.vstack([ndens_data, temp_data, phi_data, heating_data]))
    # go from ndens, then T, then phi.
    heat_cube = np.empty((len(log_ndens_arr), len(log_temp_arr),len(log_phi_arr)))
    heat_cube[:] = np.nan
    
    # fil in heating cube
    for (ndens_val, temp_val, phi_val, heating_val) in heat_table:
        # find which index these belong to
        ndens_index = np.where(log_ndens_arr == np.round(np.log10(ndens_val), decimals = 3))[0][0]
        temp_index = np.where(log_temp_arr == np.round(np.log10(temp_val), decimals = 3))[0][0]
        phi_index = np.where(log_phi_arr == np.round(np.log10(phi_val), decimals = 3))[0][0]
        # record into the cube
        heat_cube[ndens_index, temp_index, phi_index] = heating_val

    # =============================================================================
    # Step 3: create an interpolation function. 
    # Future TODO: If it fails, i.e., if it returns NaN because the values don't exist in the cooling
    # table, we do further operations. 
    # =============================================================================
    
    return log_ndens_arr, log_temp_arr, log_phi_arr, cool_cube, heat_cube


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
    
        # What are the available ages? If the given age is greater than the maximum or
        # is lower than the minimum, then use the max/min instead. Otherwise, do interpolation (in another function).
        # loop through the folder which contains all the data
        age_list = []
        for files in os.listdir(warpfield_params.path_cooling_nonCIE):
            # look for .dat
            if files[-4:] == '.dat':
                # returns i.e. '1.00e+06'.
                age_list.append(get_fileage(files))
        # array
        age_list = np.array(age_list)
        # if in array, use the file.
        if age in age_list:
            age_str = format(age, '.2e')
            # include brackets to check if there is one or two filenames
            filename = 'opiate_cooling' + '_' + rot_str + '_' + 'Z' + Z_str + '_' + 'age' + age_str + '.dat'
            return filename
        # for min/max age, use the max/min
        elif age >= max(age_list):
            age_str = format(max(age_list), '.2e')
            # include brackets to check if there is one or two filenames
            filename = 'opiate_cooling' + '_' + rot_str + '_' + 'Z' + Z_str + '_' + 'age' + age_str + '.dat'
            return filename
        elif age <= min(age_list):
            age_str = format(min(age_list), '.2e')
            # include brackets to check if there is one or two filenames
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
        

def get_fileage(filename):
    # look for the numbers after 'age'. 
    age_index_begins = filename.find('age')
    # returns i.e. '1.00e+06'.
    return float(filename[age_index_begins+3:age_index_begins+3+8])


















