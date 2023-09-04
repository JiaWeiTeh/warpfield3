
# =============================================================================
# Here is the old read_cloudy.file
# =============================================================================

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
    # Available ages: 1e6, 2e6, 3e6, 4e6, 5e6, 1e7 yr. 
    # For given time (cluster age), find the nearest available age. 
    filename = get_filename(age)
    
    # if return only one file, no need interpolation. see get_filename()
    if isinstance(filename, list) ==  False:
        Cool_Struc = create_cooling_grid(filename)
        cooling = Cool_Struc['cooling']
        heating = Cool_Struc['heating']
        
    # if two files, then it means there is interpolation. This is the nearest higher/lower file
    else:
        # pseudocode
        file_age_lower, file_age_higher = filename
        # values from higher and lower ages
        Cool_Struc = create_cooling_grid(file_age_lower)
        Cool_Struc_higher = create_cooling_grid(file_age_higher)
        # get values
        cooling_higherage = Cool_Struc_higher['cooling']
        heating_higherage = Cool_Struc_higher['heating']
        cooling_lowerage = Cool_Struc['cooling']
        heating_lowerage = Cool_Struc['heating']
        # create cooling and heating from dict if tney don't exist. 
        # get ages
        # returns i.e. '1.00e+06'.
        age_lower = float(get_fileage(file_age_lower))
        age_higher = float(get_fileage(file_age_higher))
            
        # Get cooling values for fixed points/range for this particular age
        cooling = np.interp(age, [age_lower, age_higher], [cooling_lowerage, cooling_higherage])
        heating = np.interp(age, [age_lower, age_higher], [heating_lowerage, heating_higherage])
    
    # Create interpolation functions
    phase_space = np.transpose(np.vstack([Cool_Struc['ndens'], Cool_Struc['temp'], Cool_Struc['phi']]))
    
    # get interpolation function for the coolings
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
    opiate_file = ascii.read(warpfield_params.path_cooling_nonCIE + filename)
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
        print(f'{cpr.WARN}Heating values have negative signs in {filename}. They are now changed to positive.{cpr.END}')
    if np.sign(cooling[0]) == -1:
        cooling = -1 * cooling
        print(f'{cpr.WARN}Heating values have negative signs in {filename}. They are now changed to positive.{cpr.END}')
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
        
        # sanity check: make sure that the values are constantly spaced in log-space. 
        # This should always be true, because this should be how it was defined in CLOUDY.
        # find unique difference values
        array_diff = np.array(list(set(np.diff(x))))
        # make sure they are evenly spaced
        # account for machine precision
        assert np.all(np.isclose(array_diff, array_diff[0], atol = 1e-10)), 'Structure of cooling table not recognised. Distance between grid points in log-space is not constant.'
        return x
    
    log_ndens = convert(ndens)
    log_temp = convert(temp)
    log_phi = convert(phi)
    
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




#%%

# =============================================================================
# Here is where the test stuffs begin
# =============================================================================

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
    # try:
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
    # except:
        # raise Exception("Opiate/CLOUDY file (non-CIE) for cooling curve not found. Make sure to double check parameters in the 'parameters for Starburst99 operations' and 'parameters for setting path' section.")

def get_fileage(filename):
    # look for the numbers after 'age'. 
    age_index_begins = filename.find('age')
    # returns i.e. '1.00e+06'.
    return float(filename[age_index_begins+3:age_index_begins+3+8])


filename = get_filename(1e6)
print('filename: ', filename)



# =============================================================================
# Here is where we create the cooling structure. Now, instead of the previous
# one, we create a cube! This cube has (cool, heat) as tuple.
# =============================================================================

# _data: raw values directly from the table.

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

    # TODO: maybe this isn't needed?
    # sanity check: make sure that the values are constantly spaced in log-space. 
    # This should always be true, because this should be how it was defined in CLOUDY.
    # find unique difference values
    array_diff = np.array(list(set(np.diff(array))))
    # make sure they are evenly spaced
    # account for machine precision with .isclose()
    assert np.all(np.isclose(array_diff, array_diff[0], atol = 1e-2)), 'Structure of cooling table not recognised. Distance between grid points in log-space is not constant.'

    return array



plt.scatter(np.linspace(0, len(cooling_data[::21]), len(cooling_data[::21])), cooling_data[::21],  s = 0.1)
plt.plot(np.linspace(0, len(cooling_data[::21]), len(cooling_data[::21])), cooling_data[::21],  'k-')
plt.yscale('log')

# check, there is no NaN. 
for i in cooling_data:
    if np.isnan(i):
        print('ye')



print(cooling_data.shape)


# create these lines

log_ndens_arr = create_limits(ndens_data)
log_temp_arr = create_limits(temp_data)
log_phi_arr = create_limits(phi_data)

print(len(log_ndens_arr)) # 33
print(len(log_temp_arr)) # 21
print(len(log_phi_arr)) # 22
print(33*21*22)







#%%




# =============================================================================
# Idea: create a 3D array, go through each dimension to record the tuples.
# this way we can also see where and what is missing. 
# 
# Important: the cuboid side values (e.g., ndens) are in log, but the cooling/
# heating values are in linear space.
# =============================================================================


# create rows of data
cool_table = np.transpose(np.vstack([ndens_data, temp_data, phi_data, cooling_data]))

# go from ndens, then T, then phi.
cool_cube = np.empty((len(log_ndens_arr), len(log_temp_arr),len(log_phi_arr)))
# size = (31, 21, 22), meaning 31 slices of (21x22) arrays
cool_cube[:] = np.nan

# fil in cooling cube
for (ndens_val, temp_val, phi_val, cooling_val) in cool_table:
    # find which index these belong to
    ndens_index = np.where(log_ndens_arr == np.round(np.log10(ndens_val), decimals = 5))[0][0]
    temp_index = np.where(log_temp_arr == np.round(np.log10(temp_val), decimals = 5))[0][0]
    phi_index = np.where(log_phi_arr == np.round(np.log10(phi_val), decimals = 5))[0][0]
    # record into the cube
    cool_cube[ndens_index, temp_index, phi_index] = cooling_val
    

# Interpolate these data. 

# print(cool_cube[:,:,-1])


#%%

import pandas as pd
from astropy.visualization import simple_norm

test_slice = np.log10(cool_cube[13,:,:].copy())
# test_slice = cool_cube[:,0,:]
# test_slice = cool_cube[:,:,15]

df = pd.DataFrame(test_slice)
norm = simple_norm(df, 'power')
plt.imshow(df, norm = norm )
plt.colorbar()
plt.show()

df_new = df.interpolate(
                        limit_direction = 'forward',
                        method = 'linear', 
                        # order = 3,
                        axis = 1,
                        )
norm = simple_norm(df_new, 'power')
plt.imshow(df_new, norm = norm )
plt.show()


# =============================================================================
# Idea 1: make try to catch error, then loop through axis = 0/1 and direction.
# Idea 2: make loop to make ndens/T/phi constant, then average to get better cube.
# ============================================================================




#%%

test_slice = cool_cube[10,:,:]

test_array = test_slice[0]

print(test_array)
print(np.log10(test_array))


import numpy as np
nan = np.nan


# Creating a numpy array
arr = test_array.copy()

# Display original array
print("Original Array:\n",arr,"\n")

# Making sequences for interp
ok = ~np.isnan(arr)
xp = ok.ravel().nonzero()[0]
fp = arr[~np.isnan(arr)]
x  = np.isnan(arr).ravel().nonzero()[0]

# Replacing nan values
arr[np.isnan(arr)] = np.interp(x, xp, fp)

# Display result
print ("Result:\n",arr)

plt.plot(test_array, linewidth = 5, alpha = 0.1)

plt.plot(arr)





#%%




# create rows of data
heat_table = np.transpose(np.vstack([ndens_data, temp_data, phi_data, heating_data]))

# go from ndens, then T, then phi.
heat_cube = np.empty((len(log_ndens_arr), len(log_temp_arr),len(log_phi_arr)), dtype = object)
heat_cube[:] = np.nan

# fil in heating cube
for (ndens_val, temp_val, phi_val, heating_val) in heat_table:
    # find which index these belong to
    ndens_index = np.where(log_ndens_arr == np.round(np.log10(ndens_val), decimals = 3))[0][0]
    temp_index = np.where(log_temp_arr == np.round(np.log10(temp_val), decimals = 3))[0][0]
    phi_index = np.where(log_phi_arr == np.round(np.log10(phi_val), decimals = 3))[0][0]
    # record into the cube
    heat_cube[ndens_index, temp_index, phi_index] = heating_val



# create rows of data
coolheat_table = np.transpose(np.vstack([ndens_data, temp_data, phi_data, cooling_data, heating_data]))

# go from ndens, then T, then phi.
coolheat_cube = np.empty((len(log_ndens_arr), len(log_temp_arr),len(log_phi_arr)), dtype = object)
coolheat_cube[:] = np.nan

# fil in both heating and cooling cubes as tuple
for (ndens_val, temp_val, phi_val, cooling_val, heating_val) in coolheat_table:
    # find which index these belong to
    ndens_index = np.where(log_ndens_arr == np.round(np.log10(ndens_val), decimals = 3))[0][0]
    temp_index = np.where(log_temp_arr == np.round(np.log10(temp_val), decimals = 3))[0][0]
    phi_index = np.where(log_phi_arr == np.round(np.log10(phi_val), decimals = 3))[0][0]
    # record into the cube
    coolheat_cube[ndens_index, temp_index, phi_index] = (cooling_val, heating_val)



#%%


import scipy 




def cube_linear_interpolate(x, ages, cubes):
    
    # get values
    ages_low, ages_high = ages
    cubes_low, cubes_high = cubes

    return cubes_low + (x - ages_low) * (cubes_high - cubes_low)/(ages_high - ages_low)

print(cube_linear_interpolate(1.5, [1, 2], [cool_cube, cool_cube]))



#%%




# But if I already have the interpolation function, why can't I just give values and 
# dont do any of the calculations daniel do?

from src.warpfield.cooling.non_CIE import read_cloudy
import os


# os.environ['PATH_TO_CONFIG'] = r'/Users/jwt/Documents/Code/warpfield3/outputs/example_pl/example_pl_config.yaml'



# test cases
Cool_Struc = read_cloudy.get_coolingStructure(1e6)


#%%



def test_interp(dens, T, phi):
    
    pts = np.log10(np.array([dens, T, phi]))
    
    # print(pts)

    x = log_ndens_arr
    
    y = log_temp_arr
    
    z = log_phi_arr
    
    # xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    
    data = cool_cube
    
    
    # interp = LinearNDInterpolator(list(zip(x, y, z)), data)
    interp = RegularGridInterpolator((x, y, z), data)
    
    
    answer = interp(pts)
    print(answer)
    
    # answer = 1
    
    # daniel = Cool_Struc['log_cooling_interpolation'](pts)
    
    # assert answer == true, f"Failed. answer: {answer}, true: {true}"
    # print(f'cooling - answer: {answer}, daniel: {daniel}')
    
    return




def retrieve_value(array, cube):
    ndens_val, temp_val, phi_val = array
    # find which index these belong to
    # print(np.round(np.log10(ndens_val), 3),log_ndens_arr )
    ndens_index = np.where(log_ndens_arr == np.round(np.log10(ndens_val), 3))[0][0]
    temp_index = np.where(log_temp_arr == np.round(np.log10(temp_val), 3))[0][0]
    phi_index = np.where(log_phi_arr == np.round(np.log10(phi_val), 3))[0][0]
    # record into the cube
    value = cube[ndens_index, temp_index, phi_index]
    return value


def retrieve_two_values(array, cube):
    # TODO: find neighbours with min().
    ndens_val, temp_val, phi_val = array
    # find which index these belong to
    # print(np.round(np.log10(ndens_val), 3),log_ndens_arr )
    ndens_index = np.where(log_ndens_arr == np.round(np.log10(ndens_val), 3))[0][0]
    temp_index = np.where(log_temp_arr == np.round(np.log10(temp_val), 3))[0][0]
    phi_index = np.where(log_phi_arr == np.round(np.log10(phi_val), 3))[0][0]
    # record into the cube
    value = cube[ndens_index, temp_index, phi_index]
    return value




testarray1 = np.array([-4, 3.6, 0])
testarray2 = np.array([log_ndens_arr[1], log_temp_arr[2], log_phi_arr[1]])
testarray3 = np.array([log_ndens_arr[10], log_temp_arr[18], log_phi_arr[0]])
testarray4 = np.array([log_ndens_arr[11], log_temp_arr[17], log_phi_arr[10]])
testarray5 = np.array([1, 4, 14])
testarray6 = np.array([np.log10(506663.2212419483), np.log10(294060.78931362595), np.log10(1.5473355225629384e+16)])

def test(array):
    
    # print(retrieve_value(array, cool_cube))
    test_interp(*(10**array))
    print(10**Cool_Struc['log_cooling_interpolation'](array))
    print('\n-------\n')
    # print(Cool_Struc['log_cooling_interpolation'](1, 4, 14))
    
    return 



for i in [testarray1, testarray2, testarray3, testarray4, testarray5, testarray6]:
    # test
    test(i)

# test(testarray2)




# TODO
# ValueError: One of the requested xi is out of bounds in dimension 1



#%%

# =============================================================================
# Here is the proper code
# =============================================================================




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
import matplotlib.pyplot as plt
import warnings
#--
import src.warpfield.functions.operations as operations
from src.warpfield.functions.terminal_prints import cprint as cpr

# # get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()


from src.warpfield.cooling.non_CIE import read_cloudy
import os
# test cases
Cool_Struc = read_cloudy.get_coolingStructure(1e6)

import src.warpfield.cooling.non_CIE.read_opiate_old as read_opiate_old
Cool_Struc_old = read_opiate_old.get_Cool_dat_timedep(1, 1e6)
cool_interp_old, heat_interp_old = read_opiate_old.create_onlycoolheat(1, 1e6)


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
    # try:
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
    # except:
        # raise Exception("Opiate/CLOUDY file (non-CIE) for cooling curve not found. Make sure to double check parameters in the 'parameters for Starburst99 operations' and 'parameters for setting path' section.")

def get_fileage(filename):
    # look for the numbers after 'age'. 
    age_index_begins = filename.find('age')
    # returns i.e. '1.00e+06'.
    return float(filename[age_index_begins+3:age_index_begins+3+8])


filename = get_filename(1e6)
print('filename: ', filename)



# =============================================================================
# Here is where we create the cooling structure. Now, instead of the previous
# one, we create a cube! This cube has (cool, heat) as tuple.
# =============================================================================

# _data: raw values directly from the table.

# Create cube
# def create_coolingcuboid():

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



# =============================================================================
# Idea: create a 3D array, go through each dimension to record the tuples.
# this way we can also see where and what is missing. 
# 
# Important: the cuboid side values (e.g., ndens) are in log
# =============================================================================

def retrieve_value(array, cube):
    ndens_val, temp_val, phi_val = array
    # find which index these belong to
    # print(np.round(np.log10(ndens_val), 3),log_ndens_arr )
    ndens_index = np.where(log_ndens_arr == np.round(np.log10(ndens_val), 3))[0][0]
    temp_index = np.where(log_temp_arr == np.round(np.log10(temp_val), 3))[0][0]
    phi_index = np.where(log_phi_arr == np.round(np.log10(phi_val), 3))[0][0]
    # record into the cube
    value = cube[ndens_index, temp_index, phi_index]
    return value


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


# return cool_cube_raw, cool_cube_interpolated

# =============================================================================
# Here is the important part, first, we try to interpolate. If it fails,
# i.e., if it returns NaN because the values don't exist in the cooling
# table, we do further operations. 
# =============================================================================

# Remember to log the cooling cube!!! This is what is causing the difference.
cool_interpolation = RegularGridInterpolator((log_ndens_arr, log_temp_arr, log_phi_arr), np.log10(cool_cube),
                                              method = 'linear',
                                              )



array2interp = np.array([-4, 3.6, 0])
print('----')
print('check for basic corner case:')
cool_interpolation_val = cool_interpolation(array2interp)
print('interpolation value with cube:', 10**cool_interpolation_val[0])
print('true value in cube:', retrieve_value(10**array2interp, cool_cube))
print('interpolation value in daniel:', 10**cool_interp_old(array2interp)[0])
print('interpolation value in re-write:', 10**Cool_Struc['log_cooling_interpolation'](array2interp)[0])
print('----')

array2interp = np.array([-1, 3.6, 2])
print('----')
print('check for basic corner case:')
cool_interpolation_val = cool_interpolation(array2interp)
print('interpolation value with cube:', 10**cool_interpolation_val[0])
print('true value in cube:', retrieve_value(10**array2interp, cool_cube))
print('interpolation value in daniel:', 10**cool_interp_old(array2interp)[0])
print('interpolation value in re-write:', 10**Cool_Struc['log_cooling_interpolation'](array2interp)[0])
print('----')


array2interp = np.array([-4+0.1, 3.6+0.05, 0+0.02])
print('----')
cool_interpolation_val = cool_interpolation(array2interp)
print('interpolation value with cube:', 10**cool_interpolation_val[0])
print('interpolation value in daniel:', 10**cool_interp_old(array2interp)[0])
print('interpolation value in re-write:', 10**Cool_Struc['log_cooling_interpolation'](array2interp)[0])
print('----')

array2interp = np.array([np.log10(506663.2212419483), np.log10(294060.78931362595), np.log10(1.5473355225629384e+16)])
print('----')
cool_interpolation_val = cool_interpolation(array2interp)
print('interpolation value with cube:', 10**cool_interpolation_val[0])
print('interpolation value in daniel:', 10**cool_interp_old(array2interp)[0])
print('interpolation value in re-write:', 10**Cool_Struc['log_cooling_interpolation'](array2interp)[0])
print('----')

array2interp = np.array([12 - 0.02, 5.5-0.2, 21])
print('----')
cool_interpolation_val = cool_interpolation(array2interp)
print('interpolation value with cube:', 10**cool_interpolation_val[0])
print('interpolation value in daniel:', 10**cool_interp_old(array2interp)[0])
print('interpolation value in re-write:', 10**Cool_Struc['log_cooling_interpolation'](array2interp)[0])
print('----')




array2interp = np.array([1, 4, 14])
print('----')
cool_interpolation_val = cool_interpolation(array2interp)
print('interpolation value with cube:', 10**cool_interpolation_val[0])
print('interpolation value in daniel:', 10**cool_interp_old(array2interp)[0])
print('interpolation value in re-write:', 10**Cool_Struc['log_cooling_interpolation'](array2interp)[0])
print('----')


#%%
# Now


import pandas as pd
from astropy.visualization import simple_norm



directions = ['forward', 'backward']

axis = [1, 2]

cool_cube_interpolated = cool_cube.copy()

array2interp = np.array([1, 4, 14])


def mini_cube():
    # create a 5x5 cube centered on the NaN value, and find the interpolation. 

    ndens_index = np.where(log_ndens_arr == np.round(np.log10(ndens_val), 3))[0][0]
    temp_index = np.where(log_temp_arr == np.round(np.log10(temp_val), 3))[0][0]
    phi_index = np.where(log_phi_arr == np.round(np.log10(phi_val), 3))[0][0]
    
    
    return mini_cube

# for 


test_slice = np.log10(cool_cube[1,:,:].copy())
# test_slice = cool_cube[:,0,:]
# test_slice = cool_cube[:,:,15]

df = pd.DataFrame(test_slice)
norm = simple_norm(df, 'sqrt')
plt.imshow(df, norm = norm )
plt.colorbar()
plt.show()

df_new = df.interpolate(
                        limit_direction = 'forward',
                        method = 'linear', 
                        # order = 3,
                        axis = 1,
                        )
norm = simple_norm(df_new, 'power')
plt.imshow(df_new, norm = norm )
plt.show()








#%%



def interpolate(array, filename):
    
    # cube = cooling or heating
    # array = [ndens, temp, phi] in LINEAR space
    
    
    # =============================================================================
    # Step1: read data
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

    # create lines for cube sides
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

    log_ndens_arr = create_limits(ndens_data)
    log_temp_arr = create_limits(temp_data)
    log_phi_arr = create_limits(phi_data)

    # =============================================================================
    # Step2: create cube
    # =============================================================================




    # use rectilinear grid (which allows uneven spacing!) in 3-dimensions to interpolate.
    # remember that the cube is in log space. 
    interpolation = RegularGridInterpolator((log_ndens_arr, log_temp_arr, log_phi_arr), np.log10(cool_cube),
                                               method = 'linear', # linear seems to be the best
                                              )
    
    return interpolation




























