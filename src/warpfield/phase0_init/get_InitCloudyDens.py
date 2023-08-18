#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:48:08 2023

@author: Jia Wei Teh

This script contains functions that create density law for cloudy, and 
write initial density profile to a txt. file.
"""

import numpy as np
import astropy.constants as c
import os
import csv
import src.warpfield.cloud_properties.density_profile as density_profile

def get_InitCloudyDens(path2output,
                       density_specific_param, 
                       rCloud, mCloud,
                       coll_counter=0):

    # Note:
        # old code: __cloudy__.create_dlaw()
    
    dx_small = 1e-4
    # last radius just outside cloud
    # TODO: shouldn't this be +dx_small then?
    r = np.logspace(-1.,np.log10(rCloud - dx_small), endpoint=True, num=200) 
    # join with different endpoints
    r = np.concatenate([[dx_small],r,[r[-1]+0.001], [r[-1]+5.0], [r[-1]+500.0]])
    # get density profile
    
    n = density_profile.get_density_profile(r, density_specific_param,
                                            rCloud, mCloud)[0]
    
    logn = np.log10(n)
    logr = np.log10(r * c.pc.cgs.value)
    
    # old: dlaw.txt
    # save into csv
    full_path = os.path.join(path2output, 'init_density_profile_col'+ str(coll_counter) + '.csv')
    rel_path = os.path.relpath(full_path, os.getcwd())
    np.savetxt(full_path,
               np.transpose([logr, logn]), fmt='%.6e',
               delimiter = ',',
               header="Density [log10(n)],Radius [log10(r)]", comments='')
    
    # TODO: make this sound better, and also check if the logr is in correct unit.
    return print(f'\033[1m\033[96mInitial density: {rel_path}\033[0m')
