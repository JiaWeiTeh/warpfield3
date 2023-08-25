#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:16:06 2023

@author: Jia Wei Teh

This script contains the function to obatin the cooling function.
"""

from src.warpfield.cooling import read_opiate


def get_firstCoolStruc(age):
    """
    
    
    """

    # TODO: change file names here so that they are more intuitive.
    
    # Get cooling structure for 
    Cool_Struc = read_opiate.read_opiate(age)
    
    
    # TODO: dont do this here. Do this there. 
    onlycoolfunc, onlyheatfunc = read_opiate.create_onlycoolheat(Zism, age, 
                                                              basename=basename, extension=extension, cool_folder=cool_folder)
    Cool_Struc['Cfunc'] = onlycoolfunc
    Cool_Struc['Hfunc'] = onlyheatfunc

    return Cool_Struc