#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:16:06 2023

@author: Jia Wei Teh

This script contains the function to obatin the initial cooling function.
"""

from src.warpfield.cooling import read_opiate

def get_firstCoolStruc(Zism, age, 
                       basename="opiate_cooling", 
                       extension=".dat", cool_folder="cooling_tables",
                       indiv_CH=False):
    """
    get the first cooling function to start with
    :param Zism:
    :param age:
    :param basename:
    :param extension:
    :param cool_folder:
    :param indiv_CH:
    :return:
    """

    Cool_Struc = read_opiate.get_Cool_dat_timedep(Zism, age, 
                                               basename=basename, extension=extension, cool_folder=cool_folder, indiv_CH=True)
    onlycoolfunc, onlyheatfunc = read_opiate.create_onlycoolheat(Zism, age, 
                                                              basename=basename, extension=extension, cool_folder=cool_folder)
    Cool_Struc['Cfunc'] = onlycoolfunc
    Cool_Struc['Hfunc'] = onlyheatfunc

    return Cool_Struc