#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:41:06 2023

@author: Jia Wei Teh

This script retrieves the parameters.
"""

import os
import yaml

def get_param():
    # load
    with open(os.environ['PATH_TO_CONFIG'],'r') as file:
        warpfield_params = yaml.load(file)
    
    # A simple script that turns dictionaries into objects
    class Dict2Class(object):
        # set object attribute
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
    
    # return
    return Dict2Class(warpfield_params)
