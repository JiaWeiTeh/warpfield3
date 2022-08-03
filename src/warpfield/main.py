#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:46:31 2022

@author: Jia Wei Teh

This script contains a wrapper that initialises the expansion of 
shell.
"""

import numpy as np
from src.warpfield import get_InitCloudProp

def expansion(param_dict):
    """
    This function takes in the parameters and feed them into smaller
    functions.

    Parameters
    ----------
    param_dict : dict
        Dictionary of parameters obtained from .param file.

    Returns
    -------
    None.

    """
    
    
    #  Get a dictionary of initial cloud properties
    rCore, rCloud, nEdge = get_InitCloudProp.get_InitCloudProp(sfe, 
                                                               log_mCloud, 
                                                               mCloud_beforeSF, 
                                                               nCore, 
                                                               rCore, 
                                                               mu_n, 
                                                               gamma)
    
    print('rCore', rCore)
    print('rCloud', rCloud)
    print('nEdge', nEdge)
    return