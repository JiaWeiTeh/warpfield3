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

def expansion(params_dict):
    """
    This function takes in the parameters and feed them into smaller
    functions.

    Parameters
    ----------
    params_dict : dict
        Dictionary of parameters obtained from .param file.

    Returns
    -------
    None.

    """
    
    
    #  Get a dictionary of initial cloud properties
    _, rCloud, nEdge = get_InitCloudProp.get_InitCloudProp(params_dict['sfe'], 
                                                               params_dict['log_mCloud'], 
                                                               params_dict['mCloud_beforeSF'], 
                                                               params_dict['nCore'], 
                                                               params_dict['rCore'], 
                                                               params_dict['mu_n'], 
                                                               params_dict['gamma'])
    
    return