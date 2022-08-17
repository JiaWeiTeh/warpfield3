#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:46:31 2022

@author: Jia Wei Teh

This script contains a wrapper that initialises the expansion of 
shell.
"""

import numpy as np
from src.warpfield.phase0_init import get_InitCloudProp

def expansion(params):
    """
    This function takes in the parameters and feed them into smaller
    functions.

    Parameters
    ----------
    params : Object
        An object describing WARPFIELD parameters.

    Returns
    -------
    None.

    """
    
    #  Get a dictionary of initial cloud properties
    cloudProp_dict = get_InitCloudProp.get_InitCloudProp(params.sfe, 
                                                         params.log_mCloud,
                                                         params.mCloud_beforeSF,
                                                         params.nCore,
                                                         params.rCore,
                                                         params.mu_n,
                                                         params.gamma_adia,
                                                         )
    
    return