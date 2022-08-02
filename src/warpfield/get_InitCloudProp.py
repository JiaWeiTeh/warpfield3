#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 23:42:14 2022

@author: Jia Wei Teh

This script contains a function that returns initial values for the cloud.
"""
import numpy as np

def get_InitCloudProp(sfe, log_mCloud,
                      mCloud_beforeSF, 
                      nCore,
                      rCore,
                      u_n, gamma,
                      density_profile = "pL_prof",
                      alpha = None, g = None,
                      ):
    
    
    
    

    
    # Returns initial properties like rhoEdge, rCloud, rCore.
    
    # mass of cloud
    if mCloud_beforeSF == '1':
        mCloud = np.log10(log_mCloud)
    else:
        mCloud = np.log10(log_mCloud) / (1 - sfe)
    # cluster mass
    mCluster = mCloud * sfe
    # cloud mass after SF
    mCloud_afterSF = mCloud - mCluster
    
    # =============================================================================
    # For power-law density profile    
    # =============================================================================
    if density_profile == "pL_prof":
        # If power-law, use the given (or default value) for core radius.
        rCore = rCore
        # compute cloud radius
        rCloud = (
                    (
                        log_mCloud/(4 * np.pi * nCore * u_n) - rCore**3/3
                    ) * rCore ** alpha * (alpha + 3)\
                        + rCore**(alpha + 3)
                 )**(1/(alpha + 1))
        # compute the density at edge
        nEdge = nCore * u_n * (rCloud/rCore)**alpha
        # return
        return rCore, rCloud, nEdge

    # =============================================================================
    # For Bonnor-Ebert density profile
    # =============================================================================
    elif density_profile == "bE_prof":
        
        # TODO
        
        
        
        
        
        
        
        
        
        
        return


