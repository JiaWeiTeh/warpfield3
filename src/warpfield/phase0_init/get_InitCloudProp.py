#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 23:42:14 2022

@author: Jia Wei Teh

This script contains a function that returns initial properties of the cloud.
"""
import numpy as np
import src.warpfield.bonnorEbert as bE

def get_InitCloudProp(sfe, log_mCloud,
                      mCloud_beforeSF, 
                      nCore,
                      rCore,
                      mu_n, gamma,
                      density_profile_type = "bE_prof", 
                      alpha = -2, g = 14.1,
                      T = 1e5,
                      ):
    """
    This function computes the initial properties of the cloud, including (but not all):
        - cloud radius (Units: pc)
        - cloud edge density (Units: 1/cm3)
        And either of these two (depending on density profile):
        - cloud core radius (Units: pc)  (for pL)
        - cloud bE temperature (Units: K)  (for bE)
        
    Watch out units!

    Parameters
    ----------
    sfe : float
        Star-formation efficiency.
    log_mCloud : float
        Log mass of cloud (Units: solar mass).
    mCloud_beforeSF : boolean
        Is log_mCloud given before or after star formation?
    nCore : float
        Cloud core density (Units: 1/cm3).
    rCore : float
        Cloud radius (Units: pc).
    mu_n : float
        Mean mass per nucleus (Units: g).
    gamma : float
        Adiabatic index.
    density_profile_type : string { bE_prof | pL_prof }
        The type of density profile. Currently supports either a power-law 
        distribution, or a Bonnor-Ebert sphere. The default is "bE_prof".
    alpha : float
        The exponent, if pL_prof is chosen for `profile_type`. 
        The default is -2.
    g : float
        The ratio given as g = rho_core/rho_edge. The default is 14.1.
        This will only be considered if `bE_prof` is selected.
    T : float
        The temperature of the BE sphere. (Units: K). The default value is 1e5 K.
        This will only be considered if `bE_prof` is selected.

    Returns
    -------
    rCore : float
        cloud radius (Units: pc). 
        This value is only computed if power-law profile is selected.
    bE_T: float
        Temperature of the bE sphere.
        This value is only computed if Bonnor-ebert profile is selected.
    rCloud : float
        cloud core radius (Units: pc).
    nEdge : float
        cloud edge density (Units: 1cm3).

    """
    # TODO: update docstrings
    # Returns initial properties like nEdge, rCloud, rCore.
    # density is in 1/cm3 and radius is in pc.
    
    # mass of cloud
    if mCloud_beforeSF == 1:
        mCloud = 10**(log_mCloud)
    else:
        mCloud = 10**(log_mCloud) / (1 - sfe)
    # cluster mass
    mCluster = mCloud * sfe
    # cloud mass after SF
    mCloud_afterSF = mCloud - mCluster
    # Initialise dictionary
    cloudProp_dict = {}
    # record important properties
    # cloudProp_dict[] = 
    
    # =============================================================================
    # For power-law density profile    
    # =============================================================================
    if density_profile_type == "pL_prof":
        # compute cloud radius
        rCloud = (
                    (
                        mCloud/(4 * np.pi * nCore * mu_n) - rCore**3/3
                    ) * rCore ** alpha * (alpha + 3)\
                        + rCore**(alpha + 3)
                 )**(1/(alpha + 3))
        # compute the density at edge
        nEdge = nCore * mu_n * (rCloud/rCore)**alpha
        # return
        return rCore, rCloud, nEdge

    # =============================================================================
    # For Bonnor-Ebert density profile
    # =============================================================================
    elif density_profile_type == "bE_prof":
        
        # Remembver that rCore is a property of power-law. it is bE_T for bE spheres.
        # print(mCloud, nCore, g, mu_n, gamma)
        bE_T = bE.get_bE_T(mCloud, nCore, g, mu_n, gamma)
        rCloud, nEdge = bE.get_bE_rCloud_nEdge(nCore, bE_T, mCloud, mu_n, gamma)
        
        return bE_T, rCloud, nEdge















