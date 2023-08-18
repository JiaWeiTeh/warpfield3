# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Sun Jul 24 23:42:14 2022

@author: Jia Wei Teh

This script contains a function that returns initial properties of the cloud.
"""
import numpy as np
import src.warpfield.cloud_properties.bonnorEbert as bE
import astropy.units as u
import sys
#--
from src.input_tools import get_param
warpfield_params = get_param.get_param()

def get_InitCloudProp():
    """
    This function computes the initial properties of the cloud, including (but not all):
        - cloud radius (Units: pc)
        - cloud edge density (Units: 1/cm3)
        And either of these two (depending on density profile):
        - cloud core radius (Units: pc)  (for pL)
        
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
    dens_profile : string { bE_prof | pL_prof }
        The type of density profile. Currently supports either a power-law 
        distribution, or a Bonnor-Ebert sphere. The default is "bE_prof".
    alpha : float
        The exponent, if pL_prof is chosen for `profile_type`. 
        The default is -2.

    Returns
    -------
    rCore : float
        cloud radius (Units: pc). 
        This value is only computed if power-law profile is selected.
    rCloud : float
        cloud core radius (Units: pc).
    nEdge : float
        cloud edge density (Units: 1cm3).
    mCloud_afterSF: float
        cloud mass after cluster formation (Units: Msun).
    mCluster: float 
       cluster mass (Units: Msun).

    """

    # Note:
    #   old code: get_cloudproperties() in get_startvalues.py in 
    #           expansion_main() in expansion_full.py
        
    # load parameters
    mu_n = warpfield_params.mu_n
    alpha = warpfield_params.dens_a_pL
    
    # Old code: get_cloud_Rn().
    # converting to cgs
    mCloud = warpfield_params.mCloud * u.M_sun.to(u.g)
    
    # compute cloud radius
    # use core radius/density if there is a power law. If not, use average density.
    if alpha != 0:
        nCore = warpfield_params.nCore
        rCore = warpfield_params.rCore
        rCore = rCore * u.pc.to(u.cm)
        rCloud = (
                    (
                        mCloud/(4 * np.pi * nCore * mu_n) - rCore**3/3
                    ) * rCore ** alpha * (alpha + 3) + rCore**(alpha + 3)
                 )**(1/(alpha + 3))
        # density at edge
        nEdge = nCore * (rCloud/rCore)**alpha
        
    elif alpha == 0:
        nAvg = warpfield_params.dens_navg_pL
        rCloud = (3 * mCloud / 4 / np.pi / (nAvg * mu_n))**(1/3)
        # density at edge should just be the average density
        nEdge = nAvg
        
    # converting back
    rCloud = rCloud * u.cm.to(u.pc)
    
    # print('check for initials')
    # print('rCore, mCloud, nEdge, nCore, rCloud')
    # print( (rCore * u.cm.to(u.pc)), (mCloud/u.M_sun.to(u.g)), nEdge, nCore, rCloud)
    # sys.exit()
    
    # sanity check
    if nEdge < warpfield_params.nISM:
        print(f'nCore: {nCore}, nISM: {warpfield_params.nISM}')
        sys.exit('"The density at the edge of the cloud is lower than the ISM; please consider increasing nCore."')

    # return
    return rCloud, nEdge
    

