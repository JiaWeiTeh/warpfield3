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
    dens_profile : string { bE_prof | pL_prof }
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
    mCloud_afterSF: float
        cloud mass after cluster formation (Units: Msun).
   mCluster: float 
       cluster mass (Units: Msun).

    """

    # density is in 1/cm3 and radius is in pc.
    
    # Note:
    #   old code: get_cloudproperties() in get_startvalues.py in 
    #           expansion_main() in expansion_full.py
        
    
    log_mCloud = warpfield_params.log_mCloud
    sfe = warpfield_params.sfe
    rCore = warpfield_params.rCore
    nCore = warpfield_params.nCore
    mu_n = warpfield_params.mu_n
    gamma = warpfield_params.gamma_adia
    
    # mass of cloud
    if warpfield_params.is_mCloud_beforeSF == 1:
        mCloud = 10**(log_mCloud)
    else:
        mCloud = 10**(log_mCloud) / (1 - sfe)
    # cluster mass
    mCluster = mCloud * sfe
    # cloud mass after SF
    mCloud_afterSF = mCloud - mCluster
    
    # Old code: get_cloud_Rn().
    # =============================================================================
    # For power-law density profile    
    # =============================================================================
    if warpfield_params.dens_profile == "pL_prof":
        # initialise value if not selected.
        bE_T = np.nan
        alpha = warpfield_params.dens_a_pL
        # converting to cgs
        mCloud = mCloud * u.M_sun.to(u.g)
        rCore = rCore * u.pc.to(u.cm)
        # compute cloud radius
        rCloud = (
                    (
                        mCloud/(4 * np.pi * nCore * mu_n) - rCore**3/3
                    ) * rCore ** alpha * (alpha + 3)\
                        + rCore**(alpha + 3)
                 )**(1/(alpha + 3))
        # compute the density at edge
        nEdge = nCore * mu_n * (rCloud/rCore)**alpha
        # sanity check
        print('nEdge', nEdge)
        if nEdge < warpfield_params.nISM:
            print(f'nCore: {nCore}, nISM: {warpfield_params.nISM}')
            sys.exit('"The density at the edge of the cloud is lower than the ISM; please consider increasing nCore."')
        print(f'nCore: {nCore}, nISM: {warpfield_params.nISM}')
        sys.exit()
        # converting back
        rCore = rCore * u.cm.to(u.pc)
        rCloud = rCloud * u.cm.to(u.pc)

    # =============================================================================
    # For Bonnor-Ebert density profile
    # =============================================================================
    elif warpfield_params.dens_profile == "bE_prof":
        # initialise value if not selected.
        rCore = warpfield_params.rCore
        g = warpfield_params.dens_g_bE
        # Remember that rCore is a property of power-law. it is bE_T for bE spheres.
        # print(mCloud, nCore, g, mu_n, gamma)
        bE_T = bE.get_bE_T(mCloud, nCore, g, mu_n, gamma)
        # print(mCloud, nCore, g, mu_n, gamma)
        # these are the values
        # 1000000.0 1000.0 14.1 2.1287915392418182e-24 1.6666666666666667
        # 4649.954642315685
        rCloud, nEdge = bE.get_bE_rCloud_nEdge(nCore, bE_T, mCloud, mu_n, gamma)
        
    # return
    return rCore, bE_T, rCloud, nEdge, mCloud_afterSF, mCluster
    


#%%

# # fix this

# aa = get_InitCloudProp(warpfield_params.sfe, warpfield_params.warpfield_params.log_mCloud,
#                       warpfield_params.mCloud_beforeSF, 
#                       warpfield_params.nCore,
#                       warpfield_params.rCore,
#                       warpfield_params.mu_n, warpfield_params.gamma_adia,
#                       warpfield_params,
#                        # "pL_prof", 
#                        "bE_prof",
#                       warpfield_params.dens_a_pL, warpfield_params.dens_g_bE,
#                       T = 1e5,
#                       )
    
# print(aa)









