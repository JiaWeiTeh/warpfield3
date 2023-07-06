#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 22:25:55 2022

@author: Jia Wei Teh

This script includes a mini function that helps compute density of the 
shell at the innermost radius.
"""

import numpy as np
import astropy.constants as c
import scipy.optimize


def get_nShell0(pBubble, T, warpfield_params):
    """
    This function computes density of the shell at the innermost radius.

    Parameters
    ----------
    pBubble : pressure of the bubble
        DESCRIPTION.
    T : float (units: K)
        Temperature of at inner edge of shell.
            
    Returns
    -------
    nShell0 : float
        The density of shell at inner edge/radius.
    nShell0_cloudy : float
        The density of shell at inner edge/radius, but including B-field, as
        this will be passed to CLOUDY.

    """
    # TODO: BMW and nMW are given in log units. Is this the same?
    # TODO: Add description for BMW nMW
    # convert to cgs
    # (u.M_sun/u.pc/u.Myr**2).cgs = 6.4705429e-13
    pBubble = pBubble * 6.4705429e-13
    # The density of shell at inner edge/radius
    nShell0 = warpfield_params.mu_p/warpfield_params.mu_n/(c.k_B.cgs.value * T) * pBubble
    # The density of shell at inner edge/radius
    # that is passed to cloudy (usually includes B-field)
    # Note: this is only used to pass on to CLOUDY and does not affect WARPFIELD.
    # Assuming equipartition and pressure equilibrium, such that
    # Pwind = Pshell, where Pshell = Ptherm + Pturb + Pmag
    #                              = Ptherm + 2Pmag
    # where Pmag \propro n^(gamma/2)
    
    # def pShell(n, pBubble, T, mu_n, mu_p, BMW, nMW, gamma_mag):
    #     # return function
    #     return warpfield_params.mu_n/warpfield_params.mu_p * c.k_B.cgs.value * T * n +\
    #                 BMW**2 / (4 * np.pi * nMW**gamma_mag) * n ** (4/3) - pBubble
    
    BMW = 10**(warpfield_params.log_BMW)
    nMW = 10**(warpfield_params.log_nMW)
    
    def pShell(n, pBubble, T):
        # return function
        return warpfield_params.mu_n/warpfield_params.mu_p * c.k_B.cgs.value * T * n +\
                    BMW**2 / (4 * np.pi * nMW**warpfield_params.gamma_mag) * n ** (4/3) - pBubble
  
    nShell0_cloudy = scipy.optimize.fsolve(pShell, x0 = 10,
                   args = (pBubble, T))[0]
    
    return nShell0, nShell0_cloudy


# # Uncomment to test

# #%%

# pBubble = 4.732521672817037e-05 
# T = 10000.0

# nShell0, nShell0_cloudy = get_nShell0(pBubble, T,
#                 )

# # True answer: 16401152.8251588 1227812.2967044176
# print(nShell0, nShell0_cloudy)








