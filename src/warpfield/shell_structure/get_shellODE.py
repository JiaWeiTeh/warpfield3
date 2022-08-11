#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:22:09 2022

@author: Jia Wei Teh

This script contains a function that returns ODE of the ionised number density (n), 
fraction of ionizing photons that reaches a surface with radius r (phi), and the
optical depth (tau) of the shell.
"""

import numpy as np
import astropy.constants as c

def get_shellODE(y, 
                 # nShell,
                 # phi,
                 # tau
                 r, 
                 cons,
                 # sigma_dust,
                 # mu_n, mu_p, t_ion, t_neu,
                 # alpha_B
                 f_cover = 1,
                 is_ionised = True
                 ):
    """
    A function that returns ODE of the ionised number density (n), 
    fraction of ionizing photons that reaches a surface with radius r (phi), and the
    optical depth of dust (tau) of the shell.
    
    This function works in units of cgs.

    Parameters
    ----------
    y : list
        A list of ODE variable.
        # nShell : float
            the number density of the shell.
        # phi : float
            fraction of ionizing photons that reaches a surface with radius r.
        # tau : float
            the optical depth of dust in the shell.               
    r : list
        An array of radii where y is evaluated.
    cons : list
        A list of constants used in the ODE.
        # sigma_dust : float
            Dust cross-section (scaled!).
        # mu_n : float
            Mean mass per nucleus
        # mu_p : float
            Mean mass per particle
        # t_ion : float
            Temperature of ionised region.
        # t_neu : float
            Temperature of neutral region.
        # alpha_B : float
            Case B recombination coefficient.
    f_cover: float, 0 < f_cover <= 1
            The fraction of shell that remained after fragmentation process.
            f_cover = 1: all remained.
    is_ionised: boolean
            Is this part of the shell ionised? If not, then phi = Li = 0, etc.
            This happens at r > R_ionised.

    Returns
    -------
    dndr : ODE
    dphidr : ODE
    dtaudr : ODE

    """
    
    # TODO: Add f_cover
    
    # Is this region of the shell ionised?
    # If yes:
    if is_ionised:
        # unravel
        nShell, phi, tau = y
        Ln, Li, Qi, sigma_dust, mu_n, mu_p, t_ion, alpha_B = cons
        
        # prevent underflow for very large tau values
        if tau > 700:
            neg_exp_tau = 0
        else:
            neg_exp_tau = np.exp(-tau)
            
        # number density
        dndr = mu_p/mu_n/(c.k_B.cgs.value * t_ion) * (
            nShell * sigma_dust / (4 * np.pi * r**2 * c.c.cgs.value) * (Ln * neg_exp_tau + Li * phi) +\
                nShell**2 * alpha_B * Li / Qi / c.c.cgs.value
            )
        # ionising fraction
        dphidr = - 4 * np.pi * r**2 / Qi * alpha_B * nShell**2 -\
                    nShell * sigma_dust * phi
        # optical depth
        dtaudr = nShell * sigma_dust * f_cover
        # return
        return dndr, dphidr, dtaudr
    # If not, omit ionised paramters such as Li and phi.
    else:
        # unravel
        nShell, tau = y
        Ln, Qi, sigma_dust, t_neu, alpha_B = cons
        
        # prevent underflow for very large tau values
        if tau > 700:
            neg_exp_tau = 0
        else:
            neg_exp_tau = np.exp(-tau)        
            
        # number density
        dndr = 1/(c.k_B.cgs.value * t_neu) * (
            nShell * sigma_dust / (4 * np.pi * r**2 * c.c.cgs.value) * (Ln * neg_exp_tau) 
            )
        # optical depth
        dtaudr = nShell * sigma_dust
        # return
        return dndr, dtaudr            


# # Uncomment to plot
# #%%

# import matplotlib.pyplot as plt
# import scipy.integrate

# r = np.arange(1.292570882476065e+18, 1.2929396129227028e+18, 368730446637.824)
# y0 = [16401152.8251588, 1.0, 0.0,
#       ]
# cons = [1.515015429411944e+43, 1.9364219639465924e+43, 5.395106225151267e+53,
#       1.5e-21,
#       2.1287915392418182e-24,
#       1.0181176926808696e-24,
#       1e4,
#       2.59e-13]
# f_cover = 1
# is_ionised = True

# sol = scipy.integrate.odeint(get_shellODE, y0, r,
#                              args=(cons, f_cover, is_ionised),
#                              rtol=1e-3, hmin=1e-7)
# dndr, dphidr, dtaudr = zip(*sol)
# print(dndr[~0], dphidr[~0], dtaudr[~0])
# ################################## non-ionised region ##################################
# r = np.arange(3.6440984789239613e+18, 3.644165172647624e+18, 66693723663.00373)
# y0 = [982704385.1169335, 1.6896779448838628]
# cons = [1.51501543e+43, 5.39510623e+53,
#       1.5e-21,
#       1e2,
#       2.59e-13]
# is_ionised = False

# sol = scipy.integrate.odeint(get_shellODE, y0, r,
#                              args=(cons, f_cover, is_ionised),
#                              rtol=1e-3, hmin=1e-7)
# dndr, dtaudr = zip(*sol)
# print(dndr[~0], dtaudr[~0])










