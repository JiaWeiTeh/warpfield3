#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:37:58 2022

@author: Jia Wei Teh

This script contains function which computes the mass profile of cloud.
"""

import astropy.units as u
import scipy.integrate
import numpy as np
import src.warpfield.bonnorEbert as bE
import src.warpfield.density_profile as density_profile

    
def get_mass_profile(r_arr,
                         rCore, rCloud, 
                         nCore, nISM,
                         mCloud,
                         mu_n, gamma,
                         rdot_arr = [],
                         return_rdot = False,
                         profile_type = "bE_prof", 
                         alpha = -2, g = 14.1,
                         T = 1e5,
                         ):
    """
    This function takes in basic properties of cloud and calculates the 
    radial profile of swept-up mass and time-derivative mass of the shell.
    
    Watch out the units!

    Parameters
    ----------
    r_arr : list/array of radius of interest
        Radius at which we are interested in the density (Units: pc).
    rCore : float
        Core radius. (Units: pc)
    rCloud : float
        Cloud radius. (Units: pc)
    nCore : float
        Core number density. (Units: 1/cm^3)
    nISM : float
        ISM number density. (Units: 1/cm^3)
    mCloud : float
        Mass of cloud (Units: solar mass).
    mu_n : float
        Mean mass per nucleus (Units: cgs, i.e., g)
    gamma: float
        Adiabatic index of gas.
    return_rdot: boolean { True | False }
        Whether or not to also compute the time-derivative of radius.
    rdot_arr: None or an array of float
        Time-derivative of radius. (dr/dt)
    profile_type : string { bE_prof | pL_prof }
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
    mGas: array of float
        The mass profile. (Units: Msol)
    mGasdot: array of float
        The time-derivative mass profile dM/dt. (Units: Msol/s)

    """
    
    # initialise
    mGas = r_arr * np.nan
    mGasdot = r_arr * np.nan
    # convert to np.array
    r_arr = np.array(r_arr)
    
    # Setting up values
    rhoCore = nCore * mu_n
    rhoISM = nISM * mu_n
    
    # =============================================================================
    # power-law density profile (alpha < 0)
    # =============================================================================
    if profile_type == "pL_prof":
        # Change from g/cm3 to pc/Msol for further computation
        rhoCore = rhoCore * (u.g/u.cm**3).to(u.M_sun/u.pc**3)
        rhoISM = rhoISM * (u.g/u.cm**3).to(u.M_sun/u.pc**3)
        
        # input values into mass array
        # inner sphere
        mGas[r_arr <= rCore] = 4 / 3 * np.pi * r_arr[r_arr <= rCore]**3 * rhoCore
        # composite region, see Eq25 in WARPFIELD 2.0 (Rahner et al 2018)
        # assume rho_cl \propto rho (r/rCore)**alpha
        mGas[r_arr > rCore] = 4. * np.pi * rhoCore * (
                       rCore**3/3. +\
                      (r_arr[r_arr > rCore]**(3.+alpha) - rCore**(3.+alpha))/((3.+alpha)*rCore**alpha)
                      )
        # outer sphere
        mGas[r_arr > rCloud] = mCloud + 4. / 3. * np.pi * rhoISM * (r_arr[r_arr > rCloud]**3 - rCloud**3)
        
        if return_rdot: # compute dM/dt?
            # is array given?
            if len(rdot_arr) == len(r_arr): 
                rdot_arr = np.array(rdot_arr)
                # input values into mass array
                # dm/dt, see above for expressions of m.
                mGasdot[r_arr <= rCore] = 4 * np.pi * rhoCore * r_arr[r_arr <= rCore]**2 * rdot_arr[r_arr <= rCore]
                mGasdot[r_arr > rCore] = 4 * np.pi * rhoCore * (r_arr[r_arr > rCore]**(2+alpha) / rCore**alpha) * rdot_arr[r_arr > rCore]
                mGasdot[r_arr > rCloud] = 4 * np.pi * rhoISM * r_arr[r_arr > rCloud]**2 * rdot_arr[r_arr > rCloud]
            # if rdot is not given, there is no need to calculate mGasdot
            else:
                raise Exception('return_rdot is set to True but rdot_arr is not specified.')
        # return
        return mGas, mGasdot
        
    # =============================================================================
    # Bonnor-Ebert density profile 
    # =============================================================================
    # Get density profile
    elif profile_type == "bE_prof":
        # Get density profile before altering units
        dens_arr, xi_arr = density_profile.get_density_profile(r_arr, rCore, rCloud, 
                                                       nCore, nISM, mCloud, mu_n, gamma,
                                                       profile_type, alpha,
                                                       g, T)
        # sound speed
        c_s = bE.get_bE_soundspeed(T, mu_n, gamma)
        # Convert density profile 
        dens_arr = dens_arr * (1/u.cm**3).to(1/u.m**3).value
        # Then convert all to SI units
        rCloud = rCloud * u.pc.to(u.m)
        r_arr = r_arr * u.pc.to(u.m)
        rdot_arr = rdot_arr * u.pc.to(u.m)
        rCore = rCore * u.pc.to(u.m)
        rhoCore = rhoCore * (u.g/u.cm**3).to(u.kg/u.m**3)
        rhoISM = rhoISM * (u.g/u.cm**3).to(u.kg/u.m**3)
        mCloud = mCloud * (u.M_sun).to(u.kg)
        # initial values (boundary conditions)
        # effectively 0.
        for ii, xi in enumerate(xi_arr):
            # For radius within cloud
            if r_arr[ii] <= rCloud:
                mass, _ = scipy.integrate.quad(bE.massIntegral,0,xi,args=(rhoCore,c_s))
                mGas[ii] = mass
            else:
                mGas[ii] = mCloud + 4 / 3 * np.pi * rhoISM * (r_arr[ii]**3 - rCloud**3)
        
        if return_rdot: # compute dM/dt?
            # is array given?
            if len(rdot_arr) == len(r_arr): 
                # array-ise
                rdot_arr = np.array(rdot_arr)   
                
                # # # initial condition (set to a value that is very close to zero)
                # y0 = [1e-12, 1e-12]
                # # solve ODE
                # psi, omega = zip(*scipy.integrate.odeint(bE.laneEmden, y0, xi_arr))
                # psi = np.array(psi)
                # dens_arr = rhoCore * np.exp(-psi)
                
                mGasdot[r_arr <= rCloud] = 4 * np.pi * r_arr[r_arr <= rCloud]**2 * rdot_arr[r_arr <= rCloud] * dens_arr[r_arr <= rCloud] 
                mGasdot[r_arr > rCloud]  = 4 * np.pi * r_arr[r_arr > rCloud]**2 * rdot_arr[r_arr > rCloud] * rhoISM 
            else:
                raise Exception('return_rdot is set to True but rdot_arr is not specified.')
    
        mGas = mGas * u.kg.to(u.Msun) #return to solar masses
        mGasdot = mGasdot * u.kg.to(u.Msun) #return to solar masses

        return mGas, mGasdot
        
    
    
# #%%

# # Uncomment to check out plot
    
# import matplotlib.pyplot as plt

# rCloud = 355
# rCore = 10
# mu_n = 2.1287915392418182e-24
# gamma = 5/3
# mCloud =  1e9
# r_arr = np.linspace(1e-4, 500, 201)
# rdot_arr = np.linspace(1, 100, 201) * u.km.to(u.pc)
# nCore = 1000
# nISM = 10
# alpha = -2
# T = 451690.2638133162
# g = 14.1
# profile_type = "bE_prof"
# return_rdot = True

# mGas, mGasdot = get_mass_profile(r_arr,
#                          rCore, rCloud, 
#                          nCore, nISM,
#                          mCloud,
#                          mu_n, gamma,
#                          rdot_arr,
#                          return_rdot = True,
#                          profile_type = "bE_prof", 
#                          # profile_type = "pL_prof", 
#                          alpha = -2, g = 14.1,
#                          T = 451690.2638133162,
#                          )


# fig = plt.subplots(1, 1, figsize = (7, 5), dpi = 200)
# plt.plot(r_arr, mGas, 
#           )
# plt.xlabel('$r(pc)$')
# plt.ylabel('M (Msol)')
# plt.vlines(rCloud, 0, 9e9, linestyles = '--', color = 'k', 
#            label = 'rCloud')
# plt.ylim(1e-1, 1e10)
# plt.yscale('log')
# plt.legend()


# fig = plt.subplots(1, 1, figsize = (7, 5), dpi = 200)
# plt.plot(r_arr, mGasdot, 
#           )
# plt.xlabel('$r(pc)$')
# plt.ylabel('Mdot (Msol/s)')
# # plt.vlines(rCloud, 0, 9e9, linestyles = '--', color = 'k', 
# #            label = 'rCloud')
# # plt.ylim(1e-1, 1e10)
# plt.yscale('log')
# # plt.legend()






        
    
