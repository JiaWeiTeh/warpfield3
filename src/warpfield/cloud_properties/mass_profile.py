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
import sys 

import src.warpfield.cloud_properties.bonnorEbert as bE
import src.warpfield.cloud_properties.density_profile as density_profile



from src.input_tools import get_param
warpfield_params = get_param.get_param()




# TODO: return_Mdot, not rdot. This is a misnomer. 

def get_mass_profile(r_arr,
                         density_specific_param, 
                         rCloud, 
                         mCloud,
                         rdot_arr = np.array([]),
                         return_rdot = False,
                         ):
    """
    This function takes in basic properties of cloud and calculates the 
    radial profile of swept-up mass and time-derivative mass of the shell.
    
    Watch out the units!

    Parameters
    ----------
    r_arr : list/array of radius of interest
        Radius at which we are interested in the density (Units: pc).
    density_specific_param: float
        Available parameters = { rCore | T }
        - rCore : float
            Core radius (Units: pc). 
            This parameter is only invoked if profile_type == 'pL_prof'
        - T : float
            The temperature of the BE sphere. (Units: K). The default value is 1e5 K.
            This will only be considered if `bE_prof` is selected.
            This parameter is only invoked if profile_type == 'bE_prof'
    mCloud : float
        Mass of cloud (Units: solar mass).
    return_rdot: boolean { True | False }
        Whether or not to also compute the time-derivative of radius.
    rdot_arr: None or an array of float
        Time-derivative of radius. (dr/dt)
    profile_type : string { bE_prof | pL_prof }
        The type of density profile. Currently supports either a power-law 
        distribution, or a Bonnor-Ebert sphere. The default is "bE_prof".

    Returns
    -------
    mGas: array of float
        The mass profile. (Units: Msol)
    mGasdot: array of float
        The time-derivative mass profile dM/dt. (Units: Msol/s)

    """
    
    # convert to np.array
    # array for easier operation
    if hasattr(r_arr, '__len__'):
        r_arr  = np.array(r_arr)
    else:
        r_arr  = np.array([r_arr])
        
    if hasattr(rdot_arr, '__len__'):
        rdot_arr  = np.array(rdot_arr)
    else:
        rdot_arr  = np.array([rdot_arr])
    
    # initialise
    mGas = r_arr * np.nan
    mGasdot = r_arr * np.nan

    # Setting up values
    rhoCore = warpfield_params.nCore * warpfield_params.mu_n
    rhoISM = warpfield_params.nISM * warpfield_params.mu_n
    
    # =============================================================================
    # power-law density profile (alpha < 0)
    # =============================================================================
    if warpfield_params.dens_profile == "pL_prof":
        # redefine for clarity
        rCore = density_specific_param
        alpha = warpfield_params.dens_a_pL
        # Change from g/cm3 to pc/Msol for further computation
        rhoCore = rhoCore * (u.g/u.cm**3).to(u.M_sun/u.pc**3)
        rhoISM = rhoISM * (u.g/u.cm**3).to(u.M_sun/u.pc**3)
        
        # input values into mass array
        # inner sphere
        mGas[r_arr <= rCore] = 4 / 3 * np.pi * r_arr[r_arr <= rCore]**3 * rhoCore
        # composite region, see Eq25 in WARPFIELD 2.0 (Rahååner et al 2018)
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
    elif warpfield_params.dens_profile == "bE_prof":
        # redefine for clarity
        T = density_specific_param
        # Get density profile before altering units
        dens_arr, xi_arr = density_profile.get_density_profile(r_arr,
                         density_specific_param,  rCloud, mCloud, warpfield_params)
        # sound speed
        # print('values that are fed into the c_s calculation\n', T)
        c_s = bE.get_bE_soundspeed(T, warpfield_params.mu_n, warpfield_params.gamma_adia)
        # print('sound speed and T',c_s, T)
        # Convert density profile 
        # TODO: This is not included?
        dens_arr = dens_arr  * warpfield_params.mu_n * (u.g/u.cm**3).to(u.kg/u.m**3)
        # Then convert all to SI units
        rCloud = rCloud * u.pc.to(u.m)
        r_arr = r_arr * u.pc.to(u.m)
        rhoCore = rhoCore * (u.g/u.cm**3).to(u.kg/u.m**3)
        # print('rhoISM before', rhoISM)
        rhoISM = rhoISM * (u.g/u.cm**3).to(u.kg/u.m**3)
        mCloud = mCloud * (u.M_sun).to(u.kg)
        # initial values (boundary conditions)
        # effectively 0.
        # print("we are now in calc_mass_BE")
        # print('rhoCore, rhoISM, r, rCloud, mCloud, c_s, xi')
        # print(rhoCore, rhoISM, r_arr, rCloud, mCloud, c_s, xi_arr)
        
        
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
                rdot_arr = np.array(rdot_arr) * u.pc.to(u.m)
                
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

        # print("Here is the calculation of mGas, mGasdot")
        # print(mGas, mGasdot)
        # sys.exit()
        
        return mGas, mGasdot
        
    
    
#%%

# # Uncomment to debug
    
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

# mGas = get_mass_profile(0.23790232199299727, 451690.2638133162, 355.8658723191992, 990000000.0, warpfield_params)

#%%

# params_wfld4  = [201648867747.70163, 105846321.7868845, 1.6666666666666667, 990000000.0, 31.394226159698523, 451690.2638133162, -2, 10000000.0, 0.0, 0.0, 0.0, 355.8658723191992, 1.0, 0.0, 1e+99, 1e+99, 1394.801812664602, 0.01]
# y0_wfld4 = [0.23790232199299727, 3656.200432285518, 5722974.028981317]
# r, v, E = y0_wfld4  # unpack current values of y (r, rdot, E)

# # [LW, PWDOT, GAM, Mcloud_au, RHOA, RCORE, A_EXP, MSTAR, LB, FRAD, fabs_i, rcloud_au, phase0, tcoll[coll_counter], t_frag, tscr, CS, SFE]
# [LW, PWDOT, GAM, Mcloud_au, RHOA, RCORE, A_EXP, MSTAR, LB, FRAD, fabs_i, rcloud_au,\
#   phase0, tcoll, t_frag, tscr, CS, SFE] = params_wfld4
    
# # LW, PWDOT, GAM, MCLOUD, RHOA, RCORE, A_EXP, MSTAR, LB, FRAD, FABSi,\
# #         RCLOUD, density_specific_param, warpfield_params,\
# #             tSF, tFRAG, tSCR, CS, SFE  = params  # unpack parameters
            
# VW = 2.*LW/PWDOT

# Msh, Msh_dot = get_mass_profile(r,  451690.2638133162,\
#                                 rcloud_au, Mcloud_au, warpfield_params,\
#                                     rdot_arr = v, return_rdot = True)



# print('\n\nHere are the values that are fed into mass_profile')
# print(r,  451690.2638133162,\
#                                 rcloud_au, Mcloud_au,\
#                                     v)




#%%



# fig = plt.subplots(1, 1, figsize = (7, 5), dpi = 200)
# plt.plot(r_arr, mGas, 
#           )
# plt.xlabel('$r(pc)$')
# plt.ylabel('M (Msol)')
# plt.vlines(rCloud, 0, 9e9, linestyles = '--', color = 'k', 
#             label = 'rCloud')
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


    




        
    
