#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 20:53:03 2022

@author: Jia Wei Teh

This script contains a function that evaluates the shell structure.
"""


import numpy as np
import astropy.constants as c
from src.warpfield.shell_structure import get_shellODE, get_shellParams
import scipy.integrate 

def shell_structure(
        rShell0, 
        pressure,
        Ln, Li, Qi,
        mShell_end, # at r0 0.23790232199299727?
        params,
        pBubble, T,
        sigma_dust,
        f_cover,
        # TBD these are just filler keywords
        ):
    
    # cgs units!!
    # TODO: Add also f_cover
    
    
    # initialise values at r = rShell0 = inner edge of shell
    rShell_start = rShell0
    phi0 = 1
    # tau(r) at ionised region?
    tau0_ion = 0
    mShell0 = 0
    # Obtain density at the inner edge of shell
    nShell0, nShell0_cloud = get_shellParams.get_nShell0(pBubble, T,
                                    params.mu_p, params.mu_n,
                                    10**(params.log_BMW), 10**(params.log_nMW),
                                    params.gamma_mag,
                                    )
    # define for future use, as nShell0 constantly changes in the loop.
    nShellInner = nShell0
    
    # =============================================================================
    # Before beginning the integration, initialise some logic gates.
    # =============================================================================
    # 1. Have we accounted all the shell mass in our integration? 
    # I.e., is mass(r = R) >= mShell?
    is_allMassSwept = False
    # 2. Are all ionising photons being used up? 
    # I.e., is phi(r = R) = 0?
    is_phiZero = False
    # 3. Has the shell dissolved?
    is_shellDissolved = False
    # 4. Is the shell fully ionised at r = R?
    is_fullyIonised = False
    
    # Create arrays to store values
    # shell mass at r (ionised region)
    mShell_arr_ion = np.array([])
    # cumulative shell mass at r (ionised region)
    mShell_arr_cum_ion = np.array([])
    # phi at r (ionised region)
    # Note here that phiShell_arr_ion is redundant; this is because
    # the phi parameter (where phi > 0 ) naturally denotes an ionised region.
    # However, we initialise them here... just cause.
    phiShell_arr_ion = np.array([])
    # tau at r (ionised region)
    tauShell_arr_ion = np.array([])
    # density at r (ionised region)
    nShell_arr_ion = np.array([])
    # r array of ionised region
    rShell_arr_ion = np.array([])
    
    while not is_allMassSwept and not is_phiZero:
        
        # =============================================================================
        # Define the range at which integration occurs.
        # This is necessary because, unfortunately, the ODE is very stiff and 
        # requires small steps of r. 
        # =============================================================================
        
        # First, set the end of integration, rShell_stop:
        # The integration range should not be larger than 1pc. However,
        # if the shell is lesser than 1pc, we will obviously use that instead.
        # Assuming constant density, what is the maximum possible shell thickness?
        # Old version:
        max_shellThickness = (3 * Qi / (4 * np.pi * params.alpha_B * nShell0**2) + rShell_start**3)**(1/3) - rShell_start 
        # New version(?):
        # max_shellThickness = r_stromgren - rShell0
        # max_shellThickness = (3 * Qi / (4 * np.pi * alpha_B * nShell0**2))**(1/3) - rShell_start
        # Therefore the end of integration is just rThickness + rStart
        rShell_stop = np.min(max_shellThickness, 1 * c.pc.cgs.value) + rShell_start
        
        # Then, set the step size. This will just be a very small number
        # i.e., 5e-4pc, unless the thickness itself is not sufficient to support 
        # at least 2000 steps cross the shell
        rShell_step = np.max([
            np.min([ 5e-4 * c.pc.cgs.value, max_shellThickness/1e3]),
            max_shellThickness/1e6
            ])
        
        # We now have the array at which we integrate
        rShell_arr = np.arange(rShell_start, rShell_stop, rShell_step)
        # Get arguments and parameters for integration:
        # ionised region
        is_ionised = True
        # initial values
        y0 = [nShell0, phi0, tau0_ion]
        # constants
        cons = [Ln, Li, Qi,
                sigma_dust, params.mu_n, params.mu_p, 
                params.t_ion, params.t_neu,
                params.alpha_B]
        # Run integration
        sol_ODE = scipy.integrate.odeint(get_shellODE.get_shellODE, y0, rShell_arr,
                              args=(cons, f_cover, is_ionised),
                              rtol=1e-3, hmin=1e-7)
        # solved for n(r), phi(r), and tau(r)
        nShell_arr = sol_ODE[:,0]
        phiShell_arr = sol_ODE[:,1]
        tauShell_arr = sol_ODE[:,2]
        # mass of spherical shell. Volume given by V = 4 pi r**2 * thickness
        mShell_arr = np.empty_like(rShell_arr)
        mShell_arr[0] = mShell0
        mShell_arr[1:] = nShell_arr[1:] * params.mu_n * 4 * np.pi * rShell_arr[1:]**2 * rShell_step
        mShell_arr_cum = np.cumsum(mShell_arr)
        
        # =============================================================================
        # Now, find the index at which M(r = R) = Mshell, or phi(r = R) = 0 
        # If exists, then terminate the loop. Otherwise, repeat the loop
        # with new (a continuation of) sets of steps and start/end values.
        # =============================================================================
        massCondition = mShell_arr_cum >= mShell_end
        phiCondition = phiShell_arr <= 0
        idx_array = np.nonzero(( massCondition | phiCondition ))[0]
        # If there is none, then take as last index
        if len(idx_array) == 0:
            idx = len(rShell_arr) - 1
        else:
            idx = idx_array[0]
        # Associated condition
        # True if any part of the array is true
        is_allMassSwept = any(massCondition) 
        is_fullyIonised = any(massCondition)
        is_phiZero = any(phiCondition)
        
        # Store values into arrays representing profile in the ionised region. 
        mShell_arr_ion = np.concatenate(( mShell_arr_ion, mShell_arr[:idx-1]))
        mShell_arr_cum_ion = np.concatenate(( mShell_arr_cum_ion, mShell_arr_cum[:idx-1]))
        phiShell_arr_ion = np.concatenate(( phiShell_arr_ion, phiShell_arr[:idx-1]))
        tauShell_arr_ion = np.concatenate(( tauShell_arr_ion, tauShell_arr[:idx-1]))
        nShell_arr_ion = np.concatenate(( nShell_arr_ion, nShell_arr[:idx-1]))
        rShell_arr_ion = np.concatenate(( rShell_arr_ion, rShell_arr[:idx-1]))
        
        # Reinitialise values for next integration
        nShell0 = nShell_arr_ion[idx - 1]
        phi0 = phiShell_arr_ion[idx - 1]
        tau0_ion = tauShell_arr_ion[idx - 1]
        mShell0 = mShell_arr_ion[idx - 1]
        rShell_start = rShell_arr_ion[idx - 1]
        
        # Consider the shell dissolved if the followings occur:
        # 1. The density of shell is far lower than the density of ISm.
        # 2. The shell has expanded too far.
        if nShellInner < (0.001 * params.n_ISM) or\
            rShell_stop == (1.2 * params.stop_r * c.pc.cgs.value) or\
                (rShell_start - rShell_stop) > (10 * rShell_start):
                    is_shellDissolved = True
                    break
        
        # begin next iteration if shell is not all ionised and mshell is not all accounted for.
        # if either condition is not met, move on.
        
    # append the last few values that are otherwise missed in the while loop.
    mShell_arr_ion = np.append(( mShell_arr_ion, mShell_arr[-2]))
    mShell_arr_cum_ion = np.append(( mShell_arr_cum_ion, mShell_arr_cum[-2]))
    phiShell_arr_ion = np.append(( phiShell_arr_ion, phiShell_arr[-2]))
    tauShell_arr_ion = np.append(( tauShell_arr_ion, tauShell_arr[-2]))
    nShell_arr_ion = np.append(( nShell_arr_ion, nShell_arr[-2]))
    rShell_arr_ion = np.append(( rShell_arr_ion, rShell_arr[-2]))
        
    # =============================================================================
    # If shell hasn't dissolved, continue some computation to prepare for 
    # further evaulation.
    # =============================================================================
    if not is_shellDissolved:
    
        # =============================================================================
        # First, compute the gravitational potential for the ionised part of shell
        # =============================================================================
        grav_ion_rho = nShell_arr_ion * params.mu_n
        grav_ion_r = rShell_arr_ion
        # mass of the thin spherical shell
        grav_ion_m = grav_ion_rho * 4 * np.pi * grav_ion_r**2 * rShell_step
        # cumulative mass
        grav_ion_m_cum = np.cumsum(grav_ion_m)
        # gravitational potential
        grav_ion_phi = - 4 * np.pi * c.G.cgs.value * scipy.integrate.simps(grav_ion_r * grav_ion_rho, x = grav_ion_r)
        # gravitational potential force per unit mass
        grav_ion_force_m = c.G.cgs.value * grav_ion_m_cum / grav_ion_r**2
        
        # Now, modify the array so that it matches the potential file.
        # I am not entirely sure what this section does, but it was in the
        # old code. 
        potentialFile_internalLength = 10000
        skip_idx = max(
            int(
                len(grav_ion_r)/potentialFile_internalLength
                ), 1
            )        
        # modify array
        grav_ion_r = grav_ion_r[:-1:skip_idx]
        grav_ion_force_m = grav_ion_force_m[:-1:skip_idx]
        
        # For the whole shell; but for now we have only calculated the ionised part.
        grav_force_m = grav_ion_force_m
        grav_r = grav_ion_r
        
        # How much ionising radiation is absorbed by dust and how much by hydrogen?
        dr_ion_arr = rShell_arr_ion[1:] - rShell_arr_ion[:-1]
        # We do so by integrating using left Riemann sums 
        # dust term in dphi/dr
        phi_dust = np.sum(
                        - nShell_arr_ion[:-1] * sigma_dust * phiShell_arr_ion[:-1] * dr_ion_arr
                        )
        # recombination term in dphi/dr
        phi_hydrogen = np.sum(
                        - 4 * np.pi * rShell_arr_ion[:-1]**2 / Qi * params.alpha_B * nShell_arr_ion[:-1]**2 * dr_ion_arr
                        )
        
        # If there is no ionised shell (e.g., because the ionising radiation is too weak)
        if phi_dust + phi_hydrogen == 0.0:
            f_ionised_dust = 0.0
            f_ionised_hydrogen = 0.0
        # If there is, compute the fraction.
        else:
            f_ionised_dust = phi_dust / (phi_dust + phi_hydrogen)
            f_ionised_hydrogen  = phi_hydrogen / (phi_dust + phi_hydrogen)
            
        # Create arrays to store values
        # shell mass at r (neutral region)
        mShell_arr_neu = np.array([])
        # cumulative shell mass at r (neutral region)
        mShell_arr_cum_neu = np.array([])
        # tau at r (neutral region)
        tauShell_arr_neu = np.array([])
        # density at r (neutral region)
        nShell_arr_neu = np.array([])
        # r array of neutral region
        rShell_arr_neu = np.array([])
        # reinitialise
        rShell_start = rShell_arr_ion[-1]
        
        # =============================================================================
        # If the shell is not fully ionised, calculate structure of 
        # non-ionized (neutral) part
        # =============================================================================
        if not is_fullyIonised:
            
            # Pressure equilibrium dictates that there will be a temperature and density
            # discontinuity at boundary between ionised and neutral region.
            nShell0 = nShell0 * params.mu_n / params.mu_p * params.t_ion / params.t_neu
            # tau(r) at neutral shell region
            tau0_neu = tau0_ion
            
            # =============================================================================
            # Evaluate the remaining neutral shell until all masses are accounted for.
            # =============================================================================
            # Entering this loop means not all mShell has been accounted for. Thus
            # is_phiZero can either be True or False here.
            while not is_allMassSwept:
    
                # if tau is already 100, there is no point in integrating more.
                tau_max = 100
                # the maximum width of the neutral shell, assuming constant density.
                max_shellThickness = np.abs((tau_max - tau0_ion)/(nShell0 * sigma_dust))
                # the end range of integration 
                rShell_stop = np.min([ 1 * c.pc.cgs.value, max_shellThickness ]) + rShell_start
                # Step size
                rShell_step = np.max([
                    np.min([ 5e-5 * c.pc.cgs.value, max_shellThickness/1e3]),
                    max_shellThickness/1e6
                    ])
                # range of r values
                rShell_arr = np.arange(rShell_start, rShell_stop, rShell_step)   
                # Get arguments and parameters for integration:
                # neutral region
                is_ionised = False
                # initial values
                y0 = [nShell0, tau0_neu]
                # constants
                cons = [Ln, Qi,
                        sigma_dust, 
                        params.t_neu, 
                        params.alpha_B]
                # Run integration
                sol_ODE = scipy.integrate.odeint(get_shellODE.get_shellODE, y0, rShell_arr,
                                      args=(cons, f_cover, is_ionised),
                                      rtol=1e-3, hmin=1e-7)
                # solved for n(r) and tau(r)
                nShell_arr = sol_ODE[:,0]
                tauShell_arr = sol_ODE[:,1]
                        
                # mass of spherical shell. Volume given by V = 4 pi r**2 * thickness
                mShell_arr = np.empty_like(rShell_arr)
                mShell_arr[0] = mShell0
                # FIXME: mu_p or mu_n?
                mShell_arr[1:] = nShell_arr[1:] * params.mu_p * 4 * np.pi * rShell_arr[1:]**2 * rShell_step
                mShell_arr_cum = np.cumsum(mShell_arr)
                
                # =============================================================================
                # Again, find the index at which M(r = R) = Mshell.
                # If exists, then terminate the loop. Otherwise, repeat the loop
                # with new (a continuation of) set of steps and start/end values.
                # =============================================================================                
                massCondition = mShell_arr_cum >= mShell_end
                idx_array = np.nonzero(massCondition)[0]
                # If there is none, then take as last index
                if len(idx_array) == 0:
                    idx = len(rShell_arr) - 1
                else:
                    idx = idx_array[0]
                # Associated condition
                # True if any part of the array is true
                is_allMassSwept = any(massCondition) 
                
                # Store values into arrays representing profile in the ionised region. 
                mShell_arr_neu = np.concatenate(( mShell_arr_neu, mShell_arr[:idx-1]))
                mShell_arr_cum_neu = np.concatenate(( mShell_arr_cum_neu, mShell_arr_cum[:idx-1]))
                tauShell_arr_neu = np.concatenate(( tauShell_arr_neu, tauShell_arr[:idx-1]))
                nShell_arr_neu = np.concatenate(( nShell_arr_neu, nShell_arr[:idx-1]))
                rShell_arr_neu = np.concatenate(( rShell_arr_neu, rShell_arr[:idx-1]))
                
                # Reinitialise values for next integration
                nShell0 = nShell_arr_neu[idx - 1]
                tau0_neu = tauShell_arr_neu[idx - 1]
                mShell0 = mShell_arr_neu[idx - 1]
                rShell_start = rShell_arr_neu[idx - 1]
                
            # append the last few values that are otherwise missed in the while loop.
            mShell_arr_neu = np.append(( mShell_arr_neu, mShell_arr[-2]))
            mShell_arr_cum_neu = np.append(( mShell_arr_cum_neu, mShell_arr_cum[-2]))
            tauShell_arr_neu = np.append(( tauShell_arr_neu, tauShell_arr[-2]))
            nShell_arr_neu = np.append(( nShell_arr_neu, nShell_arr[-2]))
            rShell_arr_neu = np.append(( rShell_arr_neu, rShell_arr[-2]))
            
            # =============================================================================
            # Now, compute the gravitational potential for the neutral part of shell
            # =============================================================================
            # FIXME: mu_p or mu_n?
            grav_neu_rho = nShell_arr_neu * params.mu_p
            grav_neu_r = rShell_arr_neu
            # mass of the thin spherical shell
            grav_neu_m = grav_neu_rho * 4 * np.pi * grav_neu_r**2 * rShell_step
            # cumulative mass
            grav_neu_m_cum = np.cumsum(grav_neu_m) + grav_ion_m_cum[-1]
            # gravitational potential
            grav_neu_phi = - 4 * np.pi * c.G.cgs.value * scipy.integrate.simps(grav_neu_r * grav_neu_rho, x = grav_neu_r)
            grav_phi = grav_neu_phi + grav_ion_phi
            # gravitational potential force per unit mass
            grav_neu_force_m = c.G.cgs.value * grav_neu_m_cum / grav_neu_r**2
            
            # Now, modify the array so that it matches the potential file.
            # I am not entirely sure what this section does, but it was in the
            # old code. 
            potentialFile_internalLength = 10000
            skip_idx = max(
                int(
                    len(grav_neu_r)/potentialFile_internalLength
                    ), 1
                )        
            # modify array
            grav_neu_r = grav_neu_r[:-1:skip_idx]
            grav_neu_force_m = grav_neu_force_m[:-1:skip_idx]
            # concatenate to an array which represents the whole shell
            grav_force_m = np.concatenate([grav_force_m, grav_neu_force_m])
            grav_r = np.concatenate([grav_r, grav_neu_r])
            
            
        # FIXME: What is the purpose of this line in the old code?
        # os.environ["ShTh"] = str(dRs)
        
        # Compute some shell properties
        if is_fullyIonised:
            # What is the final thickness of the shell?
            shellThickness = rShell_arr_ion[-1]
            # What is tau and phi at the outer edge of the shell?
            tau_rEnd = tauShell_arr_ion[-1]
            phi_rEnd = phiShell_arr_ion[-1]
            # What is the maximum shell density?
            nShell_max = np.max(nShell_arr_ion)
            # The ratio tau_IR/kappa_IR  = \int rho dr
            # Integrating using left Riemann sums.
            # See https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf page 45 Eq 9
            tau_kappa_IR = params.mu_n * np.sum(nShell_arr_ion[:-1] * dr_ion_arr) 
            
        else:
            shellThickness = rShell_arr_neu[-1]
            tau_rEnd = tauShell_arr_neu[-1]
            phi_rEnd = 0
            nShell_max = np.max(nShell_arr_ion)
            dr_neu_arr = rShell_arr_neu[1:] - rShell_arr_neu[:-1]
            # FIXME: Shouldnt we use mu_p?
            tau_kappa_IR = params.mu_n * (np.sum(nShell_arr_neu[:-1] * dr_neu_arr) + np.sum(nShell_arr_ion[:-1] * dr_ion_arr))
            
        # fraction of absorbed ionizing and non-ionizing radiations:
        f_absorbed_ion = 1 - phi_rEnd
        f_absorbed_neu = 1 - np.exp( - tau_rEnd )
        # total absorption fraction, defined as luminosity weighted average of 
        # f_absorbed_ion and f_absorbed_neu.
        # See https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf page 47 Eq 22, 23, 24.
        f_absorbed = (f_absorbed_ion * Li + f_absorbed_neu * Ln)/(Li + Ln)
            
    elif is_shellDissolved:
        f_absorbed_ion = 1.0
        f_absorbed_neu = 0.0
        f_absorbed = (f_absorbed_ion * Li + f_absorbed_neu * Ln)/(Li + Ln)
        f_ionised_dust = np.nan
        is_fullyIonised = True
        shellThickness = np.nan
        nShell_max = params.n_ISM
        tau_kappa_IR = 0
        grav_r = np.nan
        grav_phi = np.nan
        grav_force_m = np.nan

    # write dictionary
    shell_dict = { 
        "f_absorbed_ion": f_absorbed_ion,
        "f_absorbed_neu": f_absorbed_neu,
        "f_absorbed": f_absorbed,
        "f_ionised_dust": f_ionised_dust,
        "is_fullyIonised": is_fullyIonised,
        "shellThickness": shellThickness,
        "nShellInner": nShellInner,
        "nShell_max": nShell_max,
        "tau_kappa_IR": tau_kappa_IR,
        "nShell0_cloud": nShell0_cloud,
        "grav_r": grav_r,
        "grav_phi": grav_phi,
        "grav_force_m": grav_force_m,
        }
    
    # =============================================================================
    # Define a class for parameters as the dictionary is rather large
    # =============================================================================
    class Dict2Class(object):
        # set object attribute
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
                
    shell = Dict2Class(shell_dict)
    # return object
    return shell













