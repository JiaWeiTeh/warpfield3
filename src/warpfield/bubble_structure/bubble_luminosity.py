#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:43:02 2023

@author: Jia Wei Teh
"""




# libraries
import numpy as np
import os
import sys
import scipy.optimize
import scipy.integrate
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as c
from astropy.table import Table
#--
import src.warpfield.cooling.get_coolingFunction as get_coolingFunction
import src.warpfield.functions.operations as operations
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams
import src.warpfield.bubble_structure.bubble_structure as bubble_structure
import src.warpfield.shell_structure.shell_structure as shell_structure
import src.warpfield.cloud_properties.mass_profile as mass_profile
import src.warpfield.phase1_energy.energy_phase_ODEs as energy_phase_ODEs
import src.warpfield.functions.terminal_prints as terminal_prints
from src.warpfield.functions.operations import find_nearest_lower, find_nearest_higher
import src.warpfield.cooling.full_coolingcurve as full_coolingcurve


# get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()








def get_bubbleproperties(
        R2, Lw, Eb, vw,
        t_now
        
        ):
    
    
    
    
    
    
    # old code: get_bubbleLuminosity
    
    
    
    # =============================================================================
    # Step 1: Get necessary parameters, such as
    # =============================================================================
    
    # initial radius of discontinuity [pc]
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 
                               1e-3 * R2 * u.pc.to(u.cm), R2 * u.pc.to(u.cm), 
                               args=([Lw, 
                                      Eb, 
                                      vw * u.km.to(u.cm), 
                                      R2 * u.pc.to(u.cm)
                                      ])) * u.cm.to(u.pc) #back to pc
    
    # The bubble pressure [cgs - g/cm/s2, or dyn/cm2]
    press = get_bubbleParams.bubble_E2P(Eb, R2 * u.pc.to(u.cm) , R1 * u.pc.to(u.cm))
    
    # =============================================================================
    # Step 2: Calculate dMdt, the mass flux from the shell back into the hot region
    # =============================================================================
    
    # The mass flux from the shell back into the hot region (b, hot stellar wind)
    # if it isn't yet computed, set it via estimation from Equation 33 in Weaver+77.
    # Question: should this be mu_n instead?

            
    # Question: why is this in the old version? Why are the equation and values weird?
    # if dMdt_guess == 0:
    #     dMdt_guess = 4. / 25. * dMdt_factor * 4. * np.pi * R2 ** 3. / t_now\
    #         * 0.5 * c.m_p.cgs.value * u.g.to(u.Msun) / k_B * (t_now * c_therm / R2 ** 2.) ** (2. / 7.) * press ** (5. / 7.)

    
    # bubble_params = {"v0": v0, "cons": cons, "rgoal": r_goal,
    #           "Tgoal": TR2_prime, "R2": R2, "R_small": R_small,
    #           "press": press, "Cool_Struc": cool_struc, "path": path2bubble}

    
    # prepare wrapper (to skip 2 superflous calls in fsolve)
    # bubble_params["dMdtx0"] = dMdt_guess
    # bubble_params["dMdty0"] = compare_boundaryValues(dMdt_guess, bubble_params, warpfield_params)


    # 1. < factor_fsolve < 100.; if factor_fsolve is chose large, the rootfinder usually finds the solution faster
    # however, the rootfinder may then also enter a regime where the ODE soultion becomes unphysical
    # low factor_fsolve: slow but robust, high factor_fsolve: fast but less robust
    # factor_fsolve = 50. #50
    
            
    # compute the mass loss rate to find out how much of that is loss 
    # from shell into the shocked region.
    # dMdt = get_dMdt(dMdt_guess, bubble_params, warpfield_params, factor_fsolve = factor_fsolve, xtol = 1e-3)
    
    
# def get_dMdt(dMdt_guess, bubble_params, warpfield_params, factor_fsolve = 50., xtol = 1e-6):
    # """
    # This function employs root finder to get correct dMdt, i.e., the 
    # mass loss rate dM/dt from shell into shocked region.

    # Parameters
    # ----------
    # dMdt_guess : float
    #     initial guess for dMdt.
    # bubble_params : dict
    #     A temporary dictionary made to store necessary information of the bubble.
    #     This is defined in bubble_structure.bubble_structure()
    #     includes: [v0 = v[Rsmall], cons = [a,b,c,d,e, Phi], R2_prime, R2, Rsmall, dR2, press
    #         Rsmall: some very small radius (nearly 0)
    #         R2_prime: radius very slightly smaller than shell radius R2
    #         R2: shell radius
    #         dR2: R2 - R2_prime
    #         press: pressure inside bubble
    # warpfield_params : object
    #     Object containing WARPFIELD parameters.
    # factor_fsolve : float, optional
    #     scipy.optimize.fsolve parameter. The default is 50..
    # xtol : float, optional
    #     scipy.optimize.fsolve parameter. The default is 1e-6.

    # Returns
    # -------
    # dMdt : float
    #     mass loss rate dM/dt from shell into shocked region.

    # """
    
    # # Note
    # # old code: find_dMdt()

    # # retrieve data
    # countl = float(os.environ["COUNT"])
    # dmdt_0l = float(os.environ["DMDT"])
    # # solve for dMdt
    # dMdt = scipy.optimize.fsolve(compare_boundaryValues_wrapper, dMdt_guess, args=(bubble_params, warpfield_params), 
    #                              factor = factor_fsolve, xtol = xtol, epsfcn = 0.1 * xtol)
    # if dMdt < 0:
    #     print('rootfinder of dMdt gives unphysical result...trying to solve again with smaller step size')
    #     dMdt = scipy.optimize.fsolve(compare_boundaryValues_wrapper, dMdt_guess, args=(bubble_params, warpfield_params), 
    #                                  factor=15, xtol=xtol, epsfcn=0.1*xtol)
    #     if dMdt < 0:
    #         #if its unphysical again, take last dmdt and change it slightly for next timestep
    #         dMdt = dmdt_0l+ dmdt_0l*1e-3 
    #         # count how often you did this crude approximation
    #         countl += 1 
    #         if countl >3:
    #             sys.exit("Unable to find correct dMdt, have to abort WARPFIELD")
    # # dmdt
    # try:
    #     os.environ["DMDT"] = str(dMdt[0])
    # except:
    #     os.environ["DMDT"] = str(dMdt)
    # # counter for fsolve
    # os.environ["COUNT"] = str(countl)
    # # return
    # return dMdt


# def compare_boundaryValues(dMdt, bubble_params, warpfield_params):
#     """
#     This function compares boundary value calculated from dMdt guesses with 
#     true boundary conditions. This routine is repeatedly called with different
#     dMdt intil the true v0 and estimated v0 from this dMdt agree.
#     Finally, this yields a residual dMdt, which is nearly zero, and that 
#     is what we are looking for.

#     Parameters
#     ----------
#     dMdt : float
#         Guess for mass loss rate.
#     bubble_params : dict
#         A temporary dictionary made to store necessary information of the bubble.
#         This is defined in bubble_structure.bubble_structure()
#         includes: [v0 = v[Rsmall], cons = [a,b,c,d,e, Phi], R2_prime, R2, Rsmall, dR2, press
#             Rsmall: some very small radius (nearly 0)
#             R2_prime: radius very slightly smaller than shell radius R2
#             R2: shell radius
#             dR2: R2 - R2_prime
#             press: pressure inside bubble
#     warpfield_params : object
#         Object containing WARPFIELD parameters.

#     Returns
#     -------
#     residual : float
#         residual of true v(Rsmall)=v0 (usually 0) and estimated v(Rsmall).

#     """
    
#     # Notes:
#     # old code: comp_bv_au()
#     # bubble_params is defined in calc_Lb()
    
#     # if dMdt is given as a length one list
#     if hasattr(dMdt, "__len__"): 
#         dMdt = dMdt[0]
    
#     # get initial values
#     R2_prime, y0 = get_start_bstruc(dMdt, bubble_params, warpfield_params)
#     # unravel
#     [vR2_prime, TR2_prime, dTdrR2_prime] = y0
#     # define data structure to feed into ODE
#     Data_Struc = {"cons" : bubble_params["cons"], "Cool_Struc" : bubble_params["Cool_Struc"]}

#     # figure out at which postions to calculate solution
#     # number of extra points
#     n_extra = 0 
#     # some initial step size in pc
#     dx0 = (R2_prime - bubble_params["R_small"]) / 1e6  
#     # An array of r
#     r, _, _, _ = get_r_list(R2_prime, bubble_params["R_small"], dx0, n_extra=n_extra)
    
#     # print('comp values', vR2_prime, TR2_prime, dTdrR2_prime, r, warpfield_params.metallicity)
#     # try to solve the ODE (might not have a solution)
#     try:
#         psoln = scipy.integrate.odeint(get_bubbleODEs, y0, r, args=(Data_Struc, warpfield_params.metallicity), tfirst=True)
#         # get
#         v = psoln[:, 0]
#         T = psoln[:, 1]
    
#         # this are the calculated boundary value (velocity at r=R_small)
#         v_bot = v[-(n_extra+1)]
#         # compare these to correct calues!
#         residual = (bubble_params["v0"] - v_bot)/v[0]
    
#         # very low temperatures are not allowed! 
#         # This check is also necessary to prohibit rare fast (and unphysical) oscillations in the temperature profile
#         min_T = np.min(T)
#         if min_T < 3e3:
#             residual *= (3e4/min_T)**2
#     # should the ODE solver fail
#     except:
#         # this is the case when the ODE has no solution with chosen inital values
#         print("Giving a wrong residual here; unable to solve the ODE. Suggest to set xi_Tb to default value of 0.9.")
#         if dTdrR2_prime < 0.:
#             residual = -1e30
#         else:
#             residual = 1e30

#     return residual










 



    return











def get_dMdt(
        R2, Lw, Eb, vw,
        t_now, press
        ):
    
    """
    This routine calculates the value for dMdt. 
    """
    
    
    dMdt_init = 12 / 75 * warpfield_params.dMdt_factor**(5/2) * 4 * np.pi * R2**3 / t_now\
            * warpfield_params.mu_p / c.k_B.cgs.value * (t_now * warpfield_params.c_therm / R2**2)**(2/7) * press**(5/7)
    
    
    
    
    
    
    
    
    
    
    
        
    return
    
    
    
    
    
    
    
    
    
def get_bubble_ODE_initial_conditions(dMdt, pressure, alpha,
                                  R2, T_goal, t_now
                                  ):
    """
    dMdt_init (see above) can be used as an initial estimate of dMdt, 
    which is then adjusted until the velocity found by numerical integration (see below compare_bv) 
    remains positive and less than alpha*r/t at some chosen small radius. 
    
    For each value of dMdt, the integration of equations (42) and (43) - in get_bubbleODEs() - 
    can be initiated at a <<radius r>> slightly less than R2 by using these
    three relations for:
        T, dTdr, and v. 
        
    old code: r is R2_prime in old code.

    Parameters (in cgs units unless specified)
    ----------
    dMdt : float
        mass loss from region c (shell) into region b (shocked winds) due to thermal conduction.
    pressure : float
        bubble pressure.
    alpha : float
        dlnR2/dlnt, see Eq 39.
    R2 : float
        Radius of cold shell, which, when the shell is thin, is the outer shock or the interface with the hot region (b). 
    T_goal : float
        temperature at r where we have set to T_goal = 3e4K.
    t_now : float
        time.

    Returns (in cgs units)
    -------
    r : float
        the small radius (slightly smaller than R2) at which these values are evaluated.
    T : ODE
        T(r).
    dTdr : ODE
        T(r).
    v : ODE
        T(r).
    """
    
    
    # Important question: what is mu?
    # here we follow the original code and use mu_p, but maybe we should use mu_n since the region is ionised?
    mu = warpfield_params.mu_p
    
    # -----
    # r has to be calculated, via a temperature goal (usually 3e4 K). 
    # dR2 = (R2 - r), in Equation 44
    # -----
    # old code: r is R2_prime, i.e., radius slightly smaller than R2. 
    # TODO: For very strong winds (clusters above 1e7 Msol), this number heeds to be higher!
    dR2 = T_goal**(5/2) / (25/4 * c.k_B.cgs.value / mu / warpfield_params.c_therm * dMdt / (4 * np.pi * R2**2) )
    
    # -----
    # Now, write out the estimation equations for initial conditions for the ODE (Eq 42/43)
    # -----
    # Question: I think mu here should point to ionised region
    # T(r)
    T = (25/4 * c.k_B.cgs.value / mu / warpfield_params.c_therm * dMdt / (4 * np.pi * R2**2) )**(2/5) * dR2**(2/5)
    # v(r)
    v = alpha * R2 / t_now - dMdt / (4 * np.pi * R2**2) * c.k_B.cgs.value * T / mu / pressure
    # T'(r)
    dTdr = - 2 / 5 * T - dR2
    # Finally, calculate r for future use
    r = R2 - dR2
    # return values
    return r, T, dTdr, v
    
    

# =============================================================================
# Step1: set up the ODEs
# =============================================================================




    
def get_bubble_ODE(t, 
                   alpha, beta, delta, 
                   r, T, dTdr, v, 
                   pressure, C,
                   # Qi is from SB99 interp
                   Qi
                   ):
    
    
    """
    ignoring cooling when t < tcool and assuming immediate loss of all
    energy for t ≥ tcool is a major simplification.
    Thus, we couple the energy loss term due to cooling Lcool
    to the energy equation.


    where U is the internal energy density and where the integration
    runs from the inner shock at a radius R1 to the outer radius of the
    bubble R2 , which is also the radius of the thin shell. The rate of
    change of the radiative component of the internal energy density 
    
    old code: calc_cons() and get_bubble_ODEs() aka bubble_struct()
    """
    #
    



    # TODO: make sure the units are right
    
    # semi-correct cooling at low T
    # if T < 10.**3.61:
    #     T = 10.**3.61

    # get density and ionising flux
    ndens = pressure / (2 * c.k_B.cgs.value * T)
    phi = Qi / (4 * np.pi * r**2)
    
    # net cooling rate
    dudt = full_coolingcurve.get_dudt(t, ndens, T, phi)
    
    # old code: dTdrd
    dTdrr = pressure/(warpfield_params.c_therm * T**(5/2)) * (
        (beta + 2.5 * delta) / t   +   2.5 * (v - alpha * r / t) * dTdr / T - dudt / pressure
        ) - 2.5 * dTdr**2 / T - 2 * dTdr / r
    
    # old code: vd
    dvdr = (beta + delta) / t + (v - alpha * r / t) * dTdr / T - 2 *  v / r
    
    
    return [dvdr, dTdr, dTdrr]
    














#%%




def get_xi_Tb(l1, l2, warpfield_params):
    """
    This function extracts the relative radius xi = r/R2 at which to measure 
    the bubble temperature. Unless the bubble structure file already has an 
    input, it will assume the default value given in the initial .param file.

    Parameters
    ----------
    l1 : ratio
        R1R2.
    l2 : ratio
        R2pR2.

    """
    
    xi_Tb = warpfield_params.xi_Tb
    # check if there are any existing values in the bubble file.
    try:
        if len(l1) > 2:
            l1 = l1[l1!=0]
            l2 = l2[l2!=0]
            if len(l1) > 1:
                a = np.max(l1)
                b = np.min(l2)
                xi_Tb = b - 0.2 * (b - a)
                if np.isnan(xi_Tb):
                    xi_Tb = warpfield_params.xi_Tb
    except:
        pass
    # return
    return xi_Tb