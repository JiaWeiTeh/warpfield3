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
import time
#--
import src.warpfield.functions.operations as operations
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams
import src.warpfield.bubble_structure.bubble_structure as bubble_structure
import src.warpfield.shell_structure.shell_structure as shell_structure
import src.warpfield.cloud_properties.mass_profile as mass_profile
import src.warpfield.phase1_energy.energy_phase_ODEs as energy_phase_ODEs
import src.output_tools.terminal_prints as terminal_prints
from src.warpfield.functions.operations import find_nearest_lower, find_nearest_higher
from src.warpfield.cooling import net_coolingcurve
import src.warpfield.cooling.CIE.read_coolingcurve as CIE
import src.warpfield.cooling.non_CIE.read_cloudy as non_CIE

# get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()




def get_bubbleproperties(
        t_now,
        R2, 
        Qi, alpha, beta, delta,
        Lw, Eb, vw,
        ):
    
    # R2 is r0
    
    # old code: get_bubbleLuminosity
    
    # =============================================================================
    # Step 1: Get necessary parameters, such as 
    # =============================================================================
    
    # velocity at r ---> 0.
    v0 = 0.0 * u.km / u.s
    
        
    print('t_now', t_now)
    print('R2', R2)
    print('Qi', Qi.to(1/u.Myr))
    print('t_now', t_now)
    print('alpha', alpha)
    print('beta', beta)
    print('delta', delta)
    print('Lw', Lw.to(u.M_sun*u.pc**2/u.Myr**3))
    print('Eb', Eb.to(u.M_sun*u.pc**2/u.Myr**2))
    print('vw', vw)
        
    
    
    # TODO: make warpfield run this to see if there is actually need to call this twice both in 
    
    # initial radius of discontinuity [pc]
    R1 = (scipy.optimize.brentq(get_bubbleParams.get_r1, 
                               1e-3 * R2.to(u.cm).value, R2.to(u.cm).value, 
                               args=([Lw.to(u.erg/u.s).value, 
                                      Eb.to(u.erg).value, 
                                      vw.to(u.cm/u.s).value, 
                                      R2.to(u.cm).value
                                      ])) * u.cm)\
                                .to(u.pc)#back to pc
    
    # print(r_inner, R1)
    # 0.06669784054849992 pc
    
    # The bubble pressure [cgs - g/cm/s2, or dyn/cm2]
    press = get_bubbleParams.bubble_E2P(Eb, R2, R1)
    # print('Eb, R2, R1')
    # print(Eb, R2, R1)
    # 11858019.317814864 0.20207551764992493 0.14876975625376893
    # 11858019.317814864 pc2 solMass / yr2 0.20207551764992493 pc 0.2020754412666312 pc
    print('this is pressure')
    print(press.to(u.M_sun/u.pc/u.Myr**2))
    # 0.00024352944352531002 g / (cm s2) or 376360507.4418777 solMass / (Myr2 pc)
    # 'press': 380571798.5188472
    # sys.exit()
    
    # =============================================================================
    # Step 2: Calculate dMdt, the mass flux from the shell back into the hot region
    # =============================================================================
    
    # The mass flux from the shell back into the hot region (b, hot stellar wind)
    # if it isn't yet computed, set it via estimation from Equation 33 in Weaver+77.
    # Question: should this be mu_n instead?
 
    
    # First, guess dMdt
    # TODO: check what these variables actually should be
    dMdt_init = get_init_dMdt(R2, Lw, Eb, vw, t_now, press)
    print('dMdt_init', dMdt_init.to(u.M_sun/u.Myr))
    # dMdt_init 0.0845262588579248 solMass / yr
    # dMdt_init 84526.25885792478 solMass / Myr
    # dMdt 40416.890252523
    
    # This function evaluates dMdt guesses with the boundary conditions. 
    # Goes into loop and keeps evaluating until v0 (both estimated and true) agree.
    # This will yield a residual dMdt (which is nearly zero).
    # old code: compare_boundaryValues()
    # I presume Weaver meant: v -> 0 for R -> R1 (since in that chapter he talks about R1 being very small)
    # TODO: this shouldn't be 4e5. In the old code of bubble_structure2 line 310, T prime is set awayfrom Tgoal, because of dR2. 
    r_inner = R1
    # change dR2 < _dR2min: too.
    # T_goal = 3e4 * u.K
    T_goal = 453898 * u.K
    params = [t_now, T_goal, r_inner, R2, press, Qi, alpha, beta, delta, v0]
    
    
    # TODO: fix this
    # for now, use single value to check debug.
    #--------------
    # fix this
    rgoal = (warpfield_params.xi_Tb * R2).to(u.pc)
    
    # def calcr_Tb(l1,l2):
    # rTb=i.r_Tb
    
    # try:
    #     if len(l1)>2:
    #         l1=l1[l1!=0]
    #         l2=l2[l2!=0]
    #         if len(l1)>1:
    #             a=np.max(l1)
    #             b=np.min(l2)
    #             rTb=b-0.2*(b-a)
    #     else:
    #         rTb=i.r_Tb 
    # except:
    #     rTb=i.r_Tb 
    

        
    # if np.isnan(rTb):
    #     rTb=i.r_Tb
        
    # #print('calc_R_Tb=',rTb) 
    
    # return rTb
    # rgoal_f=aux.calcr_Tb(R1R2,R2pR2)

    # check whether R1 is smaller than the radius at which bubble temperature will be calculated
    # if (R2 * rgoal_f < R1):
    #     # for now, stop code. However, could also say that energy-driven phase is over if rgoal_f < R1/R2, for a reasonable rgoal_f (0.99 or so)
    #     # print "debug, rgoal_f", rgoal_f,
    #     sys.exit("rgoal_f is outside allowed range in bubble_structure.py (too small). Increase r_Tb in myconfig.py (<1.0)!")
    # else:
    #     rgoal = rgoal_f*R2
    #--------------




    # While evaluating for dMdt, we create a global variable to also store the values for
    # temperature and velocity. 
    global v_array, T_array, dTdr_array, r_array
    v_array = np.array([])
    T_array = np.array([])
    dTdr_array = np.array([])
    r_array = np.array([])
    dMdt = scipy.optimize.fsolve(get_velocity_residuals, dMdt_init.to(u.M_sun/u.yr), args = (params,), 
                                 full_output = 1, factor = 50, xtol = 1e-6)[0]

    print('done! dMdt (Msun/yr) and dMdt (Msun/Myr):', dMdt, dMdt * 1e6)
    # 0.0845262588579248 solMass / yr
    # dMdt 40416.890252523 [96559.42153809] - guess/real
    n_array = (press/((warpfield_params.mu_n/warpfield_params.mu_p) * c.k_B.cgs * T_array)).to(1/u.cm**3)
    print('v', v_array)
    print('T', T_array)
    print('dTdr', dTdr_array)
    print('r_array', r_array)
    print('n', n_array)

    # =============================================================================
    # In no_calc case. 
    # =============================================================================
     
      
    """
        # Test plot
        import matplotlib.pyplot as plt
        plt.plot(r_array, v_array)
        plt.xlabel('r (pc)')
        plt.ylabel('v (km/s)')
        plt.savefig('example_plots/first_bubblestruc/v.png')
        plt.clf()
        plt.plot(r_array, T_array)
        plt.xlabel('r (pc)')
        plt.ylabel('T (K)')
        plt.yscale('log')
        plt.savefig('example_plots/first_bubblestruc/T.png')
        plt.clf()
        plt.plot(r_array, -dTdr_array)
        plt.xlabel('r (pc)')
        plt.ylabel('- dTdr (K/pc)')
        plt.yscale('log')
        plt.savefig('example_plots/first_bubblestruc/dTdr.png')
        plt.clf()
        plt.plot(r_array, dens_array)
        plt.xlabel('r (pc)')
        plt.ylabel('n (1/cm3)')
        plt.yscale('log')
        plt.savefig('example_plots/first_bubblestruc/n.png')
        plt.clf()
        sys.exit()
    """
    
    # =============================================================================
    # Here we identify which index in the temperature array has cooling and which doesn't. 
    # The bubble will have these regions:
    #   1. Low resolution (bubble) region. This is the CIE region, where T > 10**5.5 K.
    #   2. High resolution (conduction) region. This is the non0-CIE region.
    #   3. Intermediate region. This is between 1e4 and T[index_cooling_switch].
    # -----
    # Goal: calculate power-loss (luminosity) in these regions due to cooling. To do this
    #       for each zone we calculate T, dTdr and n. 
    # Remember r is monotonically decreasing, so temperature increases!
    # 
    # =============================================================================

    # Temperature at which any lower will have no cooling
    _coolingswitch = 1e4 * u.K
    # Temperature of switching between CIE and non-CIE. 
    _CIEswitch = 10**5.5 * u.K

    #---------------- 0. Prep: insert entry at exactly _CIEswitch via interpolation
    
    # index of radius array at which T is closest (and higher) to _coolingswitch
    index_cooling_switch = operations.find_nearest_higher(T_array.value, _coolingswitch.value)
    # index of radius array at which T is closest (and higher) to _CIEswitch
    index_CIE_switch = operations.find_nearest_higher(T_array.value, _CIEswitch.value)
    # interpolate so that we have an entry at exactly _CIEswitch.
    
    
    if index_cooling_switch != index_CIE_switch:
        # array sliced from beginning until somewhere after _CIEswitch
        r_interpolation_bubble = r_array[:index_CIE_switch+20].value
        # interpolation function for T and dTdr.
        fdTdr_interp_bubble = interp1d(r_interpolation_bubble, dTdr_array[:index_CIE_switch+20].value, kind='linear')
        # subtract so that it is zero at _CIEswitch
        fT_interp_bubble = interp1d(r_interpolation_bubble, (T_array[:index_CIE_switch+20] - _CIEswitch).value, kind='cubic')
        
        # calculate quantities
        r_CIEswitch = scipy.optimize.brentq(fT_interp_bubble, np.min(r_interpolation_bubble), np.max(r_interpolation_bubble), xtol=1e-14) * u.pc
        n_CIEswitch = (press/((warpfield_params.mu_n/warpfield_params.mu_p) * c.k_B.cgs * _CIEswitch)).to(1/u.cm**3)
        dTdr_CIEswitch = fdTdr_interp_bubble(r_CIEswitch.value) * u.K / u.pc
        
        # insert into array
        T_array = np.insert(T_array, index_CIE_switch, _CIEswitch)
        r_array = np.insert(r_array, index_CIE_switch, r_CIEswitch)
        n_array = np.insert(n_array, index_CIE_switch, n_CIEswitch)
        dTdr_array = np.insert(dTdr_array, index_CIE_switch, dTdr_CIEswitch)

    #---------------- 1. Bubble. Low resolution region, T > 10**5.5 K. CIE is used. 
    
    # r is monotonically decreasing, so temperature increases
    T_bubble = T_array[index_CIE_switch:]
    r_bubble = r_array[index_CIE_switch:]
    n_bubble = n_array[index_CIE_switch:]
    dTdr_bubble = dTdr_array[index_CIE_switch:]
    # import values from two cooling curves
    # value T chosen here is not important, since we want the interpolation only
    _, _, _, cooling_CIE_interpolation = CIE.get_Lambda(T_bubble[1])
    # cooling rate
    Lambda_bubble = 10**(cooling_CIE_interpolation(np.log10(T_bubble.value))) * u.erg * u.cm**3 /u.s
    integrand_bubble = n_bubble**2 * Lambda_bubble * 4 * np.pi * r_bubble**2
    # get units right
    integrand_bubble = integrand_bubble.to(u.erg/u.cm/u.s)
    # calculate power loss due to cooling
    L_bubble = np.abs(np.trapz(integrand_bubble, x = r_bubble)).to(u.erg/u.s)
    # intermediate result for calculation of average temperature [K pc3]
    Tavg_bubble = np.abs(np.trapz(r_bubble**2 * T_bubble, x = r_bubble)).to(u.K * u.pc**3)
    
    # print('1st zone done')

    #---------------- 2. Conduction zone. High resolution region, 10**4 < T < 10**5.5 K. 
    
    # it is possible that index_cooling_switch = index_CIE_switch = 0 if the shock front is very steep.
    if index_cooling_switch != index_CIE_switch:
        # if this zone is not well resolved, solve ODE again with high resolution (IMPROVE BY ALWAYS INTERPOLATING)
        if index_CIE_switch - index_cooling_switch < 100:
            print('we are here')
            dx_conduction = (r_array[index_cooling_switch] - r_array[index_CIE_switch]).value/1e4
            r_conduction = np.arange(r_array[index_cooling_switch].value, np.max([r_array[index_CIE_switch].value - dx_conduction, dx_conduction]), -dx_conduction) * u.pc
            # rerun structure with greater precision
            psoln = scipy.integrate.odeint(get_bubble_ODE, [v_array[index_cooling_switch].value,T_array[index_cooling_switch].value,dTdr_array[index_cooling_switch].value], r_conduction.value,
                                           args = (params,), tfirst=True) # solve ODE again, there should be a better way (event finder!)
            # solutions
            T_conduction = psoln[:,1] * u.K
            dTdr_conduction = psoln[:,1] * u.K / u.pc
            # value at 1e4 K
            dTdR_coolingswitch = dTdr_conduction[0]

        else:
            r_conduction = r_array[:index_CIE_switch+1] 
            T_conduction = T_array[:index_CIE_switch+1]
            dTdr_conduction = dTdr_array[:index_CIE_switch+1]
            # value at 1e4 K
            dTdR_coolingswitch = dTdr_conduction[0]            
        
        # non-CIE is required here
        # import values from two cooling curves
        cooling_nonCIE, heating_nonCIE = non_CIE.get_coolingStructure(t_now)
        # calculate array
        n_conduction = (press/((warpfield_params.mu_n/warpfield_params.mu_p) * c.k_B.cgs * T_conduction)).to(1/u.cm**3)
        phi_conduction = (Qi / (4 * np.pi * r_conduction**2)).to(1/u.cm**2/u.s)
        # cooling rate
        cooling_conduction = 10 ** cooling_nonCIE.interp(np.transpose(np.log10([n_conduction.value, T_conduction.value, phi_conduction.value])))
        heating_conduction = 10 ** heating_nonCIE.interp(np.transpose(np.log10([n_conduction.value, T_conduction.value, phi_conduction.value])))
        dudt_conduction = (heating_conduction - cooling_conduction) * u.erg / u.cm**3 / u.s
        integrand_conduction = (dudt_conduction * 4 * np.pi * r_conduction**2) * u.erg / u.s / u.cm
        # calculate power loss due to cooling
        L_conduction = (np.abs(np.trapz(integrand_conduction, x = r_conduction))).to(u.erg/u.s)
        # intermediate result for calculation of average temperature
        Tavg_conduction = (np.abs(np.trapz(r_conduction**2 * T_conduction, x = r_conduction))).to(u.K*u.pc**3)
    # if there is no conduction; i.e., the shock front is very steep. 
    elif index_cooling_switch == 0 and index_CIE_switch == 0:
        # the power loss due to cooling in this region will simply be zero. 
        L_conduction = 0 * u.erg/ u.s   
        dTdR_coolingswitch = dTdr_bubble[0]
        
    # print(f'Second zone done. index_cooling_switch = {index_cooling_switch} and index_CIE_switch = {index_CIE_switch}.')
            
    #---------------- 3. Region between 1e4 K and T_array[index_cooling_switch]
    
    # If R2_prime is very close to R2 (i.e., where T ~ 1e4K), then this region is tiny (or non-existent)
    R2_coolingswitch = (_coolingswitch - T_array[index_cooling_switch])/dTdR_coolingswitch + r_array[index_cooling_switch]
    # assert R2_coolingswitch < r_array[index_cooling_switch], "Hmm? in region 3 of bubble_luminosity"
    # interpolate between R2_prime and R2_1e4, important because the cooling function varies a lot between 1e4 and 1e5K (R2_prime is above 1e4)
    fT_interp_intermediate = interp1d(np.array([r_array[index_cooling_switch].value, R2_coolingswitch.value]), 
                                      np.array([T_array[index_cooling_switch].value, _coolingswitch.value]), kind = 'linear')
    # get values
    r_intermediate = np.linspace(r_array[index_cooling_switch].value, R2_coolingswitch.value, num = 1000, endpoint=True) * u.pc
    T_intermediate = fT_interp_intermediate(r_intermediate) * u.K
    n_intermediate =  (press/((warpfield_params.mu_n/warpfield_params.mu_p) * c.k_B.cgs * T_intermediate)).to(1/u.cm**3)
    phi_intermediate = (Qi / (4 * np.pi * r_intermediate**2)).to(1/u.s/u.cm**2)
    # get cooling, taking into account for both CIE and non-CIE regimes
    # print(T_intermediate, _coolingswitch, (T_intermediate < _coolingswitch))
    regime_mask = {'non-CIE': T_intermediate < _CIEswitch, 'CIE': T_intermediate >= _CIEswitch}
    L_intermediate = {}
    for regime in ['non-CIE', 'CIE']:
        # masks
        mask = regime_mask[regime]
        
        if regime == 'non-CIE':
            # import values from cooling curves
            cooling_nonCIE, heating_nonCIE = non_CIE.get_coolingStructure(t_now)
            # cooling rate
            cooling_intermediate = 10 ** cooling_nonCIE.interp(np.transpose(np.log10([n_intermediate[mask].value, T_intermediate[mask].value, phi_intermediate[mask].value])))
            heating_intermediate = 10 ** heating_nonCIE.interp(np.transpose(np.log10([n_intermediate[mask].value, T_intermediate[mask].value, phi_intermediate[mask].value])))
            dudt_intermediate = (heating_intermediate - cooling_intermediate) * u.erg / u.s / u.cm**3
            integrand_intermediate = dudt_intermediate * 4 * np.pi * r_intermediate[mask]**2
        elif regime == 'CIE':
            # import values from cooling curves
            # value T chosen here is not important, since we want the interpolation only
            _, _, _, cooling_CIE_interpolation = CIE.get_Lambda(T_bubble[1])
            Lambda_intermediate = 10**(cooling_CIE_interpolation(np.log10(T_intermediate[mask].value))) * u.erg * u.cm**3 /u.s
            integrand_intermediate = n_intermediate[mask]**2 * Lambda_intermediate * 4 * np.pi * r_intermediate[mask]**2
        # calculate power loss due to cooling
        L_intermediate[regime] = (np.abs(np.trapz(integrand_intermediate, x = r_intermediate[mask]))).to(u.erg/u.s)
        
    # sum for both regions
    L_intermediate = L_intermediate['non-CIE'] + L_intermediate['CIE']
    # intermediate result for calculation of average temperature
    Tavg_intermediate =  (np.abs(np.trapz(r_intermediate**2 * T_intermediate,  x = r_intermediate))).to(u.K * u.pc**3)

    # print('Third zone done.')

    #---------------- 4. Finally, sum up across all regions. Calculate the average temeprature.
    # this was Lb in old code
    L_total = L_bubble + L_conduction + L_intermediate
    
    # calculate temperature
    # with conduction zone
    if index_cooling_switch != index_CIE_switch:
        Tabg = 3 * ( Tavg_bubble / (r_bubble[0]**3 - r_bubble[-1]**3) +\
                    Tavg_conduction / (r_conduction[0]**3 - r_conduction[-1]**3) +\
                    Tavg_intermediate / (r_intermediate[0]**3 - r_intermediate[-1]**3))
    # without conduction zone
    else:
        Tavg = 3. * ( Tavg_bubble / (r_bubble[0]**3 - r_bubble[-1]**3) +\
                     Tavg_intermediate / (r_intermediate[0]**3 - r_intermediate[-1]**3))

    # TODO: what is rgoal?
    # get temperature inside bubble at fixed scaled radius
    # temperature T at rgoal
    # print(rgoal)
    # print(r_array[index_cooling_switch])
    
    
    if rgoal > r_array[index_cooling_switch]: # assumes that r_cz runs from high to low values (so in fact I am looking for the highest element in r_cz)
        T_rgoal = fT_interp_intermediate(rgoal)
    elif rgoal > r_array[index_CIE_switch]: # assumes that r_cz runs from high to low values (so in fact I am looking for the smallest element in r_cz)
        idx = operations.find_nearest(r_conduction.value, rgoal.value)
        T_rgoal = T_conduction[idx] + dTdr_conduction[idx]*(rgoal - r_conduction[idx])
    else:
        idx = operations.find_nearest(r_bubble.value, rgoal.value)
        T_rgoal = T_bubble[idx] + dTdr_bubble[idx]*(rgoal - r_bubble[idx])
        
    # new factor for dMdt (used in next time step to get a good initial guess for dMdt)
    dMdt_factor_out = warpfield_params.dMdt_factor * dMdt / dMdt_init
    
    # print('Completed calculation of bubble luminosity.')
    
    return L_total, T_rgoal, L_bubble, L_conduction, L_intermediate, dMdt_factor_out, Tavg

    # TODO: original code also return these.
     # Mbubble, r_Phi, Phi_grav_r0b, f_grav


# =============================================================================
# Initial guess of dMdt
# =============================================================================

def get_init_dMdt(
        R2, Lw, Eb, vw,
        t_now, press
        ):
    
    # print('get_init_dMdt parameter', R2, Lw, Eb, vw, t_now, press)
    
    dMdt_init = 12 / 75 * warpfield_params.dMdt_factor**(5/2) * 4 * np.pi * R2**3 / t_now\
            * warpfield_params.mu_p / c.k_B.cgs * (t_now * warpfield_params.c_therm / R2**2)**(2/7) * press**(5/7)
       
     # dMdt_guess = 4. / 25. * dMdt_factor * 4. * np.pi * R2 ** 3. / t_now * 0.5 * (myc.mp / myc.Msun) / myc.kboltz_au * (t_now * myc.Cspitzer_au / R2 ** 2.) ** (2. / 7.) * press ** (5. / 7.)

    return dMdt_init.to(u.M_sun/u.yr)




def get_velocity_residuals(dMdt_init, params):
    """
    This routine calculates the value for dMdt, by comparing velocities at boundary condition.

    Parameters
    ----------
    dMdt_init [Msun/yr]: TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    T : TYPE
        DESCRIPTION.

    Old code: find_dMdt()

    """
    # unravel
    t_now, T_goal, r_inner, R2, pressure, Qi, alpha, beta, delta, v0 = params
    # units for dMdt_init! Because scipy removes them in ODE solvers
    # units in params are retained. 
    dMdt_init = dMdt_init*u.M_sun/u.yr
   
    # =============================================================================
    # Get initial bubble values for integration  
    # =============================================================================
    # Watch out! these are unitless.
    r2Prime, T_r2Prime, dTdr_r2Prime, v_r2Prime = get_bubble_ODE_initial_conditions(dMdt_init, params)
    # print(v_r2Prime, T_r2Prime, dTdr_r2Prime)
    # both in this run
    # [980.46174014] [453898.] [-6.21501862e+12]
    # [983.03480318] [30000.] [-3.65761471e+14]
    
    # [vR2_prime, TR2_prime, dTdrR2_prime] = y0
    # y0: [1003.9291826702926, 453898.8577997466, -1815595431198.9866] TR2_prime: 453898.8577997466
    
    # =============================================================================
    # radius array at which bubble structure is being evaluated.
    # =============================================================================
    
    # number of points (in the range of 1e5)
    # dx = (r2Prime - r_inner) / 1e6
    
    # # adjust minimum value
    # rmin = np.max([dx, r_inner])
    
    # array is monotonically decreasing, and sampled at higher density at larger radius
    # i.e., more datapoints near bubble's outer edge (sort of a reverse logspace).
    # these values are all in pc.
    global r_array
    # print(r2Prime)
    # print(r_inner)
    # print(np.logspace(np.log10(r_inner.to(u.pc).value), np.log10(r2Prime[0]), int(1e5)))
    # sys.exit()
    r_array = (r2Prime + r_inner.to(u.pc).value) -  np.logspace(np.log10(r_inner.to(u.pc).value), np.log10(r2Prime[0]), int(1e5))
    r_array = r_array * u.pc
    # print('r', r_array)
    # sys.exit()
    
    
    # check. For some reason get_bubble_ODE_initial_conditions() outputs an array. 
    if not (len(v_r2Prime) == 1 and len(T_r2Prime) == 1 and len(dTdr_r2Prime) == 1):
        sys.exit('Something is not right in dMdt.')

    # tfirst = True because get_bubble_ODE() is defined as f(t, y). 
    psoln = scipy.integrate.odeint(get_bubble_ODE, [v_r2Prime[0], T_r2Prime[0], dTdr_r2Prime[0]], r_array.value, 
                                   args=(params,), tfirst=True)
    
    # record arrays  
    global v_array, T_array, dTdr_array
    v_array = psoln[:, 0] * u.km/u.s
    T_array = psoln[:, 1] * u.K
    dTdr_array = psoln[:, 2] * u.K/u.pc
        
    # V0 is the velocity at r -> 0.
    residual = (v0.to(u.km/u.s).value - v_array[-1].value)/v_array[0].value
    
    return residual
    
    
    
    
def get_bubble_ODE_initial_conditions(dMdt, params):
    """
    dMdt_init (see above) can be used as an initial estimate of dMdt, 
    which is then adjusted until the velocity found by numerical integration (see below compare_bv) 
    remains positive and less than alpha*r/t at some chosen small radius. 
    
    For each value of dMdt, the integration of equations (42) and (43) - in get_bubbleODEs() - 
    can be initiated at a <<radius r>> slightly less than R2 by using these
    three relations for:
        T, dTdr, and v. 
        
    old code: r2_prime is R2_prime in old code.
    old code: get_start_bstruc

    Parameters. This operation assumes full cgs units. 
    ----------
    dMdt : float
        mass loss from region c (shell) into region b (shocked winds) due to thermal conduction.
    params contains these following parameters :
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
    r2_prime [pc]: float
        the small radius (slightly smaller than R2) at which these values are evaluated.
    T [K]: ODE
        T(r).
    dTdr [K/pc]: ODE
        T(r).
    v [km/s]: ODE
        T(r).
    """
    
    
    # Everything here has to be in the right units. However, scipy automatically strips unit
    # in their ODE solver loop. This means we need to make sure that the units is right manually. 
    
    # unravel
    t_now, T_goal, _, R2, pressure, _, alpha, _, _, _ = params
    # Unit checking
    t_now = t_now.to(u.s).value
    T_goal = T_goal.to(u.K).value
    R2 = R2.to(u.cm).value
    pressure = pressure.to(u.g/u.cm/u.s**2).value
    dMdt = dMdt.to(u.g/u.s).value
    
    # Important question: what is mu?
    # here we follow the original code and use mu_p, but maybe we should use mu_n since the region is ionised?
    mu = warpfield_params.mu_p.value
    
    # -----
    # r has to be calculated, via a temperature goal (usually 3e4 K). 
    # dR2 = (R2 - r), in Equation 44
    # -----
    # old code: r is R2_prime, i.e., radius slightly smaller than R2. 
    # TODO: For very strong winds (clusters above 1e7 Msol), this number heeds to be higher!
    dR2 = T_goal**(5/2) / (25/4 * c.k_B.cgs.value / mu / warpfield_params.c_therm.value * dMdt / (4 * np.pi * R2**2) )
    
    # Possibility: extremely small dR2. I.e., at mCluster~1e7, dR2 ~1e-11pc. 
    # What is the minimum dR2? Set a number here
    _dR2min = 1e-7 * u.pc.to(u.cm)
    if dR2 < _dR2min:
        dR2 = _dR2min * np.sign(dR2)
    
    # -----
    # Now, write out the estimation equations for initial conditions for the ODE (Eq 42/43)
    # -----
    # Question: I think mu here should point to ionised region
    # T(r)
    T = (25/4 * c.k_B.cgs.value / mu / warpfield_params.c_therm.value * dMdt / (4 * np.pi * R2**2) )**(2/5) * dR2**(2/5) 
    # v(r)
    v = (alpha * R2 / t_now - dMdt / (4 * np.pi * R2**2) * c.k_B.cgs.value * T / mu / pressure) * u.cm/u.s
    # T'(r)
    dTdr = (- 2 / 5 * T / dR2) * u.K / u.cm
    # Finally, calculate r for future use
    r2_prime = (R2 - dR2) * u.cm
    # return values without units.
    return r2_prime.to(u.pc).value, T, dTdr.to(u.K/u.pc).value, v.to(u.km/u.s).value
    


    
def get_bubble_ODE(r_arr, initial_ODEs, params):
        
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
    
    This whole section should run in cgs.
    """
    
    # unravel
    # t is t_now
    t, _, _, _, pressure, Qi, alpha, beta, delta, _ = params
    # these are v_r2Prime, T_r2Prime, dTdr_r2Prime.
    v, T, dTdr = initial_ODEs
    # semi-correct cooling at low T
    if T < 10.**3.61:
        T = 10.**3.61
    # make sure the units are right. They are unitless as input, because the ODE
    # solver does not work well with units. 
    v *= (u.km/u.s)
    dTdr *= (u.K/u.pc)
    r_arr *= u.pc
    # print('rarr', r_arr)
    
    # Test. 
    # T = 453898
    T *= u.K
    
    
    # print('pressure')
    # print(pressure)
    # print(pressure.to(u.M_sun/u.pc/u.Myr**2))
    # get density and ionising flux
    ndens = pressure / (2 * c.k_B.cgs * T)
    phi = Qi / (4 * np.pi * r_arr**2)
    
    
    # press: 380571798.5188472
    # 130336076362671.16
    
    # print(t, ndens.to(1/u.cm**3), T, phi.to(1/u.s/u.cm**2))
    # 0.00012057636642393612 Myr 5.708625650317196e+61 1 / pc3 453898.0 K 3.3118700666086394e+67 1 / (Myr pc2)
    # 0.00012057636642393612 Myr 1943031.8917922417 1 / cm3 453898.0 K 1.1022198611235203e+17 1 / (cm2 s)
    # sys.exit()
    
    
    # 0.00012057636642393612 Myr
    # 1.5733672637852578e+25 1 / cm3, 30000.000000000015 K, 1.1022198607656154e+17 1 / (cm2 s)

    
    # 0.00012057636642393612
    # 1972534.1634600703 453898.8577997466 1.104236441405574e+17
    
    # net cooling rate
    dudt = net_coolingcurve.get_dudt(t, ndens.to(1/u.cm**3), T, phi.to(1/u.s/u.cm**2))
    # print(t, ndens.to(1/u.cm**3), T, phi.to(1/u.s/u.cm**2), dudt)
    
    # dudt = nnLambda
    # dudt = 1 * u.erg * u.cm**3 / u.s / u.cm**6
    
    # old code: dTdrd
    dTdrr = pressure/(warpfield_params.c_therm * T**(5/2)) * (
        (beta + 2.5 * delta) / t   +   2.5 * (v - alpha * r_arr / t) * dTdr / T - dudt/pressure
        ) - 2.5 * dTdr**2 / T - 2 * dTdr / r_arr
    
    # old code: vd
    dvdr = (beta + delta) / t + (v - alpha * r_arr / t) * dTdr / T - 2 *  v / r_arr
    
    
    # make sure they are in correct units
    dvdr = (dvdr.to(u.km/u.s/u.pc)).value
    dTdr = (dTdr.to(u.K/u.pc)).value
    dTdrr = (dTdrr.to(u.K/u.pc**2)).value
    
    
    return dvdr, dTdr, dTdrr
    


# dMdt = scipy.optimize.fsolve




# #%%

# import numpy as np

# def get_r_list(Rhi, Rlo, dx0, n_extra=0):
#     """
#     get list or r where bubble strutcture will be calculated
#     result is monotonically decreasing
#     :param Rhi: upper limit for r (1st entry in r)
#     :param Rlo: lower entry in r (usually last entry)
#     :return: array (1D)
#     """
#     # figure out at which postions to calculate solution
#     top = Rhi
#     bot = np.max([Rlo, dx0])

#     clog = 2.  # max increase in dx (in log) that is allowed, e.g. 2.0 -> dxmax = 100.*dx0 (clog = 2.0 seems to be a good compromise between speed an precision)
#     dxmean = (10. ** clog - 1.) / (clog * np.log(10.)) * dx0  # average step size
#     # print "Ndx", dR2, R2_prime, TR2_prime, top, bot, dxmean, dMdt
#     Ndx = int((top - bot) / dxmean)  # number of steps with chosen step size
#     dxlist = np.logspace(0., clog, num=Ndx) * dx0  # list of step sizes
#     r = top + dx0 - np.cumsum(dxlist)  # position vector on which ODE will be solved
#     r = r[r > bot]  # for safety reasons
#     r = np.append(r, bot)  # make sure the last element is exactly the bottom entry

#     # extend the radius a bit further to small values, i.e. calculate a bit more than necessary
#     # if n_extra has been set to 0, no extra points will be used
#     #r_extra = np.linspace(r[-1] - dxmean, 0.95 * r[-1], num=n_extra)
#     #r = np.append(r, r_extra)

    
#     return r, top, bot, dxlist


# def get_r(top, bot, num):
    
    
#     r = (top+bot) -  np.logspace(np.log10(bot), np.log10(top), num)
    
    
#     return r


    
# def my_r(top, bot, dx, log_maxspace = 2):
    
    
#     # An array with an initial separation, and also with a maximum separation. 
    
#     bot = np.max([bot, dx])
    
#     r = (top+bot) -  np.logspace(np.log10(bot), np.log10(top), num)
    
#     return r




# import matplotlib.pyplot as plt

# top = 1000
# bot = 1
# dx = 1




# plt.plot(get_r_list(top, bot, dx)[0], label = 'daniel')
# plt.plot(np.logspace(np.log10(top), np.log10(bot), num = len(get_r_list(top, bot, dx)[0])), label = 'normal')
# plt.plot(get_r(top, bot, len(get_r_list(top, bot, dx)[0])), label = 'mine', linestyle = '--')
# # plt.yscale('log')
# plt.legend()




