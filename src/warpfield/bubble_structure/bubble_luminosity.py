#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:43:02 2023

@author: Jia Wei Teh
"""




# libraries
import numpy as np
import sys
import os
import scipy.optimize
import scipy.integrate
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as c
#--
import src.warpfield.functions.operations as operations
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams
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
        dMdt_factor
        ):
    """
    Used in run_energy_phase and ____.

    Parameters
    ----------
    t_now : TYPE
        DESCRIPTION.
    R2 : TYPE
        DESCRIPTION.
    Qi : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    delta : TYPE
        DESCRIPTION.
    Lw : TYPE
        DESCRIPTION.
    Eb : TYPE
        DESCRIPTION.
    vw : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Yields
    ------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    
    # old code: calc_Lb(); i.e., get_bubbleLuminosity
    
    # ---- remember to comment these out and paste these two back in Step 2:
    # rgoal = (warpfield_params.xi_Tb * R2).to(u.pc)
    # T_goal = 306686 * u.K
    # T_goal = 3e4 * u.K

    
    print('Here in get_bubbleproperties-----')
    # print(f't_now: {t_now}')
    # print(f'R2: {R2}')
    # print(f'Qi: {Qi.to(1/u.Myr)}')
    # print(f'alpha: {alpha}')
    # print(f'beta: {beta}')
    # print(f'delta: {delta}')
    # print(f'Lw: {Lw.to(u.M_sun*u.pc**2/u.Myr**3)}')
    # print(f'Eb: {Eb.to(u.M_sun*u.pc**2/u.Myr**2)}')
    # print(f'vw: {vw}')
    # # print(f'rgoal: {rgoal}')
    # # print(f'T_goal: {T_goal}')
    # print(f'dMdt_factor: {dMdt_factor}')
    
    
    
    # Here is thje debug. Checking with the same condition if it yields same result.
    # ---------------------------------------
    t_now = 0.003002 * u.Myr
    alpha = 0.573
    beta = 0.83
    delta = -0.17851
    Qi = 1.6993e65 / u.Myr
    Eb = 2313300.0 * u.M_sun*u.pc**2/u.Myr**2
    R2 = 0.5553 * u.pc
    Lw = 2016500000.0 * u.M_sun*u.pc**2/u.Myr**3
    vw = 3810.2 * u.km/u.s
    dMdt_factor = 4.1858
    # ---------------------------------------
    
    # R2 is r0
    # =============================================================================
    # Step 1: Get necessary parameters, such as 
    # =============================================================================
    
    # velocity at r ---> 0.
    v0 = 0.0 * u.km / u.s
    
    # TODO: make warpfield run this to see if there is actually need to call this twice both in 
    
    # initial radius of discontinuity [pc] (inner bubble radius)
    
    # print('calculation of R1')
    # print('Lw', Lw)
    # print('Eb', Lw)
    # print(f'Lw: {Lw.to(u.M_sun*u.pc**2/u.Myr**3)}')
    # print(f'Eb: {Eb.to(u.M_sun*u.pc**2/u.Myr**2)}')
    # print(f'vw: {vw}')
    # print(f'R2: {R2}')

    R1 = (scipy.optimize.brentq(get_bubbleParams.get_r1, 
                               1e-3 * R2.to(u.cm).value, R2.to(u.cm).value, 
                               args=([Lw.to(u.erg/u.s).value, 
                                      Eb.to(u.erg).value, 
                                      vw.to(u.cm/u.s).value, 
                                      R2.to(u.cm).value
                                      ])) * u.cm)\
                                .to(u.pc)#back to pc
    
    # The bubble pressure [cgs - g/cm/s2, or dyn/cm2]
    press = get_bubbleParams.bubble_E2P(Eb, R2, R1)
    
    # =============================================================================
    # Step 2: Calculate dMdt, the mass flux from the shell back into the hot region
    # =============================================================================
    
    # ----------- prepare for calculation of dMdt
    
    # The mass flux from the shell back into the hot region (b, hot stellar wind)
    # if it isn't yet computed, set it via estimation from Equation 33 in Weaver+77.
    # Question: should this be mu_n instead?
 
    # First, guess dMdt
    # TODO: check what these variables actually should be
    # print('\n\n\n----- dMdt_init first ever guess -----')
    # print('params in get_init_dMdt()')
    # print('R2', R2)
    # print('Lw', Lw)
    # print('Eb', Eb)
    # print('vw', vw)
    # print('press', t_now)
    # print('press', press)
    # print('dMdt_factor', dMdt_factor)
    
    dMdt_init = get_init_dMdt(R2, Lw, Eb, vw, t_now, press, dMdt_factor)
    
    # This function evaluates dMdt guesses with the boundary conditions. 
    # Goes into loop and keeps evaluating until v0 (both estimated and true) agree.
    # This will yield a residual dMdt (which is nearly zero).
    # old code: compare_boundaryValues()
    # I presume Weaver meant: v -> 0 for R -> R1 (since in that chapter he talks about R1 being very small)
    # TODO: this shouldn't be 4e5. In the old code of bubble_structure2 line 310, T prime is set awayfrom Tgoal, because of dR2. 
    r_inner = R1
    # change dR2 < _dR2min: too.
    T_goal = 3e4 * u.K
    # T_goal = 453898 * u.K
    
    
    # Now, we record values for R1, R2, and calculate what rgoal should be.
    # rgoal is r_Tb, i.e., radius at which temperature is calculated.
    # rgoal_f is xi_Tb. rgoal = rgoal_f * R2
    
    def get_xi_Tb(R1, R2):
        
        # First we check what we have recorded.
        # Initialisation of this file is done in get_InitBubStruc.py
        path2bubblestructure = os.path.join(warpfield_params.out_dir, 'bubble_structure' + '.csv')
        R1_R2, R2prime_R2 = np.loadtxt(path2bubblestructure, skiprows=1, delimiter='\t', usecols=(0,1), unpack=True)
        
        # initialise variable
        xi_Tb = warpfield_params.xi_Tb
        
        # if there is already a record of R1/R2, PLUS a record of R2prime/R2 (hence len > 2)
        try:
            if len(R1_R2) > 2:
                R1_R2 = R1_R2[R1_R2 != 0]
                R2prime_R2 = R2prime_R2[R2prime_R2 != 0]
                # if there is still more than one, then calculate xi_Tb
                if len(R1_R2) > 1:
                    xi_Tb = np.min(R2prime_R2) - 0.2 * (np.min(R2prime_R2) - np.max(R1_R2))
        except:
            xi_Tb = warpfield_params.xi_Tb
        
        if np.isnan(xi_Tb):
            xi_Tb = warpfield_params.xi_Tb
        
        # record R1/R2 ratio
        R1_R2 = np.append(R1_R2, (R1/R2).value)
        # record nan, because not calculated (calculated in get_velocity_residuals()), but had to add it in to ensure similar array length.
        R2prime_R2 = np.append(R2prime_R2, np.nan)
        # save data
        np.savetxt(path2bubblestructure,
                   np.c_[R1_R2, R2prime_R2],
                   delimiter = '\t',
                   header='R1/R2 (inner/outer bubble)'+'\t'+'R2prime/R2', comments='')
        # return
        return xi_Tb
    
    # get the ratio xi_Tb 
    xi_Tb = get_xi_Tb(R1, R2)
    # get rgoal (r_Tb) with this
    rgoal = xi_Tb * R2
    # sanity check: rgoal cannot be smaller than the inner bubble radius R1
    if rgoal < R1:
        # for now, stop code. However, could also say that energy-driven phase is over if rgoal_f < R1/R2, for a reasonable rgoal_f (0.99 or so)
        sys.exit(f'rgoal ({rgoal}) is smaller than the inner bubble radius {R1}. Consider increasing xi_Tb.')
    
    # ----------- calculation of dMdt

    params = [t_now, T_goal, r_inner, R2, press, Qi, alpha, beta, delta, v0]

    # While evaluating for dMdt, we create a global variable to also store the values for
    # temperature and velocity. 
    # This is so that we don't have to run the function again for the final values. 
    global v_array, T_array, dTdr_array, r_array
    v_array = np.array([])
    T_array = np.array([])
    dTdr_array = np.array([])
    r_array = np.array([])
    
    
    # print('\n\n---dMdt_init befrore get_velocity_residuals---', dMdt_init.to(u.M_sun/u.Myr))
    
    dMdt = scipy.optimize.fsolve(get_velocity_residuals, dMdt_init.to(u.M_sun/u.yr), args = (params,), 
                                 full_output = 1, factor = 50, xtol = 1e-6)[0] * u.M_sun / u.yr

    # print('done! dMdt (Msun/yr) and dMdt (Msun/Myr):', dMdt, dMdt * 1e6)
    # 0.0845262588579248 solMass / yr
    # dMdt 40416.890252523 [96559.42153809] - guess/real
    n_array = (press/((warpfield_params.mu_n/warpfield_params.mu_p) * c.k_B.cgs * T_array)).to(1/u.cm**3)
    # print('v', v_array)
    # print('T', T_array)
    # print('dTdr', dTdr_array)
    # print('r_array', r_array)        
    
    # new factor for dMdt (used in next time step to get a good initial guess for dMdt)
    # TODO: fix this. It seems tht our simulation runs into negative dMdt, and
    # so perhaps taking the absolute will fix?
    # # original:
    # dMdt_factor_out = dMdt_factor * dMdt / dMdt_init
    # # fix?
    dMdt_factor_out = dMdt_factor * dMdt / dMdt_init
    
    # print('n', n_array)

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
    
    
    # print('\n\nsecond checkpoint\n\n')
    # print(f'press: {press.to(u.M_sun/u.pc/u.Myr**2)}')
    # print(f'dMdt_init: {dMdt_init.to(u.M_sun/u.Myr)}')
    # print(f'dMdt: {dMdt.to(u.M_sun/u.Myr)}')
    # print(f'R1: {R1}')
    # print(f'v_array: {v_array}')
    # print(f'r_array: {r_array}')
    # print(f'T_array: {T_array}')
    # print(f'n_array: {n_array.to(1/u.pc**3)}')
    # sys.exit()
    
    
    
    # =============================================================================
    # Step 3: we identify which index in the temperature array has cooling and which doesn't. 
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
    
    # print(f'index_CIE_switch: {index_CIE_switch}')
    # print(T_array[index_CIE_switch-1])
    # print(T_array[index_CIE_switch])
    # print(T_array[index_CIE_switch+1])
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
    
    # TODO: maybe some verbosity here.
    # print('\n\n1st zone done\n\n')
    # print(f'Lambda_bubble: {Lambda_bubble.to(u.M_sun*u.pc**5/u.Myr**3)}')
    # print(f'integrand_bubble: {integrand_bubble.to(u.M_sun*u.pc/u.Myr**3)}')
    # print(f'L_bubble: {L_bubble.to(u.M_sun*u.pc**2/u.Myr**3)}')
    # print(f'Tavg_bubble: {Tavg_bubble}')
    # print(f'r_bubble: {r_bubble}')
    # print(f'T_bubble: {T_bubble}')
    # print(f'n_bubble: {n_bubble}')
    
    '''
    import matplotlib.pyplot as plt
    plt.plot(np.log10(T_bubble.value), np.log10(Lambda_bubble.to(u.M_sun*u.pc**5/u.Myr**3).value))
    np.save(r'/Users/jwt/Documents/Code/warpfield3/example_plots/cooling_comparison/cooling3.npy', [np.log10(T_bubble.value), np.log10(Lambda_bubble.to(u.M_sun*u.pc**5/u.Myr**3).value)])
    # plt.show()
    # sys.exit()
    '''

    #---------------- 2. Conduction zone. High resolution region, 10**4 < T < 10**5.5 K. 
    
    # it is possible that index_cooling_switch = index_CIE_switch = 0 if the shock front is very steep.
    if index_cooling_switch != index_CIE_switch:
        # if this zone is not well resolved, solve ODE again with high resolution (IMPROVE BY ALWAYS INTERPOLATING)
        if index_CIE_switch - index_cooling_switch < 100:
            
            # print('inside cz, not well-resolved')
            
            # This is the original array that is too short
            lowres_r_conduction = (r_array[:index_CIE_switch+1]).to(u.pc)
            # print(f'lowres_r_conduction: {lowres_r_conduction}')
            # Find the minimum and maximum value
            original_rmax = max(lowres_r_conduction.value)
            original_rmin = min(lowres_r_conduction.value)
            
            # how many intervales in high-res version? [::-1] included because r is reversed.
            _highres = 1e2
            r_conduction = np.arange(original_rmin, original_rmax,
                                       (original_rmax - original_rmin)/_highres
                                       )[::-1] * u.pc
            
            # rerun structure with greater precision
            # solve ODE again, though there should be a better way (event finder!)
            psoln = scipy.integrate.odeint(get_bubble_ODE, [v_array[index_cooling_switch].value,T_array[index_cooling_switch].value,dTdr_array[index_cooling_switch].value], r_conduction.value,
                                            args = (params,), tfirst=True) 
            
            # solutions
            v_conduction = psoln[:,0] 
            T_conduction = psoln[:,1] 
            dTdr_conduction = psoln[:,2]        
            
            # Here, something needs to be done. Because of the precision of the solver, 
            # it may return temperature with values > 10**5.5K eventhough that was the maximum limit (i.e., 10**5.500001).
            # This will crash the interpolator. To fix this, we simple shave away values in the array where T > 10**5.5, 
            # and concatenate to the low-rez limit. 
            
            # Actually, the final value may not be required; the value is already included
            # in the first zone, so we don't have to worry about them here.
            
            _Tmask = T_conduction < (10**5.5)
            # apply mask
            r_conduction = r_conduction[_Tmask]
            v_conduction = v_conduction[_Tmask] * u.km / u.s
            T_conduction = T_conduction[_Tmask] * u.K
            dTdr_conduction = dTdr_conduction[_Tmask] * u.K / u.pc    
            
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
        
        # The problem now seems to be the fact that, for what we wanted, the cooling curve
        # cannot be generated. Now, the original cooling cube is very cheese-like, meaning
        # that there are 'holes' everywhere. This may be the cause.
        # In previous code this wasn't a problem, because the interpolation is done manually
        # with gradients etc. Now, however, the points are probably beyond the available ranges, 
        # so scipy (the method we use) wouldn't do an interpolation because it is unsafe. 
        
        # cooling rate
        cooling_conduction = 10 ** cooling_nonCIE.interp(np.transpose(np.log10([n_conduction.value, T_conduction.value, phi_conduction.value])))
        heating_conduction = 10 ** heating_nonCIE.interp(np.transpose(np.log10([n_conduction.value, T_conduction.value, phi_conduction.value])))
        # net cooling rate
        dudt_conduction = (heating_conduction - cooling_conduction) * u.erg / u.cm**3 / u.s
        # integrand
        integrand_conduction = (dudt_conduction * 4 * np.pi * r_conduction**2)
        # calculate power loss due to cooling
        L_conduction = (np.abs(np.trapz(integrand_conduction, x = r_conduction))).to(u.erg/u.s)
        # intermediate result for calculation of average temperature
        Tavg_conduction = (np.abs(np.trapz(r_conduction**2 * T_conduction, x = r_conduction))).to(u.K*u.pc**3)
    # if there is no conduction; i.e., the shock front is very steep. 
    elif index_cooling_switch == 0 and index_CIE_switch == 0:
        # the power loss due to cooling in this region will simply be zero. 
        L_conduction = 0 * u.erg/ u.s   
        dTdR_coolingswitch = dTdr_bubble[0]
        
        
    # TODO: maybe some verbosity here.
    # print('\n\n2nd zone done\n\n')
    # print(f'index_cooling_switch: {index_cooling_switch}, index_CIE_switch: {index_CIE_switch}')
    # print(f'integrand_conduction: {integrand_conduction.to(u.M_sun*u.pc/u.Myr**3)}')
    # print(f'L_conduction: {L_conduction.to(u.M_sun*u.pc**2/u.Myr**3)}')
    # print(f'Tavg_conduction: {Tavg_conduction}')
    # sys.exit()
        
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
            # print('non-CIE')
            # print(np.log10([n_intermediate[mask].value, T_intermediate[mask].value, phi_intermediate[mask].value]))
            # import values from cooling curves
            cooling_nonCIE, heating_nonCIE = non_CIE.get_coolingStructure(t_now)
            # cooling rate
            cooling_intermediate = 10 ** cooling_nonCIE.interp(np.transpose(np.log10([n_intermediate[mask].value, T_intermediate[mask].value, phi_intermediate[mask].value])))
            heating_intermediate = 10 ** heating_nonCIE.interp(np.transpose(np.log10([n_intermediate[mask].value, T_intermediate[mask].value, phi_intermediate[mask].value])))
            dudt_intermediate = (heating_intermediate - cooling_intermediate) * u.erg / u.s / u.cm**3
            integrand_intermediate = dudt_intermediate * 4 * np.pi * r_intermediate[mask]**2
        elif regime == 'CIE':
            # print('CIE')
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

    # TODO: maybe some verbosity here.
    # print('\n\n3rd zone done\n\n')
    # print(f'R2_coolingswitch: {R2_coolingswitch}')
    # print(f'integrand_intermediate: {integrand_intermediate.to(u.M_sun*u.pc/u.Myr**3)}')
    # print(f'L_intermediate: {L_intermediate.to(u.M_sun*u.pc**2/u.Myr**3)}')
    # print(f'Tavg_intermediate: {Tavg_intermediate}')
    # sys.exit()

    #---------------- 4. Finally, sum up across all regions. Calculate the average temeprature.
    # this was Lb in old code
    L_total = L_bubble + L_conduction + L_intermediate
    
    # calculate temperature
    # with conduction zone
    if index_cooling_switch != index_CIE_switch:
        Tavg = 3 * ( Tavg_bubble / (r_bubble[0]**3 - r_bubble[-1]**3) +\
                    Tavg_conduction / (r_conduction[0]**3 - r_conduction[-1]**3) +\
                    Tavg_intermediate / (r_intermediate[0]**3 - r_intermediate[-1]**3))
    # without conduction zone
    else:
        Tavg = 3. * ( Tavg_bubble / (r_bubble[0]**3 - r_bubble[-1]**3) +\
                     Tavg_intermediate / (r_intermediate[0]**3 - r_intermediate[-1]**3))

    # TODO: what is rgoal?
    # get temperature inside bubble at fixed scaled radius
    # temperature T at rgoal
    print('rgoal', rgoal)
    # print(r_array[index_cooling_switch])
    
    
    if rgoal > r_array[index_cooling_switch]: # assumes that r_cz runs from high to low values (so in fact I am looking for the highest element in r_cz)
        T_rgoal = fT_interp_intermediate(rgoal)
    elif rgoal > r_array[index_CIE_switch]: # assumes that r_cz runs from high to low values (so in fact I am looking for the smallest element in r_cz)
        idx = operations.find_nearest(r_conduction.value, rgoal.value)
        T_rgoal = T_conduction[idx] + dTdr_conduction[idx]*(rgoal - r_conduction[idx])
    else:
        idx = operations.find_nearest(r_bubble.value, rgoal.value)
        T_rgoal = T_bubble[idx] + dTdr_bubble[idx]*(rgoal - r_bubble[idx])
    
    # print('Completed calculation of bubble luminosity.')
    
    # print('\n\nfinal\n\n')
    # print(f'rgoal: {rgoal.to(u.pc)}')
    # print(f'L_total: {L_total.to(u.M_sun*u.pc**2/u.Myr**3)}')
    # print(f'Tavg: {Tavg}')
    # print(f'dMdt_factor_out: {dMdt_factor_out}')
    # print(f'dMdt, dMdt_init: {dMdt.to(u.M_sun/u.Myr)}, {dMdt_init.to(u.M_sun/u.Myr)}')
    
    
    # =============================================================================
    # Step 4: Mass/gravitational potential
    # =============================================================================
    
    def get_mass_and_grav(n, r):
        # again: r and n is monotonically decreasing. We need to flip it here to avoid problems with np.cumsum.
        
        # r is now monotonically increasing
        r_new = r[::-1].to(u.cm)
        # so is n (rho) now
        # old code says * mp, but it should be mu. 
        rho_new = n[::-1] * warpfield_params.mu_n
        rho_new = rho_new.to(u.g/u.cm**3)
        # get mass 
        m_new = 4 * np.pi * scipy.integrate.simps(rho_new * r_new**2, x = r_new) * u.g 
        # cumulative mass 
        m_cumulative = np.cumsum(m_new)
        # gravitational potential
        grav_phi = - 4 * np.pi * c.G * scipy.integrate.simps(r_new * rho_new, x = r_new) * u.g / u.cm
        # gravitational force per mass
        grav_force_pmass = c.G * m_cumulative / r_new**2
        
        # print(f'r_new: {r_new.to(u.pc)}')
        # print(f'rho_new: {rho_new}')
        # print(f'm_new: {m_new * u.g.to(u.M_sun)}')
        # print(f'grav_phi: {grav_phi}')
        # print(f'grav_force_pmass: {grav_force_pmass}')
        
        return m_cumulative.to(u.M_sun), grav_phi.to(u.erg/u.g), grav_force_pmass.to(u.m/u.s**2)
    
    # gettem
    m_cumulative, grav_phi, grav_force = get_mass_and_grav(n_array, r_array)
    
    # TODO:
    # bubble mass
    mBubble = m_cumulative[-1]
    # here is what was in the old code. This was wrong. 
    # The problem is, if we tried to do this properly, we cannot. The code crashes.
    # Fixing this now. Should probably work. 
    # mBubble = 10 * u.M_sun

    # print(f'mBubble: {mBubble.to(u.M_sun)}')
    # print(f'mBubble: {mBubble.to(u.g)}')
    
    # T_rgoal will be T0 in the output in run_energy_phase.py
    print('final result in get_bubble_properties')
    # print(L_total, T_rgoal, L_bubble, L_conduction, L_intermediate, dMdt_factor_out, Tavg, mBubble)
    print(f'L_total: {L_total.to(u.M_sun*u.pc**2/u.Myr**3)}')
    print(f'T_rgoal: {T_rgoal}')
    print(f'L_bubble: {L_bubble.to(u.M_sun*u.pc**2/u.Myr**3)}')
    print(f'L_conduction: {L_conduction.to(u.M_sun*u.pc**2/u.Myr**3)}')
    print(f'L_intermediate: {L_intermediate.to(u.M_sun*u.pc**2/u.Myr**3)}')
    print(f'dMdt_factor_out: {dMdt_factor_out}')
    print(f'Tavg: {Tavg}')
    print(f'mBubble: {mBubble.to(u.M_sun)}')
    
    print('End get_bubble_properties----------\n\n\n\n')
    
    return L_total, T_rgoal, L_bubble, L_conduction, L_intermediate, dMdt_factor_out, Tavg, mBubble


# =============================================================================
# Initial guess of dMdt
# =============================================================================

def get_init_dMdt(
        R2, Lw, Eb, vw,
        t_now, press, dMdt_factor
        ):
    
    # print('get_init_dMdt parameter', R2, Lw, Eb, vw, t_now, press)
    
    # TODO:
    # dMdt_init = 12 / 75 * dMdt_factor**(5/2) * 4 * np.pi * R2**3 / t_now\
    #         * warpfield_params.mu_p / c.k_B.cgs * (t_now * warpfield_params.c_therm / R2**2)**(2/7) * press**(5/7)
    
    # above is the original code. But, i suspect this is why we get different answers compared to wfld.
    # the reason is because in theold code the factor is only powered once, where as here it is 5/2 according to the paper. 
    
    dMdt_init = 12 / 75 * dMdt_factor * 4 * np.pi * R2**3 / t_now\
        * warpfield_params.mu_p / c.k_B.cgs * (t_now * warpfield_params.c_therm / R2**2)**(2/7) * press**(5/7)
     
    # old code in old warpfield:
    # dMdt_guess = 4. / 25. * dMdt_factor * 4. * np.pi * R2 ** 3. / t_now * 0.5 * (myc.mp / myc.Msun) / myc.kboltz_au * (t_now * myc.Cspitzer_au / R2 ** 2.) ** (2. / 7.) * press ** (5. / 7.)

    return dMdt_init.to(u.M_sun/u.yr)




def get_velocity_residuals(dMdt_init, params):
    """
    This routine calculates the value for dMdt, by comparing velocities at boundary condition.
    Check out get_bubble_ODE_initial_conditions() below, for full description.

    Parameters
    ----------
    dMdt_init [in Msun/yr, but scipy must take in unitless value]: TYPE
        Guesses of dMdt.
    params : t_now, T_goal, r_inner, R2, pressure, Qi, alpha, beta, delta, v0
        Parameters.

    Returns
    -------
    resdual

    Old code: find_dMdt()

    """
    # unravel
    t_now, T_goal, r_inner, R2, pressure, Qi, alpha, beta, delta, v0 = params
    # units for dMdt_init! Because scipy removes them in ODE solvers
    # units in params are retained. 
    dMdt_init = dMdt_init*u.M_sun/u.yr
    
    # --- debug
    
    # for some reason, scipy in these loops return negative dMdt, with every other parameters simliar.

    # print('\ndMdt_init in get_velocity_residuals', dMdt_init.to(u.M_sun/u.Myr))

    # print('params in get_velocity_residuals')
    # print('t_now', t_now)
    # print('T_goal', T_goal)
    # print('r_inner', r_inner)
    # print('R2', R2)
    # print('pressure', pressure)
    # print('Qi', Qi)
    # print('alpha', alpha)
    # print('beta', beta)
    # print('delta', delta)
    # print('v0', v0)
    
    # perhaps one way to fix it is to just take the absolute value
    # i.e., 
    dMdt_init = np.abs(dMdt_init.value) *u.M_sun/u.yr

    
    # --- end debug
    
    
   
    # =============================================================================
    # Get initial bubble values for integration  
    # =============================================================================
    # Watch out! these are unitless.
    r2Prime, T_r2Prime, dTdr_r2Prime, v_r2Prime = get_bubble_ODE_initial_conditions(dMdt_init, params)
    
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
    
    # global, because we are also recording values here into the array in get_bubbleproperties().
    global r_array
    r_array = (r2Prime + r_inner.to(u.pc).value) -  np.logspace(np.log10(r_inner.to(u.pc).value), np.log10(r2Prime[0]), int(1e5))
    r_array = r_array * u.pc
    
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
    # return
    return residual
    
    
    
def get_bubble_ODE_initial_conditions(dMdt, params):
    """
    dMdt_init (see above) can be used as an initial estimate of dMdt, 
    which is then adjusted until the velocity found by numerical integration (see get_velocity_residuals()) 
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
    T [K]: T(r), ODE.
    dTdr [K/pc]: ODE.
    v [km/s]: ODE.
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
    
    # print('outputs in get_bubble_ODE_initial_conditions()')
    # print('R2', R2)
    # print('dMdt', dMdt)
    # print('mu', mu)
    # print('dR2', dR2)
    
    # temperature
    T = (25/4 * c.k_B.cgs.value / mu / warpfield_params.c_therm.value * dMdt / (4 * np.pi * R2**2) )**(2/5) * dR2**(2/5) 
    # print('T', T)
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
    Here is the main function that deals with ODE calculation.
    
    old code: calc_cons() and get_bubble_ODEs() aka bubble_struct()

    Parameters
    ----------
    r_arr : [pc]
        radius at which the ODE is solved.
    initial_ODEs : v_r2Prime, T_r2Prime, dTdr_r2Prime
        These are initial guesses for the ODE, obtained via get_bubble_ODE_initial_conditions().
    params : [t_now, T_goal, r_inner, R2, press, Qi, alpha, beta, delta, v0]
        Paramerers required to run the ODE. See the main function, get_bubbleproperties() for more.

    Returns
    -------
    dvdr : [km/s/pc]
        distance derivative of velocity.
    dTdr : [K/pc]
        distance derivative of temperature.
    dTdrr : [K/pc**2]
        second distance derivative of temperature.
    
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
    
    # Test. 
    # T = 453898
    T *= u.K
    
    # get density and ionising flux
    ndens = pressure / (2 * c.k_B.cgs * T)
    phi = Qi / (4 * np.pi * r_arr**2)
    
    # net cooling rate
    # dudt = nnLambda
    dudt = net_coolingcurve.get_dudt(t, ndens.to(1/u.cm**3), T, phi.to(1/u.s/u.cm**2))
    
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
    





