#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:04:16 2023

@author: Jia Wei Teh

This is a re-write of find_root_betadelta.py
"""

# libraries
import numpy as np
import astropy.units as u
import astropy.constants as c
import scipy.interpolate
import sys
import scipy.optimize
import os
#--
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams
import src.warpfield.bubble_structure.bubble_structure as bubble_structure
import src.warpfield.shell_structure.shell_structure as shell_structure
import src.warpfield.cloud_properties.mass_profile as mass_profile
import src.warpfield.phase1_energy.energy_phase_ODEs as energy_phase_ODEs
from src.output_tools import terminal_prints, verbosity
from src.warpfield.cooling.non_CIE import read_cloudy
import src.warpfield.bubble_structure.bubble_luminosity as bubble_luminosity
import src.warpfield.functions.operations as operations

from src.input_tools import get_param
warpfield_params = get_param.get_param()


# Note 08/03/2024: The simulation now stops at line 476ish, where we need to check the return of values.
# this is because the solver is terrible at running and finding solution. Not sure why.


# Note 03/03/24: This is a new script. This one basically rewrites the whole thing, and hopefully calculates beta and delta at 
# a better speed.

# Note 26/02/24: The current error are: 1) there is probably something wrong with the residual (changing
#                   normalisation for units which means np.sign is now useless since the solution without 
#                   normalisation will not be 1; and 2) in some loops Lw does not have units, which causes
#                   error to appear in get_bubble_properties. We need to find out which fsolve has that problem.


# TODO: look at every TODO in this script and change what is necessary.

# TODO: VERY IMPOTANT: need to include dictionary, and need to include the fact that 
# scipy args=[] will only accept numerical values wthout units I think. This is causing t oo much errors.
# actually this isnt true anymore. See old wfld4 and why it is not necessary to use dict.

# TODO: make sure thje comments and docstring correctly tells what they do. 
# Otherwise, in the future, once you forget this will be a mess.

# the old file has the structure of
#  zeroODE34  < zeroODE34wrap < rootfinder_bd < rootfinder_bd_wrap < phase_energy() rootf_bd_res that searches for beta delta and residual.
# and all of these are only being used in fE_tot, to find beta, delta, and the residuals. 
# here is the exmaple of how it looks:


        # rootf_bd_res = find_root_betadelta.rootfinder_bd_wrap(beta_guess, delta_guess, params, Cool_Struc,
        #                                                           xtol=1e-5, verbose=0)  # xtol=1e-5
        
        # # 
        
        
        # beta = rootf_bd_res['beta']
        # delta = rootf_bd_res['delta']
        # residual = rootf_bd_res['residual']
        # dMdt_factor = rootf_bd_res['dMdt_factor']
        # Tavg = rootf_bd_res['Tavg']
        
# Now, let's try to work this out in this rewrite. We need to break down
# every function in the chain and make sure they are properly names, and 
# remove any unnecessary interactions. 


def get_beta_delta_wrapper(beta_guess, delta_guess, wrapper_params):
    """
    # old code: rootfinder_bd_wrap()
    
    This wrapper handles get_beta_delta(), which deals with 
    getting better estimates of delta, beta and other bubble properties. It runs 
    get_beta_delta() with guesses of beta and delta, and if the solver crashes,
    it will then modify beta, delta, and dMdt_factor for a small amount, and run again. 
    
    Parameters
    ----------
    General: beta = - dPb/dt; delta = dT/dt at xi. See https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf,
    pg92 (A4-5). This is used to resolve the velocity and temperature structure,
    v' (dvdr) and T'' (dTdrr). 
    
    For this function, it takes in guess values for beta and delta, run a few
    solver to find a better estimate for the next loop. 
    
    beta_guess : float
        guess for beta.
    delta_guess : float
        guess for delta.
    wrapper_params : list of parameters. They are mainly parameters from feedbacks, which are obtained from run_energy.
        DESCRIPTION.

    Returns
    -------
    beta_delta_outputs_main : list
        [beta_out, delta_out, log_residual_out, dMdt_factor_out, Tavg],
        where:
            beta_out, delta_out = new beta delta guesses
            log_residual_out = [residuals of beta and delta before and after]
            dMdt_factor_out = factor used to improve the next dMdt_factor. 
                            This is used in bubble_luminosity where dMdt_new = dMdt_factor * dMdt
            Tavg = average temperature of bubble.
    """
    
    # Note: beta_guess, delta_guess is from the wrapper, and beta, delta is 
    # staying in the dictionary. When run into functions like scipy.optimize,
    # the value for beta_guess, delta_guess will be changing constantly due to its loopoing nature,
    # whereas beta, delta will stay the same.
    
    # TODO: check if any of these variables are actually not obtainable.
    beta, delta, Edot_residual_guess, T_residual_guess,\
                    L_wind, L_leak, Eb, Qi,\
                    v_wind, v2, R2, T0,\
                    alpha, t_now,\
                    pwdot, pwdotdot,\
                    dMdt_factor = wrapper_params
                        
    # Start with first estimate. 
    # =============================================================================
    # First attempt
    # =============================================================================
    
    # run this instead and uncomment others during debug:
    beta_delta_outputs_main = get_beta_delta(beta_guess, delta_guess, wrapper_params)
    sys.exit('here in bd-wrapper.')

    try:
        beta_delta_outputs_main = get_beta_delta(beta_guess, delta_guess, wrapper_params)
    except:
        # if bubble structure could not be calculated, try again with slightly different input (guess) values
        # better not to use np.random small changes in input parameters, so that errors remain reproducable at least
        # do three more tries
        # =============================================================================
        # Second attempt
        # =============================================================================
         
        try:
            print('/nSecond attempt at guessing beta/delta.')
            dMdt_factor_new = dMdt_factor * 0.95
            beta_guess_new = beta_guess + 0.0010
            delta_guess_new = delta_guess + 0.0010
            # recreate to update dMdt_factor_new.
            beta, delta, Edot_residual_guess, T_residual_guess,\
              L_wind, L_leak, Eb, Qi,\
              v_wind, v2, R2, T0,\
              alpha, t_now,\
              pwdot, pwdotdot,\
              dMdt_factor_new = wrapper_params 
            # rerun again!
            beta_delta_output_new = get_beta_delta(beta_guess_new, delta_guess_new, wrapper_params)
        except:
            # =============================================================================
            # Third attempt
            # =============================================================================
            try:
                print('/nThird attempt at guessing beta/delta.')
                dMdt_factor_new *= 0.90
                beta_guess_new += 0.0012
                delta_guess_new += -0.0014
                # recreate to update dMdt_factor_new.
                beta, delta, Edot_residual_guess, T_residual_guess,\
                  L_wind, L_leak, Eb, Qi,\
                  v_wind, v2, R2, T0,\
                  alpha, t_now,\
                  pwdot, pwdotdot,\
                  dMdt_factor_new = wrapper_params
                # rerun again!
                beta_delta_output_new = get_beta_delta(beta_guess_new, delta_guess_new, wrapper_params)    
            except:
                # =============================================================================
                # Fourth attempt
                # =============================================================================
                try:
                    print('/nFourth attempt at guessing beta/delta.')
                    dMdt_factor_new *= 1.20
                    beta_guess_new += -0.0030
                    delta_guess_new += -0.0030
                    # recreate to update dMdt_factor_new.
                    beta, delta, Edot_residual_guess, T_residual_guess,\
                      L_wind, L_leak, Eb, Qi,\
                      v_wind, v2, R2, T0,\
                      alpha, t_now,\
                      pwdot, pwdotdot,\
                      dMdt_factor_new = wrapper_params
                    # rerun again!
                    beta_delta_output_new = get_beta_delta(beta_guess_new, delta_guess_new, wrapper_params)    
                except:
                    # =============================================================================
                    # Fifth attempt
                    # =============================================================================
                    try:
                        print('/nFinal attempt at guessing beta/delta.')
                        dMdt_factor_new *= 1.20
                        beta_guess_new += -0.0030
                        delta_guess_new += -0.0030
                        # recreate to update dMdt_factor_new.
                        beta, delta, Edot_residual_guess, T_residual_guess,\
                          L_wind, L_leak, Eb, Qi,\
                          v_wind, v2, R2, T0,\
                          alpha, t_now,\
                          pwdot, pwdotdot,\
                          dMdt_factor_new = wrapper_params
                        # rerun again!
                        beta_delta_output_new = get_beta_delta(beta_guess_new, delta_guess_new, wrapper_params)       
                    except:
                        # =============================================================================
                        # If nothing works, take previous result
                        # =============================================================================
                        # unwrap values again
                        # TODO: make sure this works. SHould we perhaps use dictionary?
                        # the probelm is that here we need to initialise th default values of bd_res
                        # Also, we need to maybe make sure that this function is not in a loop, so we dont have to
                        # do so many weird stuffs. We could instead just initialise the outputs as zero per session.
                        print('/nNothing works. Previous values used.')
                        beta, delta, Edot_residual_guess, T_residual_guess,\
                            L_wind, L_leak, Eb, Qi,\
                            v_wind, v2, R2, T0,\
                            alpha, t_now,\
                            pwdot, pwdotdot,\
                            dMdt_factor = wrapper_params
                        
                        beta_out, delta_out, log_residual_out, dMdt_factor_out, Tavg = beta_delta_output_new
                        beta_out += 1e-3
                        delta_out += 1e-3
                        beta_delta_outputs_main = [beta_out, delta_out, log_residual_out, dMdt_factor_out, Tavg]
                    
    # old code: os.environ["BD_res"]=str(result)
    os.environ["beta_delta_result"] = str(beta_delta_outputs_main)
    
    
    return beta_delta_outputs_main




def get_beta_delta(beta_guess, delta_guess, wrapper_params):
    """
    # old code: rootfinder_bd()

    The main function of this script. 
    This function deals with getting better estimates of delta, beta and other bubble properties.
   
    Parameters
    ----------
    See get_beta_delta_wrapper().

    Returns
    -------
    See get_beta_delta_wrapper().
    """
    
    # full_params[0]['x0'] in old code will now be bd_guesses;
    # full_params[0]['y0'] will now be Edot_residual_guess, T_residual_guess.

    # =============================================================================
    # Step 1: check what is the initial residual corresponding to given beta/delta guesses.
    # =============================================================================
    
    # TODO: is beta, delta already in wrapper_params? or are they added here? Modify according to that.
    # this was my_params/full_params. The only difference is that full_params has cool_structure,
    # which we calculate independently anyway.
    
    # unwrap. This is for better clarification, and to avoid using dictionary.
    beta, delta, Edot_residual_guess, T_residual_guess,\
                        L_wind, L_leak, Eb, Qi,\
                        v_wind, v2, R2, T0,\
                        alpha, t_now,\
                        pwdot, pwdotdot,\
                        dMdt_factor = wrapper_params
    
    
    # old code: full_params[0]['x0'] = x0
    # old code: full_params[0]['y0'] = zeroODE34(x0, full_params)
    bd_guesses = np.array([beta_guess, delta_guess])
    # Waht is the residual corresponding to this?
    Edot_residual_guess, T_residual_guess = get_Edot_Tdot_residual_fast(bd_guesses, wrapper_params)
    print(Edot_residual_guess, T_residual_guess)
    print('these above should be unitless, because they are normalised.')
                        
    # the first bd_guess is to feed into solver. This was x0.
    # These two will keep changing in scipy.optimize stuffs.
    beta_guess, delta_guess = bd_guesses
    # this bd_guess inside param is to be used in wrapper so that the initial condition stays the same
    # in scipy loops.
    # I am renaming them here just to separate the variables and for clarification.
    beta = beta_guess
    delta = delta_guess
    
    
    # Update the parameters. I am restating this, because I want to avoid using ambiguous dictionary entries.
    # this is the re-write version of full_params
    # Now, this will be fed into calculating beta_fast, which runs with scipy. This means we need to strip units.

    wrapper_params = [
        beta, delta, #unitless
        Edot_residual_guess, T_residual_guess, #normalised, so unitless
        # these are in cgs units
        L_wind, L_leak, Eb, Qi,
        # v_wind is in cm.s here, v2, R2 in km/s and pc.
        v_wind, v2, R2, T0,
        # t_now in Myr, alpha is unitless
        alpha, t_now,
        # these are in cgs (see run_implicit)
        pwdot, pwdotdot,
        dMdt_factor
        ]
    
    # =============================================================================
    # Step 2: with this initial residual and guess, one can then run scipy.
    # a) first run the quicker version get_Edot_Tdot_residual_fast(), with this wrapper.
    # =============================================================================
    
    def get_Edot_Tdot_residual_fast_wrapper(bd_guesses, wrapper_params):
        # old code: zeroODE34wrap(x, full_params)
        
        # basically the same, but check if it is necessary to run get_Edot_Tdot_residual() again.
        # returns the residual of Edot and T given these beta/delta values.
        beta_guess, delta_guess = bd_guesses
        beta_current, delta_current = wrapper_params[0], wrapper_params[1]
        
        # is beta_guess == beta_current?
        if beta_guess == beta_current and delta_guess == delta_current \
            and T_residual_guess is not None and Edot_residual_guess is not None:
            # no need to run the function, just return -1 or 1 as residual.
            print('in sign loop', beta_guess, delta_guess, beta_current, delta_current, T_residual_guess, Edot_residual_guess)
            return np.array([np.sign(Edot_residual_guess), np.sign(T_residual_guess)])
        else:
            return get_Edot_Tdot_residual_fast(bd_guesses, wrapper_params)
        
    
    # try to run it once now to find the next output.
    # use this beta for the next residual.
    # TODO: change the args argument to unitless so that scipy can process. Make sure the units are correct!
    print('\n\nbegin fast_wrapper!-------')
    beta_fast, delta_fast = scipy.optimize.fsolve(get_Edot_Tdot_residual_fast_wrapper, bd_guesses,
                                        args = wrapper_params,
                                        factor = 2,
                                        xtol = 1e-5, #?
                                        epsfcn = 1e-6)
    
    print('check for values for error before ending fast_wrapper solver')
    print('[beta_fast, delta_fast]', beta_fast, delta_fast)
    print('Edot_residual_guess, T_residual_guess', Edot_residual_guess, T_residual_guess)
    sys.exit()
    
    log_residual_fast = np.max(np.log10(np.abs(np.array(
                        get_Edot_Tdot_residual_fast_wrapper(np.array([beta_fast, delta_fast]), wrapper_params)
                        * np.array([Edot_residual_guess, T_residual_guess]))
                    )))
    print('log residual', log_residual_fast)
    print('end fast_wrapper!')
    
    # =============================================================================
    # b) if the residual is not small enough, rerun with robust function (basically
    # separating and calculating delta and beta individually, consecutively). Downside
    # is that this takes a long time due to heavy computing.
    # =============================================================================
        
    if log_residual_fast > -3:
        print('not good enough. We start with robust now.')
        # this isnt good enough. Rerun with robust function.
        beta_robust, delta_robust = get_Edot_Tdot_residual_robust(bd_guesses, wrapper_params)
        # residual
        log_residual_robust = np.max(np.log10(np.abs(np.array(
                                get_Edot_Tdot_residual_fast_wrapper(np.array([beta_robust, delta_robust]), wrapper_params)))))
    
        # is this the better one?
        if log_residual_robust < log_residual_fast:
            # if this is better, take this.
            beta_out = beta_robust
            delta_out = delta_robust
            log_residual_out = log_residual_robust
        else: # only for clarity; can remove this afterwards.
            beta_out = beta_fast
            delta_out = delta_fast
            log_residual_out = log_residual_fast
    
    else: # only for clarity; can remove this afterwards.
        beta_out = beta_fast
        delta_out = delta_fast
        log_residual_out = log_residual_fast
            
    # =============================================================================
    # Use these beta, delta to run bubble properties and get properties we are interested in
    # =============================================================================
        
    # improve. The guess beta,delta value is now the output value.
    # use this value to run bubble properties and get properties we are interested in
    _, T_rgoal, L_bubble, L_conduction, L_intermediate, dMdt_factor_out, Tavg, mBubble\
        = bubble_luminosity.get_bubbleproperties(t_now, R2, Qi, alpha, beta_out, delta_out, L_wind, Eb, v_wind, dMdt_factor)
    # but, add beta/delta and stuffs into it.
    beta_delta_outputs = beta_out, delta_out, log_residual_out, dMdt_factor_out, Tavg

       #-- end rootfinder_bd()

    return beta_delta_outputs



def get_Edot_Tdot_residual_fast(bd_guesses, wrapper_params):
    """
    # old code: zeroODE34() + zeroODE34wrap()
    
    This function uses beta/delta guesses to find the residual for Edot and T, at which (in - out) = 0, with which gives the solution for true beta and delta. 
    This is the main bulk of the function as it is fast, however if this does not give the desired sensitivity, we 
    then switch to get_Edot_Tdot_residual_fast().

    Parameters
    ----------
    bd_guesses : [beta_guess, delta_guess]. These guesses are fed into equations which calculates
                the residuals between input and output. For beta, we calculate Edot; for delta, we calculate T.
                
    
    wrapper_params : TYPE
        DESCRIPTION.

    Returns
    -------
    Edot_residual_new : TYPE
        DESCRIPTION.
    T_residual_new : TYPE
        DESCRIPTION.

    """
    
    
    # old code: zeroODE34 + zeroODE34wrap + more (?)

    
    beta, delta = bd_guesses

    # I think the problem is here.
    _, _, Edot_residual_guess, T_residual_guess,\
                L_wind, L_leak, Eb, Qi,\
                v_wind, v2, R2, T0,\
                alpha, t_now,\
                pwdot, pwdotdot,\
                dMdt_factor = wrapper_params
                
    # add units here
    print('\nin get_Edot_Tdot_residual_fast()====================')
    # print('L_wind', L_wind)
    L_wind *= u.erg/u.s
    L_leak *= u.erg/u.s
    Eb *= u.erg
    Qi *= 1/u.s
    # v_wind is in cm.s here, v2, R2 in km/s and pc.
    v_wind *= u.cm/u.s
    v2 *= u.km/u.s
    R2 *= u.pc
    T0 *= u.K
    # t_now in Myr, alpha is unitless
    t_now *= u.Myr
    # these are in cgs (see run_implicit)
    pwdot *= u.g * u.cm / u.s**2
    pwdotdot *= u.g * u.cm / u.s**3
                
                
    # =============================================================================
    # First, calculate Lb and T which are used to calculate Edot and T respectively.
    # =============================================================================

    # time, shell radius ( think its r2)
    # the output is L_bubble and Trgoal
    # this used to be bstrux_result()
    L_bubble, T2, _, _, _, _, _, _ = bubble_luminosity.get_bubbleproperties(t_now, R2, Qi, alpha, beta, delta,
                                                                            L_wind, Eb, v_wind, dMdt_factor
                                                                            )
    
    # sys.exit('checking get_bubbleproperties in get_Edot_Tdot_residual_fast()')
    
    # =============================================================================
    # Then, using another method to find Edot, we require beta, which required pressure, which required R1
    # =============================================================================
        
    # 
    R1_params = [L_wind.to(u.erg/u.s).value, Eb.to(u.erg).value, v_wind.to(u.cm/u.s).value, R2.to(u.cm).value]
    
    # check units etc
    # calculate Edot from beta, which required pressure, which required R1
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1,
                               1e-3 * R2.to(u.cm).value, R2.to(u.cm).value,
                               args = (R1_params),
                               xtol=1e-18) * u.cm  # can go to high precision because computationally cheap (2e-5 s)
    Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1)
    
    # make sure these parameters are correct
    
        #-- method 1 of calculating Edot, from beta
    Edot = get_bubbleParams.beta2Edot(Pb, Eb, beta,
                                      t_now, pwdot, pwdotdot,
                                      R1, R2, v2,
                                      )
    
        #-- method 2 of calculating Edot, directly from equation
    L_gain = L_wind
    L_loss = L_bubble + L_leak
    # these should be R2, v2 and press_bubble
    # gain - loss + work done
    Edot2 = L_gain - L_loss - 4 * np.pi * R2**2 * v2 * Pb
    
    # TODO: maybe in this new version we don't normalise.
    # old code: residual0 and normalised, normalize2
    # Edot_residual_new = (Edot - Edot2) / Eb / np.abs(Edot_residual_guess)
    
    # Problem here: In the old script, the residual is normalised. However, the 
    # normalisation forgets about units, which will break everything here.
    
    # Therefore here we make an execption and turnoff units for this one small section.
    # first making sure the unit's right
    # Edot_residual_new = (Edot - Edot2).to(u.erg/u.s) / Eb.to(u.erg) / np.abs(Edot_residual_guess)
    # T_residual_new = (T0 - T2).to(u.K) / T0 / np.abs(T_residual_guess)
    
    # Now I am writing my own version, which is in the new script.
    # This, if the guesses of beta is correct, should yield Edot that is very close to Edot2.
    # Meaning, the difference should be ~1e-5 * Edot2. 
    # We are using Edot2 here because it is an analytical solution, thus serves as a nice control variable. 
    # Using Edot here instead would be disastrous as Edot could be a terrible estimate.
    Edot_residual_new = (Edot - Edot2)/Edot2
    T_residual_new = (T0 - T2)/T0
    
    
    # Note: 04/03/24: The problem with this new code now: it seems that the solver is not
    # doing anything. It repeats the same value and returns the same estimation and also the 
    # same result for beta/delta/Edot/Tdot. This causes the solver to be in a very long long loop
    # and it ends causing the code to run the _robust wrapper. 
    # Fix it! How? perhaps start seeing why it is returning the same thing. Does it 
    # have to do with dictionaries? Did it not update because in the old code dictionary was used
    # and it updated implicitly, vs now the new code that we have to declare and do it ourselves?
    # Answer: I think the problem is above: the fact that beta and delta is the same in every loop
    # because there wasn't any declaration. Now, changed _, _ ... = wrapper_param instead. So 
    # now hopefully it will work. Remember to check old code if it is necedssary though! 
    
    
    # Note: 08/03/24:
    # perhaps to make scipy run better, we take the logarithm of the estimates (i.e., deal with numbers 
    # <100 rather than 1e38.)
    
    print('Showing residuals--------')
    print('Edot', Edot, 'Edot2', Edot2)
    print('T0', T0, 'T2', T2)
    print('second normalisation residuals', 'residual0', Edot_residual_new, 'residual1', T_residual_new)
    print('end get_Edot_Tdot_residual_fast()====================')
        # -- here we end functions from zeroODE34
    
    # sys.exit()
    
    return Edot_residual_new.value, T_residual_new.value






def get_Edot_Tdot_residual_robust(bd_guesses, wrapper_params):
    
    # zero_delta(), zero_beta(), fsolveODEbd(). Instead of zeroODE34(), we separtely solve
    # for delta first, then use it to solve beta. This is much more robust, but much slower.
    
    # this function does not require fsolve, because it includes it.
    
    # =============================================================================
    # get delta
    # =============================================================================
    def get_T_residual(delta, delta_param):
        
        # old code: zero_delta()
        
        beta_old, delta_old, beta_residual_guess, delta_residual_guess,\
                L_wind, L_leak, Eb, Qi,\
                v_wind, v2, R2, T0,\
                alpha, beta, t_now,\
                pwdot, pwdotdot, dMdt_factor = delta_params
        
        # this used to be bstrux_result
        _, T2, _, _, _, _, _, _ = bubble_luminosity.get_bubbleproperties(t_now, R2, Qi, alpha, beta, delta,
                                                                            L_wind, Eb, v_wind, dMdt_factor)
        # residual
        T_residual = (T0 - T2)
        # return 
        return T_residual.to(u.K).value
    
    # =============================================================================
    # get beta
    # =============================================================================
    def get_Edot_residual(beta, beta_param):
        
        # old code: zero_beta()
        
        beta_old, delta_old, beta_residual_guess, delta_residual_guess,\
                L_wind, L_leak, Eb, Qi,\
                v_wind, v2, R2, T0,\
                alpha, delta, t_now,\
                pwdot, pwdotdot, dMdt_factor = beta_param
        
        
        R1_params = [L_wind.to(u.erg/u.s).value, Eb.to(u.erg).value, v_wind.to(u.cm/u.s).value, R2.to(u.cm).value]
    
        # check units etc
        # calculate Edot from beta, which required pressure, which required R1
        R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 1e-3 * R2.to(u.cm).value, R2.to(u.cm).value,
                                   args = (R1_params),
                                   xtol=1e-18) * u.cm  # can go to high precision because computationally cheap (2e-5 s)
        
        Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1)
        
        # make sure these parameters are correct
        # this used to be bstrux_result
        L_bubble, _, _, _, _, _, _, _ = bubble_luminosity.get_bubbleproperties(t_now, R2, Qi, alpha, beta, delta,
                                                                            L_wind, Eb, v_wind, dMdt_factor)
    
            #-- method 1 of calculating Edot, from beta
        Edot = get_bubbleParams.beta2Edot(Pb, Eb, beta,
                                          t_now, pwdot, pwdotdot,
                                          R1, R2, v2,
                                          )
        
            #-- method 2 of calculating Edot, directly from equation
        L_gain = L_wind
        L_loss = L_bubble + L_leak
        # these should be R2, v2 and press_bubble
        # gain - loss + work done
        Edot2 = L_gain - L_loss - 4 * np.pi * R2**2 * v2 * Pb
        # residual
        Edot_residual = (Edot - Edot2)
        # return 
        return Edot_residual.to(u.erg/u.s).value
    
        
    # =============================================================================
    # Start function here
    # =============================================================================

    # this is the guesses from the start of the function.
    beta_guess, delta_guess = bd_guesses

    beta, delta, Edot_residual_guess, T_residual_guess,\
                L_wind, L_leak, Eb, Qi,\
                v_wind, v2, R2, T0,\
                alpha, t_now,\
                pwdot, pwdotdot, dMdt_factor = wrapper_params
                
    # add units here
    print('\nin get_Edot_Tdot_residual_robust()')
    print('L_wind', L_wind)
    L_wind *= u.erg/u.s
    L_leak *= u.erg/u.s
    Eb *= u.erg
    Qi *= 1/u.s
    # v_wind is in cm.s here, v2, R2 in km/s and pc.
    v_wind *= u.cm/u.s
    v2 *= u.km/u.s
    R2 *= u.pc
    T0 *= u.K
    # t_now in Myr, alpha is unitless
    t_now *= u.Myr
    # these are in cgs (see run_implicit)
    pwdot *= u.g * u.cm / u.s**2
    pwdotdot *= u.g * u.cm / u.s**3
    
    
    # keep finding residuals for beta/delta until the difference is less than 1e-5
    while True:
        
        print('Now entering True loop in get_Edot_Tdot_residual_robust().')
        
        # declare the input beta delta for this loop, and save it to compare later.
        beta_old = beta
        delta_old = delta
        
        
        # create inputs
        
        #----------------------------------------------------------------------------
        #--- find delta
        
        
        # this was my_params/full_params. The only difference is that full_params has cool_structure,
        # which we calculate independently anyway.
        delta_params = [beta_old, delta_old, Edot_residual_guess, T_residual_guess,
                        L_wind, L_leak, Eb, Qi,
                        v_wind, v2, R2, T0,
                        alpha, beta, t_now, 
                        pwdot, pwdotdot, dMdt_factor]
        
        
        # this was my_params['d0'] = zero_delta(delta, full_params)
        T_residual_guess = get_T_residual(delta, delta_params)
        
        # declare again because I want to avoid using dictionary.
        # this is just to update T_residual_guess.
        delta_params = [beta_old, delta_old, Edot_residual_guess, T_residual_guess,
                        L_wind, L_leak, Eb, Qi,
                        v_wind, v2, R2, T0,
                        alpha, beta, t_now, 
                        pwdot, pwdotdot, dMdt_factor]


        # there is a wrap here. write it in.
        # the wrap means that if the beta delta pair already exist, do not calculate them again in scipy.fsolve loop.
        def get_T_residual_wrap(delta, delta_params):
            
            beta = delta_params[13]
            beta_old, delta_old = delta_params[0], delta_params[1]
            T_residual_guess = delta_params[3]
            
            # is beta_guess == beta_current?
            if beta == beta_old and delta == delta_old \
                and T_residual_guess is not None and Edot_residual_guess is not None:
                return T_residual_guess
            else:
                return get_T_residual(delta, delta_params)

        # get delta
        print('now calculating delta in True loop in get_Edot_Tdot_residual_robust()')
        # TODO: make sure args is unitless
        delta = scipy.optimize.fsolve(get_T_residual_wrap, delta, args = delta_params, factor = 0.5, xtol = 1e-5)
        print('obtain delta:', delta)
        
        #----------------------------------------------------------------------------
        #--- use this delta to get beta.
        
        # re-introduce delta into params
        beta_params = [beta_old, delta_old, Edot_residual_guess, T_residual_guess,
                        L_wind, L_leak, Eb, Qi,
                        v_wind, v2, R2, T0,
                        alpha, delta, t_now, 
                        pwdot, pwdotdot, dMdt_factor]
        
        # this was my_params['b0'] = zero_beta(beta, full_params)
        Edot_residual_guess = get_Edot_residual(beta, beta_params)
        
        # declare again because I want to avoid using dictionary.
        # this is just to update Edot_residual_guess.
        beta_params = [beta_old, delta_old, Edot_residual_guess, T_residual_guess,
                        L_wind, L_leak, Eb, Qi,
                        v_wind, v2, R2, T0,
                        alpha, delta, t_now, 
                        pwdot, pwdotdot, dMdt_factor]
        
        # there is a wrap here. write it in.
        # the wrap means that if the beta delta pair already exist, do not calculate them again in scipy.fsolve loop.
        def get_Edot_residual_wrap(beta, beta_params):
            
            delta = beta_params[13]
            beta_old, delta_old = beta_params[0], beta_params[1]
            Edot_residual_guess = beta_params[2]
            
            # is beta_guess == beta_current?
            if beta == beta_old and delta == delta_old \
                and T_residual_guess is not None and Edot_residual_guess is not None:
                return Edot_residual_guess
            else:
                return get_Edot_residual(beta, beta_params)
            
        # get beta
        print('now calculating beta in True loop in get_Edot_Tdot_residual_robust()')
        # TODO: make sure args is unitless
        beta = scipy.optimize.fsolve(get_Edot_residual_wrap, beta, args = beta_params, factor = 0.5, xtol = 1e-5)
        print('obtain beta:', beta)

        #--- calculate residual to see if we should continue or stop calculation.
        
        delta_residual = np.abs(delta - delta_old)
        beta_residual = np.abs(beta - beta_old)
        
        # repeat loop until this is satisfied
        if delta_residual < 1e-5 and beta_residual < 1e-5:
            # then end the loop
            break

    return beta, delta










