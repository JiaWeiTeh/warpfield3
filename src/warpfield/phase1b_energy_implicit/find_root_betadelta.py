#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:28:19 2023

@author: Jia Wei Teh

"""

# old code: root_betadelta.py
# module to find roots of beta and delta

# my_params["v2"] is v from phase_energy.py, where r, v, E, T = y

# It appears that xtol is always 1e-5, from phase_energy.
# xtol is used in scipy.fsovle, and also in fsolveODEbd() to compare beta-beta_guess and delta-delta_guess


import numpy as np
import scipy.optimize
import sys
import os
import ast
#--
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams

def energy_eq(Lgain, Lloss, Rsh, vsh, Pb):
    """
    time derivative of bubble energy, compare energy equation
    :param Lgain: total input luminosity (usually mechanical - winds and SNe)
    :param Lloss: total lost luminosity (cooling and leakage)
    :param Rsh: shell radius
    :param vsh: shell velocity
    :param Pb: bubble pressure
    :return: dE/dt
    """

    # 1st term: gain
    # 2nd term: loss
    # 3rd term: PdV work
    Edot = Lgain - Lloss - 4.*np.pi * Rsh**2 * vsh * Pb

    return Edot

def zero_deltawrap(delta, full_params):
    """
    wrapper for zero_delta (see ibidem)
    :param delta:
    :param full_params:
    :return:
    """
    if hasattr(delta, "__len__"): delta = delta[0]
    # print "delta", delta
    if [full_params[0]['beta'], delta] == full_params[0]["x0"] and full_params[0]["d0"] is not None:
        return full_params[0]["d0"]
    else:
        return zero_delta(delta, full_params)


def zero_betawrap(beta, full_params):
    """
    wrapper for zero_betawrap (see ibidem)
    :param beta:
    :param full_params:
    :return:
    """
    if hasattr(beta, "__len__"): beta = beta[0]
    # print "beta", beta
    if [beta, full_params[0]['delta']] == full_params[0]["x0"] and full_params[0]["b0"] is not None:
        return full_params[0]["b0"]
    else:
        return zero_beta(beta, full_params)


def zero_delta(delta, full_params):
    """
    calculates residual for Txi: Txi_in -Txi_out
    :param delta:
    :param full_params:
    :return:
    """
    Cool_Struc = full_params[1]
    my_params = dict.copy(full_params[0])
    if hasattr(delta, "__len__"): delta = delta[0]
    # not sure what this does. Probably just updating the parameter to be fed into bstrux.
    # This isn't needed inthe new function though since we declare every single time.
    my_params["delta"] = delta
    full_params = [my_params, Cool_Struc]

    bstrux_result = get_bubbleParams.bstrux(full_params)

    residual = my_params["T0"] - bstrux_result['Trgoal']
    if my_params["verbose"]>1: print("delta, residual:", delta, residual)

    return residual


def zero_beta(beta, full_params):
    """
    calculates residual for Edot: Edot_in - Edot_out
    :param beta:
    :param full_params:
    :return:
    """
    Cool_Struc = full_params[1]
    my_params = dict.copy(full_params[0])
    if hasattr(beta, "__len__"): beta = beta[0]
    my_params["beta"] = beta
    full_params = [my_params, Cool_Struc]

    bstrux_result = get_bubbleParams.bstrux(full_params)
    Lb2 = bstrux_result['Lb']

    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 1e-3 * my_params["R2"], my_params["R2"],args=([my_params["Lw"], my_params["Eb"], my_params["vw"], my_params["R2"]]), xtol=1e-18)
    Pb = get_bubbleParams.bubble_E2P(my_params["Eb"], my_params["R2"], R1)

    # calculate Edot
    Edot = get_bubbleParams.beta2Edot(Pb, R1, my_params["beta"], my_params)

    # print "%.5e"%Edot, "%.5e"%my_params["Lw"], "%.5e"%Lb2, "%.5e"%(4. * np.pi * my_params["R2"] ** 2 * my_params["v2"] * Pb)
    Lgain = my_params["Lw"]
    Lloss = Lb2 + my_params['L_leak']
    Edot2 = energy_eq(Lgain, Lloss, my_params["R2"], my_params["v2"], Pb)
    #Edot2 = my_params["Lw"] - Lb2 - 4. * np.pi * my_params["R2"] ** 2 * my_params["v2"] * Pb
    residual = Edot - Edot2

    if my_params["verbose"] > 1: print("beta, residual:", beta, residual)

    return residual


def fsolveODEbd(x, full_params, xtol=1e-4):
    """
    root finder, calculates correct beta, delta
    :param x: inital guesses for beta, delta: [beta_in, delta_in]
    :param full_params: structure with parameters needed for bubble_structure module
    :param xtol: (optional) tolerance level for fsolve
    :return:

    Warning: SLOW! Use zeroODE34wrap when possible. But more robust than zeroODE34wrap
    Example: see zeroODE34wrap
    """
    Cool_Struc = full_params[1]
    my_params = dict.copy(full_params[0])
    beta = x[0]
    delta = x[1]
    my_params["delta"] = delta
    my_params["beta"] = beta
    if hasattr(delta, "__len__"): delta = delta[0]
    if hasattr(beta, "__len__"): beta = beta[0]

    while (True):

        delta_old = delta
        beta_old = beta

        x0 = [beta, delta]
        my_params['x0'] = x0
        full_params = [my_params, Cool_Struc]
        # this is basically the residual thing, used in wrapper.
        my_params['d0'] = zero_delta(delta, full_params)
        full_params = [my_params, Cool_Struc]

        delta = scipy.optimize.fsolve(zero_deltawrap, delta, args=full_params, factor=0.5, xtol=xtol)
        # delta = scipy.optimize.brentq(zero_delta, delta-0.1,delta+0.1, args=full_params, xtol=xtol)
        # delta = scipy.optimize.brent(zero_delta, args=(full_params,), tol=xtol)
        # print "accepted delta", delta

        if hasattr(delta, "__len__"): delta = delta[0]
        my_params["delta"] = delta

        x0 = [beta, delta]
        my_params['x0'] = x0
        full_params = [my_params, Cool_Struc]
        my_params['b0'] = zero_beta(beta, full_params)
        full_params = [my_params, Cool_Struc]

        beta = scipy.optimize.fsolve(zero_betawrap, beta, args=full_params, factor=0.5, xtol=xtol)
        # beta = scipy.optimize.brentq(zero_beta, beta - 0.05, beta + 0.05, args=full_params, xtol=xtol)
        # beta = scipy.optimize.brent(zero_beta, args=(full_params,), tol=xtol)

        # print "accepted beta", beta

        if hasattr(beta, "__len__"): beta = beta[0]
        my_params["beta"] = beta
        
        # TODO: why are we not normalising here, but do so in ODEwrap? 

        # required precission achieved?
        # print "ddelta", np.abs(delta - delta_old), "dbeta", np.abs(beta - beta_old)
        if ((np.abs(delta - delta_old) < xtol) and (np.abs(beta - beta_old) < xtol)):
            # print "Good enough!"
            break
            # else:
            #    print "More improvement needed..."

    # print "returning from fsolveODEbd..."
    return beta, delta


def zeroODE34(x, full_params):
    """
    calculates normalized residuals of Edot, Txi: [Edot_in - Edot_out, Txi_in - Txi_out]
    :param x: [beta, delta]
    :param full_params: structure with parameters needed for bubble_structure module
    :return: residuals: [residual(Edot), residual(Txi)]

    for an example, see zeroODE34wrap
    """
    
    # fullparams is basicaly just params + cooling structure.
    
    Cool_Struc = full_params[1]
    my_params = dict.copy(full_params[0])
    
    
    # here is the meaning of 'y0': full_params[0]['y0'] = zeroODE34(x0, full_params)
    
    if 'y0' in my_params:
        y0 = my_params["y0"]
    else:
        y0 = np.array([1.,1.])

    my_params["beta"] = x[0]
    # I think this one is not used. 
    my_params["delta"] = x[1] 

    full_params = [my_params, Cool_Struc]

    # calculate Lb, T
    bstrux_result = get_bubbleParams.bstrux(full_params)
    Lb2 = bstrux_result['Lb']
    T2 = bstrux_result['Trgoal']

    # print "%.5e"%Lb2, "%.5e"%T2

    # calculate pressure
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 1e-3 * my_params["R2"], my_params["R2"],
                               args=([my_params["Lw"], my_params["Eb"], my_params["vw"], my_params["R2"]]),
                               xtol=1e-18)  # can go to high precision because computationally cheap (2e-5 s)
    Pb = get_bubbleParams.bubble_E2P(my_params["Eb"], my_params["R2"], R1)

    # calculate Edot
    Edot = get_bubbleParams.beta2Edot(Pb, R1, my_params["beta"], my_params)
    Lgain = my_params["Lw"]
    Lloss = Lb2 + my_params['L_leak']
    Edot2 = energy_eq(Lgain, Lloss, my_params["R2"], my_params["v2"], Pb)
    
    
    # this is declared here and used in event_cool_switch.
    # find a better way to do this - maybe just return?
    os.environ["Lcool_event"] = str(Lloss)
    os.environ["Lgain_event"] = str(Lgain)
    
    #print('*****++++++++Gain_loss',np.log10(Lgain),np.log10(Lloss))

    # print "%.5e"%Edot, "%.5e"%my_params["Lw"], "%.5e"%Lb2, "%.5e"%(4. * np.pi * my_params["R2"] ** 2 * my_params["v2"] * Pb)
    #residual0 = (Edot - (my_params["Lw"] - Lb2 - 4. * np.pi * my_params["R2"] ** 2 * my_params["v2"] * Pb))
    residual0 = Edot - Edot2
    residual1 = my_params["T0"] - T2

    # normalize residual?
    normalize = True
    normalize2 = True # if normalized by y0
    if normalize:
        residual0 = residual0/my_params['Eb']
        residual1 = residual1/ my_params['T0']
    if normalize2:
        residual0 = residual0/np.abs(y0[0])
        residual1 = residual1/np.abs(y0[1])

    res = [residual0, residual1]

    #print("%.5e"%res[0], "%.5e"%res[1],np.abs(y0[0]))

    return res


def zeroODE34wrap(x, full_params):
    """
    wrapper for zeroODE34 to circumvent superfluous calls to function in fsolve
    :param x: [beta, delta]
    :param full_params: structure with parameters needed for bubble_structure module
    :return: residuals: [residual(beta), residual(delta)]

    example:
    params = {'R2': 3.7722768501127852, 'delta': -0.13059185573497623, 
              'dt_L': 0.0017159535542888942, 'dMdt_factor': 3.0988406957317669, 
              'Lw': 20162080054.290474, 'structure_switch': True, 
              'beta': 0.83903033695041396, 'T0': 6580032.8642736319, 
              'vw': 3808.6254043470976, 'Qi': 1.7004101133849865e+66, 
              'alpha': 0.57225108377678047, 'Lres0': 17721920706.542767, 
              'Eb': 122510753.20964822, 'temp_counter': 32, '
              t_now': 0.015151270837035451, 'v2':1.4248e+02}
    params["verbose"] = 1
    Cool_Struc = coolnoeq.get_Cool_dat(1.0, indiv_CH=True)
    full_params = [params,Cool_Struc]
    beta1, delta1 = scipy.optimize.fsolve(zeroODE34wrap, x0, args=full_params, factor=2., xtol=1e-5, epsfcn=1e-6)
    """
    # print x[0], x[1]
    verbose = full_params[0]["verbose"]

    # debug
    if verbose > 1: print("zeroODE34: beta, delta:", x[0], x[1])
    
    # x = [beta, delta]
    #--
    # what is x0 and where did full_params come from?
    # x0 is input beta and delta values. This is being decalred in rootfinder_bd(), where
    # x0 is beta_guess and delta_guess, and y0 is 1. 
    if all(x == full_params[0]["x0"]) and full_params[0]["y0"] is not None:
        # return full_params[0]["y0"]
        # np.sign is used here so it basically returns [-1 or 1].
        res = np.array([np.sign(full_params[0]["y0"][0]), np.sign(full_params[0]["y0"][1])])
    else:
        res = zeroODE34(x, full_params)
        
    #--    
        
    if verbose > 1: print("zeroODE34: beta, delta:", x[0], x[1], "residuals:", res)
    
    
    return res



# TODO: remember to check the reproducibility of this function; should be straightforward:
    # just go to implicit energy phase where one finds beta/delta, and from there 
    # take the input and then feed into the new function. You should now be able to 
    # double check if it is right or not. 

def rootfinder_bd(beta_guess, delta_guess, params, Cool_Struc, verbose=0, xtol=1e-5, epsfcn=1e-6, log_error_min=-1.0, error_exit=False):
    """
    MAIN ROUTINE OF MODULE: finds roots of beta and delta
    :param beta_guess: input guess for beta
    :param delta_guess: input guess for delta
    :param params: additional parameters (see below)
    :param Cool_Struc: Cooling Object (non-CIE at low T)
    :param verbose: how much print statements
    :param xtol: tolerance level for rootfinder fsolve
    :param epsfcn: step size for Jacobian in rootfinder fsolve
    :param log_error_min: log10 of error we allow at most
    :param error_exit: if something goes wrong, exit?
    :return: dictionary with results

    example:
    params = {'R2': 3.7722768501127852, 'delta': -0.13059185573497623, 'dt_L': 0.0017159535542888942, 'dMdt_factor': 3.0988406957317669, 'Lw': 20162080054.290474, 'structure_switch': True, 'beta': 0.83903033695041396, 'T0': 6580032.8642736319, 'vw': 3808.6254043470976, 'Qi': 1.7004101133849865e+66, 'alpha': 0.57225108377678047, 'Lres0': 17721920706.542767, 'Eb': 122510753.20964822, 'temp_counter': 32, 't_now': 0.015151270837035451, 'v2':1.4248e+02}
    params["verbose"] = 1
    Cool_Struc = coolnoeq.get_Cool_dat(1.0, indiv_CH=True)
    beta, delta, residual = rootfinder_bd(beta, delta)
    """

    params["beta"] = beta_guess
    params["delta"] = delta_guess
    # what is this for? this gets overwritten anyway
    params["y0"] = np.array([1., 1.])
    params["verbose"] = verbose
    full_params = [params, Cool_Struc]
    x0 = [params["beta"], params["delta"]]
    # basically just means full_params[0]['x0'] = [beta_guess, delta_guess]
    full_params[0]['x0'] = x0
    full_params[0]['y0'] = zeroODE34(x0, full_params)

    beta1, delta1 = scipy.optimize.fsolve(zeroODE34wrap, x0, args=full_params, factor=2.0, xtol=xtol, epsfcn=epsfcn)
    log_residual1 = np.max(np.log10(np.abs(np.array(zeroODE34wrap(np.array([beta1, delta1]), full_params)) * np.array(full_params[0]['y0'])))) # if normalized by y0
    #log_residual1 = np.max(np.log10(np.abs(np.array(zeroODE34wrap(np.array([beta1, delta1]), full_params)))))

    if verbose >0:
        print("rootfinder result (beta, delta):", beta1, delta1, "residual:", log_residual1)

    if log_residual1 < -3.:
        beta_out = beta1
        delta_out = delta1
        log_residual_out = log_residual1
    else:
        # print "not good enough, try again!"

        beta2, delta2 = fsolveODEbd(x0, full_params, xtol=xtol)

        log_residual2 = np.max(np.log10(np.abs(zeroODE34wrap(np.array([beta2, delta2]), full_params))))

        # print beta2, delta2, "residual2:", log_residual2

        # print "residual:", log_residual2
        if ((log_residual2 > log_error_min) and (log_residual1 > log_error_min) and error_exit == True):
            sys.exit("Could not find correct beta, delta...")
        if log_residual2 > log_residual1:
            # print "The old attempt was better."
            beta_out = beta1
            delta_out = delta1
            log_residual_out = log_residual1
        else:
            # print "The new attempt was better."
            beta_out = beta2
            delta_out = delta2
            log_residual_out = log_residual2

    # improve!
    full_params[0]['beta'] = beta_out
    full_params[0]['delta'] = delta_out

    # prepare dictionary to pass on
    result = get_bubbleParams.bstrux(full_params)
    result['beta'] = beta_out
    result['delta'] = delta_out
    result['residual'] = 10. ** log_residual_out

    return result

def rootfinder_bd_wrap(beta_guess, delta_guess, params, Cool_Struc, verbose=0, xtol=1e-5, epsfcn=1e-6, log_error_min=-1.0, Ntry=10):
    """
    calls rootfinder_bd with two different guesses for start values
    :param beta_guess: input guess for beta
    :param delta_guess: input guess for delta
    :param params: additional parameters (see below)
    :param Cool_Struc: Cooling Object (non-CIE at low T)
    :param verbose: how much print statements
    :param xtol: tolerance level for rootfinder fsolve
    :param epsfcn: step size for Jacobian in rootfinder fsolve
    :param log_error_min: log10 of error we allow at most
    :return: dictionary with results
    
    
    
    # ONLY USEFUL RESULTS (I THINK)
    beta = rootf_bd_res['beta']
    delta = rootf_bd_res['delta']
    residual = rootf_bd_res['residual']
    dMdt_factor = rootf_bd_res['dMdt_factor']
    Tavg = rootf_bd_res['Tavg']
    
    
    """

    # I'm pretty sure BD_res_count does not need to be environment variable
    # because it is only being used in this function. 

    # BD_res is however a dictionary that includes     dic_res={'Lb': 0, 'Trgoal': 0, 'dMdt_factor': 0, 'Tavg': 0, 'beta': 0, 'delta': 0, 'residual': 0}
    # so i amnot sure. (p.s. i think actually this is also only used here!)
    # BD_res is actually just str(result) for this fucntion (see below in this function where it is defined).
    
    # result is actually just output from 
    #     beta = rootf_bd_res['beta']
    # delta = rootf_bd_res['delta']
    # residual = rootf_bd_res['residual']
    # dMdt_factor = rootf_bd_res['dMdt_factor']
    # Tavg = rootf_bd_res['Tavg']

    # these are the only needed ones.

    # debug
    #Ntry = 1
    
    #beta_l=float(os.environ["BD_res_B"])
    #delta_l=float(os.environ["BD_res_D"])
    #Res_l=float(os.environ["BD_res_Res"])
    bdcount_l=float(os.environ["BD_res_count"])
    
    res_l= ast.literal_eval(os.environ["BD_res"])
    
    flag=False
  
    #print('rootfinder_BD_wrap called',res_l['beta'],type(res_l['beta']))
    #xtol=1e-5

    if Ntry <= 1:
        error_exit = True
        # result is actually just output from bstrux.
        result = rootfinder_bd(beta_guess, delta_guess, params, Cool_Struc, verbose=verbose, xtol=xtol, epsfcn=epsfcn, log_error_min=log_error_min, error_exit=error_exit)
    else:
        error_exit = False
        # try first with input values guessed from last time step
        try:
            result = rootfinder_bd(beta_guess, delta_guess, params, Cool_Struc, verbose=verbose, xtol=xtol, epsfcn=epsfcn, log_error_min=log_error_min, error_exit=error_exit)
        except: # if bubble structure could not be calculated, try again with slightly different input (guess) values
            # better not to use random small changes in input parameters, so that errors remain reproducable at least
            try:
                print("+++++++++++++THIRD TO LAST ATTEMPT+++++++++++++++")
                params['dMdt_factor'] *= 0.95
                beta_guess += 0.0010
                delta_guess += 0.0010
                result = rootfinder_bd(beta_guess, delta_guess, params, Cool_Struc, verbose=verbose, xtol=xtol, epsfcn=epsfcn, log_error_min=log_error_min, error_exit=error_exit)
            except:
                try:
                    print("+++++++++++++SECOND TO LAST ATTEMPT+++++++++++++++")
                    params['dMdt_factor'] *= 0.90
                    beta_guess += 0.0012
                    delta_guess += -0.0014
                    result = rootfinder_bd(beta_guess, delta_guess, params, Cool_Struc, verbose=verbose, xtol=xtol, epsfcn=epsfcn, log_error_min=log_error_min, error_exit=error_exit)
                except:
                    if bdcount_l > 4:
                        error_exit = True
                        print("+++++++++++++VERY LAST ATTEMPT+++++++++++++++")
                        params['dMdt_factor'] *= 1.2
                        beta_guess += -0.0030
                        delta_guess += -0.0030
                        result = rootfinder_bd(beta_guess, delta_guess, params, Cool_Struc, verbose=verbose, xtol=xtol, epsfcn=epsfcn, log_error_min=log_error_min, error_exit=error_exit)
                    else:
                        try:
                            params['dMdt_factor'] *= 1.2
                            beta_guess += -0.0030
                            delta_guess += -0.0030
                            result = rootfinder_bd(beta_guess, delta_guess, params, Cool_Struc, verbose=verbose, xtol=xtol, epsfcn=epsfcn, log_error_min=log_error_min, error_exit=error_exit)
                        except:
                            result=res_l
                            result['beta']=result['beta']*(1+1e-3)
                            result['delta']=result['delta']*(1+1e-3)
                            flag=True #if everything fails, take result from last time step, count how often you did this crude approximation.
    #except:
    #    error_exit = True # if something goes wrong, do not exit simulation
    #    result = rootfinder_bd(beta_guess, delta_guess, params, Cool_Struc, verbose=verbose, xtol=xtol, epsfcn=epsfcn, log_error_min=log_error_min, error_exit=error_exit)
    # if not satisfied with result, try again with slightly different (guess) values
    if np.log10(result['residual']) > log_error_min:
        print("+++++++++++++not satisfied with residual+++++++++++++++")
        error_exit = True # if something goes wrong, exit simulation
        params['dMdt_factor'] = 1.05 * params['dMdt_factor']
        beta_guess += -0.002
        delta_guess += -0.002
        result = rootfinder_bd(beta_guess, delta_guess, params, Cool_Struc, verbose=verbose, xtol=xtol, epsfcn=epsfcn, log_error_min=log_error_min, error_exit=error_exit)

    #os.environ["BD_res_B"] = str()
    #os.environ["BD_res_D"] = str()
    #os.environ["BD_res_Res"] = str()
    #os.environ["BD_res_count"] = str()
        
        
    if flag:
        bdcount_l+=1
        print('++beta,delta estimated from last step++')

        
    #print('res_bd_wrap',result)
    os.environ["BD_res"]=str(result)
    os.environ["BD_res_count"]=str(bdcount_l)
    
        
    # ONLY USEFUL RESULTS (I THINK)
    # beta = rootf_bd_res['beta']
    # delta = rootf_bd_res['delta']
    # residual = rootf_bd_res['residual']
    # dMdt_factor = rootf_bd_res['dMdt_factor']
    # Tavg = rootf_bd_res['Tavg']
    
 
        
    return result

def Edot_Tdot(beta, delta, params, verbose=0):
    """
    convert beta and delta to dE/dt and dT/dt
    :param beta:
    :param delta:
    :param params:
    :param verbose:
    :return:
        
        This function should be moved to phase_energy.py
    """

    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 1e-3 * params["R2"], params["R2"],args=([params["Lw"], params["Eb"], params["vw"], params["R2"]]), xtol=1e-18)
    Pb = get_bubbleParams.bubble_E2P(params["Eb"], params["R2"], R1)
    Edot = get_bubbleParams.beta2Edot(Pb, R1, beta, params)
    Tdot = get_bubbleParams.delta2dTdt(params["t_now"],params["T0"],delta)

    return Edot, Tdot








