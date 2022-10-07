#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:36:10 2022

@author: Jia Wei Teh

"""


import numpy as np
#--
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams
import scipy.optimize

def get_bubbleStructure(Data_Struc,
                        Cool_Struc,
                        warpfield_params,
                        fit_len=5, fit_len_short = 5,
        ):
    
    
    #     structure_bubble = (structure_switch and not mom_phase)
    # bubble_wrap_struc = {'structure_switch': structure_bubble, 'alpha': alpha, 'beta': beta, 'delta': delta,
    #                      'Lres0': Lres0, 't_10list': t_10list, 'r_10list': r_10list, 'P_10list': P_10list,
    #                      'T_10list': T_10list, 'Lw': Lw, 'vterminal': vterminal, 'r0': r0,
    #                      't0': t0 - tcoll[coll_counter], 'E0': E0,
    #                      'T0': T0, 'dt_L': dt_real, 'temp_counter': temp_counter, 'dMdt_factor': dMdt_factor,
    #                      'Qi': fQi_evo(thalf)*c.Myr, 'mypath': mypath}
                    
    # Notes:
    # old code: bubble_wrap()


    structure_switch = Data_Struc['structure_switch']
    alpha = Data_Struc['alpha']
    beta = Data_Struc['beta']
    delta = Data_Struc['delta']
    Lres0 = Data_Struc['Lres0']
    t_10list = Data_Struc['t_10list']
    r_10list = Data_Struc['r_10list']
    P_10list = Data_Struc['P_10list']
    T_10list = Data_Struc['T_10list']
    Lw = Data_Struc['Lw']
    vterminal = Data_Struc['vterminal']
    dMdt_factor = Data_Struc['dMdt_factor']
    r0 = Data_Struc['r0']
    t0 = Data_Struc['t0']
    E0 = Data_Struc['E0']
    T0 = Data_Struc['T0']
    dt_L = Data_Struc['dt_L']
    temp_counter = Data_Struc['temp_counter']
    # by default assume everything went right      
    bubbleFailed = 0 
       
    # make sure: fit_len_short <= fit_len
    if (fit_len_short > fit_len):
        fit_len_short = fit_len
                    
    if structure_switch:
        # early on, use fixed initial delta
        if (temp_counter <= fit_len):
            data_struc_temp = {'alpha': alpha, 'beta': beta, 
                          'delta': delta, 'R2': r0, 't_now': t0, 
                          'Eb': E0, 'Lw': Lw, 'vw': vterminal, 
                          'dMdt_factor': dMdt_factor, 'Qi':Data_Struc['Qi'], 
                          'mypath': Data_Struc['mypath']}
            np.seterr(all='warn')
            Lb, T_rgoal, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg = get_bubbleParams.get_bubbleLuminosity(data_struc_temp, Cool_Struc, warpfield_params)
        else:
            # time step (allow certain relative change in (in-out)-luminosity)
            my_x = np.log(t_10list[-fit_len_short:]) - np.log(t_10list[-fit_len_short])
            my_y = np.log(r_10list[-fit_len_short:])
            my_y2 = -1.0 * np.log(P_10list[-fit_len_short:])
            # at early times (when far out of equilibrium), use input alpha and beta as start values for lin fit
            if temp_counter < 2*fit_len:
                alpha = calc_linfit(my_x, my_y, loss=i.myloss, old_guess=np.round(alpha, decimals=2), c_guess=np.round(my_y[0], decimals=2))
                beta = calc_linfit(my_x, my_y2, loss=i.myloss, old_guess=np.round(beta, decimals=2), c_guess=np.round(my_y2[0], decimals=2))
            # at later times use directly the input alpha and beta
            delta0 = delta
            data_struc_temp = {'alpha': alpha, 'beta': beta, 'old_delta': delta0,
                               'R2': r0, 't_now': t0, 'Eb': E0, 'Lw': Lw,
                               'vw': vterminal, 'dMdt_factor': dMdt_factor,
                               'Qi':Data_Struc['Qi'], 'mypath': Data_Struc['mypath']}

            # use small tolerance levels if last time step was small
            last_dt = t_10list[-1]-t_10list[-2]
            my_xtol = np.min([last_dt,1e-3])

            # if above 'fixpoint interation' did not converge, we need better (but more expensive) methods: use a root finder!
            params = [data_struc_temp, Cool_Struc, t_10list, T_10list, fit_len]
            
            # try once with some boundary values and hope there is a zero point in between
            try:
                delta = scipy.optimize.brentq(get_bubbleParams.get_delta_residual, 
                                              delta0 - 0.1 , delta0 + 0.1, 
                                              args=(params), 
                                              xtol=my_xtol, rtol=1e-8) # this might fail if no fixpoint exists in the given range. If so, try with a larger range
            except:
                # it seems either the boundary values were too far off (and bubble_structure crashed) or there was no zero point in between the boundary values
                # try to figure out what limits in delta are allowed

                worked_last_time_lo = True
                worked_last_time_hi = True

                iic = 0
                # maximum number of tries before we give up
                n_trymax = 30 
                # list containing the signs of the residual 
                # (if there is a sign flip between two values, there must be a zero point in between!)
                sgn_vec = np.zeros(2*n_trymax+1) 
                # list conatining all tried input deltas
                delta_in_vec = np.zeros(2*n_trymax+1) 
                ii_lo = np.nan
                ii_hi = np.nan
                # list which contains the number 2.0 where a sign flip ocurred
                diff_sgn_vec = abs(sgn_vec[1:]-sgn_vec[:-1]) 

                # stay in loop as long as sign has not flipped
                while all(diff_sgn_vec < 2.):

                    res_0 = delta_zero(delta0, params)
                    sgn_vec[n_trymax] = np.sign(res_0) # is probably not 0 (because of small numerical noise) but ensure it is not 0 further down
                    delta_in_vec[n_trymax] = delta0

                    if worked_last_time_lo:
                        try:
                            delta_in_lo = delta0 - 0.02 - float(iic) * 0.05
                            res_lo = get_bubbleParams.get_delta_residual(delta_in_lo, params)
                            ii_lo = n_trymax-iic-1
                            sgn_vec[ii_lo] = np.sign(res_lo)
                            delta_in_vec[ii_lo] = delta_in_lo
                            if (sgn_vec[n_trymax] == 0.): sgn_vec[n_trymax]= sgn_vec[n_trymax-1] # make sure 0 does not ocurr
                        except:
                            worked_last_time_lo = False

                    if worked_last_time_hi:
                        try:
                            delta_in_hi = delta0 + 0.02 + float(iic) * 0.05
                            res_hi = get_bubbleParams.get_delta_residual(delta_in_hi, params)
                            ii_hi = n_trymax+iic+1
                            sgn_vec[ii_hi] = np.sign(res_hi)
                            delta_in_vec[ii_hi] = delta_in_hi
                            if (sgn_vec[n_trymax] == 0.): sgn_vec[n_trymax] = sgn_vec[n_trymax + 1] # make sure 0 does not ocurr
                        except:
                            worked_last_time_hi = False

                    if iic > n_trymax / 2:
                        print("I am having a hard time finding delta...")
                        if iic >= n_trymax - 1:
                            sys.exit("Could not find delta.")

                    diff_sgn_vec = abs(sgn_vec[1:] - sgn_vec[:-1]) # this list contains a 2.0 where the sign flip ocurred (take abs, so that -2.0 becomes +2.0)

                    iic += 1

                # find the index where the sign flip ocurred (where the diff list has the element 2.0)
                idx_zero0 = np.argmax(diff_sgn_vec) # we could also look for number 2.0 but because there are no higher number, finding the maximum is equivalent
                delta_in_lo = delta_in_vec[idx_zero0]
                delta_in_hi = delta_in_vec[idx_zero0+1]
                # Now, retry and find delta
                try:
                    delta = scipy.optimize.brentq(get_bubbleParams.get_delta_residual,
                                                  delta_in_lo , delta_in_hi,
                                                  args=(params), xtol=0.1*my_xtol, rtol=1e-9) # this might fail if no fixpoint exists in the given range
                except:
                    delta = delta0
                    bubbleFailed = True # something went wrong

            data_struc_temp['delta'] = delta
            Lb, T_rgoal, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg = get_bubbleParams.get_bubbleLuminosity(Data_Struc, Cool_Struc, warpfield_params)

            # calculate next time step
            Lres = Lw - Lb
            # allowed error for Lw-Lb (percentage of mechanical luminosity)
            # old code: lum_error2
            Lres_err = 0.005 
            # try to achieve relative change of 1 % (default of lum_error) but do not change more than factor 1.42 or less than 0.1
            fac = np.max([np.min([Lres_err / (np.abs(Lres - Lres0) / Lw), 1.42]), 0.1])
            dt_L = fac * dt_L  # 3 per cent change

    else:
        Lb = 0.
        Lb_b = 0.
        Lb_cz = 0.
        Lb3 = 0.
        dMdt_factor_out = 1.646 # as in classical Weaver
        dt_L = dt_L
        Tavg = T0
        Mbubble = np.nan
        r_Phi = np.nan
        Phi_grav_r0b = np.nan
        f_grav = np.nan

    return bubbleFailed, Lb, T0, alpha, beta, delta, dt_L, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav





