#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:36:10 2022

@author: Jia Wei Teh

"""
# libraries
import sys
import numpy as np
#--
import src.warpfield.bubble_structure.get_bubbleParams as get_bubbleParams
import scipy.optimize

def get_bubbleStructure(Data_Struc,
                        Cool_Struc,
                        warpfield_params,
                        fit_len, 
                        fit_len_short,
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
    bubbleFailed = False
       
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
            Lb, T0, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav = get_bubbleParams.get_bubbleLuminosity(data_struc_temp, Cool_Struc, warpfield_params)
        else:
            # time step (allow certain relative change in (in-out)-luminosity)
            my_x = np.log(t_10list[-fit_len_short:]) - np.log(t_10list[-fit_len_short])
            my_y = np.log(r_10list[-fit_len_short:])
            my_y2 = -1.0 * np.log(P_10list[-fit_len_short:])
            # at early times (when far out of equilibrium), use input alpha and beta as start values for lin fit
            if temp_counter < 2*fit_len:
                alpha = get_bubbleParams.get_fitSlope(my_x, my_y, old_guess=np.round(alpha, decimals=2), c_guess=np.round(my_y[0], decimals=2))
                beta = get_bubbleParams.get_fitSlope(my_x, my_y2, old_guess=np.round(beta, decimals=2), c_guess=np.round(my_y2[0], decimals=2))
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

                    res_0 = get_bubbleParams.get_delta_residual(delta0, params)
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
            Lb, T0, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav = get_bubbleParams.get_bubbleLuminosity(Data_Struc, Cool_Struc, warpfield_params)

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




# testruns

# Data_Struc = {'structure_switch': True, 'alpha': 0.6, 'beta': 0.8, 'delta': -0.17142857142857143,
#               'Lres0': 1.0, 't_10list': np.array([6.50681839e-05]), 'r_10list': np.array([0.23790232]),
#               'P_10list': np.array([1.9699816e+08]), 'T_10list': np.array([]), 'Lw': 201648867747.70163,
#               'vterminal': 3810.2196532385897, 'r0': 0.23790232199299727, 't0': 6.506818386985495e-05,
#               'E0': 5722974.028981317, 'T0': 67741779.55773313, 'dt_L': 0.0001, 'temp_counter': 0,
#               'dMdt_factor': 1.646, 'Qi': 1.6994584609226492e+67,
#               'mypath': '/Users/jwt/Documents/Code/warpfield3/outputs/'}


# [0, 14687837971.138248, 77031671.96839908, 0.6, 0.8, -0.17142857142857143, 9.999999999999999e-05, 
# 3257072175.589282, 0.0, 11430765795.548965, 4.2666125287527645, 68383152.81415418, 10.0, 
# array([0.41889359]), array([5.]), array([5.])]


Data_Struc = {'structure_switch': True, 'alpha': 0.6, 'beta': 0.8,
              'delta': -0.17142857142857143, 'Lres0': 201648867747.70163,
              't_10list': np.array([6.50681839e-05, 1.65068184e-04]),
              'r_10list': np.array([0.23790232, 0.41889369]),
              'P_10list': np.array([1.96998160e+08, 7.28499874e+07]),
              'T_10list': np.array([67741779.55773313]), 'Lw': 201648867747.70163,
              'vterminal': 3810.2196532385897, 'r0': 0.4188936946067258,
              't0': 0.00016506818386985737, 'E0': 15649519.367987147,
              'T0': 67741779.55773313, 'dt_L': 9.999999999999999e-05,
              'temp_counter': 1, 'dMdt_factor': 1.646, 'Qi': 1.6994584609226492e+67,
              'mypath': '/Users/jwt/Documents/Code/warpfield3/outputs/'}


Cool_Struc = np.load('/Users/jwt/Documents/Code/warpfield3/outputs/cool.npy', allow_pickle = True).item()

warpfield_params = {'model_name': 'example', 
                    'out_dir': 'def_dir', 
                    'verbose': 1.0, 
                    'output_format': 'ASCII', 
                    'rand_input': 0.0, 
                    'log_mCloud': 6.0, 
                    'mCloud_beforeSF': 1.0, 
                    'sfe': 0.01, 
                    'nCore': 1000.0, 
                    'rCore': 0.099, 
                    'metallicity': 1.0, 
                    'stochastic_sampling': 0.0, 
                    'n_trials': 1.0, 
                    'rand_log_mCloud': ['5', ' 7.47'], 
                    'rand_sfe': ['0.01', ' 0.10'], 
                    'rand_n_cloud': ['100.', ' 1000.'], 
                    'rand_metallicity': ['0.15', ' 1'], 
                    'mult_exp': 0.0, 
                    'r_coll': 1.0, 
                    'mult_SF': 1.0, 
                    'sfe_tff': 0.01, 
                    'imf': 'kroupa.imf', 
                    'stellar_tracks': 'geneva', 
                    'dens_profile': 'bE_prof', 
                    'dens_g_bE': 14.1, 
                    'dens_a_pL': -2.0, 
                    'dens_navg_pL': 170.0, 
                    'frag_enabled': 0.0, 
                    'frag_r_min': 0.1, 
                    'frag_grav': 0.0, 
                    'frag_grav_coeff': 0.67, 
                    'frag_RTinstab': 0.0, 
                    'frag_densInhom': 0.0, 
                    'frag_cf': 1.0, 
                    'frag_enable_timescale': 1.0, 
                    'stop_n_diss': 1.0, 
                    'stop_t_diss': 1.0, 
                    'stop_r': 1000.0, 
                    'stop_t': 15.05, 
                    'stop_t_unit': 'Myr', 
                    'write_main': 1.0, 
                    'write_stellar_prop': 0.0, 
                    'write_bubble': 0.0, 
                    'write_bubble_CLOUDY': 0.0, 
                    'write_shell': 0.0, 
                    'xi_Tb': 0.9,
                    'inc_grav': 1.0, 
                    'f_Mcold_W': 0.0, 
                    'f_Mcold_SN': 0.0, 
                    'v_SN': 1000000000.0, 
                    'sigma0': 1.5e-21, 
                    'z_nodust': 0.05, 
                    'mu_n': 2.1287915392418182e-24, 
                    'mu_p': 1.0181176926808696e-24, 
                    't_ion': 10000.0, 
                    't_neu': 100.0, 
                    'nISM': 0.1, 
                    'kappa_IR': 4.0, 
                    'gamma_adia': 1.6666666666666667, 
                    'thermcoeff_wind': 1.0, 
                    'thermcoeff_SN': 1.0,
                    'alpha_B': 2.59e-13,
                    'gamma_mag': 1.3333333333333333,
                    'log_BMW': -4.3125,
                    'log_nMW': 2.065,
                    'c_therm': 1.2e-6,
                    }


class Dict2Class(object):
    # set object attribute
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)
            
# initialise the class
warpfield_params = Dict2Class(warpfield_params)

get_bubbleParams.initialise_bstruc(990000000, 0.01, '/Users/jwt/Documents/Code/warpfield3/outputs')

a = get_bubbleStructure(Data_Struc,
                        Cool_Struc,
                        warpfield_params,
                        fit_len=5, fit_len_short = 5,
        )


print(a)


