#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:53:30 2023

@author: Jia Wei Teh
"""
# libraries
import scipy.optimize
#--
from src.warpfield.phase_general import phase_events, phase_ODEs
# get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()

#t0 = 3.44561744 # 2.31693843
#Rsh0 = 7.67264049e+01 # 6.39478626e+01
#vsh0 = 5.91226042e+00 # 1.48466383e+01

def run_fE_momentum(t0, y0, ODEpar, SB99f):
    """
    solves fE_momentum
    :param t0:
    :param y0:
    :param ODEpar:
    :param SB99f:
    :return:
    """

    tmin = t0
    tmax = ODEpar['tStop']

    # define ODE which shall be solved
    ODE_fun = lambda t, y: fE_momentum(t, y, ODEpar, SB99f)

    # list of events which will cause the solver to terminate
    event_fun1 = lambda t, y: phase_events.event_StopTime(t, y, ODEpar['tStop']); event_fun1.terminal = True
    event_fun2 = lambda t, y: phase_events.event_Radius1(t, y); event_fun2.terminal = True; event_fun2.direction = -1.0
    event_fun3 = lambda t, y: phase_events.event_Radius1000(t, y); event_fun3.terminal = True
    event_fun7 = lambda t, y: phase_events.event_vel0(t, y); event_fun7.terminal = False; event_fun7.direction = -1.0  # no termination! and make sure velocity turns from positive to negative to capture onset of collapse
    event_fun8 = lambda t, y: phase_events.event_dissolution(t, y, ODEpar); event_fun8.terminal = True

    event_fun_list = [event_fun1, event_fun2, event_fun3, event_fun7, event_fun8]

    # call ODE solver
    psoln = scipy.integrate.solve_ivp(ODE_fun, [tmin, tmax], y0, method='LSODA', events=event_fun_list, max_step=0.1,min_step=10**(-7))

    return psoln


def fE_momentum(t, y, ODEpar, SB99f):
    """
    general energy-driven phase including stellar winds, gravity, power law density profiles, cooling, radiation pressure
    :param y: [r,v,E]: shell radius (R2), shell velocity (v2), bubble energy (Eb)
    :param t: time (since the ODE is autonomous, t does not appear. The ODE solver still expects it though)
    :param params: (see below)
    :return: time derivative of y, i.e. [rd, vd, Ed, Td]
    # parameters:
    # LW : mechanical luminosity
    # GAM : adiabatic index
    # M0T : core mass
    # RHOA : core density
    # RCORE : core radius
    # A_EXP : exponent of density profile
    # LB: luminosity lost to cooling (calculate from bubble structure)
    # FRAD: radiation pressure coupled to the shell, i.e. Lbol/c * fabs (calculate fabs from shell structure)
    # PHASE: current phase (core, gradient, ambient, collapse)
    """

    r, v, E, T = y  # unpack current values of y (r, rdot, E, T)

    # during this phase, we are actually not solving for E and T any more
    # but ODE_tot_aux.fE_tot_part1 still expects values
    ytemp = r, v, -1., 0.

    ########################## ODEs: acceleration and velocity ###############################
    if warpfield_params.frag_enabled == False:
        part1_dict = phase_ODEs.fE_tot_part1(t, ytemp, ODEpar, SB99f)
    else: 
        part1_dict = phase_ODEs.fE_tot_part1(t, ytemp, ODEpar, SB99f,10**10,cfs=True)
    vd = part1_dict['vd'] # acceleration
    rd = v # velocity
    ##########################################################################################

    print('t:', '%.4e' % t, 'R1:', '%.4e' % part1_dict['R1'], 'R2:', '%.4e' % r, 'v:', '%.4e' % v)


    derivs = [rd, vd, 0., 0.]  # list of dy/dt=f functions
    return derivs






