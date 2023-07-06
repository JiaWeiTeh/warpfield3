#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:13:28 2023

@author: Jia Wei Teh
"""

import init as i
import scipy.optimize
#--
import src.warpfield.phase_general.phase_events as phase_events
import src.warpfield.phase_general.phase_ODEs as phase_ODEs

#t0 = 2.19152965 # 1.34556241
#Rsh0 = 6.19831886e+01 # 4.72084451e+01
#vsh0 = 1.61810127e+01 # 1.90207451e+01
#Eb0 = 1.06767282e+10 # 7.23428832e+09
#T0 = 7.94396273e+06 # 8.59222090e+06
#cs = 494.24920085

def run_phase_transition(t0, y0, cs, ODEpar, SB99f):
    """
    solves fE_trans
    :param t0:
    :param y0: [Rsh0, vsh0, Eb0, T0]
    :param cs:
    :param ODEpar:
    :param SB99f:
    :return:
    """

    Rsh0 = y0[0]
    Eb0 = y0[2]
    y0[3] = 1e4 # set temperature to some value (will not change any more)

    # set dE/dt (constant value) which will be used during this phase
    tSCR = Rsh0 / cs
    ODEpar['dEdt'] = -Eb0 / tSCR

    tmin = t0
    tmax = ODEpar['tStop']

    # define ODE which shall be solved
    ODE_fun = lambda t, y: fE_trans(t, y, ODEpar, SB99f, Eb0)

    # list of events which will cause the solver to terminate
    event_fun1 = lambda t, y: phase_events.event_StopTime(t, y, ODEpar['tStop']); event_fun1.terminal = True
    event_fun2 = lambda t, y: phase_events.event_Radius1(t, y); event_fun2.terminal = True; event_fun2.direction = -1.0
    event_fun3 = lambda t, y: phase_events.event_Radius1000(t, y); event_fun3.terminal = True
    event_fun4 = lambda t, y: phase_events.event_EnergyZero(t, y); event_fun4.terminal = True
    event_fun7 = lambda t, y: phase_events.event_vel0(t, y); event_fun7.terminal = False; event_fun7.direction = -1.0  # no termination! and make sure velocity turns from positive to negative to capture onset of collapse
    event_fun8 = lambda t, y: phase_events.event_dissolution(t, y, ODEpar); event_fun8.terminal = True

    event_fun_list = [event_fun1, event_fun2, event_fun3, event_fun4, event_fun7, event_fun8]
    
    print('tscr/10',tSCR/10.)

    # call ODE solver
    psoln = scipy.integrate.solve_ivp(ODE_fun, [tmin, tmax], y0, method='LSODA', events=event_fun_list, max_step=tSCR/10.,min_step=10**(-7)) # make sure we have at least 10 time steps

    return psoln



def fE_trans(t, y, ODEpar, SB99f, Eb0):
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

    #print "Rsh_max:", ODEpar['Rsh_max']

    ########################## ODEs: acceleration and velocity ###############################
    if i.frag_cover == False:
        part1_dict = phase_ODEs.fE_tot_part1(t, y, ODEpar, SB99f)
    else: 
        part1_dict = phase_ODEs.fE_tot_part1(t, y, ODEpar, SB99f,Eb0,cfs=True)
        
    vd = part1_dict['vd'] # acceleration
    rd = v # velocity
    ##########################################################################################

    R1 = part1_dict['R1']
    # aux.printl(('t:', '%.4e'%t, 'R1:', '%.4e'%R1, 'R2:', '%.4e'%r,'v:', '%.4e'%v, 'E:', '%.4e'%E,'T:', '%.4e'%T), verbose=1)
    print('t:', '%.4e'%t, 'R1:', '%.4e'%R1, 'R2:', '%.4e'%r,'v:', '%.4e'%v, 'E:', '%.4e'%E,'T:', '%.4e'%T)

    Ed = ODEpar['dEdt']
    Td = 0.

    derivs = [rd, vd, Ed, Td]  # list of dy/dt=f functions
    return derivs






"""
import matplotlib.pyplot as plt

t = psoln.t
r = psoln.y[0]
v = psoln.y[1]
E = psoln.y[2]
T = psoln.y[3]

print psoln

R1 = np.zeros(len(psoln.t))

for ii in range(0,len(psoln.t)):
    y = [psoln.y[0][ii], psoln.y[1][ii], psoln.y[2][ii], psoln.y[3][ii]]
    part1_dict = ODE_tot_aux.fE_tot_part1(psoln.t[ii], y, ODEpar)
    R1[ii] = part1_dict['R1']

f, ax = plt.subplots(3, sharex=True)
ax[0].semilogy(t,r)
ax[0].semilogy(t,R1,'r')
ax[0].set_ylabel("R (pc)")

ax[1].plot(t,v)
ax[1].set_yscale('symlog')
ax[1].set_ylabel("v (km/s)")

ax[2].semilogy(t,E)
ax[2].set_ylabel("E (a. u.)")
ax[2].set_xlabel("t (Myr)")


plt.show()
"""