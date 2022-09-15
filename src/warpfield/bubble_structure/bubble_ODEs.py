#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 18:23:43 2022

@author: Jia Wei Teh

This script contains ODEs which dictates the strucuture (e.g., temperature, 
 velocity) of the bubble. 
"""

import numpy as np
import astropy.constants as c
import astropy.units as u
#--
import src.warpfield.cooling.get_coolingFunction as get_coolingFunction

def delta2dTdt(t, T, delta):
    """
    See Pg 79, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf.
    
    Parameters
    ----------
    t : float
        time.
    T : float
        Temperature at xi = r/R2.

    Returns
    -------
    dTdt : float
    """
    dTdt = (T/t) * delta

    return dTdt


def dTdt2delta(t, T, dTdt):
    """
    See Pg 79, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf.
    
    Parameters
    ----------
    t : float
        time.
    T : float
        DESCRIPTION.

    Returns
    -------
    delta : float
    """
    
    delta = (t/T) * dTdt
    
    return delta



def beta2Edot(bubble_P, bubble_E, 
              r1, r2, beta,
              t_now, pwdot, pwdotdot,
              r2dot,
              ):
    """
    see pg 80, A12 https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf 

    Parameters
    ----------
    bubble_P : float
        Bubble pressure.
    bubble_E : float
        Bubble energy.
    r1 : float
        Inner bubble radius.
    r2 : float
        Outer bubble radius.
    beta : float
        dbubble_P/dt.
    t_now : float
        time.
    pwdot : float
        dPw/dt.
    pwdotdot : float
        dPw/dt/dt.
    r2dot : float
        Outer bubble velocity.

    Returns
    -------
    bubble_Edot : float
        dE/dt.

    """
    # dp/dt pressure 
    pdot = - bubble_P * beta / t_now
    # define terms
    a = np.sqrt(pwdot/2)
    b = 1.5 * a**2 * r1
    d = r2**3 - r1**3
    adot = 0.25 * pwdotdot / a
    e = b / ( b + bubble_E )
    # main equation
    bubble_Edot = (2 * np.pi * pdot * d**2 + 3 * bubble_E * r2dot * r2**2 * (1 - e) -\
                    3 * adot / a * r1**3 * bubble_E**2 / (bubble_E + b)) / (d * (1 - e))
    # return 
    return bubble_Edot
    

def Edot2beta(bubble_P, bubble_E, 
              r1, r2, bubble_Edot,
              t_now, pwdot, pwdotdot,
              r2dot,
              ):
    """
    see pg 80, A12 https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf 
    
    Parameters
    ----------
    bubble_P : float
        Bubble pressure.
    bubble_E : float
        Bubble energy.
    r1 : float
        Inner bubble radius.
    r2 : float
        Outer bubble radius.
    bubble_Edot : float
        dE/dt.
    t_now : float
        time.
    pwdot : float
        dPw/dt.
    pwdotdot : float
        dPw/dt/dt.
    r2dot : float
        Outer bubble velocity.

    Returns
    -------
    beta : float
        dbubble_P/dt.

    """
    # define terms
    a = np.sqrt(pwdot/2)
    b = 1.5 * a**2 * r1
    d = r2**3 - r1**3
    adot = 0.25 * pwdotdot / a
    e = b / ( b + bubble_E ) 
    # main equation
    pdot = 1 / (2 * np.pi * d**2 ) *\
        ( d * (1 - e) * bubble_Edot - 3 * bubble_E * r2dot * r2**2 * (1 - e) + 3 * adot / a * r1**3 * bubble_E**2 / (bubble_E + b))
    beta = - pdot * t_now / bubble_P
    # return
    return beta


def get_bubbleODEs(r, y0, data_struc, metallicity):
    """
    system of ODEs for bubble structure (see Weaver+77, eqs. 42 and 43)
    :param x: velocity v, temperature T, spatial derivate of temperature dT/dr
    :param r: radius from center
    :param cons: constants
    :return: spatial derivatives of v,T,dTdr
    """
    
    # Note:
    # old code: bubble_struct()
    
    # unravel
    a = data_struc["cons"]["a"]
    b = data_struc["cons"]["b"]
    C = data_struc["cons"]["c"]
    d = data_struc["cons"]["d"]
    e = data_struc["cons"]["e"]
    Qi = data_struc["cons"]["Qi"]
    Cool_Struc = data_struc["Cool_Struc"]
    v, T, dTdr = y0

    # boltzmann constant in astronomical units 
    k_B = c.k_B.cgs.value * u.g.to(u.Msun) * u.cm.to(u.pc)**2 / u.s.to(u.Myr)**2

    Qi = Qi / u.Myr.to(u.s)
    ndens = d / (2. * k_B * T) /(u.pc.to(u.cm)**3)
    Phi = Qi / (4. * np.pi * (r*u.pc.to(u.cm)) ** 2)

    # interpolation range (currently repeated in calc_Lb --> merge?)
    log_T_interd = 0.1
    log_T_noeqmin = Cool_Struc["log_T"]["min"]+1.0001*log_T_interd
    log_T_noeqmax = Cool_Struc["log_T"]["max"] - 1.0001 * log_T_interd
    log_T_intermin = log_T_noeqmin - log_T_interd
    log_T_intermax = log_T_noeqmax + log_T_interd

    #debug (use semi-correct cooling at low T)
    if T < 10.**3.61:
        T = 10.**3.61

    # loss (or gain) of internal energy
    dudt = get_coolingFunction.cool_interp_master({"n":ndens, "T":T, "Phi":Phi}, Cool_Struc, metallicity,
                                       log_T_noeqmin=log_T_noeqmin, log_T_noeqmax=log_T_noeqmax, 
                                       log_T_intermin=log_T_intermin, log_T_intermax=log_T_intermax)
    

    vd = b + (v-a*r)*dTdr/T - 2.*v/r
    Td = dTdr
    # negative sign for dudt term (because of definition of dudt)
    dTdrd = C/(T**2.5) * (e + 2.5*(v-a*r)*dTdr/T - dudt/d) - 2.5*dTdr**2./T - 2.*dTdr/r 
    # return
    return [vd,Td,dTdrd]









