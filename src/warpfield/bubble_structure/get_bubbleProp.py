#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:36:10 2022

@author: Jia Wei Teh

This script contains useful functions that help compute properties
of the bubble, including bubble pressure and energy.
"""


import numpy as np
import astropy.constants as c
import astropy.units as u


def bubble_E2P(bubble_E, r1, r2, gamma):
    """
    This function relates bubble energy to buble pressure (all in cgs)

    Parameters
    ----------
    bubble_E : float
        Bubble energy.
    r1 : float
        Inner radius of bubble (outer radius of wind cavity).
    r2 : float
        Outer radius of bubble (inner radius of ionised shell).
    gamma : float
        Adiabatic index.

    Returns
    -------
    bubble_P : float
        Bubble pressure.

    """
    # TODO: Check if it is in cgs
    
    # Avoid division by zero
    r2 = r2 + (1e-10)
    # pressure, see https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf
    # pg71 Eq 6.
    bubble_P = (gamma - 1) * bubble_E / (r2**3 - r1**3) / (4 * np.pi / 3)
    # return
    return bubble_P
    
def bubble_P2E(bubble_P, r1, r2, gamma):
    """
    This function relates bubble pressure to buble energy (all in cgs)

    Parameters
    ----------
    bubble_P : float
        Bubble pressure.
    r1 : float
        Inner radius of bubble (outer radius of wind cavity).
    r2 : float
        Outer radius of bubble (inner radius of ionised shell).
    gamma : float
        Adiabatic index.

    Returns
    -------
    bubble_E : float
        Bubble energy.

    """
    # see bubble_E2P()
    return 4 * np.pi / 3 / (gamma - 1) * (r2**3 - r1**3)

def pRam(r, Lwind, vWind):
    """
    This function calculates the ram pressure.

    Parameters
    ----------
    r : float
        Radius of outer edge of bubble.
    Lwind : float
        Mechanical wind luminosity.
    vWind : float
        terminal velocity of wind.

    Returns
    -------
    Ram pressure
    """
    return Lwind / (2 * np.pi * r**2 * vWind)

def delta2dTdt(t, T, delta):
    """
    Parameters
    ----------
    t : float
        time.
    T : float
        Temperature at xi.

    Returns
    -------
    dTdt : float
        See Pg9, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf.
    """
    dTdt = (T/t) * delta

    return dTdt


def dTdt2delta(t, T, dTdt):
    """
    Parameters
    ----------
    t : float
        DESCRIPTION.
    T : float
        DESCRIPTION.

    Returns
    -------
    delta : float
        See Pg9, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf.
    """
    
    delta = (t/T) * dTdt
    
    return delta



def beta2Edot(bubble_P, bubbleE, 
              r1, r2, beta,
              pdot, pwdot,
              v2,
              ):
    
    
    
    
    
    
    return


def Edot2beta():
    
    
    
    
    
    return




#%%

def beta_to_Edot(Pb, R1, beta, my_params):
    """
    converts beta to dE/dt
    :param Pb: pressure of bubble
    :param R1: inner radius of bubble
    :param beta: -(t/Pb)*(dPb/dt), see Weaver+77, eq. 40
    :param my_params:
    :return:
    """
    R2 = my_params['R2']
    v2 = my_params["v2"]
    E = my_params['Eb']
    Pdot = -Pb*beta/my_params["t_now"]

    pwdot = my_params['pwdot'] # pwdot = 2.*Lw/vw

    A = np.sqrt(pwdot/2.)
    A2 = A**2
    C = 1.5*A2*R1
    D = R2**3 - R1**3
    #Adot = (my_params['Lw_dot']*vw - Lw*my_params['vw_dot'])/(2.*A*vw**2)
    Adot = 0.25*my_params['pwdot_dot']/A

    F = C / (C + E)

    #Edot = ( 3.*v2 * R2**2 * E + 2.*np.pi*Pdot*D**2 ) / D # does not take into account R1dot
    #Edot = ( 2.*np.pi*Pdot*D**2 + 3.*E*v2*R2**2 * (1.-F) ) / (D * (1.-F)) # takes into account R1dot but not time derivative of A
    Edot = ( 2.*np.pi*Pdot*D**2 + 3.*E*v2*R2**2 * (1.-F) - 3.*(Adot/A)*R1**3*E**2/(E+C) ) / (D * (1.-F)) # takes everything into account

    #print "term1", "%.5e"%(2.*np.pi*Pdot*D**2), "term2", "%.5e"%(3.*E*v2*R2**2 * (1.-F)), "term3", "%.5e"%(3.*(Adot/A)*R1**3*E**2/(E+C))

    #print "Edot", "%.5e"%Edot, "%.5e"%Edot_exact

    return Edot

def Edot_to_beta(Pb, R1, Edot, my_params):
    """
    converts Edot to beta (inverse function of beta_to_Edot)
    :param Pb: pressure of bubble
    :param R1: inner radius of bubble
    :param Edot: time derivative of bubble energy
    :param my_params:
    :return:
    """

    R2 = my_params['R2']
    v2 = my_params["v2"]
    E = my_params['Eb']

    pwdot = my_params['pwdot']  # pwdot = 2.*Lw/vw

    A = np.sqrt(pwdot / 2.)
    A2 = A ** 2
    C = 1.5 * A2 * R1
    D = R2 ** 3 - R1 ** 3
    # Adot = (my_params['Lw_dot']*vw - Lw*my_params['vw_dot'])/(2.*A*vw**2)
    Adot = 0.25 * my_params['pwdot_dot'] / A

    F = C / (C + E)

    Pdot = 1./(2.*np.pi*D**2.) * ((D * (1.-F)) * Edot - 3.*E*v2*R2**2 * (1.-F) + 3.*(Adot/A)*R1**3*E**2/(E+C) )
    beta = -Pdot*my_params["t_now"]/Pb

    return beta