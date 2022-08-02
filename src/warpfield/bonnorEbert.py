#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:04:46 2022

@author: Jia Wei Teh

This script contains helper functions to aid bonner-ebert sphere related 
calculations. See density_profile.py and mass_profile.py for more.
"""

import numpy as np
import scipy.integrate
import astropy.constants as c
import astropy.units as u

def laneEmden(y,xi):
    """
    This function specifics the Lane-Emden equation. This will then be fed
    into scipy.integrate.odeint() to be solved.
    """
    # E.g., see https://iopscience.iop.org/article/10.3847/1538-4357/abfdc8 Eq 4
    # Rearranging and let omega = dpsi/dxi, let y = [ psi, omega ],
    # we arrive at the following expression for dy/dxi
    # Syntax according to https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    psi, omega = y
    dydxi = [
        omega, 
        np.exp(-psi) - 2 * omega / xi
        ]
    
    return dydxi


def massIntegral(xi, rhoCore, c_s):
    """
    A function that outputs an expression (integral) to integrate to obtain
    the mass profile M(r).

    Parameters
    ----------
    xi : a list of xi
        xi is dimensionless radius, where:
            xi = (4 * pi * G * rho_core / c_s^2)^(0.5) * r
    rho_core : float
        The core density (in kg/m3)
    c_s : float
        Sound speed.

    Returns
    -------
    The expression for integral of M(r).

    """
    
    # An array for the range of xi for solving
    xi_arr = np.linspace(1e-12, xi, 200)
    # initial condition (set to a value that is very close to zero)
    y0 = [1e-12, 1e-12]
    # integrate the ODE to get values for psi and omega.
    psi, omega = zip(*scipy.integrate.odeint(laneEmden, y0, xi_arr))
    psi = np.array(psi)
    omega = np.array(omega)
    # choose one value to reduce the integral to non psi-dependent and 1-D.
    psipoint = psi[-1] 
    # See Eq33 http://astro1.physics.utoledo.edu/~megeath/ph6820/lecture6_ph6820.pdf
    A = 4 * np.pi * rhoCore * (c_s**2 / (4 * np.pi * c.G.value * rhoCore))**(3/2)
    # return the integral
    return A*np.exp(-psipoint)*xi**2

def get_bE_soundspeed(T, mu, gamma):
    """
    This function computes the isothermal soundspeed, c_s, given temperature
    T and mean molecular weight mu.

    Parameters
    ----------
    T : float (Units: K)
        Temperature of the gas.
    mu : float (Units: kg) 
        Mean molecular weight of the gas. Watch out for the units.
    gamma: float 
        Adiabatic index of gas. 

    Returns
    -------
    The isothermal soundspeed c_s (Units: m/s)

    """
    # return
    return np.sqrt(gamma * c.k_B.value * T / mu )
    

def get_bE_rCloud(rhoCore, T, mCloud):
    
    # SI units
    rhoCore = rhoCore * (u.Msun.to(u.kg))/((u.pc**3).to(u.m**3))
    mCloud = mCloud * u.Msun.to(u.kg)
    # 
    
    
    
    
    return


def get_BE_nEdge():
    
    
    
    return






#%%


import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import astropy.constants as c
import astropy.units as u


xi = 6.451
xi_arr = np.linspace(1e-12, xi, 200)
# initial condition (set to a value that is very close to zero)
y0 = [1e-12, 1e-12]
psi, omega = zip(*scipy.integrate.odeint(laneEmden, y0, xi_arr))
psi = np.array(psi)
omega = np.array(omega)

fig = plt.subplots(1, 1, figsize = (7, 5), dpi = 200)
plt.vlines(6.45, 0, 1, linestyles = '--', 
           colors = 'k',
           label = '$\\xi_{max} = 6.45$')
plt.ylim(0, 1)
plt.xlim(0, xi)
plt.plot(xi_arr, np.exp(psi * -1), 
         label = '$e^{-\\psi(\\xi)}$',
         )
plt.xlabel('$\\xi(r)$')
plt.legend()

#%%














#%%

# def FindRCBE(n0, T, mCloud, plint=True):
#     """
#     :param n0: core density (namb) 1/cm3
#     :param T: temperature of BEsphere K
#     :param M: cloud mass in solar masses
#     :return: cloud radius (Rc) in pc and density at Rc in 1/ccm
#     """
    
#     #t=np.linspace(10**(-5),2*(10**(9)),num=80)*c.pcSI
#     mCloud = mCloud* c.MsunSI
#     rho_0= n0*i.muiSI*10**6
#     cs= aux.sound_speed_BE(T)
#     #zet=t*(((4*np.pi*c.GravSI*rho_0)/(cs**2))**(1/2))
#     def Root(zet,rho_0,cs,Mcloud):
#         return quad(MassIntegrate3,0,zet,args=(rho_0,cs))[0]-Mcloud
#     #h=0
#     #print(Mcloud,rho_0,cs)
#     #while h < len(t):
#     # These are results after many calculations
#     sol = optimize.root_scalar(Root,args=(rho_0,cs,mCloud),bracket=[8.530955303346797e-07, 170619106.06693593], method='brentq')
#     zetsol=sol.root
#     rsol=zetsol/((((4*np.pi*c.GravSI*rho_0)/(cs**2))**(1/2)))
#     b=rsol
#     rs=rsol/c.pcSI
    
    
#     zeta=b*(((4*np.pi*c.GravSI*rho_0)/(cs**2))**(1/2))
#     w=np.linspace(0.0001*10**(-9),zeta,endpoint=True,num=int(100000/500))
#     y0=[0.0001*10**(-9), 0.0001*10**(-9)]
#     sol = odeint(laneEmden, y0, w)
#     psipoints=sol[:, 0][len(sol[:, 0])-1]
#     nedge=n0*np.exp(-psipoints)
    
#     if plint == True:
        
#         rhoavg=(3*mCloud)/(4*np.pi*(b**3))
#         navg=rhoavg/(i.muiSI*10**6)
#         print('Cloud radius in pc=',rs)
#         print('nedge in 1/ccm=',nedge)
#         print('g after=',n0/nedge)
#         print('navg=',navg)
    
#     return rs,nedge





