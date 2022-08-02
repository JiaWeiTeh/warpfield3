#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 19:36:19 2022

@author: Jia Wei Teh

This script test bonnor-ebert related properties.
"""

import numpy as np
import scipy
from scipy import optimize
from scipy import integrate
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
    The expression for the integral of M(r).

    """
    
    # An array for the range of xi for solving
    xi_arr = np.linspace(0.0001e-9, xi, 200)
    # initial condition (set to a value that is very close to zero)
    y0 = [0.0001e-9, 0.0001e-9]
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

def get_bE_soundspeed(T, mu = 2.1287915392418182e-24):
    """
    This function computes the isothermal soundspeed, c_s, given temperature
    T and mean molecular weight mu.

    Parameters
    ----------
    T : float
        Temperature of the gas.
    mu : float
        Mean molecular weight of the gas.

    Returns
    -------
    The isothermal soundspeed c_s.

    """
    
    return np.sqrt(c.R.value * T / mu)
    

def get_bE_r_dens(rhoCore, T, mCloud):
    
    
    
    rhoCore = rhoCore * (u.Msun.to(u.kg))/((u.pc**3).to(u.m**3))
    mCloud = mCloud * u.Msun.to(u.kg)
    
    
    
    return


MsunSI = 1.989e30 
pcSI=3.085677581e16 # pc in m
muiSI = 2.125362090909091e-27

params_dict = {'model_name': 'example',
             'out_dir': 'def_dir',
             'verbose': '1',
             'output_format': 'ASCII',
             'rand_input': '0',
             'log_mCloud': '6.0',
             'mCloud_beforeSF': '1',
             'sfe': '0.01',
             'n_cloud': '1000',
             'metallicity': '0.15',
             'stochastic_sampling': '0',
             'n_trials': '1',
             'rand_log_mCloud': ['5', ' 7.47'],
             'rand_sfe': ['0.01', '0.10'],
             'rand_n_cloud': ['100.', ' 1000.'],
             'rand_metallicity': ['0.15', ' 1'],
             'mult_exp': '0',
             'r_coll': '1.0',
             'mult_SF': '1',
             'sfe_tff': '0.01',
             'imf': 'kroupa.imf',
             'stellar_tracks': 'geneva',
             'dens_nCore': '1000',
             'dens_profile': 'bE_prof',
             'dens_g_bE': '14.1',
             'dens_a_pL': '-2',
             'dens_navg_pL': '170',
             'dens_rcore': '0.099',
             'frag_enabled': '0',
             'frag_r_min': '0.1',
             'frag_grav': '0',
             'frag_grav_coeff': '0.67',
             'frag_RTinstab': '0',
             'frag_densInhom': '0',
             'frag_cf': '1',
             'frag_enable_timescale': '1',
             'stop_n_diss': '1',
             'stop_t_diss': '1.0',
             'stop_r': '1e3',
             'stop_t': '15.05',
             'stop_t_unit': 'Myr',
             'write_main': '1',
             'write_stellar_prop': '0',
             'write_bubble': '0',
             'inc_grav': '1',
             'f_Mcold_W': '0.0',
             'f_Mcold_SN': '0.0',
             'v_SN': '1e9',
             'sigma0': '1.5e-21',
             'z_nodust': '0.05',
             'u_n': '2.1287915392418182e-24',
             'u_p': '1.0181176926808696e-24',
             't_ion': '1e4',
             't_neu': '100',
             'n_ISM': '0.1',
             'kappa_IR': '4',
             'thermcoeff_wind': '1.0',
             'thermcoeff_SN': '1.0'}

n0 = float(params_dict['dens_nCore'])
mCloud = 10**float(params_dict['log_mCloud'])
g = float(params_dict['dens_g_bE'])

a = autoT(mCloud, n0, g)



def Root(T,M,ncore,nend):
    rs, nedge = FindRCBE(ncore, T, M, plint=False)
    # so that root_scalar can solve for nedge - nend = 0, in other words
    # solve for x such that nedge(x)  = nend
    return nedge - nend
    


FindRCBE(n0, )


#%%

def autoT(M,ncore,g):
    # TOASK: What are the T, g params?
    # T is basically for sound speed for the equation of state. 
    # g = BonnerEbert param
    nend=ncore/g
    sol = scipy.optimize.root_scalar(Root,args=(M,ncore,nend),bracket=[8.530955303346797e-07, 170619106.06693593], method='brentq')
    Tsol=sol.root
    
    print('Tsol = ', Tsol)
    return Tsol


def FindRCBE(n0, T, mCloud, plint=True):
    """
    :param n0: core density (namb) 1/cm3
    :param T: temperature of BEsphere K
    :param M: cloud mass in solar masses
    :return: cloud radius (Rc) in pc and density at Rc in 1/ccm
    """
    
    #t=np.linspace(10**(-5),2*(10**(9)),num=80)*c.pcSI
    mCloud = mCloud*MsunSI
    rho_0= n0*muiSI*10**6
    cs= get_bE_soundspeed(T)
    #zet=t*(((4*np.pi*c.GravSI*rho_0)/(cs**2))**(1/2))
    def Root(zet,rho_0,cs,Mcloud):
        return scipy.integrate.quad(massIntegral,0,zet,args=(rho_0,cs))[0]-Mcloud
    #h=0
    #print(Mcloud,rho_0,cs)
    #while h < len(t):
    # These are results after many calculations
    sol = scipy.optimize.root_scalar(Root,args=(rho_0,cs,mCloud),bracket=[8.530955303346797e-07, 170619106.06693593], method='brentq')
    zetsol=sol.root
    rsol=zetsol/((((4*np.pi*c.GravSI*rho_0)/(cs**2))**(1/2)))
    b=rsol
    rs=rsol/pcSI
    
    
    zeta=b*(((4*np.pi*c.GravSI*rho_0)/(cs**2))**(1/2))
    w=np.linspace(0.0001*10**(-9),zeta,endpoint=True,num=int(100000/500))
    y0=[0.0001*10**(-9), 0.0001*10**(-9)]
    sol = scipy.integrate.odeint(laneEmden, y0, w)
    psipoints=sol[:, 0][len(sol[:, 0])-1]
    nedge=n0*np.exp(-psipoints)
    
    # if plint == True:
        
    #     rhoavg=(3*mCloud)/(4*np.pi*(b**3))
    #     navg=rhoavg/(i.muiSI*10**6)
    #     print('Cloud radius in pc=',rs)
    #     print('nedge in 1/ccm=',nedge)
    #     print('g after=',n0/nedge)
    #     print('navg=',navg)
    
    return rs,nedge






#%%

mui = 2.1287915392418182e-24
n = 1000
M = 1e7*2e30
nalpha = -2


# cloud radius
rcloud = (3./(4.*np.pi) * M/(n*mui))**(1./3.)

def F(x):
    myzero = n*mui/3. * rcloud**3. - n0*mui*( (1./3.-1./(3.+nalpha))*x**3. + 1./(3.+nalpha)*x**(-nalpha)*rcloud**(3.+nalpha) )
    return myzero

# core radius
rcore = scipy.optimize.brentq(F, 0., 1e70) # cgs


    
    
    



