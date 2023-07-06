import sys
import numpy as np
import constants as c
import init as i
from bubble_structure import R1_zero
import state_eq
import mass_profile

import scipy.optimize


sigmaD = i.sigmaD; clight = c.clight; pi =  np.pi; alphaB = c.alphaB; kboltz = c.kboltz; pc = c.pc; Msun = c.Msun; Lsun = c.Lsun; mp = c.mp; gamma = c.gamma; Myr = c.Myr; yr = c.yr; clight_au = c.clight_au;

# shell expansion equations


def f_weaver_grav_core(y, t, params):
    """Not used"""
    # general Weaver phase including gravity, for constant density profile
    # parameter:
    # LW : mechanical luminosity
    # GAM : adiabatic index
    # M0T : core mass
    # RHOA : core density
    # RCORE : core radius
    # A_EXP : exponent of density profile

    r, rd, rd2 = y      # unpack current values of y (r, rdot, rdotdot)
    LW, GAM, M0T, RHOA, RCORE, A_EXP, MSTAR  = params  # unpack parameters
    # since this ODE is for the core, M0T (the core mass) is not used

    exp3 = 3. + A_EXP
    exp2 = 2. + A_EXP
    exp1 = 1. + A_EXP

    G3 = 3.*(GAM - 1.)
    G2 = 3.*GAM - 2.

    # mass and mass derivatives for constant density profile
    Msh = 4./3. * pi * RHOA * r**3.
    Msh_dot = 4. * pi * RHOA * r**2. * rd
    Msh_dotdot = 4. * pi * RHOA * (2.*r*rd**2. + r**2.*rd2)

    #GRAV = c.Grav_au*i.Weav_grav
    GRAV = 0. # if you don't like gravity, just set the gravitational constant to 0

    # gravity correction
    term_grav = (GRAV/r**2.) * ( (rd/r)*(6.*GAM-7.)*(MSTAR+0.5*Msh)  +  Msh_dot*(1.+MSTAR/Msh) )

    # third time derivative of R
    rd3 = 1./(Msh*r) * (LW*G3 - Msh_dot*rd**2.*G2)  -  1./r * rd*rd2*G2  -  1./Msh * (Msh_dotdot*rd + 2.*Msh_dot*rd2) - term_grav

    derivs = [rd, rd2, rd3]      # list of dy/dt=f functions
    return derivs

def f_weaver_grav_general(y, t, params):
    """Not used"""
    # general Weaver phase including gravity (any power law density profile allowed)
    # parameter:
    # LW : mechanical luminosity
    # GAM : adiabatic index
    # M0T : core mass
    # RHOA : core density
    # RCORE : core radius
    # A_EXP : exponent of density profile

    r, rd, rd2 = y      # unpack current values of y (r, rdot, rdotdot)
    LW, GAM, M0T, RHOA, RCORE, A_EXP, MSTAR  = params  # unpack parameters

    exp3 = 3. + A_EXP
    exp2 = 2. + A_EXP
    exp1 = 1. + A_EXP

    G3 = 3.*(GAM - 1.)
    G2 = 3.*GAM - 2.

    # mass and mass derivatives for power law density profile
    Msh = M0T + 4. * pi * RHOA * (r**exp3 - RCORE**exp3) / (exp3 * RCORE**A_EXP)
    Msh_dot = 4. * pi * RHOA * r**exp2 * rd / (RCORE**A_EXP)
    Msh_dotdot = (4. * pi * RHOA / (RCORE**A_EXP)) * (exp2 * r**exp1 * rd**2. + r**exp2 * rd2)

    #GRAV = c.Grav_au*i.Weav_grav
    GRAV = 0. # if you don't like gravity, just set the gravitational constant to 0

    # gravity correction
    term_grav = (GRAV/r**2.) * ( (rd/r)*(6.*GAM-7.)*(MSTAR+0.5*Msh)  +  Msh_dot*(1.+MSTAR/Msh) )

    # third time derivative of R
    rd3 = 1./(Msh*r) * (LW*G3 - Msh_dot*rd**2.*G2)  -  1./r * rd*rd2*G2  -  1./Msh * (Msh_dotdot*rd + 2.*Msh_dot*rd2) - term_grav

    derivs = [rd, rd2, rd3]      # list of dy/dt=f functions
    return derivs


def f_mom(y, t, params):
    """Not used"""
    # winds (momentum-driven), radiation pressure and gravity
    # sweep up material
    # be aware that the time units are Myr!
    r, rd = y      # unpack current values of y (r, rdot)
    SG_mom, MT_mom, GSS_mom, WN_mom, RP_mom, RPIR_mom  = params  # unpack parameters

    # if r is very small and collapsing, set everything to 0.0
    #if r<=1e-2 and rd <= 0.0:
    #    rd = 0.0
    #    rdd = 0.0
    #else:
    rdd = (RP_mom+WN_mom) / r**3 +  (GSS_mom) / r**2  +  SG_mom * r  +  MT_mom * rd**2/r  +  RPIR_mom/r**3

    derivs = [rd, rdd]      # list of dy/dt=f functions
    return derivs

def f_kimmom(y, t, params):
    # winds (momentum-driven), radiation pressure and gravity
    # sweep up material but with lower density
    # be aware that the time units are Myr, Msun, pc!
    r, rd = y      # unpack current values of y (r, rdot)
    M0T, RHOA, LBOL_ABS, TAU_IR, PW, MSTAR = params  # unpack parameters

    Msh = M0T+4.*pi/3.*RHOA*r**3
    Msh_dot = 4.*pi*RHOA*r**2*rd

    rdd = 1./Msh * (LBOL_ABS/clight_au *(1.+TAU_IR) + PW - Msh_dot*rd) - (c.Grav_au/r**2) * (MSTAR+Msh/2.0)

    derivs = [rd, rdd]      # list of dy/dt=f functions
    return derivs

def f_mom_grad(y, t, params):
    # can be used for slopes or constant profiles

    r, rd = y  # unpack current values of y (r, rdot, rdotdot)
    M0, RHOA, LBOL_ABS, TAU_IR, PW, MSTAR, RCORE, A_EXP, RCLOUD, MCLOUD, SFE = params  # unpack parameters

    # calculate swept mass and time dedivative
    if i.dens_profile == "powerlaw":
        Msh = mass_profile.calc_mass(r, RCORE, RCLOUD, RHOA, i.rho_intercl_au, A_EXP, MCLOUD)[0]
        Msh_dot = mass_profile.calc_mass_dot(r, rd, RCORE, RCLOUD, RHOA, i.rho_intercl_au, A_EXP)[0]
    elif i.dens_profile == "BonnorEbert":
        T_BE=RCORE
        Msh = mass_profile.calc_mass_BE(r, RHOA, T_BE , i.rho_intercl_au, RCLOUD, MCLOUD)
        Msh_dot = mass_profile.calc_mass_dot_BE(r, rd, RHOA, T_BE , i.rho_intercl_au, RCLOUD)
        
    # if shell collapsed a bit before but is now expanding again, it is expanding into emptiness
    if M0 > Msh:
        Msh = M0
        Msh_dot = 0.0

    rdd = 1./Msh * (LBOL_ABS/clight_au*(1.+TAU_IR) + PW - Msh_dot*rd) - (c.Grav_au/r**2) * (MSTAR + Msh/2.0)

    derivs = [rd, rdd]  # list of dy/dt=f functions
    return derivs

def f_collapse(y,t,params):
    # use for collapse of shell
    # when mass of the shell is constant (Mdot_shell = 0)
    # be aware that the time units are Myr, Msun, pc!
    r, rd = y
    M0, LBOL_ABS, TAU_IR, PW, MSTAR = params

    rdd = 1./M0 * ( (LBOL_ABS/clight_au)*(1.+TAU_IR) + PW ) - c.Grav_au/r**2.*(MSTAR+M0/2.)

    derivs = [rd, rdd]      # list of dy/dt=f functions
    return derivs

def calc_ionpress(r, rcore, rcloud, alpha, rhoa):
    """
    calculates pressure from photoionized part of cloud at radius r
    by default assume units (Msun, Myr, pc) but in order to use cgs, just change mykboltz
    :param r: radius
    :param rcore: core radius (only important if density slope is used)
    :param rcloud: cloud radius (outside of rcloud, density slope)
    :param alpha: exponent of density slope: rho = rhoa*(r/rcore)**alpha, alpha is usually zero or negative
    :param rhoa: core density
    :param mykboltz: by default assume astro units (Myr, Msun, pc)
    :return: pressure of ionized gas outside shell
    """
    print('this is the input for calc_ionpress')
    # 0.23790232199299727 451690.2638133162 355.8658723191992 -2 31.394226159698523
    print(r, rcore, rcloud, alpha, rhoa)

    if r < rcore:
        rho_r = rhoa
    elif ((r >= rcore) and (r < rcloud)):
        rho_r = rhoa * (r/rcore)**alpha
    else:
        rho_r = i.rho_intercl_au


    n_r = rho_r/(i.mua/c.Msun) # n_r: total number density of particles (H+, He++, electrons)

    P_ion = n_r*c.kboltz_au*i.Ti
    print('n_r, kboltz_au, P_ion')
    # 6.140217543021121e+58 7.233880792172324e-60 4441.760174422003
    print(n_r, c.kboltz_au, P_ion)

    return P_ion

def calc_coveringf(t,tFRAG,ts):
    """
    estimate covering fraction cf (assume that after fragmentation, during 1 sound crossing time cf goes from 1 to 0)
    if the shell covers the whole sphere: cf = 1
    if there is no shell: cf = 0
    :param t:
    :param tFRAG:
    :param ts:
    :return:
    """
    cfmin = i.cfmin

    cf = 1. - ((t - tFRAG) / ts)**1.
    cf[cf>1.0] = 1.0
    cf[cf<cfmin] = cfmin

    #if (t < tFRAG):
    #    cf = 1.
    #elif (t > tFRAG + ts):
    #    cf = 0.
    #    # print cf # debugging
    #else:
    #    cf = 1. - (t - tFRAG) / ts
    #    # print cf  # debugging
    return cf

def calc_Lleak(r,E,t,tFRAG, cs):
    """
    calculate luminosity leaking out of the bubble due to shell fragmentation
    TO DO: do not use constant sound speed but make it temperature dependent!
    :param r: radius of bubble (i.e. shell radius R2)
    :param E: internal energy of bubble
    :param t: time
    :param tFRAG: time when fragmentation occured
    :return:
    L_leak: energy per time flowing out of the bubble
    """


    # sound speed
    # cs = some formula (what temperature to take?? average?)

    ts = r/cs # sound crossing time

    # calculate covering fraction of the shell (1 if full coverage, 0 if no shell)
    cf = calc_coveringf(t, tFRAG, ts)

    # energy leakage
    L_leak = E*cs/r * (1.-cf)

    return L_leak

def fE_gen(y, t, params):
    """
    general energy-driven phase including stellar winds, gravity, power law density profiles, cooling, radiation pressure
    :param y: [r,v,E]: shell radius (R2), shell velocity (v2), bubble energy (Eb)
    :param t: time (since the ODE is autonomous, t does not appear. The ODE solver still expects it though)
    :param params: (see below)
    :return: time derivative of y, i.e. [rd, vd, Ed]
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

    r, v, E = y  # unpack current values of y (r, rdot, E)
    LW, PWDOT, GAM, MCLOUD, RHOA, RCORE, A_EXP, MSTAR, LB, FRAD, FABSi, RCLOUD, PHASE, tSF, tFRAG, tSCR, CS, SFE  = params  # unpack parameters

    # energy should not be negative!
    #if E < 0.0:
    #    sys.exit("Energy is negative in ODEs.py")

    VW = 2.*LW/PWDOT

    # calculate shell mass and time derivative of shell mass
    if i.dens_profile == "powerlaw":
        Msh = mass_profile.calc_mass(r,RCORE,RCLOUD,RHOA,i.rho_intercl_au,A_EXP,MCLOUD)[0]
        Msh_dot = mass_profile.calc_mass_dot(r,v,RCORE,RCLOUD,RHOA,i.rho_intercl_au,A_EXP)[0]
        #print('r in pc=',r,'log10(Msh) in solar masses',np.log10(Msh),'Mdot=',Msh_dot,'R_c in pc=',RCLOUD)
    elif i.dens_profile == "BonnorEbert":
        T_BE=RCORE
        Msh = mass_profile.calc_mass_BE(r, RHOA, T_BE , i.rho_intercl_au, RCLOUD, MCLOUD)
        Msh_dot = mass_profile.calc_mass_dot_BE(r, v, RHOA, T_BE , i.rho_intercl_au, RCLOUD)
        #print('r in pc=',r,'log10(Msh) in solar masses',np.log10(Msh),'Mdot=',Msh_dot,'R_c in pc=',RCLOUD)
        
    # print("We are now in ODEs to check for the values of Msh")
    # print('Msh',Msh)
    # print('Msh_dot',Msh_dot)
    # sys.exit()

    # print("(r, RHOA, T_BE , i.rho_intercl_au, RCLOUD, MCLOUD)", r, RHOA, T_BE , i.rho_intercl_au, RCLOUD, MCLOUD)
    # print("Msh, Msh_dot", Msh, Msh_dot)
    # calc inward pressure from photoionized gas outside the shell (is zero if no ionizing radiation escapes the shell)
    if FABSi < 1.0:
        PHII = calc_ionpress(r, RCORE, RCLOUD, A_EXP, RHOA)
    else:
        PHII = 0.0
    #PHII = 0. #debugging
    #FRAD = 0. #debugging
    #LB = 0. #debugging
    print("PHII", PHII)

    # gravity correction (self-gravity and gravity between shell and star cluster)
    GRAV = c.Grav_au * i.Weav_grav  # if you don't want gravity, set Weav_grav to 0 in myconfig.py
    Fgrav = GRAV*Msh/r**2 * (MSTAR + Msh/2.)
    
    print("GRAV, Fgrav", GRAV, Fgrav)

    # get pressure from energy
    # radius of inner discontinuity
    R1 = scipy.optimize.brentq(R1_zero, 0.0, r, args=([LW, E, VW, r]))
    
    print('R1', R1)
    
    # the following if-clause needs to be rethought. for now, this prevents negative energies at very early times
    # IDEA: move R1 gradually outwards
    tmin = i.dt_switchon
    if (t > tmin + tSF):
        # equation of state
        Pb = state_eq.PfromE(E,r,R1,gamma=GAM)
    elif (t <= tmin + tSF):
        R1_tmp = (t-tSF)/tmin * R1
        Pb = state_eq.PfromE(E, r, R1_tmp, gamma=GAM)
    #else: #case pure momentum driving
    #    # ram pressure from winds
    #    Pb = state_eq.Pram(r,LW,VW)

    print('Pb, ', Pb)


    # calculate covering fraction
    cf = calc_coveringf(np.array([t]),tFRAG,tSCR)
    if hasattr(cf, "__len__"): cf = cf[0] # transform to float if necessary

    if cf < 1.:
        L_leak = (1. - cf) * 4. * np.pi * r ** 2 * Pb * CS / (GAM - 1.)
    else:
        L_leak = 0.

    # time derivatives
    rd = v
    vd = (4.*np.pi*r**2.*(Pb-PHII) - Msh_dot*v - Fgrav + FRAD)/Msh
    Ed = (LW - LB) - (4.*np.pi*r**2.*Pb) * v - L_leak # factor cf for debugging

    #print LW, LB, 4.*np.pi*r**2.*Pb*v

    derivs = [rd, vd, Ed]      # list of dy/dt=f functions
    print("derivs", derivs)
    # [3656.200432285518, -142187312.2269407, 25710057105.69037]
    sys.exit()
    return derivs
