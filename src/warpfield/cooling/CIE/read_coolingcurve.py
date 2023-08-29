#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 16:09:46 2022

@author: Jia Wei Teh

This script contains functions which compute the cooling function Lambda, given T.

old code: cool.py
"""
# libraries
import numpy as np
import sys
import scipy
import astropy.constants as c
import astropy.units as u
#--
import src.warpfield.functions.operations as operations
# get parameter
from src.input_tools import get_param
warpfield_params = get_param.get_param()


# This section summarizes the available CIE cooling curves
# 




# This is the simple case when CIE is achieved, so Lambda depends only on T. 


# TODO: add for non-solar metallicity 


def get_Lambda(T):
    """
    This function calculates Lambda assuming CIE conditions.

    Parameters
    ----------
    T : float/array
        Temperature.

    Available libraries (specified in .param file) include:
        1: CLOUDY cooling curve for HII region, solar metallicity.
        2: CLOUDY cooling curve for HII region, solar metallicity. 
            Includes the evaporative (sublimation) cooling of icy interstellar 
            grains (occurs e.g., when heated by cosmic-ray particle)
        
        3. Gnat and Ferland 2012 (slightly interpolated for values)
        4. Sutherland and Dopita 1993, for [Fe/H] = -1].
    
    These files are by default stored in path/to/warpfield/lib/cooling_tables/CIE/current/.

    Returns
    -------
    Lambda [erg/s * cm3]: float.
        Cooling.
    These values are from the file directly:
        logT: temperature (log).
        logLambda: Lambda-values (log).

    """
    
    # TODO: If the run is not solar metallicity, it is not implemented yet.
    
    # Might be a problem here because this does not support extrapolation. If
    # this happens, implement a function that does that.
    
    if warpfield_params.metallicity != 1:
        sys.exit('Need to implement non-solar metallicity.')
    # get path to library
    # See example_pl.param for more information.
    path2cooling = warpfield_params.path_cooling_CIE
    # unpack from file
    logT, logLambda = np.loadtxt(path2cooling, unpack = True)
    # create interpolation
    cooling_CIE_interpolation = scipy.interpolate.interp1d(logT, logLambda, kind = 'linear')
    # change temperature to log for interpolation
    T = np.log10(T)
    # find lambda
    Lambda = 10**(cooling_CIE_interpolation(T))

    return Lambda, logT, logLambda



#%%


def Interp3_dudt(point, Cool_Struc, element = "Netcool"):
    """
    Interpolates cooling function which depends on density, temperature, and photon number flux (ionizing)
    This is the main routine to call every time you request a cooling value for some parameter tuple
    :param point: structure (not log), containing number density "n", temperature "T", and photon number flux (ionizing) "Phi"
    :param Cool_Struc: see output of get_Cool_dat()
    :return: energy (net cooling/heating) rate du/dt, i.e. ne*np*(Lambda - Gamma)
    """

    # point at which to interpolate (not log)
    x = point["n"]
    y = point["T"]
    z = point["Phi"]

    # got to log (necessary to find corners of surrounding cuboid as distance between points in constant in log only)
    log_x = np.log10(x)
    log_y = np.log10(y)
    log_z = np.log10(z)

    # unpack tabulated data
    my_element = Cool_Struc[element]
    ln_dat = Cool_Struc["log_n"]
    lT_dat = Cool_Struc["log_T"]
    lP_dat = Cool_Struc["log_Phi"]

    # find indices of cuboid in which "point" lies
    ii_n_0 = int((log_x-ln_dat["min"])/ln_dat["d"])
    jj_T_0 = int((log_y-lT_dat["min"])/lT_dat["d"])
    kk_P_0 = int((log_z-lP_dat["min"])/lP_dat["d"])

    ii_n_1 = ii_n_0 + 1
    jj_T_1 = jj_T_0 + 1
    kk_P_1 = kk_P_0 + 1

    # to have true linear interpolation go to linear space instead of log-space
    x0 = 10. ** ln_dat["dat"][ii_n_0]
    x1 = 10. ** ln_dat["dat"][ii_n_1]

    y0 = 10. ** lT_dat["dat"][jj_T_0]
    y1 = 10. ** lT_dat["dat"][jj_T_1]

    z0 = 10. ** lP_dat["dat"][kk_P_0]
    z1 = 10. ** lP_dat["dat"][kk_P_1]

    def trilinear(x, X0, X1, data):
        """
        trilinear interpolation inside a cuboid
        need to provide function values at corners of cuboid, i.e. 8 values
        :param x: coordinates of point at which to interpolate (array or list with 3 elements: x, y, z)
        :param X0: coordinates of lower gridpoint (array or list with 3 elements: x0, y0, z0)
        :param X1: coordinates of upper gridpoint (array or list with 3 elements: x1, y1, z1)
        :param data: function values at all 8 gridpoints of cube (3x2 array)
        :return: interpolated value at (x, y, z)
        """
    
        xd = (x[0] - X0[0]) / (X1[0] - X0[0])
        yd = (x[1] - X0[1]) / (X1[1] - X0[1])
        zd = (x[2] - X0[2]) / (X1[2] - X0[2])
    
        c00 = data[0, 0, 0] * (1. - xd) + data[1, 0, 0] * xd
        c01 = data[0, 0, 1] * (1. - xd) + data[1, 0, 1] * xd
        c10 = data[0, 1, 0] * (1. - xd) + data[1, 1, 0] * xd
        c11 = data[0, 1, 1] * (1. - xd) + data[1, 1, 1] * xd
        
        c0 = c00*(1.-yd) + c10*yd
        c1 = c01*(1.-yd) + c11*yd
        
        c = c0*(1.-zd) + c1*zd
        
        return c

    # call interpolator
    dudt = trilinear([x, y, z], [x0, y0, z0], [x1, y1, z1],
                         my_element[ii_n_0:ii_n_0 + 2, jj_T_0:jj_T_0 + 2, kk_P_0:kk_P_0 + 2])

    print('\n\nto interpolate', 
          '\nx:', x, 
          '\ny:', y, 
          '\nz', z, 
          '\nsecond three',
          'x0:', x0, 
          '\ny0:', y0, 
          '\nz0', z0, 
          '\nthird three',
          '\nx1:', x1, 
          '\ny1:', y1, 
          '\nz1', z1,     
          '\ngridpoints',
          '\narray', my_element[ii_n_0:ii_n_0 + 2, jj_T_0:jj_T_0 + 2, kk_P_0:kk_P_0 + 2],
          )

    print('value', dudt)
    sys.exit()
    
    return dudt






def cool_interp_master(point, Cool_Struc, metallicity, log_T_intermin = 3.9, log_T_noeqmin = 4.0, log_T_noeqmax = 5.4, log_T_intermax=5.499):
    
    def linear(x, X, Y):
        """
        linear interpolation
        :param x: scalar, must lie between X[0] and X[1]
        :param X: list or array with 2 elements, X[0], X[1]
        :param Y: list or array with 2 elements, function values at X[0] and X[1]
        :return:
        """
        
        if (x > max(X)) or (x < min(X)):
            sys.exit("Cannot interpolate, x is not in range [X0, X1]")
        else:
            y = Y[0] + (x-X[0]) * (Y[1]-Y[0])/(X[1]-X[0])
        
        return y

    # THese if/else cases seem to be for T range for when to/not to use CIE cooling curves?
    
    if (np.log10(point["T"]) > log_T_intermax) or (np.log10(point["T"]) < log_T_intermin):
        Lambda = get_coolingFunction(point["T"], metallicity)
        dudt = -1. * (point["n"]) ** 2 *  Lambda / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))

    elif (np.log10(point["T"]) >= log_T_noeqmax):
        dudt1 = -1. * (point["n"]) ** 2 * get_coolingFunction(point["T"], metallicity) / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))
        dudt0 = -1. * Interp3_dudt({"n": point["n"], "T": point["T"], "Phi": point["Phi"]}, Cool_Struc) / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))
        dudt = linear(np.log10(point["T"]), [log_T_noeqmax, log_T_intermax], [dudt0, dudt1])

    elif (np.log10(point["T"]) <= log_T_noeqmin):
        dudt0 = -1. * (point["n"]) ** 2 * get_coolingFunction(point["T"], metallicity) / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))
        dudt1 = -1. * Interp3_dudt({"n": point["n"], "T": point["T"], "Phi": point["Phi"]}, Cool_Struc) / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))
        dudt = linear(np.log10(point["T"]), [log_T_intermin, log_T_noeqmin], [dudt0, dudt1])

    else:
        dudt = -1. * Interp3_dudt({"n": point["n"], "T": point["T"], "Phi": point["Phi"]}, Cool_Struc) / (c.M_sun.cgs.value / (c.pc.cgs.value* u.Myr.to(u.s)**3))

    return dudt


