#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:06:39 2023

@author: Jia Wei Teh

This script contains functions that will help reading in Starburst99 data.
"""

import numpy as np
import sys
import os
import collections
import warnings
import pathlib
import scipy
#--
from src.warpfield.functions import operations

# assume this is the structure of the ods-file:
# time(yr), log10(Qi), log10(Li/Lbol), log10(Lbol), log10(Lw w/ SNe), log10(pw_dot), log10(Lw w/o SNe)

# input:
#   file: path to file that contains all necessary feedback paramters as a function of time
#   f_mass: normalize to different cluster mass
#         e.g. if you are reading in a file for a 1e6 Msol cluster but want to simulate a 1e5 cluster, set f_mass = 1e5/1e6 = 0.1
#   f_Zism: normalize to reference metallicity. Only wind output is affected by this scaling
#         e.g. If using 0.7 solar Z, the code will use stellar tracks for closest reference Z and then scale winds accordingly, i.e. with f_Zism = Z/Z_reference

# output:
#   t: time in Myr
#   Qi: rate of ionizing photons (1/s)
#   Li: luminosity of ionizing radiation (erg/s)
#   Ln: luminosity of non-ionizing radiation (erg/s)
#   Lbol: Li+Ln
#   Lw: mechanical wind luminosity
#   vw: wind velocity (cm/s)
#   Mw_dot: mass loss rate due to winds (g/s)
#   pdot_SNe: momentum flux of SN ejecta (SN mass loss rate times 1e4 cm/s)

# TODO1: create getSB99 data and interp functions
#  TODO: also check all functions are included by going meticulously through the original function.

def read_SB99(warpfield_params,
              f_mass = 1e6, return_format = "array"):
    
    # Note:
        # old code: getSB99_main() and load_stellar_tracks()
        # metallicity was Zism
    
    """
    get starburst99 data and corresponding interpolation functions
    :param Zism: metallicity (in solar units)
    :param rotation: boolean
    :param f_mass: mass of cluster (in solar masses)
    :param BHcutoff: cut off mass for direct collapse black holes in solar masses (stars above this mass, will not inject energy via supernova explosions)
    :return: array (of SB99 data), dictionary (containing interpolation functions)
    """
    metallicity = warpfield_params.metallicity 
    
    # SB99_data = load_stellar_tracks(Zism, rotation=rotation, f_mass=f_mass, BHcutoff=BHcutoff)    
    # force_file = i.force_SB99file, test_plot = False, log_t = False, tmax=30., return_format="array"):
       
    # First get highest and lowest metallicity to interpolate 
    # low metallicity
    SB99_file_Z0002 = get_SB99_filename(0.15, warpfield_params.SB99_rotation, 
                                        warpfield_params.SB99_BHCUT, 
                                        warpfield_params.SB99_mass) 
    # high metallicity
    SB99_file_Z0014 = get_SB99_filename(1.0, warpfield_params.SB99_rotation, 
                                        warpfield_params.SB99_BHCUT, 
                                        warpfield_params.SB99_mass) 

    # case: specific file is forced
    if warpfield_params.SB99_forcefile != 0 and isinstance(warpfield_params.SB99_forcefile, str):  
        SB99file = warpfield_params.SB99_forcefile
        [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = getSB99_data(SB99file, warpfield_params, f_mass)
    # if file is not forced, then check metallicity and get ready to interpolate. 
    else:
        # if metallicity is between, interpolate
        if (metallicity >= 0.14 and metallicity < 1.0):
            [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = getSB99_data_interp(metallicity,
                                                                                            SB99_file_Z0002, 0.15,
                                                                                            SB99_file_Z0014, 1.0,
                                                                                            warpfield_params,
                                                                                            f_mass)
        elif metallicity == 1.0:
            SB99file = SB99_file_Z0014
            [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = getSB99_data(SB99file, warpfield_params, f_mass)


        elif (metallicity >= 0.14 and metallicity <= 0.15 ):
            SB99file = SB99_file_Z0002
            [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = getSB99_data(SB99file, warpfield_params, f_mass)

        else:
            warnings.warn("Your metallicity is either too high or too low - there are no stellar tracks! WARPFIELD will choose the closest track and scale the winds linearly; however, please be cautious of the output.")
            if metallicity < 0.15: 
                SB99file = SB99_file_Z0002
                metallicity_rel = 0.15
            elif metallicity > 1.0:
                SB99file = SB99_file_Z0014
                metallicity_rel = 1.0
            print(("Using the following SB99 file: "+SB99file))
            [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = getSB99_data(SB99file, warpfield_params, f_mass, 
                                                                                                     f_met=metallicity/metallicity_rel)

    Data = [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo]

    if return_format == 'dict':
        Data_out = arr2dict(Data)
    elif return_format == 'array':
        Data_out = dict2arr(Data)

    return Data_out



def getSB99_data(file, warpfield_params, 
                 f_mass=1.0, f_met=1.0):
    """
    subroutine for load_stellar_tracks.py
    :param file: file to read
    :param f_mass: mass scaling (as default: in units of 1e6 Msol)
    :param f_met: metallicity scale factor (NB: only use a number different from 1.0 when you are scaling down or up an SB99 file with a different metallicity to the one you set in myconfig. 
                                            E.g. If you want Zism = 0.15 and a SB99 file with that metallicity exists, f_met must be 1.0. 
                                            If you want to run Zism = 0.15 and want to scale down a SB99 file with Z = 0.3, set f_met = 0.5)
    :return:
    """
    # aux.printl("getSB99_data: mass scaling f_mass = %.3f" %(f_mass), verbose=verbose)

    #data_dict = get_data(file)
    #data_list = data_dict['Sheet1']
    #data = np.array(data_list)

    data = get_SB99_file(file)

    t = data[:,0]/1e6 # in Myr
    # all other quantities are in cgs
    Qi = 10.0**data[:,1] *f_mass # emission rate of ionizing photons (number per second)
    fi = 10**data[:,2] # fraction of ionizing radiation
    Lbol = 10**data[:,3] *f_mass # bolometric luminosity (erg/s)
    Li = fi*Lbol # luminosity in the ionizing part of the spectrum (>13.6 eV)
    Ln = (1.0-fi)*Lbol # luminosity in the non-ionizing part of the spectrum (<13.6 eV)

    #get mechanical luminosity of SNe before scaling wind luminosity according to metallicity and other factors:
    pdot_W_tmp = 10**data[:, 5] * f_mass  # momentum rate for winds before scale factors considered (other than mass scaling)
    Lmech_tmp = 10**data[:,4] * f_mass # mechanical luminosity of winds and SNe
    Lmech_W_tmp = 10 ** data[:, 6] * f_mass # only wind
    Lmech_SN_tmp = Lmech_tmp - Lmech_W_tmp # only SNe

    
    def getMdotv(pdot,Lmech):
        """
        calculate mass loss rate Mdot and terminal velocity v from momentum injection rate pdot and mechanical luminosity Lmech
        :param pdot: momentum injection rate
        :param Lmech: mechanical luminosity
        :return: mass loss rate, terminal velocity
        """
    
        Mdot = pdot**2/(2.*Lmech)
        v = 2.*Lmech/pdot
    
        return Mdot, v
    
    def getpdotLmech(Mdot,v):
        """
        calculate momentum injection rate and mechanical luminosity from mass loss rate and terminal velocity
        :param Mdot: mass loss rate
        :param v: terminal velocity
        :return: momentum injection rate, mechanical luminosity
        """
    
        pdot = Mdot * v
        Lmech = 0.5 * Mdot * v**2.0
    
        return pdot, Lmech

    # winds
    Mdot_W, v_W = getMdotv(pdot_W_tmp, Lmech_W_tmp) # convert pdot and Lmech to mass loss rate and terminal velocity
    Mdot_W *= f_met * (1. + warpfield_params.f_Mcold_W) # modify mass injection rate according to 1) metallicity and 2) cold mass content in cluster (NB: metallicity affects mainly the mass loss rate, not the terminal velocity)
    v_W *= np.sqrt(warpfield_params.thermcoeff_wind / (1. + warpfield_params.f_Mcold_W)) # modifiy terminal velocity according to 1) thermal efficiency and 2) cold mass content in cluster
    pdot_W, Lmech_W = getpdotLmech(Mdot_W, v_W)

    # supernovae
    v_SN = warpfield_params.v_SN # assuming a constant ejecta velocity (which is not quite right, TO DO: get time-dependent velocity, e.g. when mass of ejecta are known)
    Mdot_SN = 2.* Lmech_SN_tmp/v_SN**2
    Mdot_SN *= (1. + warpfield_params.f_Mcold_SN) # # modify mass injection rate according to 1) cold mass content in cluster during SN explosions, do not modify according to metallicity
    v_SN *= np.sqrt(warpfield_params.thermcoeff_SN / (1. + warpfield_params.f_Mcold_SN)) # modifiy terminal velocity according to 1) thermal efficiency and 2) cold mass content in cluster during SN explosions
    pdot_SN, Lmech_SN = getpdotLmech(Mdot_SN, v_SN)

    # add mechanical energy and momentum injection rate, respectively, from winds and supernovae
    Lmech= Lmech_W + Lmech_SN
    pdot = pdot_W + pdot_SN

    # insert 1 element at t=0
    t = np.insert(t, 0, 0.0)
    Qi = np.insert(Qi, 0, Qi[0])
    Li = np.insert(Li, 0, Li[0])
    Ln = np.insert(Ln, 0, Ln[0])
    Lbol = np.insert(Lbol, 0, Lbol[0])
    Lmech= np.insert(Lmech, 0, Lmech[0])
    pdot = np.insert(pdot, 0, pdot[0])
    pdot_SN = np.insert(pdot_SN, 0, pdot_SN[0])

    # test plot (only use for debugging)
    # if test_plot: testplot(t, Qi, Li, Ln, Lbol, Lmech, pdot, pdot_SN, log_t=log_t, t_max=tmax, ylim=ylim)

    return[t,Qi,Li,Ln,Lbol,Lmech,pdot,pdot_SN]



def getSB99_data_interp(Zism, file1, Zfile1, file2, Zfile2, warpfield_params, f_mass = 1.0):
    """
    interpolate metallicities from SB99 data
    :param Zism: metallicity you want (between metallicity 1 and metallicity 2)
    :param file1: path to file for tracks with metallicity 1
    :param Zfile1: metallicity 1
    :param file2: path to file for tracks with metallicity 2
    :param Zfile: metallicity 2
    :return:
    """

    # let's ensure that index 1 belongs to the lower metallicity
    if Zfile1 < Zfile2:
        [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = getSB99_data(file1, warpfield_params, f_mass = f_mass)
        [t2, Qi2, Li2, Ln2, Lbol2, Lw2, pdot2, pdot_SNe2] = getSB99_data(file2, warpfield_params, f_mass = f_mass)
        Z1 = Zfile1
        Z2 = Zfile2
    elif Zfile1 > Zfile2:
        [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = getSB99_data(file2, f_mass = f_mass)
        [t2, Qi2, Li2, Ln2, Lbol2, Lw2, pdot2, pdot_SNe2] = getSB99_data(file1, f_mass = f_mass)
        Z1 = Zfile2
        Z2 = Zfile1

    tend1 = t1[-1]
    tend2 = t2[-2]

    tend = np.min([tend1, tend2])

    # cut to same length

    Qi1 = Qi1[t1 <= tend]
    Li1 = Li1[t1 <= tend]
    Ln1 = Ln1[t1 <= tend]
    Lbol1 = Lbol1[t1 <= tend]
    Lw1 = Lw1[t1 <= tend]
    pdot1 = pdot1[t1 <= tend]
    pdot_SNe1 = pdot_SNe1[t1 <= tend]

    Qi2 = Qi2[t2 <= tend]
    Li2 = Li2[t2 <= tend]
    Ln2 = Ln2[t2 <= tend]
    Lbol2 = Lbol2[t2 <= tend]
    Lw2 = Lw2[t2 <= tend]
    pdot2 = pdot2[t2 <= tend]
    pdot_SNe2 = pdot_SNe2[t2 <= tend]

    t1 = t1[t1 <= tend]
    t2 = t2[t2 <= tend]

    if not all(t1 == t2):
        print("FATAL: files do not have the same time vectors")
        sys.exit("Exiting: SB99 files time arrays do not match!")

    t = t1
    # Linear interpolation
    Qi = (Qi1 * (Z2 - Zism) + Qi2 * (Zism - Z1)) / (Z2 - Z1)
    Li = (Li1 * (Z2 - Zism) + Li2 * (Zism - Z1)) / (Z2 - Z1)
    Ln = (Ln1 * (Z2 - Zism) + Ln2 * (Zism - Z1)) / (Z2 - Z1)
    Lbol = (Lbol1 * (Z2 - Zism) + Lbol2 * (Zism - Z1)) / (Z2 - Z1)
    Lw = (Lw1 * (Z2 - Zism) + Lw2 * (Zism - Z1)) / (Z2 - Z1)
    pdot = (pdot1 * (Z2 - Zism) + pdot2 * (Zism - Z1)) / (Z2 - Z1)
    pdot_SNe = (pdot_SNe1 * (Z2 - Zism) + pdot_SNe2 * (Zism - Z1)) / (Z2 - Z1)

    return[t,Qi,Li,Ln,Lbol,Lw,pdot,pdot_SNe]


def get_SB99_filename(metallicity, rotation, BHcutoff, SB99_mass = 1e6, Mmax = 120, txt = True):
    """
    Parameters
    ----------
    # See the example.param for more details, under SB99 section.
    metallicity : 
        metallicity in solar units.
    rotation : 
        stars rotating? (boolean)
    BHcutoff : 
        upper mass limit for black holes (usually 120).
    SB99_mass : 
        mass of starburst99 cluster. The default is 1e6.
    Mmax : 
        upper mass limit of IMF. The default is 120.
    txt : 
        should the returned file have the ending ".txt"?. The default is True.

    Returns
    -------
    SB99_file : string
        The resulting filename that is used for the query.

    """
    # Note:
        # old code: warp_nameparser.get_SB99_filename()
            
    # String for mass, e.g., "1e6"
    log_SB99_mass = np.log10(SB99_mass)
    SB99_mass_str = str(int(SB99_mass/10.**np.floor(log_SB99_mass))) + "e" + str(int(log_SB99_mass))
    # String for rotation
    if rotation is True:
        rot_str = "rot"
    else:
        rot_str = "norot"
    # String for metallicity
    if metallicity == 0.15:
        Z_str = "Z0002"
    elif metallicity == 1.0:
        Z_str = "Z0014"
    # String for blackhole cutoff mass
    BHcutoff_str = "BH" + str(int(BHcutoff))
    # Finale filename
    SB99_file = SB99_mass_str + "cluster_" + rot_str + "_" + Z_str + "_" + BHcutoff_str
    # only append Mmax if Mmax is not 120
    if Mmax != 120:
        SB99_file += "_Mmax" + str(Mmax)
    # filetype?
    if txt is True:
        SB99_file = SB99_file + ".txt"
    # return
    return SB99_file


def get_SB99_file(filename):
    # if it is a filename directly
    if os.path.isfile(filename):
        return np.loadtxt(filename)
    # If not, return to WARPFIELD directory
    warpfield3 = str(pathlib.Path(__file__).parent.parent.parent.parent.absolute())
    # check if we are in the right directory
    if warpfield3[-10:] == "warpfield3":
        path2sb99 = warpfield3 + '/lib/sps/starburst99/' + filename
        # if we are, try loading it
        try:
            return np.loadtxt(path2sb99)
        # if not, we try brute forcing it
        except Exception:
            for root, dirs, files in os.walk(warpfield3):
                if filename in files:
                    path2sb99 = os.path.join(root, filename)
                    return np.loadtxt(path2sb99)
    # if not found
    return sys.exit(f'starburst99 file: {filename} not found in warpfield3/lib/sps/starburst99/. Please ensure the file exists.')


def dict2arr(SB99_data):
    """
    Turns an array into a dictionary
    """
    # Note:
        # old code: dict2arr()
    # check whether SB99_data is a dictionary
    if isinstance(SB99_data, collections.Mapping):
        t_Myr = SB99_data['t_Myr']
        Qi_cgs = SB99_data['Qi_cgs']
        Li_cgs = SB99_data['Li_cgs']
        Ln_cgs = SB99_data['Ln_cgs']
        Lbol_cgs = SB99_data['Lbol_cgs']
        Lw_cgs = SB99_data['Lw_cgs']
        pdot_cgs = SB99_data['pdot_cgs']
        pdot_SNe_cgs = SB99_data['pdot_SNe_cgs']
        SB99_data = [t_Myr, Qi_cgs, Li_cgs, Ln_cgs, Lbol_cgs, Lw_cgs, pdot_cgs, pdot_SNe_cgs]
    return SB99_data
    

def arr2dict(SB99_data):
    """
    Turns an array into a dictionary
    """
    # Note:
        # old code: arr2dict()
    if not isinstance(SB99_data, collections.Mapping):
        [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = SB99_data
        SB99_data = {'t_Myr': t_evo, 'Qi_cgs': Qi_evo, 'Li_cgs': Li_evo, 'Ln_cgs': Ln_evo, 'Lbol_cgs': Lbol_evo, 'Lw_cgs': Lw_evo, 'pdot_cgs': pdot_evo, 'pdot_SNe_cgs': pdot_SNe_evo}
    return SB99_data


def make_interpfunc(SB99_data_IN):
    """
    get starburst99 interpolation functions
    :param SB99_data: array (of SB99 data) or dictionary
    :return: dictionary (containing interpolation functions)
    """

    # convert to an array
    SB99_data = dict2arr(SB99_data_IN)
    [t_Myr, Qi_cgs, Li_cgs, Ln_cgs, Lbol_cgs, Lw_cgs, pdot_cgs, pdot_SNe_cgs] = SB99_data

    fQi_cgs = scipy.interpolate.interp1d(t_Myr, Qi_cgs, kind='cubic')
    fLi_cgs = scipy.interpolate.interp1d(t_Myr, Li_cgs, kind='cubic')
    fLn_cgs = scipy.interpolate.interp1d(t_Myr, Ln_cgs, kind='cubic')
    fLbol_cgs = scipy.interpolate.interp1d(t_Myr, Lbol_cgs, kind='cubic')
    fLw_cgs = scipy.interpolate.interp1d(t_Myr, Lw_cgs, kind='cubic')
    fpdot_cgs = scipy.interpolate.interp1d(t_Myr, pdot_cgs, kind='cubic')
    fpdot_SNe_cgs = scipy.interpolate.interp1d(t_Myr, pdot_SNe_cgs, kind='cubic')

    SB99f = {'fQi_cgs': fQi_cgs, 'fLi_cgs': fLi_cgs, 'fLn_cgs': fLn_cgs, 'fLbol_cgs': fLbol_cgs, 'fLw_cgs': fLw_cgs,
             'fpdot_cgs': fpdot_cgs, 'fpdot_SNe_cgs': fpdot_SNe_cgs}

    return SB99f


def combineSFB_data(warpfield_params, file1, file2, f_mass1=1.0, f_mass2=1.0, interfile1 = ' ', interfile2 = ' ', Zfile1 = 0.15, Zfile2 = 1.0, Zism = 0.5):
    """
    adds up feedback from two cluster populations
    """
    # if file1 is "inter" or "interpolate" then an interpolation between 2 metallicities is performed

    if (file1 == 'interpolate' or file1 == 'inter'):
        [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = getSB99_data_interp(Zism, interfile1, Zfile1, interfile2, Zfile2, warpfield_params, f_mass = f_mass1)
        [t2, Qi2, Li2, Ln2, Lbol2, Lw2, pdot2, pdot_SNe2] = getSB99_data(file2, warpfield_params, f_mass = f_mass2)
    else:
        [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = getSB99_data(file1, warpfield_params, f_mass = f_mass1)
        [t2, Qi2, Li2, Ln2, Lbol2, Lw2, pdot2, pdot_SNe2] = getSB99_data(file2, warpfield_params, f_mass = f_mass2)

    tend1 = t1[-1]
    tend2 = t2[-2]

    tend = np.min([tend1, tend2])
    
    def add_vector(a,b):
        if len(a) < len(b):
            c = b.copy()
            c[:len(a)] += a
        else:
            c = a.copy()
            c[:len(b)] += b
        return c

    # cut to same length

    Qi = add_vector(Qi1, Qi2)
    Li = add_vector(Li1, Li2)
    Ln = add_vector(Ln1, Ln2)
    Lbol = add_vector(Lbol1, Lbol2)
    Lw = add_vector(Lw1, Lw2)
    pdot = add_vector(pdot1, pdot2)
    pdot_SNe = add_vector(pdot_SNe1, pdot_SNe2)

    # check that the times are the same
    t1_tmp = t1[t1 <= tend]
    t2_tmp = t2[t2 <= tend]
    if not all(t1_tmp == t2_tmp):
        print("FATAL: files do not have the same time vectors")
        sys.exit("Exiting: SB99 files time arrays do not match!")

    if tend1 > tend2: t = t1
    else: t = t2

    return[t,Qi,Li,Ln,Lbol,Lw,pdot,pdot_SNe]


def sum_SB99(SB99f, SB99_data2_IN, dtSF, return_format='array'):
    """
    sum 2 SB99 files (dictionaries).
    :param SB99f: Interpolation dictionary for 1st SB99 file
    :param SB99_data2: Data dictionary for 2nd SB99 file (cluster forms after cluster corresponding to SB99f)
    :param dtSF: time difference between 1st and 2nd file
    :return: Data dictionary
    """

    SB99_data2 = arr2dict(SB99_data2_IN)
    ttemp = SB99_data2['t_Myr']

    # mask out late times for which no interpolation exists
    mask = ttemp+dtSF <= max(SB99f['fQi_cgs'].x) # np.max(ttemp)
    t = ttemp[mask]

    # initialize Dsum
    Dsum = {'t_Myr': t}

    # loop through keys and summ up feedback
    for key in SB99_data2:
        if key != 't_Myr': # do not sum time
            Dsum[key] = SB99f['f'+key](t + dtSF) + SB99_data2[key][mask]

    # what format should the result be in?
    if return_format == 'array': # array
        Dsum_array = [Dsum['t_Myr'], Dsum['Qi_cgs'], Dsum['Li_cgs'], Dsum['Ln_cgs'], Dsum['Lbol_cgs'], Dsum['Lw_cgs'], Dsum['pdot_cgs'], Dsum['pdot_SNe_cgs']]
        return Dsum_array
    else: # dictionary
        return Dsum

def time_shift(SB99_data_IN, t):
    """
    adds a time time to time vector of SB99 data dictionary or array
    :param SB99_data: SB99 data dictionary
    :param t: time offset (float)
    :return:
    """

    # check whether input is a dictionary
    if isinstance(SB99_data_IN, collections.Mapping):
        SB99_data = SB99_data_IN.copy()
        SB99_data['t_Myr'] += t
    else: # not a dictionary
        SB99_data = np.copy(SB99_data_IN)
        SB99_data[0] += t

    return SB99_data

def SB99_conc(SB1, SB2):
    """
    concatenate 2 files (assuming file 2 has a later start time)
    :param SB1: SB99 data array or dictionary
    :param SB2: SB99 data array or dictionary
    :return: array
    """

    [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = dict2arr(SB1)
    [t2, Qi2, Li2, Ln2, Lbol2, Lw2, pdot2, pdot_SNe2] = dict2arr(SB2)

    ii_time = operations.find_nearest_lower(t1, t2[0])

    t = np.append(t1[:ii_time+1], t2)
    Qi = np.append(Qi1[:ii_time + 1], Qi2)
    Li = np.append(Li1[:ii_time + 1], Li2)
    Ln = np.append(Ln1[:ii_time + 1], Ln2)
    Lbol = np.append(Lbol1[:ii_time + 1], Lbol2)
    Lw = np.append(Lw1[:ii_time + 1], Lw2)
    pdot = np.append(pdot1[:ii_time + 1], pdot2)
    pdot_SNe = np.append(pdot_SNe1[:ii_time + 1], pdot_SNe2)

    return [t, Qi, Li, Ln, Lbol, Lw, pdot, pdot_SNe]

def full_sum(t_list, Mcluster_list, Zism, 
             warpfield_params,
             rotation=True, BHcutoff=120., return_format='array'):
    
    # this fnction is not properly implremented yet,
    # because it is in expansion_next().
        
    Data = {}
    Data_interp = {}

    t_now = t_list[-1]

    N = len(t_list)
    for ii in range(0,N):
        f_mass = Mcluster_list[ii]/warpfield_params.SB99_mass
        key = str(ii)
        Data[key] = read_SB99(Zism, rotation=rotation, f_mass=f_mass, BHcutoff=BHcutoff, return_format='dict')
        Data_interp[key] = make_interpfunc(Data[key])

    Data_tot = Data[str(N-1)]
    if N > 1:
        for ii in range(0,N-1):
            dtSF = t_now - t_list[ii]
            Data_tot = sum_SB99(Data_interp[str(ii)], Data_tot, dtSF)

    Data_tot_interp = make_interpfunc(Data_tot)

    if return_format == 'dict': 
        Data_tot = arr2dict(Data_tot)
    elif return_format == 'array': 
        Data_tot = dict2arr(Data_tot)

    return Data_tot, Data_tot_interp



