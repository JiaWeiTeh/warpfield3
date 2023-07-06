#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:11:42 2023

@author: Jia Wei Teh

This script contains functions that handle the creation and lookup of directories and filenames
"""

# some routines which handle creation and lookup of directories and filenames

import os
import numpy as np
    
    
def check_outdir(dirstring):
    """Check if output directory exists. If not, create it."""
    if not os.path.isdir(dirstring):
        os.makedirs(dirstring)

    return 0

def dir_up(dirstring):
    """
    goes one directory level up
    in contrast to os.path.dirname this routine also works if provided directory end with a slash "/"
    :param dirstring: directory as a string
    :return: direct parent directory of dirstring
    """
    if dirstring[-1] == "/": # get rid of trailing "/" because os.path.dirname fails if there is a trailing "/"
        dirstring = dirstring[:-1]
    dirup = os.path.dirname(dirstring)

    return dirup


def modelprop_to_string(navg, Zism, SFE, Mcloud, nalpha, ncore):
    """
    convert model properties (floats) to strings
    :param navg: average number density of cloud
    :param Zism: metallicty in units of solar Z
    :param SFE: star formation efficiency (number between 0 and 1)
    :param Mcloud: cloud mass in solar masses (not the log)
    :return: strings
    """

    # DON'T MESS WITH THIS!

    n_string = 'n' + '%.1f'%navg
    Z_string = 'Z' + '%.2f'%Zism
    SFE_string = 'SFE' + '%.2f'%(100.*SFE)
    Mcloud_string = "M" + '%.2f'%(np.log10(Mcloud))
    nalpha_string = "nalpha" + '%.2f'%nalpha
    ncore_string = "nc" + '%.2f'%(np.log10(ncore))

    return n_string, Z_string, SFE_string, Mcloud_string, nalpha_string, ncore_string




def get_mypath(basedir, navg, Zism, SFE, Mcloud, nalpha, ncore, g_BE, dens_profile):
    # Note:
        # old code: old code is in warp_nameparser.py, and did not include g_BE, dens_profile.
    """
    get path where data of given model will be stored
    :param basedir: full path name of project folder in which all runs (different models) will be stored
    :param navg:
    :param Zism:
    :param SFE:
    :param Mcloud:
    :return:
        
    """
    n_string, Z_string, SFE_string, Mcloud_string, nalpha_string, ncore_string = modelprop_to_string(navg, Zism, SFE, Mcloud, nalpha=nalpha, ncore=ncore)
    basedir = basedir.replace("//", "/")  # fix double slashes here
    
    mypath = os.path.join(basedir, Z_string)  # join strings intelligently
    check_outdir(mypath)
    
    mypath = os.path.join(mypath, Mcloud_string)
    check_outdir(mypath)
    
    if dens_profile == "powerlaw":
        mypath = os.path.join(mypath, n_string + '_' + nalpha_string + '_' + ncore_string)
        check_outdir(mypath)

    elif dens_profile == "BonnorEbert":
        gstring="g"+'{0:.2f}'.format(g_BE)
        mypath = os.path.join(mypath, gstring + '_' + ncore_string)
        check_outdir(mypath)
        
    mypath = os.path.join(mypath, SFE_string)
    check_outdir(mypath)
    
    return mypath


def get_fname(navg, Zism, SFE, Mcloud, nalpha=i.nalpha, ncore=i.namb):
    """
    get base of filename where data will be stored
    :param navg:
    :param Zism:
    :param SFE:
    :param Mcloud:
    :param nalpha:
    :param ncore:
    :return:
    """
    # convert model properties (floats) to strings
    n_string, Z_string, SFE_string, Mcloud_string, nalpha_string, ncore_string= modelprop_to_string(navg, Zism, SFE, Mcloud, nalpha=nalpha, ncore=ncore)

    # identifier for model
    savetitle_name = Mcloud_string + '_' + SFE_string + '_' + n_string + '_' + Z_string

    return savetitle_name




def get_dir(warpfield_params):
    
    # Note:
        # old code: savedir()
    
    # TODO
    
    return





def savedir(basedir, navg, Zism, SFE, Mcloud, nalpha=i.nalpha, ncore=i.namb):
    """
    gets all necessary directory and file names for warpfield (see below)
    :param basedir: full path name of project folder in which all runs (different models) will be stored
    :param navg: average number density of cloud
    :param Zism: metallicity in solar Z units
    :param SFE: star formation efficiency (number between 0 and 1)
    :param Mcloud: cloud mass in solar masses (not log)
    :return:
    mypath: absolute path to folder where data of given model will be stored 
    (e.g. '~/my_favorite_project/Z1.0/M5.0/SFE10.0_n1000.0/')
    cloudypath: absolute path to folder where cloudy input (and after running cloudy, also output) files  will be stored, e.g. 
    (e.g. '~/my_favorite_project/Z1.0/M5.0/SFE10.0_n1000.0/cloudy/')
    outdata_file: absolute filename of outputdata file (containing warpfield data output) 
    (e.g. '~/my_favorite_project/Z1.0/M5.0/SFE10.0_n1000.0/M5.0_SFE10.0_n1000.0_Z1.0_data.txt')
    figure_file: absolute filename of summary figure
    input_file: absolute filename of file storing the input parameters of the warpfield run
    """

    # Files currently not included here but in expansion_main.py:
    # -potential directory (storing files with gravitational potential)
    # -SB99 file (contains starburst 99 feedback parameters as a function of time as were used by warpfield (i.e. also summed feedback of multiple clusters if recollapse occured)

    # create folder where project data is stored if it does not exist
    check_outdir(basedir)

    # absolute path to folder where data of given model will be stored (e.g. '~/my_favorite_project/Z1.0/M6.0/SFE10.0_n1000.0/')
    mypath = get_mypath(basedir, navg, Zism, SFE, Mcloud, nalpha=nalpha, ncore=ncore)

    
    def get_cloudypath(mypath):
        """get the absolute path to directory where cloudy .in files will be stored"""
    
        cloudypath = os.path.join(mypath, "cloudy/")
    
        # create directory if it does not exist
        if i.write_cloudy is True:
            check_outdir(cloudypath)

        return cloudypath

    cloudypath = get_cloudypath(mypath)

    if i.write_potential is True:
        check_outdir(os.path.join(mypath, "potential"))

    if i.savebubble is True:
        check_outdir(os.path.join(mypath, "bubble"))

    # get base of filename
    savetitle_name = get_fname(navg, Zism, SFE, Mcloud, nalpha=nalpha, ncore=ncore)

    # file name of summary figure
    figtitle = savetitle_name + '.png'
    figure_file = (os.path.join(mypath, figtitle))

    # This is discontinued.
    # file name of main output data file and name of file containing used input data (myconfig.py and parameters.py)
    # if i.old_output_names:
    #     input_title = savetitle_name + "_input.txt"
    #     datafile_title = savetitle_name + "_data.txt"
    # else:
        
    input_title = "input.dat"
    datafile_title = "evo.dat"
    
    input_file = (os.path.join(mypath, input_title))
    outdata_file = (os.path.join(mypath, datafile_title))
    
    return mypath, cloudypath, outdata_file, figure_file, input_file

