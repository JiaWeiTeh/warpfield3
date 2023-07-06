#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:02:19 2023

@author: Jia Wei Teh
"""


TBD


import numpy as np




# some routines which write various output file

from scipy import interpolate
import numpy as np
from astropy.io import ascii
from astropy.table import Table

import warp_nameparser
import getSB99_data
import ODE_tot_aux
import __cloudy__

import os
import glob
import shutil

from tempfile import mkstemp





# def warp_reconstruct(t, y, ODEpar, 
#                      SB99f, ii_coll, cloudypath, 
#                      outdata_file, cloudy_write=True, data_write=False, 
#                      append=True):
#     """
#     reconstruct output parameters in warpversion 2.1;3
#     :param time: list or array of times SINCE THE LAST STAR FORMATION EVENT
#     :param y: [r,v,E,T]
#     :param ODEpar: cloud/cluster properties dictionary
#     :param SB99f: interpolation dictionary for summed cluster
#     :param ii_coll: number of re-collapses (0 if only 1 cluster present)
#     :param cloudypath: path to directory where cloudy input files will be stored
#     :return:
#     """

# warp_writedata.warp_reconstruct(t1, [r1,v1,E1,T1], ODEpar, 
#                                 SB99f, ii_coll, cloudypath, 
#                                 outdata_file, data_write=i.write_data, 
#                                 cloudy_write=i.write_cloudy, append=False)

# warp_writedata.warp_reconstruct(t1, [r1,v1,E1,T1], ODEpar, SB99f, ii_coll, cloudypath, outdata_file, data_write=i.write_data, cloudy_write=i.write_cloudy, append=True)

def warp_reconstruct(t, y, ODEpar, SB99f, ii_coll, cloudypath, outdata_file, cloudy_write=True, data_write=False, append=True):
    """
    reconstruct output parameters in warpversion 2.1;3
    :param time: list or array of times SINCE THE LAST STAR FORMATION EVENT
    :param y: [r,v,E,T]
    :param ODEpar: cloud/cluster properties dictionary
    :param SB99f: interpolation dictionary for summed cluster
    :param ii_coll: number of re-collapses (0 if only 1 cluster present)
    :param cloudypath: path to directory where cloudy input files will be stored
    :return:
    """

    r, v, E, T = y

    Ncluster = ii_coll + 1
    ODEpar['Rsh_max'] = 0.0 # set max previously achieved shell radius to 0 again; necessary because we will the max radius in the following loop
    SB99f_all = {}

    for jj in range(0, Ncluster):
        trash, SB99f_all[str(jj)] = getSB99_data.getSB99_main(init.Zism, rotation=init.rotation,
                                                              f_mass=ODEpar['Mcluster_list'][jj] / init.SB99_mass,
                                                              BHcutoff=init.BHcutoff)

    # minimum time (in Myr) to wait after a star formation event before cloudy files are created
    tmin = init.cloudy_tmin # default 1e-2
    # if cloudy_dt or small_cloudy_dt are set to lower values this will prevent output which the user requested
    # however, I fell it makes no sense to set cloudy_dt to a value smaller than this
    # TO DO: one should just prevent the user from setting a smaller cloudy_dt than this

    len_dat = len(t)

    # add time of last SF event to time vector
    Data = {'t':t+ODEpar['tSF_list'][-1], 'r':r, 'v':v, 'Eb':E, 'Tb':T}
    
    if i.frag_cover == True:
        try:
            tcf,cfv=np.loadtxt(ODEpar['mypath'] +"/FragmentationDetails/Coverfrac"+str(len(ODEpar['Mcluster_list']))+".txt", skiprows=1, delimiter='\t', usecols=(0,1), unpack=True)
        except:
            pass

    for key in ("Mshell", "fabs", "fabs_i","fabs_n", "Pb", "R1", "n0", "nmax"):
        Data[key] = np.ones(len_dat) * np.nan

    for ii in range(0, len_dat):

        t_real = t[ii] + ODEpar['tSF_list'][-1] # current time (measured from the very first SF event)

        # reconstruct values like pressure
        
        if i.frag_cover == True and int(os.environ["Coverfrac?"])==1:
            if t[ii] <= tcf[1]:
                cfr=1
            else:
                ide=aux.find_nearest_id(tcf, t[ii])
                cfr=cfv[ide]
            aux_data = ODE_tot_aux.fE_tot_part1(t[ii], [r[ii], v[ii], E[ii], T[ii]], ODEpar, SB99f,cfs='recon',cf_reconstruct=cfr)
        else: 
            aux_data = ODE_tot_aux.fE_tot_part1(t[ii], [r[ii], v[ii], E[ii], T[ii]], ODEpar, SB99f)
       

        if (cloudy_write is True and t[ii] >= tmin):  # do not write cloudy output very close after SF event; instead wait at least 1e-2 Myr

            Lbol_list = np.ones(Ncluster) * np.nan  # current bolometric luminosities of all clusters
            Age_list = t_real - ODEpar['tSF_list']  # current ages of all clusters
            for jj in range(0, Ncluster):
                Lbol_list[jj] = SB99f_all[str(jj)]['fLbol_cgs'](Age_list[jj])

            # write cloudy data to file
            __cloudy__.create_model(cloudypath, ODEpar['SFE'], ODEpar['Mcloud_au'], init.namb, init.Zism, aux_data['n0_cloudy'], r[ii], v[ii],
                         aux_data['Msh'], np.log10(SB99f['fLbol_cgs'](t[ii])), t_real,
                         ODEpar['Rcloud_au'], ODEpar['nedge'],
                         warpfield_params,
                         SB99model=init.SB99cloudy_file, shell=init.cloudy_stopmass_shell, turb=init.cloudy_turb,
                         coll_counter=ii_coll, Tarr=Age_list, Larr=Lbol_list,
                         Li_tot=SB99f['fLi_cgs'](t[ii]), Qi_tot=SB99f['fQi_cgs'](t[ii]),
                         pdot_tot=SB99f['fpdot_cgs'](t[ii]), Lw_tot=SB99f['fLw_cgs'](t[ii]),
                         Mcluster=np.sum(ODEpar['Mcluster_list']), phase=np.nan)

        Data['Mshell'][ii] = aux_data['Msh']
        for key in ("fabs", "fabs_i", "fabs_n", "Pb", "R1", "n0", "nmax"):
            Data[key][ii] = aux_data[key]

        ##########################################

        # create bubble structure
        # TO DO: check ii > 1
        """
        my_params = {"R2":r[ii], "v2":v[ii], "Eb":E[ii], "t_now":t[ii], 'pwdot':pwdot, 'pwdot_dot':pwdot_dot}

        dt = t[ii]-t[ii-1]
        Edot = (E[ii]-E[ii-1])/dt
        Tdot = (T[ii]-T[ii-1])/dt
        alpha = t[ii]/r[ii] * v[ii]
        beta = state_eq.Edot_to_beta(Data["Pb"][ii], Data["R1"][ii], Edot, my_params) # TO DO: need to pass my_params
        delta = state_eq.Tdot_to_delta(t[ii],T[ii],Tdot)

        # TO DO: need to write function which saves bubble structure
        Lb = bubble_structure2.calc_Lb(data_struc, Cool_Struc, 1, rgoal_f=init.r_Tb, verbose=0, plot=0, no_calc=False, error_exit=True,
                xtol=1e-6)
        """


        ########################

    Data['Mcloud'] = np.ones(len_dat) * ODEpar['Mcloud_au']
    Data['Mcluster'] = np.ones(len_dat) * ODEpar['Mcluster_au']

    Data['Lbol'] = SB99f['fLbol_cgs'](t)
    Data['Li'] = SB99f['fLi_cgs'](t)
    Data['Ln'] = SB99f['fLn_cgs'](t)
    Data['Qi'] = SB99f['fQi_cgs'](t)
    Data['Fram'] = SB99f['fpdot_cgs'](t)
    Data['Lw'] = SB99f['fLw_cgs'](t)

    # write data to file
    if data_write is True:
        write_outdata_table(Data, outdata_file, append=append)

    #print "DEBUG cloudy_write = ", cloudy_write

    lum_bubble_placeholder = "WARP2_LOG10_LUM_BUBBLE_CL" # placeholder name for total luminosity of bubble

    #cloudypath = "/home/daniel/Documents/work/loki/code/warpfield/output_test/warpfield2/new3/Z1.00/M6.00/n500.0_nalpha0.00_nc2.70/SFE5.00/cloudy/"
    #ODEpar = {}; ODEpar['mypath'] = "/home/daniel/Documents/work/loki/code/warpfield/output_test/warpfield2/new3/Z1.00/M6.00/n500.0_nalpha0.00_nc2.70/SFE5.00/"
    #A = ascii.read(ODEpar['mypath']+'M6.00_SFE5.00_n500.0_Z1.00_data.txt'); t = A['t'], T = A['Tb']

    # compare list of cloudy files and bubble files
    # if there is a cloudy file in the energy phase without a corresponding bubble file, we need to make that missing bubble file
    if (cloudy_write is True and init.savebubble is True):
        cloudyfiles_all = [f for f in os.listdir(cloudypath) if os.path.isfile(os.path.join(cloudypath, f))]
        cloudyfiles = sorted([ii for ii in cloudyfiles_all if ('shell' in ii and '.in' in ii)]) # take only shell input files
        cf = ["" for x in range(len(cloudyfiles))]
        # create array containing ages of cloudy files
        for ii in range(0,len(cloudyfiles)):
            mystring = cloudyfiles[ii]
            idx = [pos for pos, char in enumerate(mystring) if char == '.']
            cf[ii] = mystring[idx[-2]-1:idx[-1]]
        age_c = np.array([float(i) for i in cf]) * 1e-6 # convert to Myr
        # we now have a list with all ages for which there are cloudy (shell) files


        bubblepath = os.path.join(ODEpar['mypath'], 'bubble/')
        bubblefiles_all = [f for f in os.listdir(bubblepath) if os.path.isfile(os.path.join(bubblepath, f))]
        bubblefiles = sorted([ii for ii in bubblefiles_all if ('bubble' in ii and '.dat' in ii)])

        #print "DEBUG, bubblepath, bubblefiles", bubblepath, bubblefiles

        # find times where bubble burst
        tburst = np.array([0.])
        for ii in range(1,len(t)):
            if T[ii] <= 2e4:
                if T[ii-1] > 2e4: # if the bubble bursts, the temperature drops to 1e4 K
                    tburst = np.append(tburst, t[ii])
        if T[-1] > 2e4: # case: simulation ended in energy phsae
            tburst = np.append(tburst,1.01*t[-1]) # case where the bubble never burst (take some arbitrary high value)

        bf = ["" for x in range(len(bubblefiles))]
        # create array containing ages of bubble files
        for ii in range(0,len(bubblefiles)):
            mystring = bubblefiles[ii]
            idx = [pos for pos, char in enumerate(mystring) if char == '.']
            bf[ii] = mystring[idx[-2]-1:idx[-1]]
            #if bf[ii] not in cf:
            #    rmfile = os.path.join(bubblepath, bubblefiles[ii])
            #    os.remove(rmfile) # remove bubble files which do not have cloudy counterparts with same age
        age_b = np.array([float(i) for i in bf]) * 1e-6 # convert to Myr
        # we now have a list with all ages for which there are bubble files

        #print "DEBUG age_c, age_b", age_c, age_b, tburst

        # in case cluster winds are to be modelled: copy all files from cloudy folder to bubble folder
        # in case cluster winds are not to be modelled: copy only those shell.in and static.ini file from the cloudy folder where the expansion is in the energy limit (will be done later)
        if init.cloudy_CWcavitiy is True:
            for file in glob.glob(os.path.join(cloudypath, '*.in*')):
                shutil.copy(file, bubblepath)


        # create interpolated bubble profiles where there is a cloudy file without a bubble file in the energy phase
        for jj in range(1,len(tburst)): # go through each energy-driven phase seperately (each recollapse leads to another energy phase)
            msk = (age_b > tburst[jj-1]) * (age_b <= tburst[jj]+1e-6)
            age_b0 = age_b[msk] # list of bubble file ages in respective energy phase
            msk = (age_c >= age_b0[0]) * (age_c <= age_b0[-1])
            age_c0 = age_c[msk] # list of shell file ages in respective energy phase

            for ii in range(0,len(age_c0)):
                age1e7_str = ('{:0=5.7f}e+07'.format(age_c0[ii] / 10.))  # age in years (factor 1e7 hardcoded), naming convention matches naming convention for cloudy files
                bubble_base = os.path.join(bubblepath, "bubble_SB99age_" + age1e7_str) # base name of bubble data file
                bubble_file = bubble_base + ".in"

                # check whether there is a bubble file for this shell file
                if age_c0[ii] not in age_b0: # there is a cloudy file but not a corresponding bubble file... we need to make one
                    bubble_interp(age_c0[ii], age_b0, bubblepath, bubble_base, bubblefiles) # make new bubble file by interpolation

                # we are now sure a bubble file exists for the current t
                # now need to copy the lines with "table star" and "luminosity total" from the shell file to the bubble file (when the bubble file was created this information was not available)

                # copy shell and static file from cloudy directory to bubble directory
                # 1) copy shell file to bubble directory, modify extension, and modify prefix inside
                if init.cloudy_CWcavitiy is False:  # if colliding winds are to be modelled, copying already happened. If no colliding winds, we only want to copy files in energy phase. Do this now!
                    for file in glob.glob(os.path.join(cloudypath, 'shell*' + age1e7_str + '.in')): # this is exactly 1 file
                        #print "copy...", file, bubblepath
                        shutil.copy(file, bubblepath)
                for file in glob.glob(os.path.join(bubblepath, 'shell*' + age1e7_str + '.in')): # this is exactly 1 file
                    luminosity_line, table_star_line, warp_comments = repair_shell(file) # repair (and rename as .ini) the shell.in file in the bubble folder

                # 2) bubble file gets table star and luminosity total from old shell file
                repair_bubble(bubble_file, table_star_line, luminosity_line, warp_comments)

                # 3) copy static file to bubble directory and modify prefix inside
                if init.cloudy_CWcavitiy is False: # if colliding winds are to be modelled, copying already happened. If no colliding winds, we only want to copy files in energy phase. Do this now!
                    for file in glob.glob(os.path.join(cloudypath, 'static*'+age1e7_str+'.ini')): # this is exactly 1 file
                        shutil.copy(file, bubblepath)

        # repair all static files
        for file in glob.glob(os.path.join(bubblepath, 'static*'+'.ini')):
            repair_static(file)

        # now remove superfluous bubble files
        for ii in range(0,len(bubblefiles)):
            mystring = bubblefiles[ii]
            idx = [pos for pos, char in enumerate(mystring) if char == '.']
            bf[ii] = mystring[idx[-2]-1:idx[-1]]
            if bf[ii] not in cf:
                rmfile1 = os.path.join(bubblepath, bubblefiles[ii]) # this is the .dat file
                os.remove(rmfile1) # remove bubble .dat files
                rmfile2 = rmfile1[:-3] + "in"
                os.remove(rmfile2)  # remove bubble files which do not have cloudy counterparts with same age


        if init.cloudy_CWcavitiy is True:
            # add Chevalier and Clegg profile to existing bubble.in files
            Bfile_list = glob.glob(os.path.join(bubblepath, 'bubble_SB99age_*.in'))
            for Bfile in Bfile_list:
                Lmech, pdot, [rB_start, rB_stop], lines = get_Lmechpdotrstart(Bfile)
                rend = (1.0 - 1.e-5)*rB_start # stop radius for colliding winds (make it just slightly smaller than the start radius of the bubble (Weaver) component)
                Mdot, vterm = getSB99_data.getMdotv(pdot, Lmech)

                cluster_rad_cm = get_cluster_rad(init.fixed_cluster_radius, rB_stop) * c.pc

                R,V,rho,T,P = ClusterWind_profile.CW_profile(vterm, Mdot, cluster_rad_cm, rend*c.pc, Rstart=3.0e-3*c.pc)
                ndens = rho / init.mui # number density of H atoms
                CW_dlaw = "continue -35.0 {:.9f}\n".format(np.log10(ndens[0]))
                CW_tlaw = "continue -35.0 {:.9f}\n".format(np.log10(T[0]))
                for ll in range(0,len(R)):
                    CW_dlaw += "continue {:.9f} {:.9f}\n".format(np.log10(R[ll]), np.log10(ndens[ll]))
                    CW_tlaw += "continue {:.9f} {:.9f}\n".format(np.log10(R[ll]), np.log10(T[ll]))
                with open(Bfile, "w") as f:
                    insert_dlaw = False
                    insert_tlaw = False
                    for line in lines:
                        if "radius" in line and "linear parsec" in line and "stop" not in line: # need to modify start radius (we now start from the cluster center, i.e. r = 0)
                            f.write("radius 2.0e-02 linear parsec\n") # pick a small number here (but not too small or cloudy gets problems)
                        elif "continue " not in line: # lines with "continue" (which are part of the density or temperature profile) must not just be copied
                            f.write(line) # lines which do not include "continue" can be copied
                            if "dlaw table radius" in line:
                                insert_dlaw = True
                            elif "tlaw table radius" in line:
                                insert_tlaw = True
                        elif "continue " in line:
                            if insert_dlaw is True:
                                # now insert new profile
                                f.write(CW_dlaw)
                                insert_dlaw = False
                            elif insert_tlaw is True:
                                # now insert new profile
                                f.write(CW_tlaw)
                                insert_tlaw = False
                            else:
                                f.write(line)

            # create new bubble files for shell.in files in the momentum phase. For these new bubble files just use the Chevalier and Clegg profile
            SHfile_list = glob.glob(os.path.join(bubblepath, 'shell_SB99age_*.in'))
            for ii in range(0,len(SHfile_list)):
                SHfile = SHfile_list[ii]
                Bfile = SHfile.replace("shell_SB99age", "bubble_SB99age")
                bubble_base = Bfile[:-3]
                if Bfile not in Bfile_list: # all bubble files in Bfile_list already exist. Now go through the ones which don't exist

                    Lmech, pdot, [rSH_start, _], lines = get_Lmechpdotrstart(SHfile)

                    cluster_rad_cm = get_cluster_rad(init.fixed_cluster_radius, rSH_start) * c.pc

                    rend = (1.0 - 1e-10)*rSH_start  # stop radius for colliding winds (make it just slightly smaller than the start radius of the shell component)
                    Mdot, vterm = getSB99_data.getMdotv(pdot, Lmech)
                    R, V, rho, T, P = ClusterWind_profile.CW_profile(vterm, Mdot, cluster_rad_cm, rend * c.pc, Rstart=1.8e-2*c.pc)
                    ndens = rho / init.mui  # number density of H atoms

                    bub_savedata = {"r_cm": R, "n_cm-3": ndens, "T_K": T}
                    name_list = ["r_cm", "n_cm-3", "T_K"]
                    tab = Table(bub_savedata, names=name_list)
                    formats = {'r_cm': '%1.9e', 'n_cm-3': '%1.4e', 'T_K': '%1.4e'}
                    outname = bubble_base + ".dat"
                    tab.write(outname, format='ascii', formats=formats, delimiter="\t", overwrite=True)

                    # write bubble.in file for newly created (interpolated) bubble.dat file
                    __cloudy_bubble__.write_bubble(outname, Z=init.Zism)

                    luminosity_line, table_star_line, warp_comments = repair_shell(SHfile) # also renames shell file to .ini
                    repair_bubble(Bfile, table_star_line, luminosity_line, warp_comments)

                    #f.write('table read file = "' + prefix.replace("static_SB99age_", "shell_SB99age_") + '.con"\n')

    return 0









#%%







def my_asciiwrite(data, outname, names=[], delimiter="\t", formats={}, overwrite=True):
    try: # new versions of astropy (>=1.3.2) have the optional argument 'overwrite'. To prevent Warnings, set true.
        ascii.write(data, outname, names=names, delimiter=delimiter,formats=formats, overwrite=overwrite)
    except: # old version of astropy (<= 1.1.1) do not have the optional argument 'overwrite'
        ascii.write(data, outname, names=names, delimiter=delimiter, formats=formats)
    return 0

def bubble_interp(age, ageB_list, bubblepath, bubble_base, bubblefiles):
    """
    creates interpolated bubble.dat file
    :param age: requested age
    :param ageB_list: list of ages of existing bubble files
    :param bubblepath: path to bubble directory
    :param bubble_base: base name of bubble file without path
    :param bubblefiles: list of all existing bubble files without path
    :return: name of newly created bubble.dat file (interpolation happens silently)
    """
    # now find the closest bubble files to this cloudy file (one at earlier time, one at later time)
    kk = 1
    while ageB_list[kk] < age:
        kk += 1
    bub_lo = ascii.read(
        bubblepath + bubblefiles[kk - 1])  # this is the existing bubble file immediately before the missing bubble file
    bub_hi = ascii.read(
        bubblepath + bubblefiles[kk])  # this is the existing bubble file immediately after the missing bubble file

    t_lo = ageB_list[kk - 1]  # time of existing bubble file immediately before the missing bubble file (early time)
    t_hi = ageB_list[kk]  # time of existing bubble file immediately after the missing bubble file (late time)

    # inner and outer shell radii
    r_lo0 = bub_lo['r_cm'][0]
    r_lo1 = bub_lo['r_cm'][-1]
    r_hi0 = bub_hi['r_cm'][0]
    r_hi1 = bub_hi['r_cm'][-1]

    xi_lo = (bub_lo['r_cm'] - bub_lo['r_cm'][0]) / (
                bub_lo['r_cm'][-1] - bub_lo['r_cm'][0])  # dimensionless radii for early time
    xi_hi = (bub_hi['r_cm'] - bub_hi['r_cm'][0]) / (
                bub_hi['r_cm'][-1] - bub_hi['r_cm'][0])  # dimenstionless radii for late time

    lT_lo = np.log10(bub_lo['T_K'])
    lT_hi0 = np.log10(bub_hi['T_K'])

    ln_lo = np.log10(bub_lo['n_cm-3'])
    ln_hi0 = np.log10(bub_hi['n_cm-3'])

    fT_hi = interpolate.interp1d(xi_hi, lT_hi0, kind='linear')  # interpolate T at late t to same grid in r as at low t
    fn_hi = interpolate.interp1d(xi_hi, ln_hi0, kind='linear')  # interpolate n at late t to same grid in r as at low t
    lT_hi = fT_hi(xi_lo)
    ln_hi = fn_hi(xi_lo)

    lT_all = np.vstack((lT_lo, lT_hi))
    fT = interpolate.interp2d(xi_lo, [t_lo, t_hi], lT_all, kind='linear')  # interpolation function for temperature
    ln_all = np.vstack((ln_lo, ln_hi))
    fn = interpolate.interp2d(xi_lo, [t_lo, t_hi], ln_all, kind='linear')  # interpolation function for density

    # now: get correct radii at time of missing file (via interpolation)
    xi_mi = xi_lo
    T_mi = 10. ** fT(xi_mi, age)
    n_mi = 10. ** fn(xi_mi, age)
    r_mi0 = r_lo0 + (r_hi0 - r_lo0) * (age - t_lo) / (t_hi - t_lo)
    r_mi1 = r_lo1 + (r_hi1 - r_lo1) * (age - t_lo) / (t_hi - t_lo)
    r_mi = xi_mi * (r_mi1 - r_mi0) + r_mi0

    # save new interpolated bubble file (same naming conventions as in bubble_structure2.py)
    bub_savedata = {"r_cm": r_mi, "n_cm-3": n_mi, "T_K": T_mi}
    name_list = ["r_cm", "n_cm-3", "T_K"]
    tab = Table(bub_savedata, names=name_list)
    formats = {'r_cm': '%1.9e', 'n_cm-3': '%1.4e', 'T_K': '%1.4e'}
    outname = bubble_base + ".dat"
    tab.write(outname, format='ascii', formats=formats, delimiter="\t", overwrite=True)

    # write bubble.in file for newly created (interpolated) bubble.dat file
    __cloudy_bubble__.write_bubble(outname, Z=init.Zism)

    return outname

def get_cluster_rad(constant_Rc, shell_rad):
    if constant_Rc is True:
        cluster_rad = init.cluster_radius
    else:
        cluster_rad = init.scale_cluster_radius * shell_rad
    return cluster_rad

def get_Lmechpdotrstart(file):
    """
    read mechanical luminosity, momentum injection rate, start radius, stop radius as numbers and the lines of the file as strings
    :param file:
    :return:
    """
    rstop = None
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "additional info mechanical_luminosity_log10_erg/s" in line:
                Lmech = 10. ** float(line.split("mechanical_luminosity_log10_erg/s")[-1])
            elif "additional info momentum_injection_rate_log10_dyne" in line:
                pdot = 10. ** float(line.split("momentum_injection_rate_log10_dyne")[-1])
            elif "radius" in line and "linear parsec" in line and "stop" not in line:
                tmp = line.split("radius ")[-1]
                rstart = float(tmp.split(" linear parsec")[0])
            elif "stop radius" in line and "linear parsec" in line:
                tmp = line.split("stop radius ")[-1]
                rstop = float(tmp.split(" linear parsec")[0])
    return Lmech, pdot, [rstart, rstop], lines

def repair_static(file):
    """
    repairs an ini file which has been copied from the cloudy folder to the bubble folder
    :param file: shell.ini file in the bubble folder
    :return: 0
    """
    with open(file, "r") as f:
        lines = f.readlines()
    with open(file, "w") as f:
        for line in lines:
            if "set save prefix" in line: # modify set save prefix line
                newline = line.replace("/cloudy/", "/bubble/")
                f.write(newline)
                prefix = newline.split('"')[1] # just the path
                #f.write('set save prefix "' + file[:-4] + '"\n') # removing last 4 characters --> removing .ini
            elif "table read file" in line: #  modify table star line
                f.write('table read file = "' + prefix.replace("static_SB99age_","shell_SB99age_") + '.con"\n')
                #f.write('table read file = "' + fileB[:-4] + '.con"\n') # removing last 4 characters --> removing .ini
            elif "luminosity total" in line: # replace luminosity of star cluster with a placeholder string for the bubble luminosity
                #f.write('luminosity total ' + lum_bubble_placeholder + '\n') # commented out because EWP does wants to add the luminosity command in post processing
                None
            else:
                f.write(line)
    return 0

def repair_bubble(file, table_star_line, luminosity_line, warp_comments):
    """
    repair a bubble file in the bubble folder
    :param file:
    :param table_star_line:
    :param luminosity_line:
    :param warp_comments:
    :return:
    """

    with open(file, "r") as f:
        lines = f.readlines()
    lines.insert(3, table_star_line)
    lines.insert(4, luminosity_line)
    N_warp_comments = len(warp_comments)  # insert warpfield comments into bubble file
    for pp in range(0, N_warp_comments):
        lines.insert(5 + pp, warp_comments[pp])
    with open(file, "w") as f:
        lines = "".join(lines)
        f.write(lines)

    return 0

def repair_shell(file):
    """
    repairs a shell.in file which has been copied from the cloudy folder to the bubble folder, also renames it to .ini
    :param file: shell.in file in the bubble folder
    :return: the line with the "luminosity total" command (string), the line with the "table star" command (as string), and the warpfield comments
    """
    fileB = file + "i"
    os.rename(file, fileB)  # now the file is called shell*.ini instead of shell*.in
    with open(fileB, "r") as f:
        lines = f.readlines()
    warp_comments = np.array([])  # array in which warpfield comments are stored
    with open(fileB, "w") as f:
        for line in lines:
            if "set save prefix" in line:  # modify set save prefix line
                newline = line.replace("/cloudy/", "/bubble/")
                f.write(newline)
                prefix = newline.split('"')[1]  # just the path
                # f.write('set save prefix "' + fileB[:-4] + '"\n') # removing last 4 characters --> removing .ini
            elif "table star" in line:  # modify table star line
                table_star_line = line
                f.write('table read file = "' + prefix.replace("shell_SB99age_", "bubble_SB99age_") + '.con"\n')
                # f.write('table read file = "' + bubble_base + '.con"\n')
            elif "luminosity total" in line:  # replace luminosity of star cluster with a placeholder string for the bubble luminosity
                # f.write('luminosity total ' + lum_bubble_placeholder + '\n') # commented out because EWP does wants to add the luminosity command in post processing
                luminosity_line = line
            elif "c warpfield" in line or "# warpfield" in line:  # remember lines with additional comments
                warp_comments = np.append(warp_comments, line)
                f.write(line)
            else:
                f.write(line)

    return luminosity_line, table_star_line, warp_comments

def get_age(filename):
    """
    get age (as float) from filename assuming hardcoded naming convention where there is 1 digit before the decimal point and after the age immediately follows another point and the file type follows
    :param filename:
    :return:
    """
    idx = [pos for pos, char in enumerate(filename) if char == '.']
    age = float(filename[idx[-2] - 1:idx[-1]])
    return age

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with os.fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Remove original file
    os.remove(file_path)
    #Move new file
    shutil.move(abs_path, file_path)
    return 0


def create_inputfile(input_file):
    """
    create input_file (stores parameters used in warpfield run)
    :param input_file: absolute file name of input_file
    :return: 0
    """

    my_initparameter_list = sorted(np.append(["navg", "tff", "sigmaD"], init.parameter_list), key=lambda s: s.lower()) # case-insensitive sorting
    with open(input_file, 'w') as the_file:
        for var in my_initparameter_list:
            a = 'init.%s' % (var)
            b = eval(a)
            mystring = a.split(".")[1] + ' = ' + str(b)
            the_file.write(mystring + '\n')

    return 0


def write_warpSB99(SB99_data, mypath):
    """
    write SB99 file
    :param SB99_data:
    :param mypath:
    :return:
    """

    [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = SB99_data
    namesSB99_list = ['t', 'Qi', 'Lw', 'Lbol', 'Li', 'Ln', 'pdot', 'pdot_SN']
    formatsSB99_list = {'t': '%1.4e', 'Qi': '%1.5e', 'Lw': '%1.5e', 'Lbol': '%1.5e', 'Li': '%1.5e', 'Ln': '%1.5e','pdot': '%1.5e', 'pdot_SN': '%1.5e'}
    SB99data_write = {'t': t_evo, 'Qi': Qi_evo, 'Lw': Lw_evo, 'Lbol': Lbol_evo, 'Li': Li_evo, 'Ln': Ln_evo,'pdot': pdot_evo, 'pdot_SN': pdot_SNe_evo}
    my_asciiwrite(SB99data_write, os.path.join(mypath, "SB99.txt"), names=namesSB99_list, delimiter="\t",formats=formatsSB99_list, overwrite=True)

    return 0

def write_outdata(Data, outdata_file):
    """
    write output data to file
    :param Data:
    :param outdata_file:
    :return: 0
    """
    


    aux.printl("*** saving data to " + outdata_file)

    parameter_list = ['t', 'r', 'v', 'Eb', 'Tb', 'Pb', 'R1', 'fabs', 'fabs_i', 'fabs_n', 'Mshell', 'Mcloud', 'Mcluster']

    names_long = parameter_list
    formats_long = {'t': '%1.6e', 'r': '%1.4e', 'v': '%1.4e', 'Eb': '%1.4e', 'Tb': '%1.4e', 'Pb': '%1.4e', 'R1': '%1.4e', 'fabs': '%1.4e', 'fabs_i': '%1.4e', 'fabs_n': '%1.4e', 'Mshell': '%1.4e', 'Mcloud': '%1.4e', 'Mcluster': '%1.4e'}
    
    my_asciiwrite(Data, outdata_file, names=names_long, delimiter="\t", formats=formats_long, overwrite=True)

    return 0

def write_outdata_table(Data, outdata_file, append=False):
    parameter_list = ['t', 'r', 'v', 'Eb', 'Tb', 'Pb', 'R1', 'n0', 'nmax', 'fabs', 'fabs_i', 'fabs_n', 'Mshell', 'Mcloud', 'Mcluster', 'Qi', 'Lbol', 'Li', 'Ln', 'Lw', 'Fram']

    formats_long = {'t': '%1.6e', 'r': '%1.4e', 'v': '%1.4e', 'Eb': '%1.4e', 'Tb': '%1.4e', 'Pb': '%1.4e',
                    'R1': '%1.4e', 'fabs': '%1.4e', 'fabs_i': '%1.4e', 'fabs_n': '%1.4e', 'Mshell': '%1.4e',
                    'Mcloud': '%1.4e', 'Mcluster': '%1.4e', 'n0': '%1.4e', 'nmax': '%1.4e',
                    'Qi': '%1.4e', 'Lbol': '%1.4e', 'Li': '%1.4e', 'Ln': '%1.4e', 'Lw': '%1.4e', 'Fram': '%1.4e'}

    tab = Table(Data, names=parameter_list)

    # new file
    if append is False:
        tab.write(outdata_file, format='ascii', formats=formats_long, delimiter="\t")
    else: # append to file
        with open(outdata_file, mode='a') as f:
            f.seek(0, os.SEEK_END)  # Some platforms don't automatically seek to end when files opened in append mode
            tab.write(f, format='ascii.no_header', formats=formats_long, delimiter="\t")

    return 0

def getmake_dir(basedir, navg, Zism, SFE, Mcloud, SB99data, nalpha=init.nalpha, ncore=init.namb):
    """
    creates directories and writes some basic files (like SB99 data)
    :param basedir:
    :param navg:
    :param Zism:
    :param SFE:
    :param Mcloud:
    :param SB99data:
    :return:
    """
    mypath, cloudypath, outdata_file, figure_file, input_file = warp_nameparser.savedir(basedir, navg, Zism, SFE, Mcloud, nalpha=nalpha, ncore=ncore)
    create_inputfile(input_file)  # create input file (where input parameters of warpfield run are stored)
    if init.write_SB99 is True:
        write_warpSB99(SB99data, mypath)  # create file containing SB99 feedback info

    return mypath, cloudypath, outdata_file, figure_file



def warp_reconstruct(t, y, ODEpar, SB99f, ii_coll, cloudypath, outdata_file, cloudy_write=True, data_write=False, append=True):
    """
    reconstruct output parameters in warpversion 2.1;3
    :param time: list or array of times SINCE THE LAST STAR FORMATION EVENT
    :param y: [r,v,E,T]
    :param ODEpar: cloud/cluster properties dictionary
    :param SB99f: interpolation dictionary for summed cluster
    :param ii_coll: number of re-collapses (0 if only 1 cluster present)
    :param cloudypath: path to directory where cloudy input files will be stored
    :return:
    """

    r, v, E, T = y

    Ncluster = ii_coll + 1
    ODEpar['Rsh_max'] = 0.0 # set max previously achieved shell radius to 0 again; necessary because we will the max radius in the following loop
    SB99f_all = {}

    for jj in range(0, Ncluster):
        trash, SB99f_all[str(jj)] = getSB99_data.getSB99_main(init.Zism, rotation=init.rotation,
                                                              f_mass=ODEpar['Mcluster_list'][jj] / init.SB99_mass,
                                                              BHcutoff=init.BHcutoff)

    # minimum time (in Myr) to wait after a star formation event before cloudy files are created
    tmin = init.cloudy_tmin # default 1e-2
    # if cloudy_dt or small_cloudy_dt are set to lower values this will prevent output which the user requested
    # however, I fell it makes no sense to set cloudy_dt to a value smaller than this
    # TO DO: one should just prevent the user from setting a smaller cloudy_dt than this

    len_dat = len(t)

    # add time of last SF event to time vector
    Data = {'t':t+ODEpar['tSF_list'][-1], 'r':r, 'v':v, 'Eb':E, 'Tb':T}
    
    if i.frag_cover == True:
        try:
            tcf,cfv=np.loadtxt(ODEpar['mypath'] +"/FragmentationDetails/Coverfrac"+str(len(ODEpar['Mcluster_list']))+".txt", skiprows=1, delimiter='\t', usecols=(0,1), unpack=True)
        except:
            pass

    for key in ("Mshell", "fabs", "fabs_i","fabs_n", "Pb", "R1", "n0", "nmax"):
        Data[key] = np.ones(len_dat) * np.nan

    for ii in range(0, len_dat):

        t_real = t[ii] + ODEpar['tSF_list'][-1] # current time (measured from the very first SF event)

        # reconstruct values like pressure
        
        if i.frag_cover == True and int(os.environ["Coverfrac?"])==1:
            if t[ii] <= tcf[1]:
                cfr=1
            else:
                ide=aux.find_nearest_id(tcf, t[ii])
                cfr=cfv[ide]
            aux_data = ODE_tot_aux.fE_tot_part1(t[ii], [r[ii], v[ii], E[ii], T[ii]], ODEpar, SB99f,cfs='recon',cf_reconstruct=cfr)
        else: 
            aux_data = ODE_tot_aux.fE_tot_part1(t[ii], [r[ii], v[ii], E[ii], T[ii]], ODEpar, SB99f)
       

        if (cloudy_write is True and t[ii] >= tmin):  # do not write cloudy output very close after SF event; instead wait at least 1e-2 Myr

            Lbol_list = np.ones(Ncluster) * np.nan  # current bolometric luminosities of all clusters
            Age_list = t_real - ODEpar['tSF_list']  # current ages of all clusters
            for jj in range(0, Ncluster):
                Lbol_list[jj] = SB99f_all[str(jj)]['fLbol_cgs'](Age_list[jj])

            # write cloudy data to file
            __cloudy__.create_model(cloudypath, ODEpar['SFE'], ODEpar['Mcloud_au'], init.namb, init.Zism, aux_data['n0_cloudy'], r[ii], v[ii],
                         aux_data['Msh'], np.log10(SB99f['fLbol_cgs'](t[ii])), t_real,
                         ODEpar['Rcloud_au'], ODEpar['nedge'],
                         warpfield_params,
                         SB99model=init.SB99cloudy_file, shell=init.cloudy_stopmass_shell, turb=init.cloudy_turb,
                         coll_counter=ii_coll, Tarr=Age_list, Larr=Lbol_list,
                         Li_tot=SB99f['fLi_cgs'](t[ii]), Qi_tot=SB99f['fQi_cgs'](t[ii]),
                         pdot_tot=SB99f['fpdot_cgs'](t[ii]), Lw_tot=SB99f['fLw_cgs'](t[ii]),
                         Mcluster=np.sum(ODEpar['Mcluster_list']), phase=np.nan)

        Data['Mshell'][ii] = aux_data['Msh']
        for key in ("fabs", "fabs_i", "fabs_n", "Pb", "R1", "n0", "nmax"):
            Data[key][ii] = aux_data[key]

        ##########################################

        # create bubble structure
        # TO DO: check ii > 1
        """
        my_params = {"R2":r[ii], "v2":v[ii], "Eb":E[ii], "t_now":t[ii], 'pwdot':pwdot, 'pwdot_dot':pwdot_dot}

        dt = t[ii]-t[ii-1]
        Edot = (E[ii]-E[ii-1])/dt
        Tdot = (T[ii]-T[ii-1])/dt
        alpha = t[ii]/r[ii] * v[ii]
        beta = state_eq.Edot_to_beta(Data["Pb"][ii], Data["R1"][ii], Edot, my_params) # TO DO: need to pass my_params
        delta = state_eq.Tdot_to_delta(t[ii],T[ii],Tdot)

        # TO DO: need to write function which saves bubble structure
        Lb = bubble_structure2.calc_Lb(data_struc, Cool_Struc, 1, rgoal_f=init.r_Tb, verbose=0, plot=0, no_calc=False, error_exit=True,
                xtol=1e-6)
        """


        ########################

    Data['Mcloud'] = np.ones(len_dat) * ODEpar['Mcloud_au']
    Data['Mcluster'] = np.ones(len_dat) * ODEpar['Mcluster_au']

    Data['Lbol'] = SB99f['fLbol_cgs'](t)
    Data['Li'] = SB99f['fLi_cgs'](t)
    Data['Ln'] = SB99f['fLn_cgs'](t)
    Data['Qi'] = SB99f['fQi_cgs'](t)
    Data['Fram'] = SB99f['fpdot_cgs'](t)
    Data['Lw'] = SB99f['fLw_cgs'](t)

    # write data to file
    if data_write is True:
        write_outdata_table(Data, outdata_file, append=append)

    #print "DEBUG cloudy_write = ", cloudy_write

    lum_bubble_placeholder = "WARP2_LOG10_LUM_BUBBLE_CL" # placeholder name for total luminosity of bubble

    #cloudypath = "/home/daniel/Documents/work/loki/code/warpfield/output_test/warpfield2/new3/Z1.00/M6.00/n500.0_nalpha0.00_nc2.70/SFE5.00/cloudy/"
    #ODEpar = {}; ODEpar['mypath'] = "/home/daniel/Documents/work/loki/code/warpfield/output_test/warpfield2/new3/Z1.00/M6.00/n500.0_nalpha0.00_nc2.70/SFE5.00/"
    #A = ascii.read(ODEpar['mypath']+'M6.00_SFE5.00_n500.0_Z1.00_data.txt'); t = A['t'], T = A['Tb']

    # compare list of cloudy files and bubble files
    # if there is a cloudy file in the energy phase without a corresponding bubble file, we need to make that missing bubble file
    if (cloudy_write is True and init.savebubble is True):
        cloudyfiles_all = [f for f in os.listdir(cloudypath) if os.path.isfile(os.path.join(cloudypath, f))]
        cloudyfiles = sorted([ii for ii in cloudyfiles_all if ('shell' in ii and '.in' in ii)]) # take only shell input files
        cf = ["" for x in range(len(cloudyfiles))]
        # create array containing ages of cloudy files
        for ii in range(0,len(cloudyfiles)):
            mystring = cloudyfiles[ii]
            idx = [pos for pos, char in enumerate(mystring) if char == '.']
            cf[ii] = mystring[idx[-2]-1:idx[-1]]
        age_c = np.array([float(i) for i in cf]) * 1e-6 # convert to Myr
        # we now have a list with all ages for which there are cloudy (shell) files


        bubblepath = os.path.join(ODEpar['mypath'], 'bubble/')
        bubblefiles_all = [f for f in os.listdir(bubblepath) if os.path.isfile(os.path.join(bubblepath, f))]
        bubblefiles = sorted([ii for ii in bubblefiles_all if ('bubble' in ii and '.dat' in ii)])

        #print "DEBUG, bubblepath, bubblefiles", bubblepath, bubblefiles

        # find times where bubble burst
        tburst = np.array([0.])
        for ii in range(1,len(t)):
            if T[ii] <= 2e4:
                if T[ii-1] > 2e4: # if the bubble bursts, the temperature drops to 1e4 K
                    tburst = np.append(tburst, t[ii])
        if T[-1] > 2e4: # case: simulation ended in energy phsae
            tburst = np.append(tburst,1.01*t[-1]) # case where the bubble never burst (take some arbitrary high value)

        bf = ["" for x in range(len(bubblefiles))]
        # create array containing ages of bubble files
        for ii in range(0,len(bubblefiles)):
            mystring = bubblefiles[ii]
            idx = [pos for pos, char in enumerate(mystring) if char == '.']
            bf[ii] = mystring[idx[-2]-1:idx[-1]]
            #if bf[ii] not in cf:
            #    rmfile = os.path.join(bubblepath, bubblefiles[ii])
            #    os.remove(rmfile) # remove bubble files which do not have cloudy counterparts with same age
        age_b = np.array([float(i) for i in bf]) * 1e-6 # convert to Myr
        # we now have a list with all ages for which there are bubble files

        #print "DEBUG age_c, age_b", age_c, age_b, tburst

        # in case cluster winds are to be modelled: copy all files from cloudy folder to bubble folder
        # in case cluster winds are not to be modelled: copy only those shell.in and static.ini file from the cloudy folder where the expansion is in the energy limit (will be done later)
        if init.cloudy_CWcavitiy is True:
            for file in glob.glob(os.path.join(cloudypath, '*.in*')):
                shutil.copy(file, bubblepath)


        # create interpolated bubble profiles where there is a cloudy file without a bubble file in the energy phase
        for jj in range(1,len(tburst)): # go through each energy-driven phase seperately (each recollapse leads to another energy phase)
            msk = (age_b > tburst[jj-1]) * (age_b <= tburst[jj]+1e-6)
            age_b0 = age_b[msk] # list of bubble file ages in respective energy phase
            msk = (age_c >= age_b0[0]) * (age_c <= age_b0[-1])
            age_c0 = age_c[msk] # list of shell file ages in respective energy phase

            for ii in range(0,len(age_c0)):
                age1e7_str = ('{:0=5.7f}e+07'.format(age_c0[ii] / 10.))  # age in years (factor 1e7 hardcoded), naming convention matches naming convention for cloudy files
                bubble_base = os.path.join(bubblepath, "bubble_SB99age_" + age1e7_str) # base name of bubble data file
                bubble_file = bubble_base + ".in"

                # check whether there is a bubble file for this shell file
                if age_c0[ii] not in age_b0: # there is a cloudy file but not a corresponding bubble file... we need to make one
                    bubble_interp(age_c0[ii], age_b0, bubblepath, bubble_base, bubblefiles) # make new bubble file by interpolation

                # we are now sure a bubble file exists for the current t
                # now need to copy the lines with "table star" and "luminosity total" from the shell file to the bubble file (when the bubble file was created this information was not available)

                # copy shell and static file from cloudy directory to bubble directory
                # 1) copy shell file to bubble directory, modify extension, and modify prefix inside
                if init.cloudy_CWcavitiy is False:  # if colliding winds are to be modelled, copying already happened. If no colliding winds, we only want to copy files in energy phase. Do this now!
                    for file in glob.glob(os.path.join(cloudypath, 'shell*' + age1e7_str + '.in')): # this is exactly 1 file
                        #print "copy...", file, bubblepath
                        shutil.copy(file, bubblepath)
                for file in glob.glob(os.path.join(bubblepath, 'shell*' + age1e7_str + '.in')): # this is exactly 1 file
                    luminosity_line, table_star_line, warp_comments = repair_shell(file) # repair (and rename as .ini) the shell.in file in the bubble folder

                # 2) bubble file gets table star and luminosity total from old shell file
                repair_bubble(bubble_file, table_star_line, luminosity_line, warp_comments)

                # 3) copy static file to bubble directory and modify prefix inside
                if init.cloudy_CWcavitiy is False: # if colliding winds are to be modelled, copying already happened. If no colliding winds, we only want to copy files in energy phase. Do this now!
                    for file in glob.glob(os.path.join(cloudypath, 'static*'+age1e7_str+'.ini')): # this is exactly 1 file
                        shutil.copy(file, bubblepath)

        # repair all static files
        for file in glob.glob(os.path.join(bubblepath, 'static*'+'.ini')):
            repair_static(file)

        # now remove superfluous bubble files
        for ii in range(0,len(bubblefiles)):
            mystring = bubblefiles[ii]
            idx = [pos for pos, char in enumerate(mystring) if char == '.']
            bf[ii] = mystring[idx[-2]-1:idx[-1]]
            if bf[ii] not in cf:
                rmfile1 = os.path.join(bubblepath, bubblefiles[ii]) # this is the .dat file
                os.remove(rmfile1) # remove bubble .dat files
                rmfile2 = rmfile1[:-3] + "in"
                os.remove(rmfile2)  # remove bubble files which do not have cloudy counterparts with same age


        if init.cloudy_CWcavitiy is True:
            # add Chevalier and Clegg profile to existing bubble.in files
            Bfile_list = glob.glob(os.path.join(bubblepath, 'bubble_SB99age_*.in'))
            for Bfile in Bfile_list:
                Lmech, pdot, [rB_start, rB_stop], lines = get_Lmechpdotrstart(Bfile)
                rend = (1.0 - 1.e-5)*rB_start # stop radius for colliding winds (make it just slightly smaller than the start radius of the bubble (Weaver) component)
                Mdot, vterm = getSB99_data.getMdotv(pdot, Lmech)

                cluster_rad_cm = get_cluster_rad(init.fixed_cluster_radius, rB_stop) * c.pc

                R,V,rho,T,P = ClusterWind_profile.CW_profile(vterm, Mdot, cluster_rad_cm, rend*c.pc, Rstart=3.0e-3*c.pc)
                ndens = rho / init.mui # number density of H atoms
                CW_dlaw = "continue -35.0 {:.9f}\n".format(np.log10(ndens[0]))
                CW_tlaw = "continue -35.0 {:.9f}\n".format(np.log10(T[0]))
                for ll in range(0,len(R)):
                    CW_dlaw += "continue {:.9f} {:.9f}\n".format(np.log10(R[ll]), np.log10(ndens[ll]))
                    CW_tlaw += "continue {:.9f} {:.9f}\n".format(np.log10(R[ll]), np.log10(T[ll]))
                with open(Bfile, "w") as f:
                    insert_dlaw = False
                    insert_tlaw = False
                    for line in lines:
                        if "radius" in line and "linear parsec" in line and "stop" not in line: # need to modify start radius (we now start from the cluster center, i.e. r = 0)
                            f.write("radius 2.0e-02 linear parsec\n") # pick a small number here (but not too small or cloudy gets problems)
                        elif "continue " not in line: # lines with "continue" (which are part of the density or temperature profile) must not just be copied
                            f.write(line) # lines which do not include "continue" can be copied
                            if "dlaw table radius" in line:
                                insert_dlaw = True
                            elif "tlaw table radius" in line:
                                insert_tlaw = True
                        elif "continue " in line:
                            if insert_dlaw is True:
                                # now insert new profile
                                f.write(CW_dlaw)
                                insert_dlaw = False
                            elif insert_tlaw is True:
                                # now insert new profile
                                f.write(CW_tlaw)
                                insert_tlaw = False
                            else:
                                f.write(line)

            # create new bubble files for shell.in files in the momentum phase. For these new bubble files just use the Chevalier and Clegg profile
            SHfile_list = glob.glob(os.path.join(bubblepath, 'shell_SB99age_*.in'))
            for ii in range(0,len(SHfile_list)):
                SHfile = SHfile_list[ii]
                Bfile = SHfile.replace("shell_SB99age", "bubble_SB99age")
                bubble_base = Bfile[:-3]
                if Bfile not in Bfile_list: # all bubble files in Bfile_list already exist. Now go through the ones which don't exist

                    Lmech, pdot, [rSH_start, _], lines = get_Lmechpdotrstart(SHfile)

                    cluster_rad_cm = get_cluster_rad(init.fixed_cluster_radius, rSH_start) * c.pc

                    rend = (1.0 - 1e-10)*rSH_start  # stop radius for colliding winds (make it just slightly smaller than the start radius of the shell component)
                    Mdot, vterm = getSB99_data.getMdotv(pdot, Lmech)
                    R, V, rho, T, P = ClusterWind_profile.CW_profile(vterm, Mdot, cluster_rad_cm, rend * c.pc, Rstart=1.8e-2*c.pc)
                    ndens = rho / init.mui  # number density of H atoms

                    bub_savedata = {"r_cm": R, "n_cm-3": ndens, "T_K": T}
                    name_list = ["r_cm", "n_cm-3", "T_K"]
                    tab = Table(bub_savedata, names=name_list)
                    formats = {'r_cm': '%1.9e', 'n_cm-3': '%1.4e', 'T_K': '%1.4e'}
                    outname = bubble_base + ".dat"
                    tab.write(outname, format='ascii', formats=formats, delimiter="\t", overwrite=True)

                    # write bubble.in file for newly created (interpolated) bubble.dat file
                    __cloudy_bubble__.write_bubble(outname, Z=init.Zism)

                    luminosity_line, table_star_line, warp_comments = repair_shell(SHfile) # also renames shell file to .ini
                    repair_bubble(Bfile, table_star_line, luminosity_line, warp_comments)

                    #f.write('table read file = "' + prefix.replace("static_SB99age_", "shell_SB99age_") + '.con"\n')

    return 0