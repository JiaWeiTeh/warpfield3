import constants as myc
import init as i
import getSB99_data
import os
import auxiliary_functions as aux
import phase_energy
import phase_transition
import phase_momentum
import phase_solver2
import numpy as np
import warp_writedata
import get_startvalues
import phase_lookup as ph
import datetime
import __cloudy__
import sys



def expansion_main(Mcloud_au_INPUT,SFE, model_id, n_models):
    """
    main routine for evolution of individual object
    TODO: Re-expansion after re-collapse
    :param Mcloud_au_INPUT:
    :param SFE:
    :param model_id:
    :param n_models:
    :return:
    """
    # TOASK: will recollapse be implemented in v3? Though I think it is reimplemented.

    
    startdatetime = datetime.datetime.now()
    
    if i.dens_profile == "BonnorEbert":
        aux.printl("{}: start model {} with Mcloud={} ({}), SFE={}, n0={}, g={} and Z={}".format(startdatetime, model_id+1, Mcloud_au_INPUT, np.log10(Mcloud_au_INPUT), SFE, i.namb, i.g_BE, i.Zism), verbose = -1)
        aux.printl("Initialization of Bonnor-Ebert spheres may take a few moments...." , verbose = -1)
    else:    
        aux.printl("{}: start model {} with Mcloud={} ({}), SFE={}, navg={}, and Z={}".format(startdatetime, model_id+1, Mcloud_au_INPUT, np.log10(Mcloud_au_INPUT), SFE, i.navg, i.Zism), verbose = -1)
    
    #aux.printl("{}: start model {} with Mcloud={} ({}), SFE={}, navg={}, and Z={}".format(startdatetime, model_id + 1, Mcloud_au_INPUT, np.log10(Mcloud_au_INPUT), SFE, i.navg, i.Zism), verbose=-1)

    ######## STEP 0: initialize some variables ############

    # time where SF event happens (?) TOASK
    tSF = 0.
    # number of re-collapses (0 if only 1 cluster present)
    ii_coll = 0
    # time
    t = np.array([tSF])
    # radius
    r = np.array([1.e-7])
    # shell velocity (?)
    v = np.array([3000.])
    # energy of bubble/shell (?)
    E = np.array([1.])
    # temperature
    T = np.array([100.])
    
    

    ######## STEP 1: cloud properties and SB99 ############
    
    # Set dictionary to initialise properties
    # cloud_before_SF is a boolean.
    ODEpar = get_startvalues.get_cloudproperties(Mcloud_au_INPUT, SFE, cloud_before_SF=i.Mcloud_beforeSF)
    
    ODEpar['gamma'] = myc.gamma
    ODEpar['tSF_list'] = np.array([tSF])
    ODEpar['Mcluster_list'] = np.array([ODEpar['Mcluster_au']])
    ODEpar['tStop'] = i.tStop
    ODEpar['Rsh_max'] = 0.

    # correct till here
    
    aux.printl(("ODEpar:", ODEpar), verbose=1)
    
    # Scaling factor for cluster masses. Though this might only be accurate for
    # high mass clusters (~>1e5) in which the IMF is fully sampled.
    factor_feedback = ODEpar['Mcluster_au'] / i.SB99_mass
    # Get SB99 data. This function returns data and interpolation functions.
    SB99_data, SB99f = getSB99_data.getSB99_main(i.Zism, 
                                                 rotation=i.rotation, 
                                                 f_mass=factor_feedback, BHcutoff=i.BHcutoff)
    # if tSF != 0.: we would actually need to shift the feedback parameters by tSF
    

    # prepare directories and write some basic files (like the file cotaining the input parameters)
    mypath, cloudypath, outdata_file, figure_file = warp_writedata.getmake_dir(i.basedir, i.navg, i.Zism, SFE,
                                                                               Mcloud_au_INPUT, SB99_data)
    # sys.exit()
    # remove success file (in case such a file was left over from previous run in this folder); will create success file at end of this run again
    # If a folder contains such a file, we know that the run was successful
    success_file = os.path.join(mypath, "Success.txt")
    if os.path.isfile(success_file): os.remove(success_file) # check whether file exists. If yes, remove (will be written when code finished)

    # create density law for cloudy
    print('create density law for cloudy')
    __cloudy__.create_dlaw(mypath, i.namb, i.nalpha, ODEpar['Rcore_au'], ODEpar['Rcloud_au'], ODEpar['Mcluster_au'], SFE, coll_counter=ii_coll)
    print('completed density law for cloudy')
    ODEpar['mypath'] = mypath # mypath is also stored in params, but better keep the info also here
    # optimal bubble structure (?)
    aux.optimalbstrux(Mcloud_au_INPUT, SFE,ODEpar['mypath'])

    # debug
    #t1 = np.array([1.,2.])
    #r1 = np.array([0.5,1.])
    #v1 = np.array([10.,-100.])
    #E1 = np.array([10.,2.])
    #T1 = np.array([1e5,1e4])

    ########### Simulate the Evolution (until end time reached, cloud dissolves, or re-collapse happens) ###############

    # MAIN WARPFIELD CODE
    t1, r1, v1, E1, T1 = run_expansion(ODEpar, SB99_data, SB99f, mypath, cloudypath)
    t = np.append(t, t1)
    r = np.append(r, r1)
    v = np.append(v, v1)
    E = np.append(E, E1)
    T = np.append(T, T1)
    
    # print("\n\n\n\n\nDEBUG\n\n\n\n\n")
    # print("Here is the shell velocity")
    # print(v)
    # print(r)
    # sys.exit("Done")

    # write data (make new file) and cloudy data
    # (this must be done after the ODE has been solved on the whole interval between 0 and tcollapse (or tdissolve) because the solver is implicit)
    warp_writedata.warp_reconstruct(t1, [r1,v1,E1,T1], ODEpar, SB99f, ii_coll, cloudypath, outdata_file, data_write=i.write_data, cloudy_write=i.write_cloudy, append=False)

    ########### STEP 2: In case of recollapse, prepare next expansion ##########################

    # did the expansion stop because a recollapse occured? If yes, start next expansion
    while (aux.check_continue(t[-1], r[-1], v[-1], i.tStop) == ph.coll):

        ii_coll += 1

        # debug
        #ODEpar['tSF_list'] = np.array([0.,1.])
        #ODEpar['Mcluster_list'] = np.array([1e6,1e5])
        #t1 = np.array([1., 2.])
        #r1 = np.array([0.5, 1.])
        #v1 = np.array([10., -100.])
        #E1 = np.array([10., 2.])
        #T1 = np.array([1e5, 1e4])

        # run expansion_next
        t1, r1, v1, E1, T1, ODEpar, SB99_data, SB99f = expansion_next(t[-1], ODEpar, SB99_data, SB99f, mypath, cloudypath, ii_coll)
        t = np.append(t, t1 + t[-1])
        r = np.append(r, r1)
        v = np.append(v, v1)
        E = np.append(E, E1)
        T = np.append(T, T1)

        # write data (append to old file) and cloudy data
        # TOASK: WHY RERUN RECONSTRUCT?
        # how to call cf in reconstruct?
        warp_writedata.warp_reconstruct(t1, [r1,v1,E1,T1], ODEpar, SB99f, ii_coll, cloudypath, outdata_file, data_write=i.write_data, cloudy_write=i.write_cloudy, append=True)

    # write success message to file
    with open(success_file, "w") as text_file:
        text_file.write("Stopped because...")
        if abs(t[-1]-i.tStop) < 1e-3: text_file.write("end time reached")
        elif (abs(r[-1]-i.rcoll) < 1e-3 and v[-1] < 0.): text_file.write("recollapse")
        else: text_file.write("unknown")

    return 0


def run_expansion(ODEpar, SB99_data, SB99f, mypath, cloudypath):
    """
    Model evolution of the cloud (both energy- and momentum-phase) until next recollapse or (if no re-collapse) until end of simulation
    :param ODEpar:
    :param SB99_data:
    :param SB99f:
    :param mypath:
    :param cloudypath:
    :return:
    """

    print('here we enter run_expansion.')
    ii_coll = 0
    tcoll = [0.]
    tStop = ODEpar['tStop']

    ######## STEP A: energy-phase (explicit) ###########
    # t0 = start time for Weaver phase
    # y0 = [r0, v0, E0, T0]
    # r0 = initial separation (pc)
    # v0 = initial velocity (km/s)
    t0, y0 = get_startvalues.get_y0(0., SB99f)
    # print(t0)
    # # 2.0576366423937075e-05 Myr
    # print(y0)
    # # [0.07523131981406396, 3656.200432285518, 1809763.2921571438 (Msun pc2 / yr2), 94126897.47628851 K]
    # sys.exit()
    # print('remember to delete all prints in coolstruc')
    Cool_Struc = get_startvalues.get_firstCoolStruc(i.Zism, t0 * 1e6)

    shell_dissolved = False
    t_shdis = 1e99

    tfinal = t0 + 30. * i.dt_Estart
    #print "tfinal:", tfinal

    [Dw, shell_dissolved, t_shdis] = phase_solver2.Weaver_phase(t0, y0, ODEpar['Rcloud_au'], SB99_data,
                                                                ODEpar['Mcloud_au'], ODEpar['SFE'], mypath,
                                                                cloudypath, shell_dissolved, t_shdis,
                                                                ODEpar['Rcore_au'], i.nalpha, ODEpar['Mcluster_au'],
                                                                ODEpar['nedge'], Cool_Struc, tcoll=tcoll,
                                                                coll_counter=ii_coll, tfinal=tfinal)

    ######## STEP B: energy-phase (implicit) ################
    # if (aux.check_continue(Dw['t_end'], Dw['r_end'], Dw['v_end']) == ph.cont):

    params = {'t_now': Dw['t_end'], 'R2': Dw['r_end'], 'Eb': Dw['E_end'], 'T0': Dw['Tb'][-1], 'beta': Dw['beta'][-1],
              'delta': Dw['delta'][-1], 'alpha': Dw['alpha'][-1], 'dMdt_factor': Dw['dMdt_factor_end']}
    params['alpha'] = Dw['v_end'] * (params["t_now"]) / params['R2'] # this is not quite consistent (because we are only using rough guesses for beta and delta) but using a wrong value here means also using the wrong velocity
    params['temp_counter'] = 0
    params['mypath'] = mypath

    '''
    try:
        warnings.filterwarnings("error")
        psoln_energy, params = phase_energy.run_phase_energy(params, ODEpar, SB99f)
    except:
        print('Catched Warning!!!!!!!!!!!')
    '''
    
    
    #pathf=os.environ["Fpath"]
    tfeed=np.linspace(5e-03, 40,num=6000)
    #Qifee=Lifee=Lnfee=Lbolfee=Lwfee=pdotfee=pdot_SNefee=[]
    #for d in range(len(tfeed)):
    Qifee=SB99f['fQi_cgs'](tfeed)
    Lifee=SB99f['fLi_cgs'](tfeed)
    Lnfee=SB99f['fLn_cgs'](tfeed)
    Lbolfee=SB99f['fLbol_cgs'](tfeed)
    Lwfee=SB99f['fLw_cgs'](tfeed)
    pdotfee=SB99f['fpdot_cgs'](tfeed)
    pdot_SNefee=SB99f['fpdot_SNe_cgs'](tfeed)

   
    
    
    #{'fQi_cgs': fQi_cgs, 'fLi_cgs': fLi_cgs, 'fLn_cgs': fLn_cgs, 'fLbol_cgs': fLbol_cgs, 'fLw_cgs': fLw_cgs,
     #        'fpdot_cgs': fpdot_cgs, 'fpdot_SNe_cgs': fpdot_SNe_cgs}
    
    #np.savetxt(pathf, np.c_[tfeed,Qifee,Lifee,Lnfee,Lbolfee,Lwfee,pdotfee,pdot_SNefee],delimiter='\t')
     
    # I think Daniel created these new files to run energy with gates. 
    # that said, the new files do not have option to record feedbak parameters.
    # we need to think of how to add them (should be straightforward)
    
    psoln_energy, params = phase_energy.run_phase_energy(params, ODEpar, SB99f)

    t = psoln_energy.t
    r = psoln_energy.y[0]
    v = psoln_energy.y[1]
    E = psoln_energy.y[2]
    T = psoln_energy.y[3]
    #print psoln_energy

    ######### STEP C: transition phase ########################
    if (aux.check_continue(t[-1], r[-1], v[-1], tStop) == ph.cont):


        t0 = psoln_energy.t[-1]
        y0 = [psoln_energy.y[0][-1], psoln_energy.y[1][-1], psoln_energy.y[2][-1], psoln_energy.y[3][-1]]
        cs = params['cs_avg']

        psoln_transition = phase_transition.run_phase_transition(t0, y0, cs, ODEpar, SB99f)
        #print psoln_transition

        t = np.append(t, psoln_transition.t)
        r = np.append(r, psoln_transition.y[0])
        v = np.append(v, psoln_transition.y[1])
        E = np.append(E, psoln_transition.y[2])
        T = np.append(T, psoln_transition.y[3])

        ######### STEP D: momentum phase ##############################
        if (aux.check_continue(t[-1], r[-1], v[-1], tStop) == ph.cont):
            t0 = psoln_transition.t[-1]
            y0 = [psoln_transition.y[0][-1], psoln_transition.y[1][-1], psoln_transition.y[2][-1],
                  psoln_transition.y[3][-1]]

            psoln_momentum = phase_momentum.run_fE_momentum(t0, y0, ODEpar, SB99f)
            #print psoln_momentum

            t = np.append(t, psoln_momentum.t)
            r = np.append(r, psoln_momentum.y[0]);
            v = np.append(v, psoln_momentum.y[1]);
            E = np.append(E, psoln_momentum.y[2]);
            T = np.append(T, psoln_momentum.y[3]);

    return t, r, v, E, T




def expansion_next(tStart, ODEpar, SB99_data_old, SB99f_old, mypath, cloudypath, ii_coll):

    aux.printl("Preparing new expansion...")
    
    print('next_expansion!!')

    ODEpar['tSF_list'] = np.append(ODEpar['tSF_list'], tStart) # append time of this SF event
    dtSF = ODEpar['tSF_list'][-1] - ODEpar['tSF_list'][-2] # time difference between this star burst and the previous
    ODEpar['tStop'] = i.tStop - tStart

    aux.printl(('list of collapses:', ODEpar['tSF_list']), verbose=0)

    # get new cloud/cluster properties and overwrite those keys in old dictionary
    CloudProp2 = get_startvalues.make_new_cluster(ODEpar['Mcloud_au'], ODEpar['SFE'], ODEpar['tSF_list'], ii_coll)
    for key in CloudProp2:
        ODEpar[key] = CloudProp2[key]

    # create density law for cloudy
    __cloudy__.create_dlaw(mypath, i.namb, i.nalpha, ODEpar['Rcore_au'], ODEpar['Rcloud_au'], ODEpar['Mcluster_au'], ODEpar['SFE'], coll_counter=ii_coll)

    ODEpar['Mcluster_list'] = np.append(ODEpar['Mcluster_list'], CloudProp2['Mcluster_au']) # contains list of masses of indivdual clusters

    # new method to get feedback
    SB99_data, SB99f = getSB99_data.full_sum(ODEpar['tSF_list'], ODEpar['Mcluster_list'], i.Zism, rotation=i.rotation, BHcutoff=i.BHcutoff, return_format='array')

    # get new feedback parameters
    #factor_feedback = ODEpar['Mcluster_au'] / i.SB99_mass
    #SB99_data2 = getSB99_data.load_stellar_tracks(i.Zism, rotation=i.rotation, f_mass=factor_feedback,BHcutoff=i.BHcutoff, return_format="dict") # feedback of only 2nd cluster
    #SB99_data = getSB99_data.sum_SB99(SB99f_old, SB99_data2, dtSF, return_format = 'array') # feedback of summed cluster --> use this
    #SB99f = getSB99_data.make_interpfunc(SB99_data) # make interpolation functions for summed cluster (allowed range of t: 0 to (99.9-tStart))

    if i.write_SB99 is True:
        SB99_data_write = getSB99_data.SB99_conc(SB99_data_old, getSB99_data.time_shift(SB99_data, tStart))
        warp_writedata.write_warpSB99(SB99_data_write, mypath)  # create file containing SB99 feedback info

    # for the purpose of gravity it is important that we pass on the summed mass of all clusters
    # we can do this as soon as all all other things (like getting the correct feedback) have been finished
    ODEpar['Mcluster_au'] = np.sum(ODEpar['Mcluster_list'])

    aux.printl(("ODEpar:", ODEpar), verbose=1)

    t, r, v, E, T = run_expansion(ODEpar, SB99_data, SB99f, mypath, cloudypath)

    return t, r, v, E, T, ODEpar, SB99_data, SB99f

