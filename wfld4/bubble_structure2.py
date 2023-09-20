import numpy as np
import scipy.optimize
import scipy.integrate
import constants as myc
import time
import cool
from scipy.interpolate import interp1d
import state_eq
import auxiliary_functions as aux
import warp_nameparser as wn
import sys
import os
import contextlib
import init as i
import coolnoeq
import deltmod
from astropy.table import Table
import __cloudy_bubble__
# if i.plot_data is True: import matplotlib.pyplot as plt
#import random

#import time




# cooling and heating interpolation functions
# this is temporary (move to higher level routine and only call once per run!)
#onlycoolfunc, onlyheatfunc = coolnoeq.create_onlycoolheat(i.Zism, 1e6)

# create interpolation function for CIE cooling curve (CIE curve does not change during run time as metallicity does not change --> create the CIE curve only once here)
f_logLambdaCIE = cool.create_coolCIE(i.Zism)

# a constant which we will need
temp_au = (25./4.) *myc.kboltz_au/(0.5 * (myc.mp/myc.Msun)*myc.Cspitzer_au)
#temp = 25. / 4. * myc.kboltz / (0.5 * myc.mp * myc.Cspitzer)

#next 2 routines are for supression of lsoda warnings
def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied



def T_from_R(R2,R2_prime_guess, dMdt):
    T = (25./4.*myc.kboltz/(0.5*myc.mp*myc.Cspitzer) * dMdt/(4.*np.pi*R2**2.))**0.4 * (R2-R2_prime_guess)**0.4
    return T

def R1_zero(x, params):
    # root of this sets R1 (here: x) (see Weaver77, eq. 55)
    Lw, Eb, vw, R2 = params
    Eb_min = 1e-4
    if Eb < Eb_min: Eb = Eb_min # safety
    residual = (Lw*(R2**3.-x**3.)/(Eb*vw))**0.5 - x

    return residual

def set_dx():
    dx = 0.01*myc.pc
    return dx

def TR2_prime1e4(R2_prime_guess,params):
    # zero function for R2_prime for which TR2_prime = Tgoal
    # Tgoal should be close to 1e4K
    dMdt, R2, Tgoal = params
    TR2_prime = T_from_R(R2,R2_prime_guess,dMdt)
    residual = TR2_prime - Tgoal
    return residual

def constrainedFunction(x, f, lower, upper, minIncr=0.001):
    x = np.asarray(x)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    xBorder = np.where(x < lower, lower, x)
    xBorder = np.where(x > upper, upper, xBorder)
    fBorder = f(xBorder)
    distFromBorder = (np.sum(np.where(x < lower, lower - x, 0.))
                      + np.sum(np.where(x > upper, x - upper, 0.)))
    return (fBorder + (fBorder
                       + np.where(fBorder > 0, minIncr, -minIncr)) * distFromBorder)

def calc_cons(alpha,beta,delta,t_now, press, units = 'au'):
    if units == 'cgs':
        my_Cspitzer = myc.Cspitzer
        my_kboltz = myc.kboltz
    elif units == 'au':
        my_Cspitzer = myc.Cspitzer_au
        my_kboltz = myc.kboltz_au
    a = alpha/t_now
    b = (beta+delta)/t_now
    c = press/my_Cspitzer
    d = press
    e = (beta+2.5*delta)/t_now

    cons={"a":a, "b":b, "c":c, "d":d, "e":e, 't_now': t_now}

    return cons

def bubble_struct(r, x, Data_Struc, units = 'au'):
    """
    system of ODEs for bubble structure (see Weaver+77, eqs. 42 and 43)
    :param x: velocity v, temperature T, spatial derivate of temperature dT/dr
    :param r: radius from center
    :param cons: constants
    :return: spatial derivatives of v,T,dTdr
    """

    #start = time.time()

    a = Data_Struc["cons"]["a"]
    b = Data_Struc["cons"]["b"]
    c = Data_Struc["cons"]["c"]
    d = Data_Struc["cons"]["d"]
    e = Data_Struc["cons"]["e"]
    Qi = Data_Struc["cons"]["Qi"]
    Cool_Struc = Data_Struc["Cool_Struc"]

    v, T, dTdr = x

    #end = time.time()
    #print(end - start)

    #start = time.time()

    if units == 'cgs':
        my_kboltz = myc.kboltz
        sys.exit("cgs units not implemented in bubble_structure2")
    elif units == 'au':
        my_kboltz = myc.kboltz_au

        Qi = Qi/myc.Myr
        ndens = d / (2. * my_kboltz * T) /(myc.pc**3)
        Phi = Qi / (4. * np.pi * (r*myc.pc) ** 2)

    #end = time.time()
    #print(end - start)


    # old version
    #mycool = 10.**onlycoolfunc(np.log10([ndens, T, Phi]))
    #myheat = 10.**onlyheatfunc(np.log10([ndens, T, Phi]))
    #dudt = (myheat-mycool)/ myc.dudt_cgs # negative if more cooling than heating
    # end of old version

    #start = time.time()

    # interpolation range (currently repeated in calc_Lb --> merge?)
    log_T_interd = 0.1
    log_T_noeqmin = Cool_Struc["log_T"]["min"]+1.0001*log_T_interd
    log_T_noeqmax = Cool_Struc["log_T"]["max"] - 1.0001 * log_T_interd
    log_T_intermin = log_T_noeqmin - log_T_interd
    log_T_intermax = log_T_noeqmax + log_T_interd
    #print log_T_noeqmin, log_T_noeqmax, log_T_intermin, log_T_intermax
    #log_T_intermin = 3.501
    #log_T_noeqmin = 3.601
    #log_T_noeqmax = 5.399
    #log_T_intermax = 5.499

    #debug (use semi-correct cooling at low T)
    if T < 10.**3.61:
        T = 10.**3.61

    # loss (or gain) of internal energy
    # print('\n\n\nEntering dudt calculation. ')
    # print(Data_Struc["cons"]["t_now"])
    # print(ndens, T, Phi)
    # np.save('cool_interp3dudtmyelement', Cool_Struc['Netcool'])
    
    # I think dudt is also in au units?
    # print(ndens, T, Phi)
    # sys.exit()
    dudt = coolnoeq.cool_interp_master({"n":ndens, "T":T, "Phi":Phi, 't_now': Data_Struc["cons"]["t_now"]}, Cool_Struc, log_T_noeqmin=log_T_noeqmin, log_T_noeqmax=log_T_noeqmax, log_T_intermin=log_T_intermin, log_T_intermax=log_T_intermax)
    # testing purpose
    # dudt = 1
    
    # # =============================================================================
    # # comment this out to check for cooling
    # # =============================================================================
    # T_arr = np.logspace(4, 8, 100)
    # dudt_arr = []
    # print('ndens', ndens, 'Phi', Phi, 't_now', Data_Struc["cons"]["t_now"])
    # for T in T_arr:
    #     dudt = coolnoeq.cool_interp_master({"n":ndens, "T":T, "Phi":Phi, 't_now': Data_Struc["cons"]["t_now"]}, Cool_Struc, log_T_noeqmin=log_T_noeqmin, log_T_noeqmax=log_T_noeqmax, log_T_intermin=log_T_intermin, log_T_intermax=log_T_intermax)
    #     dudt_arr.append(dudt)
    
    # import matplotlib.pyplot as plt
    # plt.plot(T_arr, dudt_arr)
    # plt.vlines(10**(5.5), min(dudt_arr), max(dudt_arr), linestyle = '--')
    # plt.xscale('log')
    # plt.show()
    # sys.exit()
    
    #end = time.time()
    #print(end - start)

    #start = time.time()

    vd = b + (v-a*r)*dTdr/T - 2.*v/r
    Td = dTdr
    dTdrd = c/(T**2.5) * (e + 2.5*(v-a*r)*dTdr/T - dudt/d) - 2.5*dTdr**2./T - 2.*dTdr/r # negative sign for dudt term (because of definition of dudt)

    #print Td

    #end = time.time()
    #print(end - start)
    #print '####################'

    return [vd,Td,dTdrd]

def get_r_list(Rhi, Rlo, dx0, n_extra=0):
    """
    get list or r where bubble strutcture will be calculated
    result is monotonically decreasing
    :param Rhi: upper limit for r (1st entry in r)
    :param Rlo: lower entry in r (usually last entry)
    :return: array (1D)
    """
    # figure out at which postions to calculate solution
    top = Rhi
    bot = np.max([Rlo, dx0])

    clog = 2.  # max increase in dx (in log) that is allowed, e.g. 2.0 -> dxmax = 100.*dx0 (clog = 2.0 seems to be a good compromise between speed an precision)
    dxmean = (10. ** clog - 1.) / (clog * np.log(10.)) * dx0  # average step size
    # print "Ndx", dR2, R2_prime, TR2_prime, top, bot, dxmean, dMdt
    Ndx = int((top - bot) / dxmean)  # number of steps with chosen step size
    dxlist = np.logspace(0., clog, num=Ndx) * dx0  # list of step sizes
    r = top + dx0 - np.cumsum(dxlist)  # position vector on which ODE will be solved
    r = r[r > bot]  # for safety reasons
    r = np.append(r, bot)  # make sure the last element is exactly the bottom entry
    # print('in get_r_list')
    # print('Ndx', Ndx)
    # print('top, bot', top, bot)
    # print('dxmean', dxmean)
    # print('r')
    # sys.exit()

    # extend the radius a bit further to small values, i.e. calculate a bit more than necessary
    # if n_extra has been set to 0, no extra points will be used
    #r_extra = np.linspace(r[-1] - dxmean, 0.95 * r[-1], num=n_extra)
    #r = np.append(r, r_extra)

    return r, top, bot, dxlist

def calc_bstruc_start(dMdt, params):
    """
    start values for bubble structure measured at R2_prime (upper limit of integration): velocity, temperature, dT/dr
    :param dMdt: mass flow rate into bubble
    :param params:
    :return: float (R2_prime) and list (v, T, dT/dr at r2_prime)
    """
    
    
    #path=wn.get_mypath(i.basedir, i.navg, i.Zism, SFEb, MCb)

    # initiate integration at radius R2_prime slightly less than R2
    dR2 = (params["Tgoal"]**2.5)/(temp_au * dMdt / (4. * np.pi * params["R2"] ** 2.)) # spatial separation between R2 and the point where the ODE solver is initialized (cannot be exactly at R2)
    
    path = params["path"]
    
    R1R2,R2pR2=np.loadtxt(path, skiprows=1, delimiter='\t', usecols=(0,1), unpack=True)
    
    dR2min = 1.0e-7 # IMPORTANT: this number might have to be higher in case of very strong winds (clusters above 1e7 Msol)! TO DO: figure out, what to set this number to...
    
    
    
    MC= float(os.environ["Mcl_aux"])
    SFE=float(os.environ["SF_aux"])
    
    Mclus = MC * SFE
    if Mclus>1.0e7:
        dR2min = 1.0e-14*Mclus + 1.0e-7
        
    
    
    if dR2 < dR2min: dR2 = np.sign(dR2)*dR2min # prevent super small dR2
    
    
    R2_prime = params["R2"] - dR2  # radius at which ODE solver is initialized. At this radius the temperature is Tgoal (usually set to 30,000 K)
    TR2_prime = (temp_au * dMdt * dR2/ (4. * np.pi * params["R2"] ** 2.)) ** 0.4  # should be Tgoal (usually set to 30,000 K)
    
    

    
    #print('bubstruc',R2_prime, params["rgoal"], params["R2"],dR2,params["rgoal"]/R2_prime)
    
    
    
    
    #MC=np.append(MC, 0)
    #SFE=np.append(SFE, 0)
    R1R2=np.append(R1R2, 0)
    #dmdt=np.append(dmdt, 0)
   #count=np.append(count, 0)
    R2pR2=np.append(R2pR2,R2_prime/params["R2"])
    
   
    np.savetxt(path, np.c_[R1R2,R2pR2],delimiter='\t',header='R1/R2'+'\t'+'R2p/R2')

   

    if (params["rgoal"] > R2_prime):
        sys.exit("rgoal_f is outside allowed range in bubble_structure.py (too large). Decrease r_Tb in myconfig.py (<1.0)!")

    # np.seterr(all='raise')

    #print dMdt, TR2_prime, params["R2"], R2_prime, dR2 #debugging

    dTdrR2_prime = -2. / 5. * TR2_prime / dR2  # temperature gradient at R2_prime, this is not the correct boundary condition (only a guess). We will determine the correct value using a shooting method
    vR2_prime = params["cons"]["a"] * params["R2"] - dMdt * myc.kboltz_au * TR2_prime / (4. * np.pi * params["R2"] ** 2. * 0.5 * (myc.mp/myc.Msun) * params["press"])  # velocity at R2_prime

    # y0: initial conditions for bubble structure
    y0 = [vR2_prime, TR2_prime, dTdrR2_prime]

    return R2_prime, y0

def comp_bv_au_wrap(dMdt, params):
    if dMdt == params["dMdtx0"] and params["dMdty0"] is not None:
        return params["dMdty0"]
    else:
        return comp_bv_au(dMdt, params)

def comp_bv_au(dMdt, params):
    """
    -compare boundary value calculated from guessed dMdt with true boundary conditions
    -the velocity at r=0 is supposed to be v0 (usually 0)
    -this routine is called repeatedly with different mass loss rate dMdt until the true v0 and the v0 estimated from this dMdt agree
    -then the dMdt which yields a residual which is (nearly) 0 is the dMdt we are looking for
    :param dMdt: guess for mass loss rate dM/dt
    :param params: additional parameters [v0 = v[Rsmall], cons = [a,b,c,d,e, Phi], R2_prime, R2, Rsmall, dR2, press
            Rsmall: some very small radius (nearly 0)
            R2_prime: radius very slightly smaller than shell radius R2
            R2: shell radius
            dR2: R2 - R2_prime
            press: pressure inside bubble
    :return: residual of true v(Rsmall)=v0 (usually 0) and estimated v(Rsmall)
    """
    # only include next line if using scipy.optimize.root
    # dMdt = dMdt[0]  # root finder passes an array but I want a float here

    #start = time.time()

    if hasattr(dMdt, "__len__"): dMdt = dMdt[0]
    # debug
    # print "dMdt", dMdt

    # get initial values
    R2_prime, y0 = calc_bstruc_start(dMdt, params)
    [vR2_prime, TR2_prime, dTdrR2_prime] = y0

    Data_Struc = {"cons":params["cons"], "Cool_Struc":params["Cool_Struc"]}

    # figure out at which postions to calculate solution
    n_extra = 0 # number of extra points
    dx0 = (R2_prime - params["R_small"]) / 1e6  # some initial step size in pc
    r, top, bot, dxlist = get_r_list(R2_prime, params["R_small"], dx0, n_extra=n_extra)

    # print('comp values', vR2_prime, TR2_prime, dTdrR2_prime, r)
    # try to solve the ODE (might not have a solution)
    try:
        ############## Option1: call ODE solver "odeint" ###########################
        
        
        
        
        psoln = scipy.integrate.odeint(bubble_struct, y0, r, args=(Data_Struc,), tfirst=True)
        v = psoln[:, 0]
        T = psoln[:, 1]

    #     print(    'a' , Data_Struc["cons"]["a"],
    # '\nb' , Data_Struc["cons"]["b"],
    # '\nc' , Data_Struc["cons"]["c"],
    # '\nd' , Data_Struc["cons"]["d"],
    # '\ne' , Data_Struc["cons"]["e"],
    # '\nt_now' , Data_Struc["cons"]["t_now"],
    # '\nQi' , Data_Struc["cons"]["Qi"],
    # '\ny0', y0,
    # '\nR2_prime', R2_prime, 
    # '\nparams["R_small"]', params["R_small"],
    # '\ndMdt', dMdt,
    # '\nparams["R2"]', params["R2"],
    # '\nparams["T_goal"]', params["Tgoal"],
    # '\nparams["press"]', params["press"],
    # '\nv', v,
    # )
    #     sys.exit()
        # a 4976.099527584466
        # b 5213.056647945631
        # c 6.2274324244100785e+25
        # d 380571798.5188472
        # e 3080.442564695146
        # t_now 0.00012057636642393612
        # Qi 1.6994584609226492e+67
        # y0 [1003.9291826702926, 453898.8577997466, -1815595431198.9866]
        # R2_prime 0.20207541764992493
        # params["R_small"] 0.14876975625376893
        # dMdt 40416.890252523
        # params["R2"] 0.20207551764992493

        ############################################################################

        #end = time.time()
        #print "time2:", end - start

        #start = time.time()

        ################ Option2: call ODE solver "solve_ivp" ######################
        # PROBLEM: internal step size is not adjusted accordning to the cooling function --> taken step sizes might be too large
        # Thus, we have to provide a list of points where to calculate the solution anyway and letting the solver figure out the time step is not a good idea
        # Thus, no benefit to use solve_ivp instead of odeint (except maybe event location) but solve_ivp is actually slower

        #max_step = (top - bot)/1000.0 # set max step size, otherwise steep velocity gradient at small r might be a bit undersampled
        #fun = lambda r, y: bubble_struct(r, y, Data_Struc)
        #psoln = scipy.integrate.solve_ivp(fun, [top, bot], y0, method='BDF', max_step=max_step) # use LSODA because ODE can be stiff
        #r = psoln.t
        #v = psoln.y[0]
        #T = psoln.y[1]

        ############################################################################

        #end = time.time()
        #print "time2:", end - start

        # this are the calculated boundary value (velocity at r=R_small)
        v_bot = v[-(n_extra+1)]
        #print r[-(n_extra+1)] - bot

        # compare these to correct calues!
        residual = (params["v0"] - v_bot)/v[0]
        #print dMdt, v_bot

        # very low temperatures are not allowed! This check is also necessary to prohibit rare fast (and unphysical) oscillations in the temperature profile
        min_T = np.min(T)
        #print "min_T", min_T
        if min_T < 3e3:
            residual *= (3e4/min_T)**2
            print('T_min,',min_T)

        #end = time.time()
        #print(end - start)
        #print '####################'

        #print dMdt, dMdt*(myc.Msun/myc.Myr), TR2_prime, dTdrR2_prime, vR2_prime, residual

        #print dMdt, residual

        """
        n0 = 1.
        T_weaver = 1.51e6 * 1.27 ** (8. / 35.) * n0 ** (2. / 35.) * 1.0 ** (-6. / 35.) * (1. - r/params["R2"]) ** 0.4
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()
        ax1.plot(r, v,'k')
        ax1.plot([bot,bot], [max(v[0],v_bot),0.], ':', color="gray")
        ax1.set_ylabel('km/s')
        ax1.set_yscale('symlog')
        ax1.set_xlabel('pc')
        ax2 = ax1.twinx()
        ax2.semilogy(r, T, 'r')
        ax2.semilogy(r, T_weaver, 'r:')
        ax2.set_ylabel('K', color='r')
        ax2.tick_params('y', colors='r')
        plt.show()
        """



    except:
        print('here in bub2 but now is exiting.')
        sys.exit()
        # this is the case when the ODE has no solution with chosen inital values
        print("giving a wrong residual here")
        if dTdrR2_prime < 0.:
            residual = -1e30
        else:
            residual = 1e30


    return residual

def find_dMdt(dMdt_guess, params, factor_fsolve=50., xtol=1e-6):
    """
    employ root finder to get correct dMdt (mass loss rate dM/dt from shell into shocked region)
    :param dMdt_guess: initial guess for dMdt
    :param params:
    :param factor_fsolve:
    :param xtol:
    :return:
    """
    
    path = params["path"]
    
    
    flag=False

        
    countl = float(os.environ["COUNT"])
    dmdt_0l = float(os.environ["DMDT"])
    #print('DMT',dmdt_0l)    
    

    dMdt = scipy.optimize.fsolve(comp_bv_au_wrap, dMdt_guess, args=(params), factor=factor_fsolve, xtol=xtol, epsfcn=0.1*xtol)
    if dMdt < 0:
        print('rootfinder of dMdt gives unphysical result...trying to solve again with smaller step size')
        dMdt = scipy.optimize.fsolve(comp_bv_au_wrap, dMdt_guess, args=(params), factor=15, xtol=xtol, epsfcn=0.1*xtol)
        if dMdt < 0:
            dMdt=dmdt_0l+ dmdt_0l*1e-3 #if its unphysical again, take last dmdt and change it slightly for next timestep
            countl+=1 # count how often you did this crude approximation
            flag=True
            if countl >3:
                sys.exit("Unable to find correct dMdt, have to abort WARPFIELD")
                
    
    #dMdt = scipy.optimize.brentq(comp_bv_au, 0.99 * dMdt_guess, 1.01 * dMdt_guess, args=(params), xtol=xtol)

    """
    try:
        #print "going with first guess"
        dMdt = scipy.optimize.fsolve(comp_bv_au_wrap, dMdt_guess, args=(params), factor=factor_fsolve, xtol=xtol)
    except:
        #print "changing dMdt_guess"
        dMdt_guess *= 0.99
        dMdt = scipy.optimize.fsolve(comp_bv_au_wrap, dMdt_guess, args=(params), factor=factor_fsolve, xtol=xtol)
        
 
    """
    try:
        os.environ["DMDT"] = str(dMdt[0])
    except:
        os.environ["DMDT"] = str(dMdt)
        
    
    if flag ==True:
        os.environ["COUNT"] = str(countl)
    else:
        os.environ["COUNT"] = str(0)
    

    return dMdt

# counter only here temporary
def calc_Lb(data_struc, Cool_Struc, counter, rgoal_f=i.r_Tb, verbose=0, plot=0, no_calc=False, error_exit = True, xtol=1e-6):
    """
    calculate luminosity lost to cooling, and bubble temperature at radius rgoal_f*R2
    whole routine assumes units are Myr, Msun, pc and also returns result (Lb) in those units
    :param data_struc: for alpha, beta, delta, see definitions in Weaver+77, eq. 39, 40, 41
    :param rgoal_f: optional, sets location where temperature of bubble is reported: r = rgoal_f * R2; R1/R2 < rgoal_f < 1.
    :return: cooling luminity Lb, temperature at certain radius T_rgoal
    """
    #np.seterr(all='raise')
    # unpack input data
    # cgs units!!!
    alpha = data_struc['alpha']
    beta = data_struc['beta']
    delta = data_struc['delta']
    Eb = data_struc['Eb'] # bubble energy
    R2 = data_struc['R2'] # shell radius in pc
    t_now = data_struc['t_now'] # current time in Myr
    Lw = data_struc['Lw'] # mechanical luminosity
    vw = data_struc['vw'] # wind luminosity (and SNe ejecta)
    dMdt_factor = data_struc['dMdt_factor'] # guess for dMdt_factor (classical Weaver is 1.646)
    Qi = data_struc['Qi'] # current photon flux of ionizing photons
    v0 = 0.0 # velocity at r --> 0.
    #Cool_Struc = data_struc['Cool_Struc']
    
    
    print('here are the parameters for calc_Lb. This can be used for debugging in new code.')
    
    print('\nalpha' , data_struc["alpha"])
    print('beta' , data_struc["beta"])
    print('delta' , data_struc["delta"])
    print('Eb' , data_struc["Eb"])
    print('R2' , data_struc["R2"])
    print('t_now' , data_struc["t_now"])
    print('Lw' , data_struc["Lw"])
    print('vw' , data_struc["vw"])
    print('dMdt_factor' , data_struc["dMdt_factor"])
    print('Qi' , data_struc["Qi"])
    print('v0' , v0)
    sys.exit()
          
    onlycoolfunc = Cool_Struc['Cfunc']
    onlyheatfunc = Cool_Struc['Hfunc']

    #start = time.time()
    #onlycoolfunc, onlyheatfunc = coolnoeq.create_onlycoolheat(i.Zism, t_now*myc.Myr)
    #end = time.time()
    #print "time", end - start

    bub_error = 0

    # interpolation range (currently repeated in bubble_struct --> merge?)
    log_T_interd = 0.1
    log_T_noeqmin = Cool_Struc["log_T"]["min"] + 1.01 * log_T_interd
    log_T_noeqmax = Cool_Struc["log_T"]["max"] - 1.01 * log_T_interd
    log_T_intermin = log_T_noeqmin - log_T_interd
    log_T_intermax = log_T_noeqmax + log_T_interd

    ## debugging
    if verbose==1: print(data_struc)

    # calculate R1 (inner discontinuity)
    # R1 = aux.cubic_eq(1., vw*Eb/Lw, 0., -R2**3.) # does not always work
    R1 = scipy.optimize.brentq(R1_zero, 1e-3*R2, R2, args=([Lw, Eb, vw, R2]), xtol=1e-18) # can go to high precision because computationally cheap (2e-5 s)

    np.save('/Users/jwt/Documents/Code/warpfield3/outputs/cool.npy', Cool_Struc)
    # print('debug starts here')
    # print("data", data_struc)
    # print('r1',R1)
    
    #print "beta, delta", beta, delta, "R1, R2", R1, R2

    #press = (5./3. - 1.) * 3. * Eb / (4. * np.pi * (R2**3.)) # old, used for debugging
    press = state_eq.PfromE(Eb, R2, R1)
    
    # print('pressure', press)
    # print('constants', alpha, beta, delta, t_now, press)
    # 0.6 0.8 -0.17142857142857143 0.00012057636642393612 380571798.5188472
    cons = calc_cons(alpha, beta, delta, t_now, press, units='au')
    cons["Qi"] = Qi
    
    # print('cons', cons)

    # eq. 33
    
    dMdt_guess=float(os.environ["DMDT"])
    
    if dMdt_guess == 0:
        dMdt_guess = 4. / 25. * dMdt_factor * 4. * np.pi * R2 ** 3. / t_now * 0.5 * (myc.mp / myc.Msun) / myc.kboltz_au * (t_now * myc.Cspitzer_au / R2 ** 2.) ** (2. / 7.) * press ** (5. / 7.)

    # initiate integration at radius R2_prime slightly less than R2 (we define R2_prime by T(R2_prime) = TR2_prime
    TR2_prime = 3e4 # this is the temperature at R2_prime (important: must be > 1e4 K)
    # print "debug", R1, R2*rgoal_f
    
    path=os.environ["Bstrpath"]
    #MC,SFE,R1R2,R2pR2,dmdt,count=np.loadtxt(path, skiprows=1, delimiter='\t', usecols=(0,1,2,3,4,5), unpack=True)
    
    
    R1R2,R2pR2=np.loadtxt(path, skiprows=1, delimiter='\t', usecols=(0,1), unpack=True)
    
    
    rgoal_f=aux.calcr_Tb(R1R2,R2pR2)

    # print('r1r2 here')
    # print(R1R2, R2pR2, R1/R2)'
    
    
    R1R2=np.append(R1R2, R1/R2)
    R2pR2=np.append(R2pR2,0)
    
    np.savetxt(path, np.c_[R1R2,R2pR2],delimiter='\t',header='R1/R2'+'\t'+'R2p/R2')  

    # check whether R1 is smaller than the radius at which bubble temperature will be calculated
    if (R2*rgoal_f < R1):
        # for now, stop code. However, could also say that energy-driven phase is over if rgoal_f < R1/R2, for a reasonable rgoal_f (0.99 or so)
        # print "debug, rgoal_f", rgoal_f,
        sys.exit("rgoal_f is outside allowed range in bubble_structure.py (too small). Increase r_Tb in myconfig.py (<1.0)!")
    else:
        rgoal = rgoal_f*R2

    # print("rgoal", rgoal_f, R2)
    # for the purpose of solving initially, try to satisfy BC v0 = 0 at some small radius R_small
    #R_small = np.min([R1,0.015]) #0.011 # in an old version, R_small = R1
    R_small = R1 # I presume Weaver meant: v -> 0 for R -> R1 (since in that chapter he talks about R1 being very small)
    params = {"v0":v0, "cons":cons, "rgoal":rgoal, "Tgoal":TR2_prime, "R2":R2, "R_small":R_small, "press":press, "Cool_Struc":Cool_Struc, "path":path}

    #try:
    #    #with stdout_redirected():
    #    dMdt = scipy.optimize.brentq(comp_bv_au, 3e-1 * dMdt_guess, 3e0 * dMdt_guess, args=(params), xtol=1e-5)
    #except:
    #    try:
    #        dMdt = scipy.optimize.brentq(comp_bv_au, 1e-1 * dMdt_guess, 1e1 * dMdt_guess, args=(params))
    #    except:
    #        try:
    #            dMdt = scipy.optimize.brentq(comp_bv_au, 3e-2 * dMdt_guess, 3e1 * dMdt_guess, args=(params))
    #        except:
    #            bub_error = 1

    # other variants:
    #dMdt = scipy.optimize.brentq(comp_bv_au, 2e-1 * dMdt_guess, 5e0 * dMdt_guess, args=(params))

    #start = time.time()

    # prepare wrapper (to skip 2 superflous calls in fsolve)
    params["dMdtx0"] = dMdt_guess
    params["dMdty0"] = comp_bv_au(dMdt_guess, params)
    #dMdt = scipy.optimize.fsolve(comp_bv_au_wrap, dMdt_guess, args=(params), factor = 1.5, xtol=xtol)

    # print("params[\"dMdty0\"]", params["dMdty0"])

    # 1. < factor_fsolve < 100.; if factor_fsolve is chose large, the rootfinder usually finds the solution faster
    # however, the rootfinder may then also enter a regime where the ODE soultion becomes unphysical
    # low factor_fsolve: slow but robust, high factor_fsolve: fast but less robust
    factor_fsolve = 50. #50

    # find correct dMdt
    '''
    if i.output_verbosity <= 0:
        with stdout_redirected():
            dMdt = find_dMdt(dMdt_guess, params, factor_fsolve=factor_fsolve, xtol=1e-6)
    else:
        dMdt = find_dMdt(dMdt_guess, params, factor_fsolve=factor_fsolve, xtol=1e-6)
        
    '''
    
    if i.output_verbosity <= 0:
        with stdout_redirected():
            dMdt = find_dMdt(dMdt_guess, params, factor_fsolve=factor_fsolve, xtol=1e-3)
    else:
        dMdt = find_dMdt(dMdt_guess, params, factor_fsolve=factor_fsolve, xtol=1e-3) #1e-6
        # print('\nhere is the dMdt estimate in bstructure2.py.\n')
        # print('inputs')
        # print(dMdt_guess, params, factor_fsolve)
        # print(dMdt)
        # sys.exit()

    # print("dMdt", dMdt_guess, dMdt)

    # if output is an array, make it a float
    if hasattr(dMdt, "__len__"): dMdt = dMdt[0]
    #print dMdt

    # 2 kinds of problem can occur:
    #   Problem 1 (for very high beta): dMdt becomes negative, the cooling luminosity diverges towards infinity
    #   Problem 2 (for very low beta): the velocity profile has negative velocities

    #end = time.time()
    #print "time", end - start


    # CHECK 1: negative dMdt must not happen! (negative velocities neither, check later)
    """
    if dMdt <= 0.:
        #data_struc.pop("Cool_Struc")
        bub_error = 1
        if error_exit:
            print "data_struc in bubble_structure2:", data_struc
            print "counter, etc. in bubble_structure2:", counter, rgoal_f, verbose, plot, no_calc
            sys.exit("could not find correct dMdt in bubble_structure.py")
        else:
            print("Warning: Negative dMdt")
            Lb = 100.*max(1.,np.abs((dMdt/100.)))*Lw # will be positive
            T_rgoal = -1.*max(1.,np.abs(dMdt/100.))*1e6 #max(1e5/max(np.abs(dMdt)**2,1e2),0.1) # low temperature (K)
            #Lb = np.nan
            #T_rgoal = np.nan

            Lb_b = np.nan
            Lb_cz = np.nan
            Lb3 = np.nan
            dMdt_factor_out = np.nan
            Tavg = np.nan
            Mbubble = np.nan
            r_Phi = np.nan
            Phi_grav_r0b = np.nan
            f_grav = np.nan

            return Lb, T_rgoal, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav
    """

    # new factor for dMdt (used in next time step to get a good initial guess for dMdt)
    dMdt_factor_out = dMdt_factor * dMdt/dMdt_guess



    """ # old get inital values
    # recalculate R2_prime (was already calculated in the routine above which calculated dMdt, but was not passed on
    dR2 = (TR2_prime ** 2.5) / (temp_au * dMdt / (4. * np.pi * R2 ** 2.))
    dR2min = 1.0e-7 # IMPORTANT: this number might have to be higher in case of very strong winds (clusters above 1e7 Msol)! TO DO: figure out, what to set this number to...
    if dR2 < dR2min: dR2 = np.sign(dR2)*dR2min # prevent super small dR2

    R2_prime = R2 - dR2

    dTdrR2_prime = -2. / 5. * TR2_prime / dR2  # temperature gradient at R2_prime
    vR2_prime = alpha * R2 / t_now - dMdt * myc.kboltz_au * TR2_prime / (4. * np.pi * R2 ** 2. * 0.5 * (myc.mp / myc.Msun) * press)  # velocity at R2_prime

    y0 = [vR2_prime,TR2_prime,dTdrR2_prime]
    """

    # get initial values
    R2_prime, y0 = calc_bstruc_start(dMdt, params)
    [vR2_prime, TR2_prime, dTdrR2_prime] = y0
    
    # print("R2_prime, y0", R2_prime, y0)


    # now we know the correct dMdt, but since we did not store the solution for T(r), calculate the solution once again (IMPROVE?)
    # figure out at which positions to calculate solution
    n_extra = 0  # number of extra points
    deltaT_min = 5000. # resolve at least temperature differences of deltaT_min
    dx0 = min( np.abs(deltaT_min/dTdrR2_prime), (R2_prime-R1)/1e6 )
    #dx0 = (R2_prime - params["R_small"]) / 1e6  # some initial step size in pc
    r, top, bot, dxlist = get_r_list(R2_prime, params["R_small"], dx0, n_extra=n_extra)

    rgoal_idx = np.argmin(np.abs(r-rgoal)) # find index where r is closest to rgoal
    r[rgoal_idx] = rgoal # replace that entry in r with rgoal (with that we can ensure that we get the correct Tgoal


    Data_Struc = {"cons": cons, "Cool_Struc": Cool_Struc}

    # if output verbosity is low, do not show warnings
    if i.output_verbosity <= 0:
        with stdout_redirected():
            psoln = scipy.integrate.odeint(bubble_struct, y0, r, args=(Data_Struc,), tfirst=True)
    else:
        psoln = scipy.integrate.odeint(bubble_struct, y0, r, args=(Data_Struc,), tfirst=True)
            


    v = psoln[:,0]
    T = psoln[:,1]
    dTdr = psoln[:,2]
    n = press/((i.mui/i.mua) * myc.kboltz_au*T) # electron density (=proton density), assume astro units (Msun, pc, Myr)

        
    # print('v', v)
    # print('T', T)
    # print('dTdr', dTdr)
    # print('n', n)


    """
    print "dMdt_guess:", dMdt_guess, "dMdt:", dMdt # debug
    print "v(R1):", v[-1]
    print "T:", T[0:3]

    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax1.semilogy(r,psoln[:,0],'k')
    ax1.semilogy(r, -1.*psoln[:, 0], '--k')
    ax1.set_ylabel('km/s')
    ax1.set_xlabel('pc')
    ax2 = ax1.twinx()
    ax2.semilogy(r, psoln[:, 1], 'r')
    #ax2.semilogy(r, T_weaver, 'r:')
    ax2.set_ylabel('K', color='r')
    ax2.tick_params('y', colors='r')
    plt.show()
    """


    # CHECK 2: negative velocities must not happen! (??)
    min_v = np.min(v[r > (bot + 0.05 * (top - bot))])
    """
    if (min_v/v[0] < -0.1):
        bub_error = 1
        if error_exit:
            print "data_struc in bubble_structure2:", data_struc
            print "counter, etc. in bubble_structure2:", counter, rgoal_f, verbose, plot, no_calc
            sys.exit("could not find correct dMdt in bubble_structure.py")
        else:
            print("Warning: Negative velocity")
            Lb = 100.*max(1.,np.abs(min_v)**2)*Lw # will be positive
            T_rgoal = max(1.,np.abs(min_v)**2)*1e9  # high temperature 1e9
            #Lb = np.nan
            #T_rgoal = np.nan

            Lb_b = np.nan
            Lb_cz = np.nan
            Lb3 = np.nan
            dMdt_factor_out = np.nan
            Tavg = np.nan
            Mbubble = np.nan
            r_Phi = np.nan
            Phi_grav_r0b = np.nan
            f_grav = np.nan

            return Lb, T_rgoal, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav
    """

    # CHECK 3: temperatures lower than 1e4K should not happen
    min_T = np.min(T)
    if (min_T < 1e4):
        bub_error = 1
        if error_exit:
            print("data_struc in bubble_structure2:", data_struc)
            print("counter, etc. in bubble_structure2:", counter, rgoal_f, verbose, plot, no_calc)
            sys.exit("could not find correct dMdt in bubble_structure.py")
        else:
            print("Warning: temperature below 1e4K", min_T)
            Lb = 100. * max(1., np.abs(min_T-1e4) ** 2) * Lw  # will be positive
            T_rgoal = -1.*(min_T-1e4)*1e6  # high temperature 1e9
            # Lb = np.nan
            # T_rgoal = np.nan

            Lb_b = np.nan
            Lb_cz = np.nan
            Lb3 = np.nan
            dMdt_factor_out = np.nan
            Tavg = np.nan
            Mbubble = np.nan
            r_Phi = np.nan
            Phi_grav_r0b = np.nan
            f_grav = np.nan

            return Lb, T_rgoal, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav

    if no_calc:
        Lb = None
        Lb_b = None
        Lb_cz = None
        Lb3 = None
        Tavg = None
        Mbubble = None
        T_rgoal = None
        Phi_grav_r0b = None  # gravitational potential
        f_grav = None  # gravitational force per unit mass
    else:
        # find 1st index where temperature is above Tborder ~ 3e5K (above this T, cooling becomes less efficient and less resolution is ok)
        #idx_6 = aux.find_nearest_higher(T,0.99*Cool_Struc["log_T"]["max"]) # WRONG

        # at Tborder we will switch between usage of CIE and non-CIE cooling curves
        Tborder = 10 ** log_T_noeqmax

        # find index of radius at which T is closest (and higher) to Tborder
        idx_6 = aux.find_nearest_higher(T, Tborder)

        # find index of radius at which T is closest (and higher) to 1e4K (no cooling below!), needed later
        idx_4 = aux.find_nearest_higher(T, 1e4)

        #print "idx", idx_4, idx_6, dTdr[0]

        # interpolate and insert, so that we do have an entry with exactly Tborder
        if (idx_4 != idx_6):
            iplus = 20
            r46_interp = r[:idx_6+iplus]
            fT46 = interp1d(r46_interp, T[:idx_6+iplus] - Tborder, kind='cubic') # zero-function for T=Tborder
            fdTdr46 = interp1d(r46_interp, dTdr[:idx_6 + iplus], kind='linear')

            # calculate radius where T==Tborder
            rborder = scipy.optimize.brentq(fT46, np.min(r46_interp), np.max(r46_interp), xtol=1e-14)
            nborder = press/((i.mui/i.mua) * myc.kboltz_au*Tborder)
            dTdrborder = fdTdr46(rborder)

            # insert quantities at rborder to the full vectors
            dTdr = np.insert(dTdr,idx_6,dTdrborder)
            T = np.insert(T, idx_6, Tborder)
            r = np.insert(r, idx_6, rborder)
            n = np.insert(n, idx_6, nborder)

        ######################## 1) low resolution region (bubble) (> 3e5 K) ##############################
        r_b = r[idx_6:] # certain that we need -1?
        T_b = T[idx_6:]
        dTdr_b = dTdr[idx_6:]
        n_b = n[idx_6:] # electron density (=proton density)

        # at temperatures above 3e5 K assumption of CIE is valid. Assuming CIE, the cooling can be calculated much faster than if CIE is not valid.

        #print("Cooling")
        #start = time.time()

        #print "min T_b, max T_b", np.min(T_b), np.max(T_b)

        Lambda_b = 10.**(f_logLambdaCIE(np.log10(T_b))) / myc.Lambda_cgs
        #Lambda_b = cool.coolfunc_arr(T_b) / myc.Lambda_cgs  # assume astro units (Msun, pc, Myr) # old (slow) version
        integrand = n_b ** 2 * Lambda_b * 4. * np.pi * r_b ** 2

        #end = time.time()
        #print(end - start)


        # power lost to cooling in bubble without conduction zone (where 1e4K < T < 3e5K)
        Lb_b = np.abs(np.trapz(integrand, x=r_b))

        # intermediate result for calculation of average temperature
        Tavg_tmp_b = np.abs(np.trapz(r_b**2*T_b, x=r_b))

        #if verbose==1:
        #    #print "####### v(0) is ", v[-1]
        #    print "####### v(R1) is ", v[-1]
        #if abs(v[-1]) > 1e-6:
        #    print "******************* Warning! v(R1) should be 0 km/s but is ", v[-1]
        #    print dMdt_guess, dMdt
        #if abs(v[-1] - v0) > 1e-3:
        #    print "Input Data: data_struc, rgoal_f"
        #    print data_struc
        #    print rgoal_f
        #    sys.exit('Wrong dM/dt chosen, velocity wrong')

        ######################### 2) high resolution region (conduction zone, 1e4K - 3e5K) #######################


        # there are 2 possibilities:
        # 1. the conduction zone extends to temperatures below 1e4K (unphysical, photoionization!)
        # 2. the conduction zone extends to temperatures above 1e4K

        # in any case, take the index where temperature is just above 1e4K

        if (idx_4 != idx_6): # it could happen that idx_4 == idx_6 == 0 if the shock front is very, very steep
            #start = time.time()
            if idx_6 - idx_4 < 100: # if this zone is not well resolved, solve ODE again with high resolution (IMPROVE BY ALWAYS INTERPOLATING)
                dx = (r[idx_4]-r[idx_6])/1e4 # want high resolution here
                top = r[idx_4]
                bot = np.max([r[idx_6]-dx,dx])

                r_cz = np.arange(top, bot, -dx)

                # since we are taking very small steps in r, the solver might bitch around --> shut it up
                with stdout_redirected():
                    psoln = scipy.integrate.odeint(bubble_struct, [v[idx_4],T[idx_4],dTdr[idx_4]], r_cz, args=(Data_Struc,), tfirst=True) # solve ODE again, there should be a better way (event finder!)

                T_cz = psoln[:,1]
                dTdr_cz = psoln[:,2]
                dTdR_4 = dTdr_cz[0]
            ######################
            else:
                r_cz = r[:idx_6+1]
                T_cz = T[:idx_6+1]
                dTdr_cz = dTdr[:idx_6+1]
                dTdR_4 = dTdr_cz[0]

            # TO DO: include interpolation

            n_cz = press/((i.mui/i.mua)*myc.kboltz_au*T_cz) # electron density (=proton density), assume astro units (Msun, pc, Myr)
            Phi_cz = (Qi/myc.Myr) / (4. * np.pi * (r_cz*myc.pc) ** 2)

            mycool = 10. ** onlycoolfunc(np.transpose(np.log10([n_cz / myc.pc ** 3, T_cz, Phi_cz])))
            myheat = 10. ** onlyheatfunc(np.transpose(np.log10([n_cz / myc.pc ** 3, T_cz, Phi_cz])))
            dudt_cz = (myheat - mycool) / myc.dudt_cgs
            integrand = dudt_cz * 4. * np.pi * r_cz ** 2

            # power lost to cooling in conduction zone
            #Lb_cz = - scipy.integrate.simps(integrand,x=r_cz) # negative sign because I want to luminosity lost as a positive number
            Lb_cz = np.abs(np.trapz(integrand, x=r_cz))

            #print "Lb_cz:", Lb_cz

            # intermediate result for calculation of average temperature
            Tavg_tmp_cz = np.abs(np.trapz(r_cz ** 2 * T_cz, x=r_cz))

        elif ((idx_4 == idx_6) and (idx_4 == 0)):
            dTdR_4 = dTdr_b[0]
            Lb_cz = 0.

        ############################## 3) region between 1e4K and T[idx_4] #######################################

        #(use triangle approximation, i.e.interpolate)
        # If R2_prime was very close to R2 (where the temperature should be 1e4K), then this region is tiny (or non-existent)

        # find radius where temperature would be 1e4 using extrapolation from the measured T just above 1e4K, i.e. T[idx_4]
        T4 = 1e4
        R2_1e4 = (T4-T[idx_4])/dTdR_4 + r[idx_4]
        # this radius should be larger than r[idx_4] since temperature is monotonically decreasing towards larger radii
        #print(dTdR_4)
        #print(r_b[0], T_b[0])
        if R2_1e4 < r[idx_4]:
            sys.exit("Something went wrong in the calculation of radius at which T=1e4K in bubble_structure.py")

        # interpolate between R2_prime and R2_1e4 (triangle)
        # it's important to interpolate because the cooling function varies a lot between 1e4 and 1e5K
        f3 = interp1d(np.array([r[idx_4],R2_1e4]), np.array([T[idx_4],T4]), kind = 'linear')
        #f = interp1d(np.append(r_cz[5:0:-1], [R2_1e4]), np.append(T_cz[5:0:-1], [T4]), kind='quadratic')
        r3 = np.linspace(r[idx_4],R2_1e4,num=1000,endpoint=True)
        T3 = f3(r3)
        n3 = press/((i.mui/i.mua)*myc.kboltz_au*T3) # electron density (=proton density), assume astro units (Msun, pc, Myr)
        Phi3 = (Qi/myc.Myr) / (4. * np.pi * (r3*myc.pc) ** 2)

        mask = {'loT': T3 < Tborder, 'hiT': T3 >= Tborder}
        Lb3 = {}
        for mask_key in ['loT', 'hiT']:
            msk = mask[mask_key]

            if mask_key == "loT":
                mycool = 10. ** onlycoolfunc(np.transpose(np.log10([n3[msk] / myc.pc ** 3, T3[msk], Phi3[msk]])))
                myheat = 10. ** onlyheatfunc(np.transpose(np.log10([n3[msk] / myc.pc ** 3, T3[msk], Phi3[msk]])))
                dudt3 = (myheat - mycool) / myc.dudt_cgs
                integrand = dudt3 * 4. * np.pi * r3[msk] ** 2
            elif mask_key == "hiT":
                Lambda_b = 10. ** (f_logLambdaCIE(np.log10(T3[msk]))) / myc.Lambda_cgs
                # Lambda_b = cool.coolfunc_arr(T_b) / myc.Lambda_cgs  # assume astro units (Msun, pc, Myr) # old (slow) version
                integrand = n3[msk] ** 2 * Lambda_b * 4. * np.pi * r3[msk] ** 2

            #Lb3 = - scipy.integrate.simps(integrand,x=r3) # negative sign because I want to luminosity lost as a positive number
            Lb3[mask_key] = np.abs(np.trapz(integrand, x=r3[msk]))

        Lb3 = Lb3['loT'] + Lb3['hiT']
        #print('Lb3', Lb3)

        # intermediate result for calculation of average temperature
        Tavg_tmp_3 = np.abs(np.trapz(r3 ** 2 * T3, x=r3))

        ################################################################################################################

        # add up cooling luminosity from the 3 regions
        Lb = Lb_b + Lb_cz + Lb3
        # print("Lb_b + Lb_cz + Lb3", Lb_b, Lb_cz,Lb3)
        # sys.exit('stop')

        if (idx_4 != idx_6):
            Tavg = 3.* (Tavg_tmp_b/(r_b[0]**3. - r_b[-1]**3.) + Tavg_tmp_cz/(r_cz[0]**3. - r_cz[-1]**3.) + Tavg_tmp_3/(r3[0]**3. - r3[-1]**3.))
        else:
            Tavg = 3. * (Tavg_tmp_b / (r_b[0] ** 3. - r_b[-1] ** 3.) + Tavg_tmp_3 / (r3[0] ** 3. - r3[-1] ** 3.))

        #print dMdt

        # find solution at rgoal
        #idx = aux.find_nearest_higher(r, rgoal)
        #top = r[idx]
        #bot = r[idx-1]
        #dx = (top - bot) / 1e6
        #r_around_rgoal = np.arange(top, bot, -dx)
        #psoln = scipy.integrate.odeint(bubble_struct, [v[idx], T[idx], dTdr[idx]], r_around_rgoal, args=(Data_Struc,), tfirst=True)
        #T_around_rgoal = psoln[:,1]
        #dTdr_around_rgoal = psoln[:,2]
        #idx = aux.find_nearest(r_around_rgoal, rgoal)
        #T_rgoal = T_around_rgoal[idx] + dTdr_around_rgoal[idx] * (rgoal - r_around_rgoal[idx])



        # get temperature inside bubble at fixed scaled radius
        if rgoal > r[idx_4]: # assumes that r_cz runs from high to low values (so in fact I am looking for the highest element in r_cz)
            T_rgoal = f3(rgoal)
        elif rgoal > r[idx_6]: # assumes that r_cz runs from high to low values (so in fact I am looking for the smallest element in r_cz)
            idx = aux.find_nearest(r_cz, rgoal)
            T_rgoal = T_cz[idx] + dTdr_cz[idx]*(rgoal - r_cz[idx])
        else:
            idx = aux.find_nearest(r_b, rgoal)
            T_rgoal = T_b[idx] + dTdr_b[idx]*(rgoal - r_b[idx])

        ################### gravitational potential #######################################

        # some random debug values
        r_Phi = np.array([r[0]])
        Phi_grav_r0b = np.array([5.0])
        f_grav = np.array([5.0])
        Mbubble = 10.
        """
        # get graviational potential (in cgs units)
        # first we need to flip the r and n vectors (otherwise the cumulative mass will be wrong)
        r_Phi_tmp = np.flip(r,0)*myc.pc # now r is monotonically increasing
        rho_tmp =  (np.flip(n,0)/(myc.pc**3))*myc.mp # mass density (monotonically increasing)
        dx = np.flip(dxlist,0)
        m_r_tmp = rho_tmp * 4.*np.pi*r_Phi_tmp**2 * dx*myc.pc # mass per bin (number density n was in 1/pc**3)
        Mcum_tmp = np.cumsum(m_r_tmp) # cumulative mass
        #Phi_grav_tmp = myc.Grav*Mcum_tmp[0]/r_Phi_tmp[0] # gravitational potential WRONG
        Phi_grav_r0b = -4.*np.pi*myc.Grav * scipy.integrate.simps(r_Phi_tmp*rho_tmp,x=r_Phi_tmp)
        f_grav_tmp = myc.Grav*Mcum_tmp/r_Phi_tmp**2. # gravitational force per unit mass

        # skip some entries, so that length becomes 100, then concatenate the last 10 entries (potential varies a lot there)
        len_r = len(r_Phi_tmp)
        skip = max(int(float(len_r) / float(i.pot_len_intern)),1)
        r_Phi = np.concatenate([r_Phi_tmp[0:-10:skip], r_Phi_tmp[-10:]]) # flip lists (r was monotonically decreasing)
        #Phi_grav = np.concatenate([Phi_grav_tmp[0:-10:skip], Phi_grav_tmp[-10:]])
        f_grav = np.concatenate([f_grav_tmp[0:-10:skip], f_grav_tmp[-10:]])

        # mass of material inside bubble (in solar masses)
        Mbubble = Mcum_tmp[-1]/myc.Msun
        """

        #end = time.time()
        #print(end - start)

    if i.savebubble is True:
        # save bubble structure as .txt file (radius, density, temperature)
        # only save Ndat entries (equally spaced in index, skip others)
        Ndat = 450
        len_r_b = len(r_b)
        Nskip = int(max(1,len_r_b/Ndat))

        rsave = np.append(r_b[-1:Nskip:-Nskip],r_b[0])
        nsave = np.append(n_b[-1:Nskip:-Nskip],n_b[0])
        Tsave = np.append(T_b[-1:Nskip:-Nskip],T_b[0])
        if idx_6 != idx_4: # conduction zone is resolved
            Ndat = 50
            len_r_cz = len(r_cz)
            Nskip = int(max(1, len_r_cz / Ndat))

            rsave = np.append(rsave,r_cz[-Nskip-1:Nskip:-Nskip]) # start at -2 to make sure not to have the same radius as in r_b again
            nsave = np.append(nsave,n_cz[-Nskip-1:Nskip:-Nskip])
            Tsave = np.append(Tsave,T_cz[-Nskip-1:Nskip:-Nskip])

            rsave = np.append(rsave,r_cz[0])
            nsave = np.append(nsave, n_cz[0])
            Tsave = np.append(Tsave, T_cz[0])

        # convert units to cgs for cloudy
        rsave *= myc.pc
        nsave *= myc.pc**(-3.)


        bub_savedata = {"r_cm": rsave, "n_cm-3":nsave, "T_K":Tsave}
        name_list = ["r_cm", "n_cm-3", "T_K"]
        tab = Table(bub_savedata, names=name_list)
        mypath = data_struc["mypath"]
        age1e7_str = ('{:0=5.7f}e+07'.format(t_now / 10.)) # age in years (factor 1e7 hardcoded), naming convention matches naming convention for cloudy files
        outname = os.path.join(mypath, "bubble/bubble_SB99age_"+age1e7_str+".dat")
        formats = {'r_cm': '%1.9e', 'n_cm-3': '%1.5e', 'T_K': '%1.5e'}
        #ascii.write(bubble_data,outname,names=names,overwrite=True)
        tab.write(outname, format='ascii', formats=formats, delimiter="\t", overwrite=True)

        if i.write_cloudy is True:
            # create cloudy bubble.in file
            __cloudy_bubble__.write_bubble(outname,Z=i.Zism)


    # if (plot==1 and i.plot_data):

    #     """
    #     # comparison to classical weaver result (Weaver 77, eq. 37, fig. 3)
    #     if Lw < 1e30: #astro units!
    #         L36 = Lw*myc.L_cgs / 1e36
    #     else: # cgs units
    #         L36 = Lw/1e36
    #     Lb_weaver = 0.
    #     n0 = 1. # ambient density (should read this in, but only need n0 here, so not worth it)

    #     r_temp = r_b
    #     xi = r_temp/R2
    #     T_weaver = 1.51e6 * L36 ** (8. / 35.) * n0 ** (2. / 35.) * t_now ** (-6. / 35.) * (1. - xi) ** 0.4
    #     n_weaver = press / (2. * myc.kboltz_au * T_weaver)
    #     Lambda_weaver = cool.coolfunc_arr(T_weaver) / myc.Lambda_cgs
    #     integrand = n_weaver**2*Lambda_weaver*4.*np.pi*r_temp**2
    #     Lb_weaver += np.abs(np.trapz(integrand,x=r_temp))

    #     r_temp = r_cz
    #     xi = r_temp / R2
    #     T_weaver = 1.51e6 * L36 ** (8. / 35.) * n0 ** (2. / 35.) * t_now ** (-6. / 35.) * (1. - xi) ** 0.4
    #     n_weaver = press / (2. * myc.kboltz_au * T_weaver)
    #     Lambda_weaver = cool.coolfunc_arr(T_weaver) / myc.Lambda_cgs
    #     integrand = n_weaver ** 2 * Lambda_weaver * 4. * np.pi * r_temp ** 2
    #     Lb_weaver += np.abs(np.trapz(integrand, x=r_temp))

    #     print 'Lb_weaver', '%.2e' %Lb_weaver

    #     xi = r / R2
    #     T_weaver = 1.51e6 * L36 ** (8. / 35.) * n0 ** (2. / 35.) * t_now ** (-6. / 35.) * (1. - xi) ** 0.4
    #     """

    #     # plot results
    #     fig, ax1 = plt.subplots()

    #     ax1.loglog(r, n/myc.pc**3 , 'k')
    #     ax1.set_ylabel('1/ccm')
    #     ax1.set_xlabel('pc')
    #     ax1.set_ylim([1e-1, 1e6])

    #     #ax1.loglog(r, v , 'k')
    #     #ax1.set_ylabel('km/s')
    #     #ax1.set_xlabel('pc')
    #     #ax1.set_ylim([1e-1, 1e3])

    #     ax2 = ax1.twinx()
    #     ax2.loglog(r, T, 'r')
    #     #ax2.loglog(r, T_weaver,'r:')
    #     ax2.set_ylabel('K', color='r')
    #     ax2.tick_params('y', colors='r')
    #     ax2.loglog(np.array([rgoal, rgoal]), np.array([1e5, T_rgoal]), 'k--')
    #     ax2.set_ylim([1e4,1e8])
    #     #plt.show()
    #     if counter < 10: str_counter = '000'+str(counter)
    #     elif counter < 100: str_counter = '00'+str(counter)
    #     elif counter < 1000: str_counter = '0' + str(counter)
    #     else: str_counter = str(counter)
    #     figpath = os.path.join('/home/daniel/Documents/work/temp/T_profile/', str_counter+'.png')
    #     plt.savefig(figpath)
    #     plt.close(fig)

    # debug
    #print '%.4e' % Lb_b, '%.4e' % Lb_cz, '%.4e' % Lb3, '%.4e' % Lb

    #print "##############################"

    return Lb, T_rgoal, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav

def calc_alpha_beta_delta(t, r, P, T, loss = 'soft_l1', alpha_guess = 0.6, beta_guess = 0.8, delta_guess = -0.2):
    """
    Not used.
    calculate alpha, beta, delta (see Weaver+77, eq. 39, 40, 41)
    use last 10 values of radius, pressure, temperature to calculate the time derivates
    neglect outliers for the fits
    :param t: time list (np array)
    :param r: radius list (np array)
    :param P: pressure list (np array)
    :param T: temperature list (np array)
    :param loss: correction function for increasing robustness: 'linear' gives you normal least_squares (not robust), 'soft_l1', 'huber' medium robustness, 'cauchy', 'arctan' high robustness
                (for more info, see http://scipy-cookbook.readthedocs.io/items/robust_regression.html)
    :return: alpha, beta, delta
    """

    log_t = np.log(t)
    log_r = np.log(r)
    log_P = np.log(P)
    log_T = np.log(T) # 1 element fewer than the other vectors because current value is not known yet

    res_robust = scipy.optimize.least_squares(aux.f_lin, [0., alpha_guess], loss=loss, f_scale=0.1, args=(log_t, log_r))
    alpha = res_robust.x[1]

    res_robust = scipy.optimize.least_squares(aux.f_lin, [0., beta_guess], loss=loss, f_scale=0.1, args=(log_t, log_P))
    beta = -res_robust.x[1]

    res_robust = scipy.optimize.least_squares(aux.f_lin, [0., delta_guess], loss=loss, f_scale=0.1, args=(log_t[:-1], log_T)) # temperature has 1 less element and is shifted: do not consider last time element
    delta = res_robust.x[1]

    #alpha = (np.log(r[-1]) - np.log(r0)) / (np.log(t[-1]) - np.log(tStart_i))  # in classical Weaver: alpha = 0.6
    #beta = -(np.log(state_eq.PfromE(Eb[-1], r[-1])) - np.log(P0)) / (
    #np.log(t[-1]) - np.log(tStart_i))  # in classical Weaver: beta = 0.8
    #delta_1 = (np.log(T_bubble) - np.log(T0)) / (np.log(t[-1]) - np.log(tStart_i))
    # if delta_counter > 10: delta = (np.log(T_bubble)-np.log(T0))/(np.log(t[-1])-np.log(tStart_i))
    #delta_counter += 1
    #print alpha, beta, delta_1

    return alpha, beta, delta

def calc_linfit(x, y, loss=i.myloss, old_guess = np.nan, c_guess=0.):
        """
        calculate slope of linear fit
        neglect outliers for the fits
        :param x: e.g. time list (np array)
        :param y: e.g. temperature list (np array)
        :param loss: correction function for increasing robustness: 'linear' gives you normal least_squares (not robust), 'soft_l1' and 'huber' have medium robustness, 'cauchy' and 'arctan' have high robustness
                    (for more info, see http://scipy-cookbook.readthedocs.io/items/robust_regression.html)
        :return: slope m
        """

        # IT SEEMS SAFER NOT TO USE GUESSED VALUES PROVIDED BY THE USER. WE ARE JUST USING OUR OWN ONES (but keep in mind that this means, this routine only works to calculate alpha and beta)

        # rounding seems to help
        #c_guess_round = float("%.2g" % c_guess)
        #m_guess_round = float("%.2g" % m_guess)

        # old_guess = np.nan

        #print x
        #print y

        my_c_guess = c_guess
        my_m_guess = 0.7
        # we need to guess what the soft threshold between inliners and outliers is
        # very rough order of magnitude approximation: use standard deviation
        # (better: use standard deviation from fit curve, but for that we would need to know the fit beforehand)
        my_fscale = np.std(y) # my_fscale = 0.1

        res_robust = scipy.optimize.least_squares(aux.f_lin, [my_c_guess, my_m_guess], loss=loss, f_scale=my_fscale, args=(x, y))
        m_temp1 = res_robust.x[1]

        # maybe we picked the wrong guess for m?
        if ((not np.isnan(old_guess)) and abs(m_temp1-old_guess) > 0.05):
            my_m_guess = 0.0
            res_robust = scipy.optimize.least_squares(aux.f_lin, [my_c_guess, my_m_guess], loss=loss, f_scale=my_fscale, args=(x, y))
            m_temp2 = res_robust.x[1]

            my_m_guess = 2.0
            res_robust = scipy.optimize.least_squares(aux.f_lin, [my_c_guess, my_m_guess], loss=loss, f_scale=my_fscale, args=(x, y))
            m_temp3 = res_robust.x[1]

            #my_m_guess = -4.0
            #res_robust = scipy.optimize.least_squares(aux.f_lin, [my_c_guess, my_m_guess], loss=loss, f_scale=my_fscale, args=(x, y))
            #m_temp4 = res_robust.x[1]

            #my_m_guess = 1.1
            #res_robust = scipy.optimize.least_squares(aux.f_lin, [my_c_guess, my_m_guess], loss=loss, f_scale=my_fscale, args=(x, y))
            #m_temp5 = res_robust.x[1]

            m_temp_list = np.array([m_temp1, m_temp2, m_temp3])
            idx = np.argmin(abs(m_temp_list-old_guess))
            m = m_temp_list[idx]
        else:
            m = m_temp1

        return m

        #D = {'alpha': alpha1, 'beta': beta1, 'delta': delta1, 'R2': R21, 't_now': t_now1, 'Eb': Eb1, 'Lw': Lw1, 'vw': vw1, 'v0': v01}

#Lb = calc_Lb(D)

def calc_linfit2(x, y, loss=i.myloss, old_guess=np.nan, c_guess=0.):
    """
    Not used
    
    calculate slope of linear fit
    neglect outliers for the fits
    :param x: e.g. time list (np array)
    :param y: e.g. temperature list (np array)
    :param loss: correction function for increasing robustness: 'linear' gives you normal least_squares (not robust), 'soft_l1' and 'huber' have medium robustness, 'cauchy' and 'arctan' have high robustness
                (for more info, see http://scipy-cookbook.readthedocs.io/items/robust_regression.html)
    :return: slope m
    """

    # IT SEEMS SAFER NOT TO USE GUESSED VALUES PROVIDED BY THE USER. WE ARE JUST USING OUR OWN ONES (but keep in mind that this means, this routine only works to calculate alpha and beta)

    # rounding seems to help
    # c_guess_round = float("%.2g" % c_guess)
    # m_guess_round = float("%.2g" % m_guess)

    # old_guess = np.nan

    #print x
    #print y

    my_c_guess = c_guess
    my_m_guess = 0.7
    my_fscale = np.std(y)

    res_robust = scipy.optimize.least_squares(aux.f_lin, [my_c_guess, my_m_guess], loss=loss, f_scale=my_fscale, args=(x, y))
    m_temp1 = res_robust.x[1]
    c_temp1 = res_robust.x[0]

    # maybe we picked the wrong guess for m?
    if ((not np.isnan(old_guess)) and abs(m_temp1 - old_guess) > 0.05):
        my_m_guess = 0.0
        res_robust = scipy.optimize.least_squares(aux.f_lin, [my_c_guess, my_m_guess], loss=loss, f_scale=my_fscale,
                                                  args=(x, y))
        m_temp2 = res_robust.x[1]
        c_temp2 = res_robust.x[0]

        my_m_guess = 2.0
        res_robust = scipy.optimize.least_squares(aux.f_lin, [my_c_guess, my_m_guess], loss=loss, f_scale=my_fscale,
                                                  args=(x, y))
        m_temp3 = res_robust.x[1]
        c_temp3 = res_robust.x[0]

        # my_m_guess = -4.0
        # res_robust = scipy.optimize.least_squares(aux.f_lin, [my_c_guess, my_m_guess], loss=loss, f_scale=my_fscale, args=(x, y))
        # m_temp4 = res_robust.x[1]

        # my_m_guess = 1.1
        # res_robust = scipy.optimize.least_squares(aux.f_lin, [my_c_guess, my_m_guess], loss=loss, f_scale=my_fscale, args=(x, y))
        # m_temp5 = res_robust.x[1]

        m_temp_list = np.array([m_temp1, m_temp2, m_temp3])
        c_temp_list = np.array([c_temp1, c_temp2, c_temp3])
        idx = np.argmin(abs(m_temp_list - old_guess))
        m = m_temp_list[idx]
        c = c_temp_list[idx]
    else:
        m = m_temp1
        c = c_temp1

    return m, c


def delta_zero(delta_in, params, verbose_dzero=0):

    Cool_Struc, t_10list, T_10list, fit_len, verbose2 = params[1:5+1]
    data_struc_temp = dict.copy(params[0])
    data_struc_temp['delta'] = delta_in

    [Lb, T0, Lb_b, Lb_cz, Lb3, dMdt_factor, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav] = calc_Lb(data_struc_temp, Cool_Struc, 999, rgoal_f=i.r_Tb, verbose=verbose2, plot=0)

    # use temperature of the bubble T_temp which has been calculated with a slightly wrong delta 
    # (delta_old, the predictor) to calculate a better estimate of delta
    T_10list_temp = aux.del_append(T_10list, T0, maxlen=fit_len)
    my_x = np.log(t_10list)
    my_y = np.log(T_10list_temp)
    c_guess = np.round(my_y[0], decimals=2)
    m_guess = np.round(data_struc_temp['old_delta'], decimals=2)
    delta_out = deltmod.cdelta(my_x, my_y, c_guess, m_guess, loss=i.myloss, verb=verbose_dzero)

    #if verbose_dzero > 0:
    # debugging
    #print "delta_in", delta_in, "delta_out", delta_out

    residual = delta_in - delta_out

    return residual

def new_zero_delta(delta_in, params):
    Cool_Struc = params[2]
    param0 = dict.copy(params[0])
    param1 = dict.copy(params[1])
    t0 = param0['t_now']
    t1 = param1['t_now']
    param0['delta'] = delta_in
    param1['delta'] = delta_in
    Lb_temp1, T_rgoal1, dMdt_factor_out1 = bstrux([param1, Cool_Struc])
    Lb_temp0, T_rgoal0, dMdt_factor_out0 = bstrux([param0, Cool_Struc])
    delta_out = (T_rgoal1 - T_rgoal0)/(t1-t0) * t1/T_rgoal1
    residual = delta_out - delta_in
    #print delta_in, delta_out
    return residual

def bubble_wrap(data, Cool_Struc, fit_len=5, fit_len_short = 5, verbose = 0):
    # wrapper to calculate bubble structure
    # make sure: fit_len_short <= fit_len
    if (fit_len_short > fit_len):
        fit_len_short = fit_len
    
    # print('this is input', data)
    
    structure_switch = data['structure_switch']
    alpha = data['alpha']
    beta = data['beta']
    delta = data['delta']
    Lres0 = data['Lres0']
    t_10list = data['t_10list']
    r_10list = data['r_10list']
    P_10list = data['P_10list']
    T_10list = data['T_10list']
    Lw = data['Lw']
    vterminal = data['vterminal']
    dMdt_factor = data['dMdt_factor']
    # Qi = data['Qi']  # current photon ionizing photon emission rate (in 1/Myr)
    r0 = data['r0']
    t0 = data['t0']
    E0 = data['E0']
    T0 = data['T0']
    # print(T0)
    dt_L = data['dt_L']
    temp_counter = data['temp_counter']
    # Cool_Struc = data['Cool_Struc']

    bubble_check = 0 # by default assume everything went right

    if (structure_switch):
        # after a couple of start time steps, calculate alpha, beta, delta
        #if (temp_counter > fit_len):
        use_root_finder = False
        if (temp_counter > fit_len):
            # time step (allow certain relative change in (in-out)-luminosity)
            my_x = np.log(t_10list[-fit_len_short:]) - np.log(t_10list[-fit_len_short])
            my_y = np.log(r_10list[-fit_len_short:])
            my_y2 = -1.0 * np.log(P_10list[-fit_len_short:])
            # at early times (when far out of equilibrium), use input alpha and beta as start values for lin fit
            if temp_counter < 2*fit_len:
                alpha = calc_linfit(my_x, my_y, loss=i.myloss, old_guess=np.round(alpha, decimals=2), c_guess=np.round(my_y[0], decimals=2))
                beta = calc_linfit(my_x, my_y2, loss=i.myloss, old_guess=np.round(beta, decimals=2), c_guess=np.round(my_y2[0], decimals=2))
            # at later times use directly the input alpha and beta
            delta0 = delta
            #delta_old = 1000.  # ONLY USED FOR ENTERING LOOP, arbitrary high to make sure the following while loop is always entered at least once
            data_struc_temp = {'alpha': alpha, 'beta': beta, 'old_delta': delta0, 'R2': r0, 't_now': t0, 'Eb': E0, 'Lw': Lw, 'vw': vterminal, 'dMdt_factor': dMdt_factor, 'Qi':data['Qi'], 'mypath': data['mypath']}
            # the following is a fixpoint iteration (should be improved). Q: Does it always converge. For fit_len = 5, it doesn't!
            fixpoint_counter = 0

            # use small tolerance levels if last time step was small
            last_dt = t_10list[-1]-t_10list[-2]
            my_xtol = np.min([last_dt,1e-3])

            use_root_finder = True # debugging
            """
            while ((np.abs((delta - delta_old) / delta) > i.delta_error) and (fixpoint_counter < 3)):  # stay in loop as long as estimate of delta changes more than 5 % (default of delta_error)
                delta_old = delta
                # make sure you use correct Lw, vw
                # delta_old is probably not the correct delta (since no temperature info at the current time t[-1] was available), so calculate T_bubble using the (slightly) wrong delta_old
                data_struc_temp['delta'] = delta_old
                try:
                    [Lb, T0, Lb_b, Lb_cz, Lb3, dMdt_factor_out] = calc_Lb(data_struc_temp, Cool_Struc, temp_counter, rgoal_f=i.r_Tb, verbose=verbose,plot=1)
                except:
                    use_root_finder = True
                    break
                # I cannot use Lb_temp and T_temp as Lb and T0 for the next time step because I don't know, how Lw and vw will change

                # use temperature of the bubble T_temp which has been calculated with a slightly wrong delta (delta_old, the predictor) to calculate a better estimate of delta
                T_10list_temp = aux.del_append(T_10list, T0, maxlen=fit_len)
                delta = deltmod.cdelta(t_10list, T_10list_temp, loss=i.myloss, delta_guess=delta)
                #print "delta, delta_old, rel. change", delta, delta_old, (delta - delta_old) / delta
                fixpoint_counter += 1
            """
            # if above 'fixpoint interation' did not converge, we need better (but more expensive) methods: use a root finder!
            if (use_root_finder is True):
                print('entering use_root_finder == True')
                #print("using root finder to estimate delta") # debugging
                params = [data_struc_temp, Cool_Struc, t_10list, T_10list, fit_len, verbose]

                # try once with some boundary values and hope there is a zero point in between
                try:
                    delta = scipy.optimize.brentq(delta_zero, delta0 - 0.1 , delta0 + 0.1, args=(params), xtol=my_xtol, rtol=1e-8) # this might fail if no fixpoint exists in the given range. If so, try with a larger range
                except:
                    # it seems either the boundary values were too far off (and bubble_structure crashed) or there was no zero point in between the boundary values
                    # try to figure out what limits in delta are allowed
                    #########################################

                    worked_last_time_lo = True
                    worked_last_time_hi = True

                    iic = 0
                    n_trymax = 30 # maximum number of tries before we give up
                    sgn_vec = np.zeros(2*n_trymax+1) # list containing the signs of the residual (if there is a sign flip between two values, there must be a zero point in between!)
                    delta_in_vec = np.zeros(2*n_trymax+1) # list conatining all tried input deltas
                    verbose_dzero = 0
                    ii_lo = np.nan
                    ii_hi = np.nan

                    diff_sgn_vec = abs(sgn_vec[1:]-sgn_vec[:-1]) # list which contains the number 2.0 where a sign flip ocurred

                    # stay in loop as long as sign has not flipped
                    while all(diff_sgn_vec < 2.):

                        res_0 = delta_zero(delta0, params, verbose_dzero=verbose_dzero)
                        sgn_vec[n_trymax] = np.sign(res_0) # is probably not 0 (because of small numerical noise) but ensure it is not 0 further down
                        delta_in_vec[n_trymax] = delta0

                        if worked_last_time_lo:
                            try:
                                delta_in_lo = delta0 - 0.02 - float(iic) * 0.05
                                res_lo = delta_zero(delta_in_lo, params, verbose_dzero=verbose_dzero)
                                ii_lo = n_trymax-iic-1
                                sgn_vec[ii_lo] = np.sign(res_lo)
                                delta_in_vec[ii_lo] = delta_in_lo
                                if (sgn_vec[n_trymax] == 0.): sgn_vec[n_trymax]= sgn_vec[n_trymax-1] # make sure 0 does not ocurr
                            except:
                                worked_last_time_lo = False

                        if worked_last_time_hi:
                            try:
                                delta_in_hi = delta0 + 0.02 + float(iic) * 0.05
                                res_hi = delta_zero(delta_in_hi, params, verbose_dzero=verbose_dzero)
                                ii_hi = n_trymax+iic+1
                                sgn_vec[ii_hi] = np.sign(res_hi)
                                delta_in_vec[ii_hi] = delta_in_hi
                                if (sgn_vec[n_trymax] == 0.): sgn_vec[n_trymax] = sgn_vec[n_trymax + 1] # make sure 0 does not ocurr
                            except:
                                worked_last_time_hi = False



                        if iic > n_trymax / 2:
                            print("I am having a hard time finding delta...")
                            verbose_dzero = 1
                            if iic >= n_trymax - 1:
                                sys.exit("Could not find delta.")

                        diff_sgn_vec = abs(sgn_vec[1:] - sgn_vec[:-1]) # this list contains a 2.0 where the sign flip ocurred (take abs, so that -2.0 becomes +2.0)

                        # print statements for debugging
                        #print(worked_last_time_lo, ii_lo, worked_last_time_hi, ii_hi)
                        #print("iic, sgn_vec", iic, sgn_vec, end=' ')
                        #print("diff_sgn_vec",diff_sgn_vec)

                        iic += 1

                    # find the index where the sign flip ocurred (where the diff list has the element 2.0)
                    idx_zero0 = np.argmax(diff_sgn_vec) # we could also look for number 2.0 but because there are no higher number, finding the maximum is equivalent
                    delta_in_lo = delta_in_vec[idx_zero0]
                    delta_in_hi = delta_in_vec[idx_zero0+1]
                #########################################
                    try:
                        delta = scipy.optimize.brentq(delta_zero, delta_in_lo , delta_in_hi, args=(params), xtol=0.1*my_xtol, rtol=1e-9) # this might fail if no fixpoint exists in the given range
                    except:
                        # print(data_struc_temp, t_10list, T_10list, fit_len, verbose)
                        delta = delta0
                        bubble_check = 1 # something went wrong

                data_struc_temp['delta'] = delta
                #[Lb, T0, Lb_b, Lb_cz, Lb3] = calc_Lb(data_struc_temp, Cool_Struc, temp_counter, rgoal_f=i.r_Tb, verbose=verbose,plot=1)
                #print('Calc_Lb called from bubble_wrap')
                [Lb, T0, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav] = calc_Lb(data_struc_temp, Cool_Struc, temp_counter, rgoal_f=i.r_Tb, verbose=verbose, plot=0)

            # calculate next time step
            Lres = Lw - Lb
            #fac = np.max([np.min([i.lum_error2 / (np.abs(Lres - Lres0) / Lres), 1.42]),0.1])  # try to achieve relative change of 1 % (default of lum_error) but do not change more than factor 1.42 or less than 0.1
            fac = np.max([np.min([i.lum_error2 / (np.abs(Lres - Lres0) / Lw), 1.42]), 0.1])
            dt_L = fac * dt_L  # 3 per cent change
        # early on, use fixed initial delta
        else:
            # print(T0)
            data_struc = {'alpha': alpha, 'beta': beta, 'delta': delta, 'R2': r0, 't_now': t0, 'Eb': E0, 'Lw': Lw, 'vw': vterminal, 'dMdt_factor': dMdt_factor, 'Qi':data['Qi'], 'mypath': data['mypath']}
            np.seterr(all='warn')
            [Lb, T0, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav] = calc_Lb(data_struc, Cool_Struc, temp_counter, rgoal_f=i.r_Tb, verbose=verbose,
                                                plot=0)  # cgs units
            # Lb = Lb_cgs * c.Myr / (c.Msun * c.kms ** 2)
            # print "T0 (MK) = ", T0/1e6
    # if not calculate bubble structure:
    else:
        Lb = 0.
        Lb_b = 0.
        Lb_cz = 0.
        Lb3 = 0.
        dMdt_factor_out = 1.646 # as in classical Weaver
        dt_L = dt_L
        Tavg = T0
        Mbubble = np.nan
        r_Phi = np.nan
        Phi_grav_r0b = np.nan
        f_grav = np.nan
    print("This is output")
    # print([bubble_check, Lb, T0, alpha, beta, delta, dt_L, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav])
    return [bubble_check, Lb, T0, alpha, beta, delta, dt_L, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav]



def bstrux(full_params):
    Cool_Struc = full_params[1]
    my_params = dict.copy(full_params[0])
    counter = 789

    # call calc_Lb with or without set xtol?
    if 'xtolbstrux' in my_params:
        xtol = my_params['xtolbstrux']
        [Lb, Trgoal, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav] = calc_Lb(my_params, Cool_Struc, counter, error_exit=False, rgoal_f=i.r_Tb, xtol=xtol)
    else:
        [Lb, Trgoal, Lb_b, Lb_cz, Lb3, dMdt_factor_out, Tavg, Mbubble, r_Phi, Phi_grav_r0b, f_grav] = calc_Lb(my_params, Cool_Struc, counter, error_exit=False, rgoal_f=i.r_Tb)

    bstrux_result = {'Lb':Lb, 'Trgoal':Trgoal, 'dMdt_factor': dMdt_factor_out, 'Tavg': Tavg}

    return bstrux_result

def delta_new_root(delta_old, params):

    bubble_check = 0
    my_xtol=1e-9
    # try once with some boundary values and hope there is a zero point in between
    try:
        delta = scipy.optimize.brentq(new_zero_delta, delta_old - 0.1 , delta_old + 0.1, args=(params), xtol=my_xtol, rtol=1e-8) # this might fail if no fixpoint exists in the given range. If so, try with a larger range
    except:
        # it seems either the boundary values were too far off (and bubble_structure crashed) or there was no zero point in between the boundary values
        # try to figure out what limits in delta are allowed
        #########################################

        worked_last_time_lo = True
        worked_last_time_hi = True

        iic = 0
        n_trymax = 30 # maximum number of tries before we give up
        sgn_vec = np.zeros(2*n_trymax+1) # list containing the signs of the residual (if there is a sign flip between two values, there must be a zero point in between!)
        delta_in_vec = np.zeros(2*n_trymax+1) # list containing all tried input deltas
        #verbose_dzero = 0
        ii_lo = np.nan
        ii_hi = np.nan

        diff_sgn_vec = abs(sgn_vec[1:]-sgn_vec[:-1]) # list which contains the number 2.0 where a sign flip ocurred

        # stay in loop as long as sign has not flipped
        while all(diff_sgn_vec < 2.):

            res_0 = new_zero_delta(delta_old, params)
            sgn_vec[n_trymax] = np.sign(res_0) # is probably not 0 (because of small numerical noise) but ensure it is not 0 further down
            delta_in_vec[n_trymax] = delta_old

            if worked_last_time_lo:
                try:
                    delta_in_lo = delta_old - 0.02 - float(iic) * 0.05
                    res_lo = new_zero_delta(delta_in_lo, params)
                    ii_lo = n_trymax-iic-1
                    sgn_vec[ii_lo] = np.sign(res_lo)
                    delta_in_vec[ii_lo] = delta_in_lo
                    if (sgn_vec[n_trymax] == 0.): sgn_vec[n_trymax]= sgn_vec[n_trymax-1] # make sure 0 does not ocurr
                except:
                    worked_last_time_lo = False

            if worked_last_time_hi:
                try:
                    delta_in_hi = delta_old + 0.02 + float(iic) * 0.05
                    res_hi = new_zero_delta(delta_in_hi, params)
                    ii_hi = n_trymax+iic+1
                    sgn_vec[ii_hi] = np.sign(res_hi)
                    delta_in_vec[ii_hi] = delta_in_hi
                    if (sgn_vec[n_trymax] == 0.): sgn_vec[n_trymax] = sgn_vec[n_trymax + 1] # make sure 0 does not ocurr
                except:
                    worked_last_time_hi = False



            if iic > n_trymax / 2:
                print("I am having a hard time finding delta...")
                verbose_dzero = 1
                if iic >= n_trymax - 1:
                    sys.exit("Could not find delta.")

            diff_sgn_vec = abs(sgn_vec[1:] - sgn_vec[:-1]) # this list contains a 2.0 where the sign flip ocurred (take abs, so that -2.0 becomes +2.0)

            # print statements for debugging
            #print(worked_last_time_lo, ii_lo, worked_last_time_hi, ii_hi)
            #print("iic, sgn_vec", iic, sgn_vec, end=' ')
            #print("diff_sgn_vec",diff_sgn_vec)

            iic += 1

        # find the index where the sign flip ocurred (where the diff list has the element 2.0)
        idx_zero0 = np.argmax(diff_sgn_vec) # we could also look for number 2.0 but because there are no higher number, finding the maximum is equivalent
        delta_in_lo = delta_in_vec[idx_zero0]
        delta_in_hi = delta_in_vec[idx_zero0+1]
    #########################################
        try:
            delta = scipy.optimize.brentq(delta_zero, delta_in_lo , delta_in_hi, args=(params), xtol=0.1*my_xtol, rtol=1e-9) # this might fail if no fixpoint exists in the given range
        except:
            print(params)
            delta = delta_old
            bubble_check = 1 # something went wrong

    return delta, bubble_check