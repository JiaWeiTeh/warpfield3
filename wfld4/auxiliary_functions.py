import constants as c
import init as i
import numpy as np
import os
import sys
import re
import phase_lookup as ph
import scipy.optimize
import scipy.integrate
import warnings
from scipy import spatial
import density_profile
import mass_profile
import warp_nameparser as wn
from numpy import diff


def printl(mystring, verbose = 0):
    # print function that only prints if specified verbosity level is higher than the one in myconfig/parameters
    if i.output_verbosity >= verbose:
        print(mystring)
    return

def cool_time(Z,n,Lw, t0 = 0.):
    # cooling time of shocked wind zone (Mac Low & McCray 1988, see also Silich & Tenorio-Tagle 2013)
    # t0 is optional time offset (e.g. for 2nd star formation event)
    Lw38_au = 1e38*c.Myr/(c.Msun*c.kms**2)
    tcool = t0 + ( 16. * Z**(-35./22.) * n**(-8./11.) * (Lw/Lw38_au)**(3./11.) )
    return tcool

def active_branch(r0, t0, rfinal, tfinal, active = True):
    # check time and radius
    #return all([r0<rfinal, (tfinal-t0) > i.dt_Emax/500., active])
    return all([r0 < rfinal, (tfinal - t0) > i.dt_Emin, active])


def set_phase(r,rcore,rcloud,t, v, dens_grad = True):
    #set the current phase

    if type(r) is not np.ndarray:
        r = np.array([r])
    if type(t) is not np.ndarray:
        t = np.array([t])
    if type(v) is not np.ndarray:
        v = np.array([v])

    phase = np.nan*r

    ph_collapse = (v < 0.0)
    #ph_weaver = (t<tcool and ((dens_grad and r < rcore) | (not dens_grad and r < rcloud)))
    ph_amb = (r >= rcloud) & (~ph_collapse)
    ph_grad = dens_grad & (r < rcloud) & (r > rcore) & (~ph_collapse)
    ph_core = ~(ph_collapse | ph_amb | ph_grad)

    phase[ph_collapse] = ph.collapse #collapse
    #phase[ph_weaver] = 1.0
    phase[ph_amb] = ph.ambient
    phase[ph_grad] = ph.gradient
    phase[ph_core] = ph.core

    # dissolved phase: 0.0
    return phase

def check_outdir(dirstring):
    """Check if output directory exists. If not, create it."""
    if not os.path.isdir(dirstring):
        os.makedirs(dirstring)

    return 0

def calc_rcloud(Mcloud_au, namb, mui, rcore_au, nalpha, n_intercl, verbose = 0):

    if i.dens_profile == "powerlaw":
        #calculates cloud radius (assuming power law profile with core)
        rcloud_au = (
            
            (Mcloud_au / (4. * np.pi * namb * mui * (c.pc ** 3 / c.Msun)) - rcore_au ** 3. / 3.)\
                * (3. + nalpha) * rcore_au**nalpha + rcore_au**(3. + nalpha)
                
                )** (1. / (3. + nalpha))
        
            
        
        printl("cloud radius: %.2f pc" %(rcloud_au), verbose = verbose)

        # check whether density at edge of cloud is below intercloud medium density (we don't want that)
        nedge = namb * (rcloud_au / rcore_au) ** (nalpha)  # density at edge of cloud
        
        # print('check for initials')
        # print('rCore, mCloud, nEdge, nCore, rCloud')
        # print(rcore_au, Mcloud_au, nedge, namb, rcloud_au)
        # sys.exit()
        
        printl("density at edge of cloud: %.1f /ccm" % (nedge), verbose=verbose)
        printl("cloud radius: %.1f pc" % (rcloud_au), verbose=verbose)
        printl("core radius: %.1f pc" % (rcore_au), verbose=verbose)


    elif i.dens_profile == "BonnorEbert":
        T_BE=rcore_au
        print("Test")
        print(namb, T_BE, Mcloud_au)
        # sys.exit()
        rcloud_au,nedge = density_profile.FindRCBE(namb, T_BE, Mcloud_au)
        
        
        # to be safe: slightly higher than intercloud material 

    if nedge < n_intercl:
        print(
            "Density at the edge of the cloud is below the density of the intercloud medium. Increase core density!")
        sys.exit("Exiting: Density at cloud edge too low! Mcloud=%.1f, namb=%.1f, rcore=%.2f" %(Mcloud_au, namb, rcore_au))

    return rcloud_au, nedge

def reset_ii_cloudy(ii_cloudy, T, dt, reset=True):
    # reset cloudy counter
    if reset:
        ii_cloudy = int(T / dt)
        reset = False
    return ii_cloudy, reset

def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass


def rcore_powdens(n0, M, n, nalpha):
    # This routine calculates core radius (and cloud radius)
    # Provide a total cloud mass, a core number density, and the average number density you are aiming for!
    # The routine assumes a power law density profile outside the constant density core

    # M : clump mass in g (1e8 * c.Msun) after SF
    # n : average density in /ccm (e.g. 1e3)
    # n0 : core density in /ccm ( e.g. 1e5 )
    # nalpha: power law index: rho = rho0*(r/r0)**nalpha, usually nalpha is negative

    #if nalpha != -2.:
    #    sys.exit("Exit: In rcore_powdens: nalpha must be -2.0!")
    if nalpha > 0.:
        warnings.warn("Warning: Power law index is positive. This has not been tested.")

    if M <= 0.:
        warnings.warn("Warning! Cloud mass negative in rcore_powdens.")
        sys.exit("Exiting: Cloud mass is zero or negative.")
    elif M < 1e25:  # if M is very low, I guess you meant solar masses. However, I need grams!
        warnings.warn("Warning! Cloud mass has been provided in solar masses instead of grams: rcore_powdens. Converting to grams now...")
        M = M * c.Msun

    # cloud radius
    rcloud = (3./(4.*np.pi) * M/(n*i.mui))**(1./3.)

    def F(x):
        myzero = n*i.mui/3. * rcloud**3. - n0*i.mui*( (1./3.-1./(3.+nalpha))*x**3. + 1./(3.+nalpha)*x**(-nalpha)*rcloud**(3.+nalpha) )
        return myzero

    # core radius
    rcore = scipy.optimize.brentq(F, 0., rcloud) # cgs

    M_control = mass_profile.calc_mass((1. - 1e-9) * rcloud, rcore, rcloud, n0 * i.mui,0.0, nalpha, M)[0] # control mass

    if abs(M_control - M)/M > 1e-3:
        sys.exit("Core radius does not produce correct cloud mass")
    if rcore <= 0.:
        sys.exit("Core radius is negative.")
    if rcore > rcloud:
        sys.exit("Core radius is larger than the cloud radius.")
        


    # return core radius in pc
    return rcore / c.pc

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))

def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

#@nb.njit
def find_nearest_lower(array, value):
    """
    finds index idx in array for which array[idx] is 1) smaller or equal to value, and 2) closest to value
    elements in array need be monotonically increasing or decreasing!
    """
    # check whether array is monotonically increasing (by only checking the first 2 elements)
    if array[1]>array[0]:
        mon_incr = True
    else:
        mon_incr = False
    idx = find_nearest(array, value)
    if array[idx]-value > 0: # then this element is the closest, but it is larger than value
        if mon_incr: idx += -1 # take the element before, it will be smaller than value (if array is monotonically increasing)
        else: idx += 1
    if idx >= len(array): idx = len(array) - 1
    if idx < 0: idx = 0
    return idx

#@nb.njit
def find_nearest_higher(array, value):
    """
    finds index idx in array for which array[idx] is 1) larger or equal to value, and 2) closest to value
    elements in array need be monotonically increasing or decreasing!
    """
    # check whether array is monotonically increasing (by only checking the first 2 elements)
    if array[1]>array[0]:
        mon_incr = True
    else:
        mon_incr = False
    idx = find_nearest(array, value)
    if array[idx]-value < 0: # then this element is the closest, but it is smaller than value
        if mon_incr: idx += 1 # take the element after, it will be larger than value (if array is monotonically increasing)
        else: idx += -1
    if idx >= len(array): idx = len(array) - 1
    if idx < 0: idx = 0
    return idx

#@nb.njit
def find_nearest(array, value):
    """
    finds index idx in array for which array[idx] is closest to value
    """

    # make sure that we deal with an numpy array
    if type(array) == list:
        array = np.array(array)

    idx = (np.abs(array-value)).argmin()
    return idx

def del_append(mylist, value, maxlen = 10):
    """
    appends value to end of list and removes first element of list if list would become longer than a given maximum length
    :param mylist: list which you want to modify
    :param value: element to append (scalar)
    :param maxlen: optional, maximum length (default is 10)
    :return:
    """
    mylist = np.append(mylist, value)
    while (len(mylist) > maxlen):
        mylist = np.delete(mylist, 0)
    return  mylist

def cubic_eq(a,b,c,d):
    """
    returns real solution of cubic equation ax^3 + bx^2 + cx + d = 0
    :param a: factor of cubic term
    :param b: factor of quadratic term
    :param c: factor of linear term
    :param d: constant term
    :return: real solution (not the 2 complex solutions)
    """

    term1 = -b**3./(27.*a**3.) + b*c/(6.*a**2.) - d/(2.*a)
    term2 = c/(3.*a) - b**2./(9.*a**2.)

    return (term1 + np.sqrt(term1**2 + term2**3.))**(1./3.) + (term1 - np.sqrt(term1**2 + term2**3.))**(1./3.) - b/(3.*a)

def f_lin(x,t,y):
    """
    function for linear regression
    :param x:
    :param t:
    :return:
    """
    return x[0] + x[1]*t - y

def sound_speed(T, unit="kms"):
    """
    calculates sound speed
    :param T: temperature
    :param unit: unit of sound speed (default is km/s), if unit != "kms", the unit will be cm/s
    :return: float
    """

    if T>1e3: mu = i.mua
    else: mu = i.mui

    cs = np.sqrt(c.gamma*c.kboltz*T/mu)
    if unit == "kms": cs *= 1e-5

    return cs


def sound_speed_BE(T):
    """
    calculates sound speed
    :param T: temperature
    :return: sound speed (default is m/s)
    """
    mu = i.mui

    cs = np.sqrt(c.gamma*c.kboltz*T/mu)
    cs *= 1e-2

    return cs

def my_differentiate(x,y):
    """
    calculate derivative via finite differences
    :param x:
    :param y:
    :return:
    """
    dydx = (y[1:] - y[0:-1])/(x[1:] - x[0:-1])
    dydx = np.concatenate([dydx,[0.]])
    return dydx

def find_NN(x,y_NN):
    """
    find nearest neighbors
    :param x: 1D-array of values for which nearest neighbors shall be determined
    :param y_NN: 1D-array of possible neighbors
    :return: array of indices of nearest neighbors
    """

    # first need to reshape arrays (need array of arrays)
    x_rsh = np.reshape(x, (len(x), 1))
    y_NN_rsh = np.reshape(y_NN, (len(y_NN), 1))

    # construct tree and query it
    tree = spatial.cKDTree(y_NN_rsh)
    idx = tree.query(x_rsh)[1]

    return idx

def make_arr(x):
    """ convert scalars and lists to arrays"""
    if not hasattr(x, "__len__"):
        x= np.array([x])
    elif isinstance(x,list):
        x = np.array(x)
    return x

def write_pot_to_file(mypath, t0, r_Phi, Phi_grav_r0, f_grav, rcloud_au, rcore_au, nalpha, Mcloud_au, Mcluster_au, SFE, Rmax = 2001.):
    """
    write files with graviational potential and force per unit mass as a function of radius
    :param mypath: path to output folder
    :param t0: time of output (in Myr)
    :param r_Phi: list-like np.array of radii at which Phi has already been evaluated (shell and possibly bubble)
    :param Phi_grav_r0: normalization of grav. potential (integral from r0 to rsh_out over r*rho(r); integral from rsh_out to r_infinity will be done here)
    :param f_grav: spatial derivative of potential (force per unit mass)
    :param rcloud_au: cloud radius in pc
    :param rcore_au: core radius in pc
    :param nalpha: power law exponent of density profile
    :param Mcloud_au: cloud mass in Msun
    :param Rmax: (optional) maximum radius (in pc) up to which ISM is considered for grav. potential and up to which the potential file extends
    :return:
    """

    # convert time from Myr to yr, and always save as e07
    age_str = ('{:0=5.9f}e+07'.format(t0 / 1e1))
    potfile = mypath + '/potential/pot_' + age_str + '.txt'

    # innermost radius at which mass appears (radius r0 of normalization)
    r0 = r_Phi[0]

    Nlarge = int(10 * i.pot_len_intern)
    Nsmall = float(i.pot_len_intern)
    skip = max(int(float(Nlarge) / float(Nsmall)), 1)

    # set force at the central, evacuated wind cavity to 0
    N_inner = max(int(3.0*Nsmall / float(i.pot_len_write)), 1)

    r_inner = np.concatenate([np.linspace(0.,0.999*r_Phi[0],num=5*N_inner, endpoint=True), np.linspace(0.9991*r_Phi[0],0.99999*r_Phi[0],num=5*N_inner)])
    # Phi_grav_inner = np.zeros(len(r_inner))
    f_grav_inner = np.zeros(len(r_inner))


    # outside region
    # outside cloud or outside shell (whatever is larger)
    r_outer_tmp = np.logspace(np.log10(max(1.0001*r_Phi[-1],rcloud_au*c.pc)),np.log10(Rmax*c.pc),num=Nlarge, endpoint=True) # go out to Rmax
    # inside cloud but outside shell (only exists at early times before the whole cloud has been swept)
    if (1.0001 * r_Phi[-1] < rcloud_au * c.pc):
        r_outer1_tmp = np.logspace(np.log10(1.0001 * r_Phi[-1]), np.log10(rcloud_au * c.pc), num=Nlarge, endpoint=False)
        r_outer_tmp = np.concatenate([r_outer1_tmp, r_outer_tmp])

    if i.dens_profile == "powerlaw":
        M_cum_tmp = mass_profile.calc_mass(r_outer_tmp/c.pc, rcore_au, rcloud_au, i.rhoa_au, i.rho_intercl_au, nalpha, Mcloud_au)*c.Msun # cgs
        rho_tmp = i.mui * density_profile.f_dens(r_outer_tmp, i.namb, i.n_intercl, rcloud_au * c.pc, nalpha=i.nalpha,rcore=rcore_au * c.pc)
    elif i.dens_profile == "BonnorEbert":
        M_cum_tmp = mass_profile.calc_mass_BE(r_outer_tmp/c.pc, i.rhoa_au, float(os.environ["TBE"]) , i.rho_intercl_au, rcloud_au, Mcloud_au)*c.Msun
        rho_tmp =  i.mui *density_profile.f_densBE(r_outer_tmp/c.pc, i.namb, float(os.environ["TBE"]) , i.n_intercl, rcore_au)
    # Phi_grav = c.Grav * M_cum / r_outer  # gravitational potential

    Phi_grav_r0a = -4. * np.pi * c.Grav * scipy.integrate.trapz(r_outer_tmp * rho_tmp, x=r_outer_tmp)
    f_grav_outer_tmp = c.Grav * M_cum_tmp / r_outer_tmp**2.  # gravitational force per unit mass

    r_outer = r_outer_tmp[0:-1:skip]
    # Phi_grav_noion = Phi_grav_noion_tmp[0:-1:skip]
    f_grav_outer = f_grav_outer_tmp[0:-1:skip]

    # add integrals to get final normalization of potential
    Phi_grav_r0 += Phi_grav_r0a

    # concatenate inner evacuated bubble part, reverse shock region of bubble (if warpversion2) + shell, and outer cloud/intercloud ambient ISM regions
    r_Phi = np.concatenate([r_inner, r_Phi, r_outer])
    # Phi_grav = np.concatenate([Phi_grav_inner, Phi_grav, Phi_grav_outer])
    f_grav = np.concatenate([f_grav_inner, f_grav, f_grav_outer])

    # potential with normalization
    Phi_grav = Phi_grav_r0 + scipy.integrate.cumtrapz(f_grav, x=r_Phi)
    Phi_grav = np.concatenate([ [Phi_grav[0]], Phi_grav]) # double first entry to get same length as r_Phi

    # reduce number of data points when writing to file (save some disk space)
    skip2 = max(int(float(len(r_Phi)) / float(i.pot_len_write)), 1)
    r_Phi_write = r_Phi[0:-1:skip2]
    Phi_grav_write = Phi_grav[0:-1:skip2]
    f_grav_write = f_grav[0:-1:skip2]

    # save potential file
    np.savetxt(potfile, np.transpose(np.concatenate([[r_Phi_write], [Phi_grav_write], [f_grav_write]])), fmt=' %.9e %.5e %.5e' )

    return 0

def check_continue(t, r, v, tStop):
    """
    check how to continue (stop simulation, continue normally, or form a new cluster due to collapse)
    :param t: current time (float)
    :param r: current shell radius (float)
    :param v: current velocity (float)
    :return: one of the following:
        -1: collapse (more more stars!)
        0: continue normally (probably in a new phase)
        1: stop simulation (end time reached, max radius reached, or shell dissolved)
        REMARK: values are set by phase_lookup.py and can be changed there
    """

    eps = 1e-2

    if (v < 0. and r < i.rcoll+eps and t < tStop-eps):
        # collapse!
        if i.mult_exp:
            return ph.coll
        else:  # if simulation should be stopped after re-collapse, return stop command
            return ph.stop
    elif (r > i.rstop-eps or t>tStop-eps): # MISSING: shell has dissolved
        # stop simulation (very large shell radius or end time reached or shell dissolved)
        return ph.stop
    else:
        # continue simulation in new phase
        return ph.cont

def surfdens_to_voldens(M, surfdens, mu = c.mp):
    """
    converts surface mass density to volume number density (of HI by default)
    :param M: mass of cloud (in solar masses)
    :param surfdens: surface density (in solar masses per square parsec)
    :param mu: molecular weight (default: proton mass, i.e. HI)
    :return: volume number density in particles per ccm (HI number density by default)
    """
    voldens = 3. / (4. * mu) * (np.pi / (M * c.Msun)) ** 0.5 * (surfdens * c.Msun / (c.pc ** 2.)) ** 1.5

    return voldens

def voldens_to_surfdens(M, voldens, mu = c.mp):
    """
    converts volume number density (of HI by default) to surface mass density
    :param M: mass of cloud (in solar masses)
    :param volume number density in particles per ccm (HI number density by default)
    :param mu: molecular weight (default: proton mass, i.e. HI)
    :return: surfdens: surface density (in solar masses per square parsec)
    """
    surfdens = ((4.*mu*voldens/3.)**2. * (M*c.Msun)/np.pi)**(1./3.) * (c.pc**2./c.Msun)

    return surfdens

def gaussian(x, mu, sig):
    """
    Gaussian distribution
    :param x: list or array
    :param mu: mean
    :param sig: standard deviation
    :return:
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def bind_energy(r, r0, rhocore, alpha, Mstar=0.):
    """
    binding energy for clouds with the following density profile:

    rho = rhocore                   if r <= r0
    rho = rhocore*(r/r0)**alpha     if r > r0

    :param r: radius up to which binding energy shall be calculated (don't set r to radius larger than cloud radius)
    :param r0: core radius
    :param rhocore: core density
    :param alpha: density power law exponent
    :param Mstar: central point mass (star cluster)
    :return: binding energy (float)
    """

    if r <= r0:
        U = (2.*np.pi * rhocore * c.Grav * Mstar * r**2.) + ( 4.*np.pi*rhocore )**2./15. * c.Grav * r**5.
    else:
        if alpha == -3.0:
            sys.exit("binding energy not implemented for alpha = -3.0")

        U0 = (2.*np.pi * rhocore * c.Grav * Mstar * r0**2.) + ( 4.*np.pi*rhocore )**2./15. * c.Grav * r0**5.


        M0 = (4.*np.pi/3.)*rhocore*r0**3. + Mstar
        fac10 = M0 - 4.*np.pi*rhocore*r0**3/(3.+alpha)
        if alpha == -2.:
            U10 = fac10 * np.log(r/r0)
        else:
            U10 = fac10 * 1./(2.+alpha) * (r**(2.+alpha) - r0**(2.+alpha))

        fac11 = 4.*np.pi*rhocore/(r0**alpha * (3.+alpha))
        if alpha == -2.5:
            U11 = fac11 * np.log(r/r0)
        else:
            U11 = fac11 * 1./(5.+2.*alpha) * (r**(5.+2.*alpha) - r0**(5.+2.*alpha))

        U = U0 + 4.*np.pi*rhocore*c.Grav/(r0**alpha) * (U10 + U11)

    return U


def coverfrac(E,E0,cfe):
    
    if int(os.environ["Coverfrac?"])==1:
        if (1-cfe)*(E/E0)+cfe < cfe:    # just to be safe, that 'overshooting' is not happening. 
            return cfe
        else:
            return (1-cfe)*(E/E0)+cfe
    else:
        return 1

def find_nearest_id(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
def optimalbstrux(Mcloud,SFE,path):
    # Optimal bubble structure
    # Looks like an initialisation function

    R1R2=R2pR2=np.array([0])
    t=Qi=Li=Ln=Lbol=Lw=pdot=pdot_SNe=np.array([0])

    
    check=os.path.join(path, "BubDetails")
    wn.check_outdir(check)
    
    pstr=path +"/BubDetails/Bstrux.txt"
    
    #pstr2=path +"/BubDetails/Feedbackfiles.txt"
    #np.savetxt(pstr2, np.c_[t,Qi,Li,Ln,Lbol,Lw,pdot,pdot_SNe],delimiter='\t')
    #os.environ["Fpath"] = pstr2
    
    np.savetxt(pstr, np.c_[R1R2,R2pR2],delimiter='\t',header='R1/R2'+'\t'+'R2p/R2')
    
    os.environ["Bstrpath"] = pstr

    
    os.environ["DMDT"] = str(0)
    os.environ["COUNT"] = str(0)
    
    os.environ["Lcool_event"] = str(0)
    os.environ["Lgain_event"] = str(0)
    
    os.environ["Coverfrac?"] = str(0)
    #os.environ["BD_res_B"] = str(0)
    #os.environ["BD_res_D"] = str(0)
    #os.environ["BD_res_Res"] = str(0)
    os.environ["BD_res_count"] = str(0)
    
    os.environ["Mcl_aux"] = str(Mcloud)
    os.environ["SF_aux"]= str(SFE)
    
    
    dic_res={'Lb': 0, 'Trgoal': 0, 'dMdt_factor': 0, 'Tavg': 0, 'beta': 0, 'delta': 0, 'residual': 0}

    os.environ["BD_res"]=str(dic_res)
    
    #print('SF**',os.environ["SF_aux"],os.environ["Mcl_aux"])
  
    return 0


def calcr_Tb(l1,l2):
    rTb=i.r_Tb
    
    try:
        if len(l1)>2:
            l1=l1[l1!=0]
            l2=l2[l2!=0]
            if len(l1)>1:
                a=np.max(l1)
                b=np.min(l2)
                rTb=b-0.2*(b-a)
        else:
            rTb=i.r_Tb 
    except:
        rTb=i.r_Tb 
    

        
    if np.isnan(rTb):
        rTb=i.r_Tb
        
    #print('calc_R_Tb=',rTb) 
    
    return rTb
    

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def trunc_auto(f, n):
    try:
        if f < 0:
            f*=-1
            print(f)
            m=True
            log=int(np.log10(f))-1
        else:
            log=int(np.log10(f))
            m=False
        #print(log)
        f=f*10**(-log)
        #print(f)
        trunc=float(truncate(f, n))
        #print(trunc)
        
        if m==True:
            res=float(str(-1*trunc)+'e'+str(log))
        else:
            res=float(str(trunc)+'e'+str(log))
    
    except:
        res=f
        
    #print(res) 
    return res   


def find_rdens(r,n):
    ''' Finds radius where density profile is steeper than r**(-2) 
    r = x-coordinate of dens profile
    n = y-coordinate of dens profile
    
    '''
    n=np.log10(n) #take logarithm 
    r=np.log10(r)
    
    dr = np.mean(diff(r)[5:10])

    fdot=diff(n)/dr #derivative
    
    index =  np.where(np.sqrt(fdot**2) > 2) #find index
    
    return 10**r[index[0][0]] #return first r where density profile is steeper


# def Larson_Mn(M):
#     # Larson (1981) laws with correction of Solomon (1987): 1st law (power law index 0.38 --> 0.5, amplitude 1.1 --> 0.72) The amplitude might be 0.78 (Bolatto+2008)
#     n = 3400.*((0.42*M**0.2/0.72)**(1./0.5))**(-1.1)
#     return n

# def my_sound_speed(T,gamma = 5./3.,heliumhydrogen = 14./11.):
#     hehy = heliumhydrogen
#     cs = np.sqrt(gamma*c.kboltz*T/(hehy*c.mp))
#     return cs

# def Bonnor_Ebert_rc(T,ncore, gamma = 5./3., heliumhydrogen = 14./11.):
#     hehy = heliumhydrogen
#     cs = my_sound_speed(T,gamma = gamma,heliumhydrogen = hehy)
#     #characteristic radius
#     rchar = cs/np.sqrt(4.*np.pi*c.Grav*ncore*hehy*c.mp)
#     rcross = 1.92*rchar
#     return rcross

# def F(x):
#     # M : clump mass
#     # n : average density
#     # n0 : core density ( 1e5 )
#     n0 = 1e5
#     M = 1e7*c.Msun
#     n = 100.
#     return n*(M/(4.*np.pi*n0*mui*x**2.) + 2./3.*x)**3. - 3.*M/(4.*np.pi*mui)





