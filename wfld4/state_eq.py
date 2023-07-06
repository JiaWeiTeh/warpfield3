import numpy as np
import constants as c


def PfromE(Eb, R2, R1, gamma=c.gamma):
    R2=R2+(10**(-10)) #avoid division by 0
    Pb = (gamma - 1.) * 3. * Eb / (4. * np.pi * (R2**3.-R1**3.))
    #print('Pb=', Pb ,'R2=',R2,';R1=',R1,';')
    return Pb

def EfromP(Pb, R2, R1, gamma=c.gamma):
    Eb = 4. * np.pi * (R2**3. - R1**3.) * Pb / ((gamma - 1.) * 3.)
    return Eb

def Pram(r,Lw,vw):
    return Lw / (2. * np.pi * r ** 2 * vw)

def delta_to_Tdot(t,T,delta):
    """
    converts delta to Tdot
    :param t: time
    :param T: temprature (at xi)
    :param delta: (t/T)*(dT/dt)
    :return:
    """

    Tdot = (T/t)*delta

    return Tdot

def Tdot_to_delta(t,T,Tdot):
    """
    converts Tdot to delta (inverse function of delta_to_Tdot)
    :param t: time
    :param T: temprature (at xi)
    :param Tdot: time derivative of temperature
    :return:
    """

    delta = Tdot*(t/T)

    return delta

def beta_to_Edot(Pb, R1, beta, my_params):
    """
    converts beta to dE/dt
    :param Pb: pressure of bubble
    :param R1: inner radius of bubble
    :param beta: -(t/Pb)*(dPb/dt), see Weaver+77, eq. 40
    :param my_params:
    :return:
    """
    R2 = my_params['R2']
    v2 = my_params["v2"]
    E = my_params['Eb']
    Pdot = -Pb*beta/my_params["t_now"]

    pwdot = my_params['pwdot'] # pwdot = 2.*Lw/vw

    A = np.sqrt(pwdot/2.)
    A2 = A**2
    C = 1.5*A2*R1
    D = R2**3 - R1**3
    #Adot = (my_params['Lw_dot']*vw - Lw*my_params['vw_dot'])/(2.*A*vw**2)
    Adot = 0.25*my_params['pwdot_dot']/A

    F = C / (C + E)

    #Edot = ( 3.*v2 * R2**2 * E + 2.*np.pi*Pdot*D**2 ) / D # does not take into account R1dot
    #Edot = ( 2.*np.pi*Pdot*D**2 + 3.*E*v2*R2**2 * (1.-F) ) / (D * (1.-F)) # takes into account R1dot but not time derivative of A
    Edot = ( 2.*np.pi*Pdot*D**2 + 3.*E*v2*R2**2 * (1.-F) - 3.*(Adot/A)*R1**3*E**2/(E+C) ) / (D * (1.-F)) # takes everything into account

    #print "term1", "%.5e"%(2.*np.pi*Pdot*D**2), "term2", "%.5e"%(3.*E*v2*R2**2 * (1.-F)), "term3", "%.5e"%(3.*(Adot/A)*R1**3*E**2/(E+C))

    #print "Edot", "%.5e"%Edot, "%.5e"%Edot_exact

    return Edot

def Edot_to_beta(Pb, R1, Edot, my_params):
    """
    converts Edot to beta (inverse function of beta_to_Edot)
    :param Pb: pressure of bubble
    :param R1: inner radius of bubble
    :param Edot: time derivative of bubble energy
    :param my_params:
    :return:
    """

    R2 = my_params['R2']
    v2 = my_params["v2"]
    E = my_params['Eb']

    pwdot = my_params['pwdot']  # pwdot = 2.*Lw/vw

    A = np.sqrt(pwdot / 2.)
    A2 = A ** 2
    C = 1.5 * A2 * R1
    D = R2 ** 3 - R1 ** 3
    # Adot = (my_params['Lw_dot']*vw - Lw*my_params['vw_dot'])/(2.*A*vw**2)
    Adot = 0.25 * my_params['pwdot_dot'] / A

    F = C / (C + E)

    Pdot = 1./(2.*np.pi*D**2.) * ((D * (1.-F)) * Edot - 3.*E*v2*R2**2 * (1.-F) + 3.*(Adot/A)*R1**3*E**2/(E+C) )
    beta = -Pdot*my_params["t_now"]/Pb

    return beta