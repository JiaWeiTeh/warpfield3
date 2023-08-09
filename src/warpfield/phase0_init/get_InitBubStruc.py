#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:14:13 2023

@author: Jia Wei Teh
"""

import os
import numpy as np

def get_InitBubStruc(Mcloud, SFE, path):
    """
    This function initialises environmental variables to help calculate
    bubble structures.

    Parameters
    ----------
    Mcloud : TYPE
        DESCRIPTION.
    SFE : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.

    Returns
    -------

    """
    # Notes
    # old code: optimalbstrux in aux_func()
    
    # Initialise this
    R1R2 = R2pR2 = np.array([0])
    # check if directory exists
    dirstring = os.path.join(path, "BubDetails")
    if not os.path.isdir(dirstring):
        os.makedirs(dirstring)
    # path to bubble details
    pstr = path +"/BubDetails/Bstrux.txt"
    # save to path
    # TODO
    # np.savetxt(pstr, np.c_[R1R2,R2pR2],delimiter='\t',header='R1/R2'+'\t'+'R2p/R2')
    
    # initialise some environment variables. 
    # path
    os.environ["Bstrpath"] = pstr
    # dMdt
    os.environ["DMDT"] = str(0)
    # count
    os.environ["COUNT"] = str(0)
    # Lcool/gain
    os.environ["Lcool_event"] = str(0)
    os.environ["Lgain_event"] = str(0)
    # If coverfraction
    os.environ["Coverfrac?"] = str(0)
    # ??
    os.environ["BD_res_count"] = str(0)
    # ??
    os.environ["Mcl_aux"] = str(Mcloud)
    os.environ["SF_aux"]= str(SFE)
    # ??
    dic_res={'Lb': 0, 'Trgoal': 0, 'dMdt_factor': 0, 'Tavg': 0, 'beta': 0, 'delta': 0, 'residual': 0}
    os.environ["BD_res"]=str(dic_res)
    # return
    return 0
