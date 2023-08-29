#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:30:36 2023

@author: Jia Wei Teh
"""


import numpy as np

x =  233229.2805318061
y = 303977.2092233737
z = 6287647631993850.0

x0 = 100000.0
y0 = 251188.6431509582
z0 = 1000000000000000.0
x1 = 316227.7660168379
y1 = 316227.7660168379
z1 = 1e+16


data = np.array( [
                    [
                        [3.84404480e-12, 3.84176530e-12],
                        [2.31919700e-12, 2.31400780e-12] 
                    ],
                 [ 
                    [3.80803497e-11, 3.80310120e-11],
                    [2.29206415e-11, 2.28381210e-11], 
                    ]
                 
                 ])



# =============================================================================
# Test 1
# =============================================================================

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
    
    
    
    x, y, z = x
    x0, y0, z0 = X0
    x1, y1, z1 = X1
    
    
    return c



dudt = trilinear([x, y, z], [x0, y0, z0], [x1, y1, z1], data)
    
print('own function', dudt)



# =============================================================================
# Test 2
# =============================================================================

import scipy.interpolate

xarr = np.linspace(x0, x1, 2)
yarr = np.linspace(y0, y1, 2)
zarr = np.linspace(z0, z1, 2)

f_dudt = scipy.interpolate.RegularGridInterpolator((xarr, yarr, zarr), data)

print('with scipy', f_dudt([x, y, z]))




#%%


# But if I already have the interpolation function, why can't I just give values and 
# dont do any of the calculations daniel do?




















