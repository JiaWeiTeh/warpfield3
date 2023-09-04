#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:30:36 2023

@author: Jia Wei Teh
"""


import numpy as np




x =  506663.2212419483
y = 294060.78931362595
z = 1.5473355225629384e+16

x0 = 316227.7660168379
y0 = 251188.6431509582
z0 = 1e+16
x1 = 1000000.0
y1 = 316227.7660168379
z1 = 1e+17


# #%%



data = np.array( [
                    [
                        [3.8031012e-11, 3.8123006e-11],
                        [2.2838121e-11, 2.2858937e-11] 
                    ],
                  [ 
                    [3.7416466e-10, 3.7678686e-10],
                    [2.2416979e-10, 2.2486096e-10], 
                    ]
                 
                  ])



# # =============================================================================
# # Test 1
# # =============================================================================

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



dudt_mine = trilinear([x, y, z], [x0, y0, z0], [x1, y1, z1], data)
    


# # =============================================================================
# # Test 2
# # =============================================================================

import scipy.interpolate

xarr = np.linspace(x0, x1, 2)
yarr = np.linspace(y0, y1, 2)
zarr = np.linspace(z0, z1, 2)

f_dudt = scipy.interpolate.RegularGridInterpolator((xarr, yarr, zarr), data)

dudt_scipy = f_dudt([x, y, z])[0] 


assert np.isclose(dudt_scipy, dudt_mine), f"values dont match\nmine: {dudt_mine}\nscipy: {dudt_scipy}"
print(f"\nmine: {dudt_mine}\nscipy: {dudt_scipy}")
print('test case 1 succeeded!')



# %%




big_cooling_table = r'/Users/jwt/Documents/Code/WARPFIELDv4/cool_interp3dudtmyelement.npy'

# how does the data look like?

cooling_table = np.load(big_cooling_table)

print(cooling_table)



#%%






# But if I already have the interpolation function, why can't I just give values and 
# dont do any of the calculations daniel do?

from src.warpfield.cooling.non_CIE import read_cloudy
import os


# os.environ['PATH_TO_CONFIG'] = r'/Users/jwt/Documents/Code/warpfield3/outputs/example_pl/example_pl_config.yaml'



# test cases
Cool_Struc = read_cloudy.get_coolingStructure(1e6)



#%%




print(Cool_Struc['log_n'])
print(Cool_Struc['log_T'])
print(Cool_Struc['log_phi'])


#%%


# print(Cool_Struc)
# print(Cool_Struc['log_cooling_interpolation'])
print(Cool_Struc['log_cooling_interpolation'](1, 4, 14))
print(Cool_Struc['log_cooling_interpolation'](1, 4, 14))
print(Cool_Struc['log_heating_interpolation'](1, 4, 14))


# remember to log the values

# netcooling = Cool_Struc['log_cooling_interpolation'](x, y, z) - Cool_Struc['log_heating_interpolation'](x, y, z)

# print(Cool_Struc['log_cooling_interpolation'](x, y, z))


# print(netcooling)




# Cool_Struc = read_cloudy.get_coolingStructure(1.5e6)
# print(logT)
# print(logLambda)








#%%


# example 1


from scipy.interpolate import LinearNDInterpolator

import numpy as np

import matplotlib.pyplot as plt

rng = np.random.default_rng(100)

x = rng.random(10) - 0.5

y = rng.random(10) - 0.5

z = np.hypot(x, y)

X = np.linspace(min(x), max(x))

Y = np.linspace(min(y), max(y))

X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation

interp = LinearNDInterpolator(list(zip(x, y)), z)

z = interp(x[0], y[0])

print(z)


# Z = interp(X, Y)

# plt.pcolormesh(X, Y, Z, shading='auto')

# plt.plot(x, y, "ok", label="input point")

# plt.legend()

# plt.colorbar()

# plt.axis("equal")

# plt.show()









#%%

# example 2





from scipy.interpolate import RegularGridInterpolator

import numpy as np

def f(x, y, z):

    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 10)

y = np.linspace(4, 7, 10)

z = np.linspace(7, 9, 10)

xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)

data = f(xg, yg, zg)


interp = RegularGridInterpolator((x, y, z), data)


pts = np.array([2.1, 6.2, 8.3])

print(interp(pts))




#%%










from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator

import numpy as np

def f(x, y, z):

    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 10)

y = np.linspace(4, 7, 10)

z = np.linspace(7, 9, 10)

# xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)

data = f(x, y, z)


interp = LinearNDInterpolator((x, y, z), data)


pts = np.array([2.1, 6.2, 8.3])

print(interp(pts))








#%%





























