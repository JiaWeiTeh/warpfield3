#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:36:36 2023

@author: Jia Wei Teh

This script contains useful functions that help compute stuffs
"""

import numpy as np

def find_nearest(array, value):
    """
    finds index idx in array for which array[idx] is closest to value
    """
    # make sure that we deal with an numpy array
    array = np.array(array)
    # index
    idx = (np.abs(array-value)).argmin()
    # return
    return idx

def find_nearest_lower(array, value):
    """
    This fucntion finds idx in array for which array[idx] satisfies:
        1) smaller or equal to value; and
        2) closest to value.
    Elements in array need be monotonically increasing or decreasing!
    """
    # check whether array is monotonically increasing (by only checking the first 2 elements)
    if array[1] > array[0]:
        mon_incr = True
    else:
        mon_incr = False
    # get index
    idx = find_nearest(array, value)
    #---
    #---
    if array[idx] - value > 0: # then this element is the closest, but it is larger than value
        if mon_incr: 
            idx += -1 # take the element before, it will be smaller than value (if array is monotonically increasing)
        else: 
            idx += 1
    # Notes: boundary conditions, just in case. Although when these happen, it means that
    # the returned idx is actually higher than the value instead of the desired 
    # lower. Not quite sure what to do with that for now, but this part of 
    # the code shouldnt need to run anyway.
    if idx >= len(array): 
        idx = len(array) - 1
    if idx < 0: 
        idx = 0
    # return
    return idx

def find_nearest_higher(array, value):
    """
    This fucntion finds idx in array for which array[idx] satisfies:
        1) smaller or equal to value; and
        2) closest to value.
    Elements in array need be monotonically increasing or decreasing!
    """
    # check whether array is monotonically increasing (by only checking the first 2 elements)
    if array[1] > array[0]:
        mon_incr = True
    else:
        mon_incr = False
    # get index
    idx = find_nearest(array, value)
    #---
    #---
    if array[idx] - value < 0: # then this element is the closest, but it is larger than value
        if mon_incr: 
            idx += 1 # take the element before, it will be smaller than value (if array is monotonically increasing)
        else: 
            idx += -1
    # Notes: boundary conditions, just in case. Although when these happen, it means that
    # the returned idx is actually higher than the value instead of the desired 
    # lower. Not quite sure what to do with that for now, but this part of 
    # the code shouldnt need to run anyway.
    if idx >= len(array): 
        idx = len(array) - 1
    if idx < 0: 
        idx = 0
    # return
    return idx



