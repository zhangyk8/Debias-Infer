#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Yikun Zhang
# Last Editing: September 3, 2023

# Description: This script contains the key functions for our debiasing method.

import numpy as np
import scipy.stats

#=======================================================================================#

def SoftThres(theta, lamb):
    '''
    Soft-thresholding function.
    
    Parameters
    ----------
        theta: (d,)-array
            The input vector for soft-thresholding.
            
        lamb: float
            The thresholding parameter.
    
    Return
    ----------
        theta: (n,)-array
            The output vector after soft-thresholding.
    '''
    try:
        return np.sign(theta)*max([abs(theta) - lamb, 0])
    except ValueError:
        res = np.zeros((theta.shape[0], 2))
        res[:,0] = np.abs(theta) - lamb
        return np.sign(theta)*np.max(res, axis=1)
    
