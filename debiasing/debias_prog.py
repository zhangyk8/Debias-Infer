#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: September 3, 2023

Description: This script contains the key functions for our debiasing method.
"""

import numpy as np
import scipy.stats
from sklearn.linear_model import Lasso
import statsmodels.api as sm
import cvxpy as cp

#=======================================================================================#


def DebiasProg(X, x, Pi, gamma_n=0.1):
    '''
    Our proposed Debiasing (primal) program.
    
    Parameters
    ----------
        X: (n,d)-array
            The input design matrix.
            
        x: (d,)-array
            The current query point.
            
        Pi: (n,n)-array
            A diagonal matrix with (estimated) propensity scores as its diagonal
            entries.
            
        gamma_n: float
            The regularization parameter "\gamma/n". (Default: gamma_n=0.1.)
            
    Return
    ----------
        w: (n,)-array
            The estimated weights by our debiasing program.
    '''
    n = X.shape[0]
    w = cp.Variable(n)
    debias_obj = cp.Minimize(cp.quad_form(w, Pi))
    constraints = [x - (1/np.sqrt(n))*(w @ Pi @ X) <= gamma_n, 
                   x - (1/np.sqrt(n))*(w @ Pi @ X) >= -gamma_n]
    debias_prog = cp.Problem(debias_obj, constraints)
    try:
        debias_prog.solve(solver=cp.MOSEK)
    except cp.SolverError:
        debias_prog.solve(solver=cp.CVXOPT, max_iters=30000)
    if debias_prog.value == np.inf:
        print('The primal debiasing program is infeasible!')
        return np.nan*np.ones((n,))
    else:
        return w.value
