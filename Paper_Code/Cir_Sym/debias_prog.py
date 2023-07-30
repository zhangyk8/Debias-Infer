#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: March 20, 2023

Description: This script contains the key functions for our debiasing program.
"""

import numpy as np
import scipy.stats
from sklearn.linear_model import Lasso
import statsmodels.api as sm
import cvxpy as cp
import ray

#==========================================================================================#

def ScaledLasso(X, Y, lam0=None, return_lamb=False):
    '''
    Scaled sparse linear regression (Translated from R package 'scalreg').
    
    Parameters
    ----------
        lam0: str or float
            The regularization parameter, which can be "univ", "quantile" or 
            other specified numerical value. (Default: if d<10^6, lam0="quantile"; 
            otherwise, lam0="univ".)
    '''
    n = X.shape[0]
    d = X.shape[1]
    if lam0 is None:
        if d > 1e6:
            lam0 = 'univ'
        else:
            lam0 = 'quantile'
            
    if lam0 == 'univ':
        lam0 = np.sqrt(2*np.log(d)/n)
    if lam0 == 'quantile':
        L = 0.1
        Lold = 0
        while abs(L - Lold) > 1e-4:
            k = L**4 + 2*(L**2)
            Lold = L
            L = -scipy.stats.norm.ppf(min([k/d, 0.99]))
            L = (L + Lold)/2
        if d == 1:
            L = 0.5
        lam0 = np.sqrt(2/n) * L
    
    sigma_int = 0.1
    sigma_new = 5
    flag = 0
    while (abs(sigma_int - sigma_new) > 1e-4) and (flag <= 500):
        flag += 1
        sigma_int = sigma_new
        lam = lam0 * sigma_int
        lasso_fit = Lasso(alpha=lam, fit_intercept=False, tol=1e-9, 
                          max_iter=int(1e7)).fit(X, Y)
        sigma_new = np.sqrt(np.mean((Y - lasso_fit.predict(X))**2))
    beta_est = lasso_fit.coef_
    if return_lamb:
        return beta_est, sigma_new, lam
    else:
        return beta_est, sigma_new


def LassoRefit(X, Y, x, method='scaled_lasso', return_beta=False):
    '''
    Lasso refitting function.
    
    Parameters
    ----------
        method: string
            The actual method for fitting the Lasso regression. It can be either 
            'sqrt_lasso' or the default value 'scaled_lasso'.
    '''
    if method == 'sqrt_lasso':
        n = X.shape[0]
        d = X.shape[1]
        sqrt_lasso = sm.OLS(Y,X).fit_regularized(method='sqrt_lasso', 
                                                 alpha=1.1*np.sqrt(n)*scipy.stats.norm.ppf(1-0.05/(2*d)), 
                                                 L1_wt=1.0, refit=False)
        beta_pilot = sqrt_lasso.params
    else:
        beta_pilot, sigma_pilot = ScaledLasso(X=X, Y=Y, lam0='quantile')
    flag = 1
    thres = 1e-7
    while flag == 1:
        X_sel = X[:,abs(beta_pilot) > thres]
        inv_Sigma = np.linalg.inv(np.dot(X_sel.T, X_sel))
        beta_new = np.dot(inv_Sigma, np.dot(X_sel.T, Y))
        x_new = x[abs(beta_pilot) > thres]
        
        m_refit = np.dot(beta_new, x_new)
        asym_var = np.sqrt(np.dot(x_new, np.dot(inv_Sigma, x_new)))
        df = X_sel.shape[0] - X_sel.shape[1]
        if df > 0:
            flag = 0
            sigma_hat = np.sqrt(np.sum((Y - np.dot(X_sel, beta_new))**2)/df)
        else:
            thres = 10*thres
            print('The degree of freedom is negative for Lasso refitting. '+
                  'We will raise the threshold for nonzero regression coefficients')
    if return_beta:
        return m_refit, beta_pilot, asym_var, sigma_hat, df
    else:
        return m_refit, asym_var, sigma_hat, df


def DebiasProg(X, x, Pi, gamma_n=0.1):
    '''
    Debiasing (primal) program.
    
    Parameters
    ----------
        gamma_n: float
            The regularization parameter "\gamma/n". (Default: gamma_n=0.1.)
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
    

def SoftThres(theta, lamb):
    '''
    Thresholding function.
    
    Parameters
    ----------
        lamb: float
            The thresholding parameter.
    '''
    try:
        return np.sign(theta)*max([abs(theta) - lamb, 0])
    except ValueError:
        res = np.zeros((theta.shape[0], 2))
        res[:,0] = np.abs(theta) - lamb
        return np.sign(theta)*np.max(res, axis=1)
    
def DualObj(X, x, Pi, ll_cur, gamma_n=0.05):
    '''
    Objective function of the dual form of our debiasing program.
    
    Parameters
    ----------
        gamma_n: float
            The regularization parameter "\gamma/n". (Default: gamma_n=0.05.)
    '''
    n = X.shape[0]
    A = np.dot(X.T, np.dot(Pi, X))
    return np.dot(np.dot(ll_cur.reshape(1, -1), A), ll_cur)/(4*n) + np.dot(x, ll_cur) \
        + gamma_n*np.sum(np.abs(ll_cur))
    

def DualCD(X, x, Pi=None, gamma_n=0.05, ll_init=None, eps=1e-9, max_iter=5000):
    '''
    Coordinate descent algorithm for solving the dual form of our debiasing program.
    
    Parameters
    ----------
        gamma_n: float
            The regularization parameter "\gamma/n". (Default: gamma_n=0.05.)
            
        max_iter: int
            Maximum number of coordinate descent iterations. (Default: max_iter=5000.)
    '''
    n = X.shape[0]
    d = X.shape[1]
    if Pi is None:
        Pi = np.eye(n)
    A = np.dot(X.T, np.dot(Pi, X))
    if ll_init is None:
        ll_new = np.ones((d,))
    else:
        ll_new = ll_init
    ll_old = 100*np.ones_like(ll_new)
    cnt = 0
    flag = 0
    while (np.linalg.norm(ll_old - ll_new) > eps) and ((cnt <= max_iter) or (flag == 0)):
        ll_old = np.copy(ll_new)
        cnt += 1
        # Coordinate descent
        for j in range(d):
            ll_cur = ll_new.copy()
            mask = np.ones(ll_cur.shape, dtype=bool)
            mask[j] = 0
            A_kj = A[mask, j]
            ll_cur = ll_cur[mask]
            up_val = SoftThres(-np.dot(A_kj, ll_cur)/(2*n) - x[j], lamb=gamma_n)/(A[j,j]/(2*n))
            if np.isnan(up_val):
                ll_new[j] = 0
            else:
                ll_new[j] = up_val
        if (cnt > max_iter) and (flag == 0):
            print('The coordinate descent algorithm has reached its maximum number of iterations: '\
                  +str(max_iter)+'!')
            A = A + 1e-9*np.eye(d)
            cnt = 0
            flag = 1
    return ll_new


def DualADMM(X, x, Pi=None, gamma_n=0.05, rho=1, ll_init=None, eps=1e-9):
    '''
    ADMM algorithm for solving the dual form of our debiasing program.
    
    Parameters
    ----------
        gamma_n: float
            The regularization parameter "\gamma/n". (Default: gamma_n=0.05.)
    '''
    n = X.shape[0]
    d = X.shape[1]
    if Pi is None:
        Pi = np.eye(n)
    A = np.dot(X.T, np.dot(Pi, X))
    if ll_init is None:
        ll_new = np.ones((d,))
    else:
        ll_new = ll_init
    ll_old = 100*np.ones_like(ll_new)
    nu_aug = np.ones_like(ll_new)
    mu_aug = np.zeros_like(ll_new)
    while np.linalg.norm(ll_old - ll_new) > eps:
        ll_old = np.copy(ll_new)
        ll_new = np.dot(np.linalg.inv(A/(2*n) + rho*np.eye(d)), rho*nu_aug - mu_aug - x)
        nu_aug = SoftThres(ll_new + mu_aug/rho, lamb=gamma_n/rho)
        mu_aug = mu_aug + rho*(ll_new - nu_aug)
    return ll_new

@ray.remote
def DualComp_Ray(X_train, X_test, x, Pi_train, Pi_test, gamma_n=0.05, ll_init=None, eps=1e-9):
    '''
    Parallel implemetation (Ray) of the dual form of our debiasing program.
    
    Parameters
    ----------
        gamma_n: float
            The regularization parameter "\gamma/n". (Default: gamma_n=0.05.)
    '''
    w_prime = DebiasProg(X=X_train.copy(), x=x, Pi=Pi_train, gamma_n=gamma_n)
    if any(np.isnan(w_prime)):
        print(r'The primal debiasing program for this fold of the data is '\
              'not feasible when \gamma/n='+str(round(gamma_n, 4))+'!')
        return np.array([np.nan])
    ll_train = DualCD(X=X_train, x=x, Pi=Pi_train, gamma_n=gamma_n, 
                      ll_init=None, eps=1e-9)
    dual_obj = DualObj(X_test, x=x, Pi=Pi_test, ll_cur=ll_train, gamma_n=gamma_n)
    return dual_obj