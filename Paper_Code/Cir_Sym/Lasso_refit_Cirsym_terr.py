#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: April 19, 2023

Description: Lasso refitting (circulant symmetric covariance).
"""

import numpy as np
import scipy.stats
from sklearn.linear_model import LogisticRegression, Lasso
import pandas as pd
import sys

from debias_prog import ScaledLasso

job_id = int(sys.argv[1])
print(job_id)

def LassoRefit(X, Y, x):
    beta_pilot, sigma_pilot = ScaledLasso(X=X, Y=Y, lam0='univ')
    X_sel = X[:,abs(beta_pilot) > 1e-7]
    inv_Sigma = np.linalg.inv(np.dot(X_sel.T, X_sel))
    beta_new = np.dot(inv_Sigma, np.dot(X_sel.T, Y))
    x_new = x[abs(beta_pilot) > 1e-7]
    
    m_refit = np.dot(beta_new, x_new)
    asym_var = np.sqrt(np.dot(x_new, np.dot(inv_Sigma, x_new)))
    df = X_sel.shape[0] - X_sel.shape[1]
    sigma_hat = np.sqrt(np.sum((Y - np.dot(X_sel, beta_new))**2)/df)
    return m_refit, asym_var, sigma_hat, df


## Homoscedastic case
d = 1000
n = 900

Sigma = np.zeros((d,d)) + np.eye(d)
rho = 0.1
for i in range(d):
    for j in range(i+1, d):
        if (j < i+6) or (j > i+d-6):
            Sigma[i,j] = rho
            Sigma[j,i] = rho

## Consider different simulation settings
for i in range(5):
    if i == 0:
        ## x0
        x = np.zeros((d,))
        x[0] = 1
    if i == 1:
        ## x1
        x = np.zeros((d,))
        x[0] = 1
        x[1] = 1/2
        x[2] = 1/4
        x[6] = 1/2
        x[7] = 1/8
    if i == 2:
        ## x2
        x = np.zeros((d,))
        x[99] = 1
    if i == 3:
        ## x3
        x = 1/np.linspace(1, d, d)
    if i == 4:
        ## x4
        x = 1/np.linspace(1, d, d)**2
    for k in range(3):
        if k == 0:
            s_beta = 5
            beta_0 = np.zeros((d,))
            beta_0[:s_beta] = np.sqrt(5)
        if k == 1:
            beta_0 = 1/np.sqrt(np.linspace(1, d, d))
            beta_0 = 5*beta_0/np.linalg.norm(beta_0)
        if k == 2:
            beta_0 = 1/np.linspace(1, d, d)
            beta_0 = 5*beta_0/np.linalg.norm(beta_0)

        # True regression function
        m_true = np.dot(x, beta_0)
        
        np.random.seed(job_id)

        # True regression function
        m_true = np.dot(x, beta_0)
        
        # Significance level
        alpha = 0.05
        
        X_sim = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
        eps_err_sim = np.random.standard_t(df=2, size=n)
        Y_sim = np.dot(X_sim, beta_0) + eps_err_sim
        
        ## MCAR
        obs_prob1 = 0.7
        R1 = np.random.choice([0,1], size=n, replace=True, p=[1-obs_prob1, obs_prob1])
        
        ## MAR
        obs_prob2 = 1/(1 + np.exp(-1+X_sim[:,6]-X_sim[:,7]))
        R2 = np.ones((n,))
        R2[np.random.rand(n) >= obs_prob2] = 0
        
        ### Complete-case data
        X = X_sim[R1 == 1,:].copy()
        Y = Y_sim[R1 == 1].copy()
        m_refit_obs1, asym_var_obs1, sigma_hat_obs1, df = LassoRefit(X, Y, x)
        if (i == 0) or (i == 2):
            ci_len_obs1 = 2*sigma_hat_obs1*asym_var_obs1*scipy.stats.t.ppf(1 - alpha/2, df=df)
        else:
            ci_len_obs1 = 2*sigma_hat_obs1*asym_var_obs1*scipy.stats.norm.ppf(1 - alpha/2)
        
        X = X_sim[R2 == 1,:].copy()
        Y = Y_sim[R2 == 1].copy()
        m_refit_obs2, asym_var_obs2, sigma_hat_obs2, df = LassoRefit(X, Y, x)
        if (i == 0) or (i == 2):
            ci_len_obs2 = 2*sigma_hat_obs2*asym_var_obs2*scipy.stats.t.ppf(1 - alpha/2, df=df)
        else:
            ci_len_obs2 = 2*sigma_hat_obs2*asym_var_obs2*scipy.stats.norm.ppf(1 - alpha/2)
        
        ### IPW
        X = np.dot(np.diag(R1/np.sqrt(obs_prob1)), X_sim)[R1 == 1,:].copy()
        Y = (Y_sim * (R1/np.sqrt(obs_prob1)))[R1 == 1].copy()
        m_refit_ipw1, asym_var_ipw1, sigma_hat_ipw1, df = LassoRefit(X, Y, x)
        if (i == 0) or (i == 2):
            ci_len_ipw1 = 2*sigma_hat_ipw1*asym_var_ipw1*scipy.stats.t.ppf(1 - alpha/2, df=df)
        else:
            ci_len_ipw1 = 2*sigma_hat_ipw1*asym_var_ipw1*scipy.stats.norm.ppf(1 - alpha/2)
        
        X = np.dot(np.diag(R2/np.sqrt(obs_prob2)), X_sim)[R2 == 1,:].copy()
        Y = (Y_sim * (R2/np.sqrt(obs_prob2)))[R2 == 1].copy()
        m_refit_ipw2, asym_var_ipw2, sigma_hat_ipw2, df = LassoRefit(X, Y, x)
        if (i == 0) or (i == 2):
            ci_len_ipw2 = 2*sigma_hat_ipw2*asym_var_ipw2*scipy.stats.t.ppf(1 - alpha/2, df=df)
        else:
            ci_len_ipw2 = 2*sigma_hat_ipw2*asym_var_ipw2*scipy.stats.norm.ppf(1 - alpha/2)
        
        ### Full (oracle) data
        m_refit_full, asym_var_full, sigma_hat_full, df = LassoRefit(X_sim, Y_sim, x)
        if (i == 0) or (i == 2):
            ci_len_full = 2*sigma_hat_full*asym_var_full*scipy.stats.t.ppf(1 - alpha/2, df=df)
        else:
            ci_len_full = 2*sigma_hat_full*asym_var_full*scipy.stats.norm.ppf(1 - alpha/2)
        
        refit_res1 = pd.DataFrame({'m_obs1': m_refit_obs1, 'asym_se_obs1': asym_var_obs1, 
                                   'sigma_hat_obs1': sigma_hat_obs1, 'ci_len_obs1': ci_len_obs1, 
                                   'm_obs2': m_refit_obs2, 'asym_se_obs2': asym_var_obs2, 
                                   'sigma_hat_obs2': sigma_hat_obs2, 'ci_len_obs2': ci_len_obs2, 
                                   'm_ipw1': m_refit_ipw1, 'asym_se_ipw1': asym_var_ipw1, 
                                   'sigma_hat_ipw1': sigma_hat_ipw1, 'ci_len_ipw1': ci_len_ipw1, 
                                   'm_ipw2': m_refit_ipw2, 'asym_se_ipw2': asym_var_ipw2, 
                                   'sigma_hat_ipw2': sigma_hat_ipw2, 'ci_len_ipw2': ci_len_ipw2, 
                                   'm_full': m_refit_full, 'asym_se_full': asym_var_full, 
                                   'sigma_hat_full': sigma_hat_full, 'ci_len_full': ci_len_full}, index=[0])
        refit_res1.to_csv('./refit_res/refit_Cirsym_d'+str(d)+'_n'+str(n)+'_'+str(job_id)+\
                          '_x'+str(i)+'_beta'+str(k)+'_terr.csv', index=False)
