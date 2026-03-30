#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Jul 15, 2023

Description: Obtaining the Lasso pilot estimates (Autoregressive covariance 
case).
"""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
import sys

from debias_prog import ScaledLasso

job_id = int(sys.argv[1])
print(job_id)

#==========================================================================================#

## Homoscedastic case (Autoregressive covariance)
d = 1000
n = 900

Sigma = np.zeros((d,d)) + np.eye(d)
rho = 0.9
for i in range(d):
    for j in range(i+1, d):
        Sigma[i,j] = rho**(abs(i-j))
        Sigma[j,i] = rho**(abs(i-j))
sig = 1

## Consider different simulation settings
for i in range(6):
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
    if i == 5:
        ## x5
        x = np.ones((d,))/np.sqrt(d)
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
        
        for error in ['gauss', 'laperr', 'terr', 'unif']:
            np.random.seed(job_id)

            X_sim = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
            if error == 'gauss':
                eps_err_sim = sig*np.random.randn(n)
            if error == 'laperr':
                eps_err_sim = np.random.laplace(loc=0, scale=1/np.sqrt(2), size=n)
            if error == 'terr':
                eps_err_sim = np.random.standard_t(df=2, size=n)
            if error == 'unif':
                eps_err_sim = np.random.rand(n)*2*np.sqrt(3) - np.sqrt(3)
            Y_sim = np.dot(X_sim, beta_0) + eps_err_sim
            Sigma_hat = np.dot(X_sim.T, X_sim)/n
                
            ## MCAR
            obs_prob1 = 0.7
            R1 = np.random.choice([0,1], size=n, replace=True, p=[1-obs_prob1, obs_prob1])
                
            ## MAR
            obs_prob2 = 1/(1 + np.exp(-1+X_sim[:,6]-X_sim[:,7]))
            R2 = np.ones((n,))
            R2[np.random.rand(n) >= obs_prob2] = 0
                
            ## Lasso pilot estimator (Complete-case)
            beta_cc1, sigma_cc1 = ScaledLasso(X=X_sim[R1 == 1,:], Y=Y_sim[R1 == 1], lam0='univ') 
            beta_cc2, sigma_cc2 = ScaledLasso(X=X_sim[R2 == 1,:], Y=Y_sim[R2 == 1], lam0='univ') 
            
            ## Propensity score estimation (logistic regression CV)
            zeta1 = np.logspace(-1, np.log10(300), 40)*np.sqrt(np.log(d)/n)
            lr1 = LogisticRegressionCV(Cs=1/zeta1, cv=5, penalty='l1', scoring='neg_log_loss', 
                                       solver='liblinear', tol=1e-6, max_iter=10000).fit(X_sim, R1)
            prop_score1 = lr1.predict_proba(X_sim)[:,1]
                
            zeta2 = np.logspace(-1, np.log10(300), 40)*np.sqrt(np.log(d)/n)
            lr2 = LogisticRegressionCV(Cs=1/zeta2, cv=5, penalty='l1', scoring='neg_log_loss', 
                                       solver='liblinear', tol=1e-6, max_iter=10000).fit(X_sim, R2)
            prop_score2 = lr2.predict_proba(X_sim)[:,1]
        
            ### Lasso pilot estimator (IPW)
            X = np.dot(np.diag(R1/np.sqrt(obs_prob1)), X_sim)[R1 == 1,:].copy()
            Y = (Y_sim * (R1/np.sqrt(obs_prob1)))[R1 == 1].copy()
            beta_ipw1, sigma_ipw1 = ScaledLasso(X=X, Y=Y, lam0='univ') 
            
            X = np.dot(np.diag(R2/np.sqrt(obs_prob2)), X_sim)[R2 == 1,:].copy()
            Y = (Y_sim * (R2/np.sqrt(obs_prob2)))[R2 == 1].copy()
            beta_ipw2, sigma_ipw2 = ScaledLasso(X=X, Y=Y, lam0='univ') 
            
            ## Lasso pilot estimator (Oracle)
            X = X_sim.copy()
            Y = Y_sim.copy()
            beta_full, sigma_full = ScaledLasso(X=X, Y=Y, lam0='univ') 
            
            pilot_res1 = pd.DataFrame({'m_obs1': np.dot(x, beta_cc1), 'sigma_hat_obs1': sigma_cc1,
                                       'm_obs2': np.dot(x, beta_cc2), 'sigma_hat_obs2': sigma_cc2, 
                                       'm_ipw1': np.dot(x, beta_ipw1), 'sigma_hat_ipw1': sigma_ipw1,
                                       'm_ipw2': np.dot(x, beta_ipw2), 'sigma_hat_ipw2': sigma_ipw2,  
                                       'm_full': np.dot(x, beta_full), 'sigma_hat_full': sigma_full, 
                                       'm_var_oracle': np.dot(np.dot(x, Sigma), x), 
                                       'm_var_mcar': np.dot(np.dot(x, Sigma), x)/obs_prob1}, index=[0])
            pilot_res1.to_csv('./pilot_res/lasso_pilot_AR_d'+str(d)+'_n'+str(n)+'_'+str(job_id)+\
                              '_x'+str(i)+'_beta'+str(k)+'_'+str(error)+'.csv', index=False)