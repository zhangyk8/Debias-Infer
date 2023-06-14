#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: June 13, 2023

Description: Simulations on our debiasing program (Equi-correlated covariance 
with Gaussian noises).
"""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
import pickle
import sys

from debias_prog import ScaledLasso, DebiasProg, DualObj, DualCD, LassoRefit

job_id = int(sys.argv[1])
print(job_id)

#==========================================================================================#

## Homoscedastic case (Equi-correlated covariance)
d = 1000
n = 900

rho = 0.8
Sigma = rho*np.ones((d,d)) + (1-rho)*np.eye(d)
sig = 1

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

        X_sim = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
        eps_err_sim = sig*np.random.randn(n)
        Y_sim = np.dot(X_sim, beta_0) + eps_err_sim
        Sigma_hat = np.dot(X_sim.T, X_sim)/n
            
        ## MCAR
        obs_prob1 = 0.7
        R1 = np.random.choice([0,1], size=n, replace=True, p=[1-obs_prob1, obs_prob1])
            
        ## MAR
        obs_prob2 = 1/(1 + np.exp(-1+X_sim[:,6]-X_sim[:,7]))
        R2 = np.ones((n,))
        R2[np.random.rand(n) >= obs_prob2] = 0
            
        ## Lasso pilot estimator
        # beta_pilot1, sigma_pilot1 = ScaledLasso(X=X_sim[R1 == 1,:], Y=Y_sim[R1 == 1], lam0='univ') 
        # beta_pilot2, sigma_pilot2 = ScaledLasso(X=X_sim[R2 == 1,:], Y=Y_sim[R2 == 1], lam0='univ') 
        m_refit1, beta_pilot1, asym_var1, sigma_pilot1, df1 = LassoRefit(X=X_sim[R1 == 1,:], 
                                                                         Y=Y_sim[R1 == 1], x=x, 
                                                                         method='sqrt_lasso', 
                                                                         return_beta=True)
        m_refit2, beta_pilot2, asym_var2, sigma_pilot2, df2 = LassoRefit(X=X_sim[R2 == 1,:], 
                                                                         Y=Y_sim[R2 == 1], x=x, 
                                                                         method='sqrt_lasso', 
                                                                         return_beta=True)
            
        ## Propensity score estimation (logistic regression CV)
        zeta1 = np.logspace(-1, np.log10(300), 40)*np.sqrt(np.log(d)/n)
        lr1 = LogisticRegressionCV(Cs=1/zeta1, cv=5, penalty='l1', scoring='neg_log_loss', 
                                   solver='liblinear', tol=1e-6, max_iter=10000).fit(X_sim, R1)
        prop_score1 = lr1.predict_proba(X_sim)[:,1]
            
        zeta2 = np.logspace(-1, np.log10(300), 40)*np.sqrt(np.log(d)/n)
        lr2 = LogisticRegressionCV(Cs=1/zeta2, cv=5, penalty='l1', scoring='neg_log_loss', 
                                   solver='liblinear', tol=1e-6, max_iter=10000).fit(X_sim, R2)
        prop_score2 = lr2.predict_proba(X_sim)[:,1]
            
        gamma_n_lst = np.linspace(0.001, np.max(abs(x)), 41)
        cv_fold = 5
        
        kf = KFold(n_splits=cv_fold, shuffle=True, random_state=0)
        f_ind = 0
        dual_loss1 = np.zeros((cv_fold, len(gamma_n_lst)))
        dual_loss2 = np.zeros((cv_fold, len(gamma_n_lst)))
        for train_ind, test_ind in kf.split(X_sim):
            X_train = X_sim[train_ind,:]
            X_test = X_sim[test_ind,:]
            prop_score1_train = prop_score1[train_ind]
            prop_score1_test = prop_score1[test_ind]
            
            prop_score2_train = prop_score2[train_ind]
            prop_score2_test = prop_score2[test_ind]
            
            for j in range(len(gamma_n_lst)):
                w_train1 = DebiasProg(X=X_train.copy(), x=x, Pi=np.diag(prop_score1_train), gamma_n=gamma_n_lst[j])
                if any(np.isnan(w_train1)):
                    print(r'The primal debiasing program for this fold of the data is '\
                          'not feasible when \gamma/n='+str(round(gamma_n_lst[j], 4))+'!')
                    dual_loss1[f_ind, j] = np.nan
                else:
                    ll_train1 = DualCD(X=X_train, x=x, Pi=np.diag(prop_score1_train), gamma_n=gamma_n_lst[j], 
                                       ll_init=None, eps=1e-8, max_iter=5000)
                    if sum(abs(w_train1 + np.dot(X_train, ll_train1)/(2*np.sqrt(X_train.shape[0])))>1e-3) > 0:
                        print(r'The strong duality between primal and dual programs does not satisfy '\
                              'when \gamma/n='+str(round(gamma_n_lst[j], 4))+'!')
                        dual_loss1[f_ind, j] = np.nan
                    else:
                        dual_loss1[f_ind, j] = DualObj(X_test, x=x, Pi=np.diag(prop_score1_test), 
                                                       ll_cur=ll_train1, gamma_n=gamma_n_lst[j])
                w_train2 = DebiasProg(X=X_train.copy(), x=x, Pi=np.diag(prop_score2_train), gamma_n=gamma_n_lst[j])
                if any(np.isnan(w_train2)):
                    print(r'The primal debiasing program for this fold of the data is '\
                          'not feasible when \gamma/n='+str(round(gamma_n_lst[j], 4))+'!')
                    dual_loss2[f_ind, j] = np.nan
                else:
                    ll_train2 = DualCD(X=X_train, x=x, Pi=np.diag(prop_score2_train), gamma_n=gamma_n_lst[j], 
                                       ll_init=None, eps=1e-8, max_iter=5000)
                    if sum(abs(w_train2 + np.dot(X_train, ll_train2)/(2*np.sqrt(X_train.shape[0])))>1e-3) > 0:
                        print(r'The strong duality between primal and dual programs does not satisfy '\
                              'when \gamma/n='+str(round(gamma_n_lst[j], 4))+'!')
                        dual_loss2[f_ind, j] = np.nan
                    else:
                        dual_loss2[f_ind, j] = DualObj(X_test, x=x, Pi=np.diag(prop_score2_test), 
                                                       ll_cur=ll_train2, gamma_n=gamma_n_lst[j])
            f_ind = f_ind + 1
        mean_dual_loss1 = np.mean(dual_loss1, axis=0)
        mean_dual_loss2 = np.mean(dual_loss2, axis=0)
        std_dual_loss1 = np.std(dual_loss1, axis=0, ddof=1)/np.sqrt(cv_fold)
        std_dual_loss2 = np.std(dual_loss2, axis=0, ddof=1)/np.sqrt(cv_fold)
        
        # Different rules for choosing the tuning parameter
        para_rule = ['1se', 'mincv', 'minfeas']
        for rule in para_rule:
            if rule == 'mincv':
                gamma_n1_opt = gamma_n_lst[np.nanargmin(mean_dual_loss1)]
                gamma_n2_opt = gamma_n_lst[np.nanargmin(mean_dual_loss2)]
            if rule == '1se':
                One_SE1 = (mean_dual_loss1 > np.nanmin(mean_dual_loss1) + std_dual_loss1[np.nanargmin(mean_dual_loss1)]) & \
                (gamma_n_lst < gamma_n_lst[np.nanargmin(mean_dual_loss1)])
                if sum(One_SE1) == 0:
                    One_SE1 = np.full((len(gamma_n_lst),), True)
                gamma_n_lst1 = gamma_n_lst[One_SE1]
                gamma_n1_opt = gamma_n_lst1[np.nanargmin(mean_dual_loss1[One_SE1])]
                
                One_SE2 = (mean_dual_loss2 > np.nanmin(mean_dual_loss2) + std_dual_loss2[np.nanargmin(mean_dual_loss2)]) & \
                (gamma_n_lst < gamma_n_lst[np.nanargmin(mean_dual_loss2)])
                if sum(One_SE2) == 0:
                    One_SE2 = np.full((len(gamma_n_lst),), True)
                gamma_n_lst2 = gamma_n_lst[One_SE2]
                gamma_n2_opt = gamma_n_lst2[np.nanargmin(mean_dual_loss2[One_SE2])]
            if rule == 'minfeas':
                gamma_n1_opt = np.min(gamma_n_lst[~np.isnan(mean_dual_loss1)])
                gamma_n2_opt = np.min(gamma_n_lst[~np.isnan(mean_dual_loss2)])
                
            # Solve the primal and dual on the original dataset
            w_obs1 = DebiasProg(X=X_sim.copy(), x=x, Pi=np.diag(prop_score1), gamma_n=gamma_n1_opt)
            ll_obs1 = DualCD(X=X_sim, x=x, Pi=np.diag(prop_score1), gamma_n=gamma_n1_opt, ll_init=None, eps=1e-9)
            
            w_obs2 = DebiasProg(X=X_sim.copy(), x=x, Pi=np.diag(prop_score2), gamma_n=gamma_n2_opt)
            ll_obs2 = DualCD(X=X_sim, x=x, Pi=np.diag(prop_score2), gamma_n=gamma_n2_opt, ll_init=None, eps=1e-9)
            
            # Store the results
            m_deb1 = np.dot(x, beta_pilot1) + np.sum(w_obs1 * R1 * (Y_sim - np.dot(X_sim, beta_pilot1)))/np.sqrt(n)
            asym_var1 = np.sqrt(np.sum(prop_score1 * w_obs1**2)/n)
            sigma_hat1 = sigma_pilot1
            
            m_deb2 = np.dot(x, beta_pilot2) + np.sum(w_obs2 * R2 * (Y_sim - np.dot(X_sim, beta_pilot2)))/np.sqrt(n)
            asym_var2 = np.sqrt(np.sum(prop_score2 * w_obs2**2)/n)
            sigma_hat2 = sigma_pilot2
                
                
            with open('./debias_res/DebiasProg_Equi_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR'\
                      +str(job_id)+'_x'+str(i)+'_beta'+str(k)+'_'+str(rule)+'.dat', "wb") as file:
                pickle.dump([m_deb1, asym_var1, sigma_hat1], file)
                
            with open('./debias_res/DebiasProg_Equi_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR'\
                      +str(job_id)+'_x'+str(i)+'_beta'+str(k)+'_'+str(rule)+'.dat', "wb") as file:
                pickle.dump([m_deb2, asym_var2, sigma_hat2], file)
            
