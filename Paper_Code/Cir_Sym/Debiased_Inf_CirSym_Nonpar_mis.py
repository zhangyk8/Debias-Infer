#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Jul 2, 2023

Description: Simulations on our debiasing program (circulant symmetric covariance 
with Gaussian noises). Here, we consider estimating the propensity scores using 
several nonparametric methods.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
import pickle
import sys

from debias_prog import ScaledLasso, DebiasProg, DualObj, DualCD

job_id = int(sys.argv[1])
print(job_id)

#==========================================================================================#

## Homoscedastic case (Circulant symmetric covariance)
d = 1000
n = 900

Sigma = np.zeros((d,d)) + np.eye(d)
rho = 0.1
for i in range(d):
    for j in range(i+1, d):
        if (j < i+6) or (j > i+d-6):
            Sigma[i,j] = rho
            Sigma[j,i] = rho
sig = 1


## Consider different simulation settings
for i in [0,1,2,4]:
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
    for k in [0,2]:
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
            
        ## MAR
        inter_mat = PolynomialFeatures(degree=2, interaction_only=False, 
                                       include_bias=False).fit_transform(X_sim[:,:8])
        obs_prob2 = scipy.stats.norm.cdf(-4 + np.sum(inter_mat, axis=1))
        R2 = np.ones((n,))
        R2[np.random.rand(n) >= obs_prob2] = 0
            
        ## Lasso pilot estimator
        beta_pilot2, sigma_pilot2 = ScaledLasso(X=X_sim[R2 == 1,:], Y=Y_sim[R2 == 1], lam0='univ') 
            
        ## Propensity score estimation (LR, Naive Bayes, Random Forests, SVM, MLP neural network)
        for non_met in ['Oracle', 'LR', 'NB', 'NBcal', 'RF', 'RFcal', 'SVM', 'SVMcal', 'NN', 'NNcal']:
            if non_met == 'Oracle':
                prop_score2 = obs_prob2.copy()
                MAE_prop = np.mean(prop_score2 - obs_prob2)
            if non_met == 'LR':
                zeta2 = np.logspace(-1, np.log10(300), 40)*np.sqrt(np.log(d)/n)
                lr2 = LogisticRegressionCV(Cs=1/zeta2, cv=5, penalty='l1', scoring='neg_log_loss', 
                                           solver='liblinear', tol=1e-6, max_iter=10000).fit(X_sim, R2)
                prop_score2 = lr2.predict_proba(X_sim)[:,1]
                MAE_prop = np.mean(prop_score2 - obs_prob2)
            if non_met == 'NB':
                lr2_NB = GaussianNB().fit(X_sim, R2)
                prop_score2 = lr2_NB.predict_proba(X_sim)[:,1]
                MAE_prop = np.mean(prop_score2 - obs_prob2)
                
            if non_met == 'NBcal':
                NB_base = GaussianNB()
                lr2_NB = CalibratedClassifierCV(NB_base, method='sigmoid', cv=5).fit(X_sim, R2)
                prop_score2 = lr2_NB.predict_proba(X_sim)[:,1]
                MAE_prop = np.mean(prop_score2 - obs_prob2)
                
            if non_met == 'RF':
                lr2_RF = RandomForestClassifier(n_estimators=100, max_depth=None, 
                                                random_state=None, n_jobs=-1).fit(X_sim, R2)
                prop_score2 = lr2_RF.predict_proba(X_sim)[:,1]
                MAE_prop = np.mean(prop_score2 - obs_prob2)
                
            if non_met == 'RFcal':
                RF_base = RandomForestClassifier(n_estimators=100, max_depth=None, 
                                                 random_state=None, n_jobs=-1)
                lr2_RF = CalibratedClassifierCV(RF_base, method='sigmoid', cv=5).fit(X_sim, R2)
                prop_score2 = lr2_RF.predict_proba(X_sim)[:,1]
                MAE_prop = np.mean(prop_score2 - obs_prob2)
            
            if non_met == 'SVM':
                lr2_SVM = SVC(kernel='rbf', gamma='scale', probability=True).fit(X_sim, R2)
                prop_score2 = lr2_SVM.predict_proba(X_sim)[:,1]
                MAE_prop = np.mean(prop_score2 - obs_prob2)
                
            if non_met == 'SVMcal':
                SVM_base = SVC(kernel='rbf', gamma='scale', probability=True)
                lr2_SVM = CalibratedClassifierCV(SVM_base, method='sigmoid', cv=5).fit(X_sim, R2)
                prop_score2 = lr2_SVM.predict_proba(X_sim)[:,1]
                MAE_prop = np.mean(prop_score2 - obs_prob2)
                
            if non_met == 'NN':
                lr2_NN = MLPClassifier(hidden_layer_sizes=(80,50,), activation='relu', 
                                       random_state=None, learning_rate='adaptive', 
                                       learning_rate_init=0.001, max_iter=1000).fit(X_sim, R2)
                prop_score2 = lr2_NN.predict_proba(X_sim)[:,1]
                MAE_prop = np.mean(prop_score2 - obs_prob2)
                
            if non_met == 'NNcal':
                NN_base = MLPClassifier(hidden_layer_sizes=(80,50,), activation='relu', random_state=None, 
                                        learning_rate='adaptive', learning_rate_init=0.001, max_iter=1000)
                lr2_NN = CalibratedClassifierCV(NN_base, method='sigmoid', cv=5).fit(X_sim, R2)
                prop_score2 = lr2_NN.predict_proba(X_sim)[:,1]
                MAE_prop = np.mean(prop_score2 - obs_prob2)
            
            gamma_n_lst = np.linspace(0.001, np.max(abs(x)), 41)
            cv_fold = 5
            
            kf = KFold(n_splits=cv_fold, shuffle=True, random_state=0)
            f_ind = 0
            dual_loss1 = np.zeros((cv_fold, len(gamma_n_lst)))
            dual_loss2 = np.zeros((cv_fold, len(gamma_n_lst)))
            for train_ind, test_ind in kf.split(X_sim):
                X_train = X_sim[train_ind,:]
                X_test = X_sim[test_ind,:]
                
                prop_score2_train = prop_score2[train_ind]
                prop_score2_test = prop_score2[test_ind]
                
                for j in range(len(gamma_n_lst)):
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
            mean_dual_loss2 = np.mean(dual_loss2, axis=0)
            std_dual_loss2 = np.std(dual_loss2, axis=0, ddof=1)/np.sqrt(cv_fold)
            
            # Different rules for choosing the tuning parameter
            para_rule = ['1se', 'mincv', 'minfeas']
            for rule in para_rule:
                if rule == 'mincv':
                    gamma_n2_opt = gamma_n_lst[np.nanargmin(mean_dual_loss2)]
                if rule == '1se':
                    One_SE2 = (mean_dual_loss2 > np.nanmin(mean_dual_loss2) + std_dual_loss2[np.nanargmin(mean_dual_loss2)]) & \
                    (gamma_n_lst < gamma_n_lst[np.nanargmin(mean_dual_loss2)])
                    if sum(One_SE2) == 0:
                        One_SE2 = np.full((len(gamma_n_lst),), True)
                    gamma_n_lst2 = gamma_n_lst[One_SE2]
                    gamma_n2_opt = gamma_n_lst2[np.nanargmin(mean_dual_loss2[One_SE2])]
                if rule == 'minfeas':
                    gamma_n2_opt = np.min(gamma_n_lst[~np.isnan(mean_dual_loss2)])
                    
                # Solve the primal and dual on the original dataset
                w_obs2 = DebiasProg(X=X_sim.copy(), x=x, Pi=np.diag(prop_score2), gamma_n=gamma_n2_opt)
                ll_obs2 = DualCD(X=X_sim, x=x, Pi=np.diag(prop_score2), gamma_n=gamma_n2_opt, ll_init=None, eps=1e-9)
                
                # Store the results
                m_deb2 = np.dot(x, beta_pilot2) + np.sum(w_obs2 * R2 * (Y_sim - np.dot(X_sim, beta_pilot2)))/np.sqrt(n)
                asym_var2 = np.sqrt(np.sum(prop_score2 * w_obs2**2)/n)
                sigma_hat2 = sigma_pilot2
                
                    
                with open('./debias_res/DebiasProg_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR'\
                          +str(job_id)+'_x'+str(i)+'_beta'+str(k)+'_prop_'+str(non_met)+'_'+str(rule)+'_mis.dat', "wb") as file:
                    pickle.dump([m_deb2, asym_var2, sigma_hat2, MAE_prop], file)