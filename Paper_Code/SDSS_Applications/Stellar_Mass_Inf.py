#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Aug 23, 2023

Description: Applications of our debiasing program to the stellar mass inference
problem with SDSS galaxies.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
import sys

from debias_prog import ScaledLasso, DebiasProg, DualObj, DualCD

job_id = int(sys.argv[1])
print(job_id)


## Read the data
gal = pd.read_csv('./SDSS_Data/SDSS_gal_redshift04.csv')
# Subset a tiny redshift range:
gal_sel = gal.loc[(gal.redshift >= 0.4) & (gal.redshift <= 0.4005)].reset_index()
gal_sel = gal_sel.drop(['index'], axis=1)

np.random.seed(job_id)
gal_ind = np.random.choice(gal.loc[(gal.redshift > 0.4005)].shape[0], size=1)[0]
gal_new = gal.loc[gal.redshift > 0.4005]
x_gal = gal_new.iloc[gal_ind]

Y = gal_sel['Chabrier_MILES_total_mass'].values
R = np.ones((len(Y),))
R[Y <= 0] = 0
log_Y = np.log(Y)
log_Y[Y <= 0] = -9999

# Subset all the covariates
dat = gal_sel[['RA', 'DEC', 'dist_DirFila_angdiam', 'dist_DirKnots_angdiam', 'dist_DirModes_angdiam', 
               'err_u','err_g','err_r','err_i','err_z', 
               'u','g','r','i','z',
               'modelMag_u','modelMag_g','modelMag_r','modelMag_i','modelMag_z',
               'cModelMag_u','cModelMag_g','cModelMag_r','cModelMag_i','cModelMag_z',
               'extinction_u', 'extinction_g','extinction_r','extinction_i','extinction_z', 
               'spectroFlux_u', 'spectroFlux_g', 'spectroFlux_r', 'spectroFlux_i', 'spectroFlux_z', 
               'spectroFluxIvar_u', 'spectroFluxIvar_g', 'spectroFluxIvar_r', 'spectroFluxIvar_i', 'spectroFluxIvar_z', 
               'spectroSynFlux_u', 'spectroSynFlux_g', 'spectroSynFlux_r', 'spectroSynFlux_i', 'spectroSynFlux_z', 
               'spectroSynFluxIvar_u', 'spectroSynFluxIvar_g', 'spectroSynFluxIvar_r', 'spectroSynFluxIvar_i', 
               'spectroSynFluxIvar_z', 
               'spectroSkyFlux_u', 'spectroSkyFlux_g', 'spectroSkyFlux_r', 'spectroSkyFlux_i', 'spectroSkyFlux_z', 
               'sn1_g', 'sn1_r', 'sn1_i', 'sn2_g', 'sn2_r', 'sn2_i', 
               'snMedian_u', 'snMedian_g', 'snMedian_r', 'snMedian_i', 'snMedian_z', 'snMedian']]


def signlog(x):
    return np.sign(x) * np.log(1 + np.abs(x))

dat['u_log'] = signlog(dat['u'])
dat['g_log'] = signlog(dat['g'])
dat['r_log'] = signlog(dat['r'])
dat['i_log'] = signlog(dat['i'])
dat['z_log'] = signlog(dat['z'])

dat['extinction_u_log'] = signlog(dat['extinction_u'])
dat['extinction_g_log'] = signlog(dat['extinction_g'])
dat['extinction_r_log'] = signlog(dat['extinction_r'])
dat['extinction_i_log'] = signlog(dat['extinction_i'])
dat['extinction_z_log'] = signlog(dat['extinction_z'])

dat['spectroFlux_u_log'] = signlog(dat['spectroFlux_u'])
dat['spectroFlux_g_log'] = signlog(dat['spectroFlux_g'])
dat['spectroFlux_r_log'] = signlog(dat['spectroFlux_r'])
dat['spectroFlux_i_log'] = signlog(dat['spectroFlux_i'])
dat['spectroFlux_z_log'] = signlog(dat['spectroFlux_z'])

dat['spectroSynFlux_u_log'] = signlog(dat['spectroSynFlux_u'])
dat['spectroSynFlux_g_log'] = signlog(dat['spectroSynFlux_g'])
dat['spectroSynFlux_r_log'] = signlog(dat['spectroSynFlux_r'])
dat['spectroSynFlux_i_log'] = signlog(dat['spectroSynFlux_i'])
dat['spectroSynFlux_z_log'] = signlog(dat['spectroSynFlux_z'])

dat['spectroSkyFlux_u_log'] = signlog(dat['spectroSkyFlux_u'])
dat['spectroSkyFlux_g_log'] = signlog(dat['spectroSkyFlux_g'])
dat['spectroSkyFlux_r_log'] = signlog(dat['spectroSkyFlux_r'])
dat['spectroSkyFlux_i_log'] = signlog(dat['spectroSkyFlux_i'])
dat['spectroSkyFlux_z_log'] = signlog(dat['spectroSkyFlux_z'])


# Create correlation matrix
corr_matrix = dat.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
dat.drop(to_drop, axis=1, inplace=True)


X_inter = dat.iloc[:,10:(dat.shape[1]+1)].values
inter_mat = SplineTransformer(degree=3, n_knots=40, extrapolation="periodic", 
                              include_bias=True).fit_transform(X_inter)


X = np.concatenate([dat[['RA', 'DEC', 'dist_DirFila_angdiam', 'dist_DirKnots_angdiam', 'dist_DirModes_angdiam']].values, 
                    inter_mat], axis=1)
x_gal = X[X.shape[0]-1,:]
y_gal = Y[Y.shape[0]-1]
X = X[:(X.shape[0]-1),:]
Y = Y[:(Y.shape[0]-1)]
log_Y = log_Y[:(log_Y.shape[0]-1)]
R = R[:(R.shape[0]-1)]


## Lasso pilot estimator
beta_pilot, sigma_pilot = ScaledLasso(X=X[R == 1,:], Y=log_Y[R == 1], lam0='univ') 


d = X.shape[1]
n = X.shape[0]

for prop_met in ['LR', 'NN', 'NNcal']:
    if prop_met == 'LR':
        ## Propensity score estimation (logistic regression CV)
        zeta = np.logspace(-1, np.log10(300), 20)*np.sqrt(np.log(d)/n)
        lr = LogisticRegressionCV(Cs=1/zeta, cv=5, penalty='l1', scoring='neg_log_loss', 
                                   solver='liblinear', tol=1e-6, max_iter=10000).fit(X, R)
        prop_score = lr.predict_proba(X)[:,1]
    if prop_met == 'NN':
        ## Propensity score estimation (NN)
        lr_NN = MLPClassifier(hidden_layer_sizes=(80,50,), activation='relu', 
                               random_state=None, learning_rate='adaptive', 
                               learning_rate_init=0.001, max_iter=1000).fit(X, R)
        prop_score = lr_NN.predict_proba(X)[:,1]
    if prop_met == 'NNcal':
        ## Propensity score estimation (NNcal)
        NN_base = MLPClassifier(hidden_layer_sizes=(80,50,), activation='relu', 
                               random_state=None, learning_rate='adaptive', 
                               learning_rate_init=0.001, max_iter=1000)
        lr2_NN = CalibratedClassifierCV(NN_base, method='sigmoid', cv=5).fit(X, R)
        prop_score = lr2_NN.predict_proba(X)[:,1]
        
    # ## x1 (color err v.s. stellar mass)
    # x = np.zeros((d,))
    # x[5:10] = 1
    # gamma_n_lst = np.linspace(1e-4, np.max(abs(x)), 41)
        
    # ## x2 (dist to filaments v.s. stellar mass)
    # x = np.zeros((d,))
    # x[2] = 1
    # gamma_n_lst = np.linspace(1e-4, np.max(abs(x)), 41)
        
    ## x3 (new queried galaxy)
    x = x_gal
    gamma_n_lst = np.logspace(-4, np.log10(np.max(abs(x))), 41)

    cv_fold = 5
    kf = KFold(n_splits=cv_fold, shuffle=True, random_state=0)
    f_ind = 0
    dual_loss = np.zeros((cv_fold, len(gamma_n_lst)))
    for train_ind, test_ind in kf.split(X):
        X_train = X[train_ind,:]
        X_test = X[test_ind,:]
        prop_score_train = prop_score[train_ind]
        prop_score_test = prop_score[test_ind]
        
        for j in range(len(gamma_n_lst)):
            try:
                w_train = DebiasProg(X=X_train.copy(), x=x, Pi=np.diag(prop_score_train), gamma_n=gamma_n_lst[j])
            except:
                w_train = np.nan*np.ones((n,))
            if any(np.isnan(w_train)):
                print(r'The primal debiasing program for this fold of the data is '\
                      'not feasible when \gamma/n='+str(round(gamma_n_lst[j], 4))+'!')
                dual_loss[f_ind, j] = np.nan
            else:
                ll_train = DualCD(X=X_train, x=x, Pi=np.diag(prop_score_train), gamma_n=gamma_n_lst[j], 
                                   ll_init=None, eps=1e-5, max_iter=5000)
                dual_gap = abs(w_train + np.dot(X_train, ll_train)/(2*np.sqrt(X_train.shape[0])))
                if sum(dual_gap > 1e-3) > 0:
                    print(r'The strong duality between primal and dual programs does not satisfy '\
                          'when \gamma/n='+str(round(gamma_n_lst[j], 4))+'!')
                    dual_loss[f_ind, j] = np.nan
                else:
                    dual_loss[f_ind, j] = DualObj(X_test, x=x, Pi=np.diag(prop_score_test), 
                                                   ll_cur=ll_train, gamma_n=gamma_n_lst[j])
        f_ind = f_ind + 1
    mean_dual_loss = np.mean(dual_loss, axis=0)
    std_dual_loss = np.std(dual_loss, axis=0, ddof=1)/np.sqrt(cv_fold)
    
    # Different rules for choosing the tuning parameter
    para_rule = ['1se', 'mincv', 'minfeas']
    for rule in para_rule:
        if rule == 'mincv':
            gamma_n_opt = gamma_n_lst[np.nanargmin(mean_dual_loss)]
        if rule == '1se':
            One_SE = (mean_dual_loss > np.nanmin(mean_dual_loss) + std_dual_loss[np.nanargmin(mean_dual_loss)]) & \
            (gamma_n_lst < gamma_n_lst[np.nanargmin(mean_dual_loss)])
            if sum(One_SE) == 0:
                One_SE = np.full((len(gamma_n_lst),), True)
            gamma_n_lst1 = gamma_n_lst[One_SE]
            gamma_n_opt = gamma_n_lst1[np.nanargmin(mean_dual_loss[One_SE])]
        if rule == 'minfeas':
            gamma_n_opt = np.min(gamma_n_lst[~np.isnan(mean_dual_loss)])
            
        # Solve the primal and dual on the original dataset
        w_obs = DebiasProg(X=X.copy(), x=x, Pi=np.diag(prop_score), gamma_n=gamma_n_opt)
        ll_obs = DualCD(X=X, x=x, Pi=np.diag(prop_score), gamma_n=gamma_n_opt, ll_init=None, eps=1e-5, max_iter=5000)
        
        # Store the results
        m_deb = np.dot(x, beta_pilot) + np.sum(w_obs * R * (log_Y - np.dot(X, beta_pilot)))/np.sqrt(n)
        asym_var = np.sqrt(np.sum(prop_score * w_obs**2)/n)
        sigma_hat = sigma_pilot
        
        debias_res = pd.DataFrame({'m_deb': [m_deb], 'asym_sd': [asym_var], 
                                   'sigma_hat': [sigma_hat], 'gamma_n':[gamma_n_opt]})
        debias_res.to_csv('./sdss_res/SDSS_stellar_mass_inf_x3_'+str(rule)+'_'+prop_met+'_'+str(job_id)+'.csv', index=False)
        