#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: May 11, 2023

Description: Synthesize the slurm array results (AR covariance).
"""

import numpy as np
import pandas as pd
import pickle


## Homoscedastic case
d = 1000
n = 900

## Proposed debiased program (gaussian noise)
for i in range(5):
    for k in range(3):
        # Different rules for choosing the tuning parameter
        para_rule = ['1se', 'mincv', 'minfeas']
        for rule in para_rule:
            m_deb1_1se = []
            asym_se1_1se = []  ## Already scaled by sqrt(n)
            sigma_hat1_1se = []
            m_deb2_1se = []
            asym_se2_1se = []  ## Already scaled by sqrt(n)
            sigma_hat2_1se = []
            B = 1000
            for b in range(1, B+1):
                try:
                    with open('./debias_res/DebiasProg_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR'\
                      +str(b)+'_x'+str(i)+'_beta'+str(k)+'_'+str(rule)+'.dat', "rb") as file:
                        m_deb, asym_se, sigma_hat = pickle.load(file)
                    m_deb1_1se.append(m_deb)
                    asym_se1_1se.append(asym_se)
                    sigma_hat1_1se.append(sigma_hat)
                except:
                    print('Gaussian')
                    print(b)
                    continue
            
            for b in range(1, B+1):
                try:
                    with open('./debias_res/DebiasProg_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR'\
                      +str(b)+'_x'+str(i)+'_beta'+str(k)+'_'+str(rule)+'.dat', "rb") as file:
                        m_deb, asym_se, sigma_hat = pickle.load(file)
                    m_deb2_1se.append(m_deb)
                    asym_se2_1se.append(asym_se)
                    sigma_hat2_1se.append(sigma_hat)
                except:
                    print('Gaussian')
                    print(b)
                    continue
                    
            deb_prog_1se = pd.DataFrame({'m_deb1': m_deb1_1se, 'asym_se1': asym_se1_1se, 'sigma_hat1': sigma_hat1_1se, 
                                      'm_deb2': m_deb2_1se, 'asym_se2': asym_se2_1se, 'sigma_hat2': sigma_hat2_1se})
            deb_prog_1se.to_csv('./Results/DebiasProg_AR_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                                '_x'+str(i)+'_beta'+str(k)+'_'+str(rule)+'.csv', 
                                index=False)
                

# ## Proposed debiased program (other types of noises)
for i in range(5):
    for k in range(3):
        # Different rules for choosing the tuning parameter
        para_rule = ['1se', 'mincv', 'minfeas']
        for rule in para_rule:
            # Different types of noises
            noises_type = ['laperr', 'uniferr', 'terr']
            for noise in noises_type:
                m_deb1_1se = []
                asym_se1_1se = []  ## Already scaled by sqrt(n)
                sigma_hat1_1se = []
                m_deb2_1se = []
                asym_se2_1se = []  ## Already scaled by sqrt(n)
                sigma_hat2_1se = []
                B = 1000
                for b in range(1, B+1):
                    try:
                        with open('./debias_res/DebiasProg_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR'\
                          +str(b)+'_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'_'+str(rule)+'.dat', "rb") as file:
                            m_deb, asym_se, sigma_hat = pickle.load(file)
                        m_deb1_1se.append(m_deb)
                        asym_se1_1se.append(asym_se)
                        sigma_hat1_1se.append(sigma_hat)
                    except:
                        print(noise)
                        print(b)
                        continue
                
                for b in range(1, B+1):
                    try:
                        with open('./debias_res/DebiasProg_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR'\
                          +str(b)+'_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'_'+str(rule)+'.dat', "rb") as file:
                            m_deb, asym_se, sigma_hat = pickle.load(file)
                        m_deb2_1se.append(m_deb)
                        asym_se2_1se.append(asym_se)
                        sigma_hat2_1se.append(sigma_hat)
                    except:
                        print(noise)
                        print(b)
                        continue
                        
                deb_prog_1se = pd.DataFrame({'m_deb1': m_deb1_1se, 'asym_se1': asym_se1_1se, 'sigma_hat1': sigma_hat1_1se, 
                                          'm_deb2': m_deb2_1se, 'asym_se2': asym_se2_1se, 'sigma_hat2': sigma_hat2_1se})
                deb_prog_1se.to_csv('./Results/DebiasProg_AR_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                                    '_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'_'+str(rule)+'.csv', 
                                    index=False)
        


# ## Debiased Lasso (Javanmard and Montarani, 2014)
for i in range(5):
    for k in range(3):
        debl_res1 = pd.DataFrame()
        B = 1000
        for b in range(1, B+1):
            debl = pd.read_csv('./debl_res/debl_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'\
                                +str(i)+'_beta'+str(k)+'.csv')
            debl_res1 = pd.concat([debl_res1, debl])
        debl_res1.to_csv('./Results/debl_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
                          +str(k)+'.csv', index=False)
        
        ## t-dist noise
        debl_res1 = pd.DataFrame()
        B = 1000
        for b in range(1, B+1):
            debl = pd.read_csv('./debl_res/debl_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'
                                +str(i)+'_beta'+str(k)+'_terr.csv')
            debl_res1 = pd.concat([debl_res1, debl])
        debl_res1.to_csv('./Results/debl_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
                          +str(k)+'_terr.csv', index=False)
        
        # ## uniform-dist noise
        debl_res1 = pd.DataFrame()
        B = 1000
        for b in range(1, B+1):
            debl = pd.read_csv('./debl_res/debl_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'\
                                +str(i)+'_beta'+str(k)+'_uniferr.csv')
            debl_res1 = pd.concat([debl_res1, debl])
        debl_res1.to_csv('./Results/debl_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
                          +str(k)+'_uniferr.csv', index=False)
            
        ## laplace-dist noise
        debl_res1 = pd.DataFrame()
        B = 1000
        for b in range(1, B+1):
            debl = pd.read_csv('./debl_res/debl_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'\
                                +str(i)+'_beta'+str(k)+'_laperr.csv')
            debl_res1 = pd.concat([debl_res1, debl])
        debl_res1.to_csv('./Results/debl_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
                          +str(k)+'_laperr.csv', index=False)


## Debiased Lasso (van de geer et al., 2014)
for i in range(5):
    for k in range(3):
        lproj_res1 = pd.DataFrame()
        B = 1000
        cnt = 0
        for b in range(1, B+1):
            try:
                lproj = pd.read_csv('./lproj_res/lproj_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'\
                                +str(i)+'_beta'+str(k)+'.csv')
                lproj_res1 = pd.concat([lproj_res1, lproj])
            except:
                print('DL-Van')
                print(b)
                continue
            # cnt += 1
            # if cnt >= 500:
            #     break
        lproj_res1.to_csv('./Results/lproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
                          +str(k)+'.csv', index=False)
            
        ## t-dist noise
        lproj_res1 = pd.DataFrame()
        B = 1000
        for b in range(1, B+1):
            lproj = pd.read_csv('./lproj_res/lproj_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'
                                +str(i)+'_beta'+str(k)+'_terr.csv')
            lproj_res1 = pd.concat([lproj_res1, lproj])
        lproj_res1.to_csv('./Results/lproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
                          +str(k)+'_terr.csv', index=False)
            
        ## laplace noise
        lproj_res1 = pd.DataFrame()
        B = 1000
        for b in range(1, B+1):
            lproj = pd.read_csv('./lproj_res/lproj_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'
                                +str(i)+'_beta'+str(k)+'_laperr.csv')
            lproj_res1 = pd.concat([lproj_res1, lproj])
        lproj_res1.to_csv('./Results/lproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
                          +str(k)+'_laperr.csv', index=False)



## Ridge projection (Buhlmann, 2013)
for i in range(5):
    for k in range(3):
        if (i == 0) or (i == 2):
            rproj_res1 = pd.DataFrame()
            B = 1000
            for b in range(1, B+1):
                rproj = pd.read_csv('./rproj_res/rproj_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'+str(i)+'_beta'+str(k)+'.csv')
                rproj_res1 = pd.concat([rproj_res1, rproj])
            rproj_res1.to_csv('./Results/rproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'.csv', 
                              index=False)
            
            ## t-dist noise
            rproj_res1 = pd.DataFrame()
            B = 1000
            for b in range(1, B+1):
                rproj = pd.read_csv('./rproj_res/rproj_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'+str(i)+'_beta'+str(k)+'_terr.csv')
                rproj_res1 = pd.concat([rproj_res1, rproj])
            rproj_res1.to_csv('./Results/rproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_terr.csv', 
                              index=False)
            
            ## uniform-dist noise
            rproj_res1 = pd.DataFrame()
            B = 1000
            for b in range(1, B+1):
                rproj = pd.read_csv('./rproj_res/rproj_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'+str(i)+'_beta'+str(k)+'_uniferr.csv')
                rproj_res1 = pd.concat([rproj_res1, rproj])
            rproj_res1.to_csv('./Results/rproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_uniferr.csv', 
                              index=False)
            
            ## laplace-dist noise
            rproj_res1 = pd.DataFrame()
            B = 1000
            for b in range(1, B+1):
                rproj = pd.read_csv('./rproj_res/rproj_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'+str(i)+'_beta'+str(k)+'_laperr.csv')
                rproj_res1 = pd.concat([rproj_res1, rproj])
            rproj_res1.to_csv('./Results/rproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_laperr.csv', 
                              index=False)



## Lasso refit
for i in range(5):
    for k in range(3):
        refit_res0 = pd.DataFrame()
        B = 1000
        for b in range(1, B+1):
            refit0 = pd.read_csv('./refit_res/refit_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'+str(i)+'_beta'+str(k)+'.csv')
            refit_res0 = pd.concat([refit_res0, refit0])
        refit_res0.to_csv('./Results/refit_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'.csv', 
                          index=False)
        
        ## t-dist noise
        refit_res0 = pd.DataFrame()
        B = 1000
        for b in range(1, B+1):
            refit0 = pd.read_csv('./refit_res/refit_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'+str(i)+'_beta'+str(k)+'_terr.csv')
            refit_res0 = pd.concat([refit_res0, refit0])
        refit_res0.to_csv('./Results/refit_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_terr.csv', 
                          index=False)
        
        ## uniform-dist noise
        refit_res0 = pd.DataFrame()
        B = 1000
        for b in range(1, B+1):
            refit0 = pd.read_csv('./refit_res/refit_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'+str(i)+'_beta'+str(k)+'_uniferr.csv')
            refit_res0 = pd.concat([refit_res0, refit0])
        refit_res0.to_csv('./Results/refit_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_uniferr.csv', 
                          index=False)
        
        ## laplace-dist noise
        refit_res0 = pd.DataFrame()
        B = 1000
        for b in range(1, B+1):
            refit0 = pd.read_csv('./refit_res/refit_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'+str(i)+'_beta'+str(k)+'_laperr.csv')
            refit_res0 = pd.concat([refit_res0, refit0])
        refit_res0.to_csv('./Results/refit_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_laperr.csv', 
                          index=False)


## Lasso Pilot Estimates
for i in range(5):
    for k in range(3):
        for error in ['gauss', 'laperr', 'terr', 'uniferr']:
            lasso_pilot_res = pd.DataFrame()
            B = 1000
            for b in range(1, B+1):
                lasso_pilot = pd.read_csv('./pilot_res/lasso_pilot_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+\
                                  '_x'+str(i)+'_beta'+str(k)+'_'+str(error)+'.csv')
                lasso_pilot_res = pd.concat([lasso_pilot_res, lasso_pilot])
            lasso_pilot_res.to_csv('./Results/lasso_pilot_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+\
                              '_x'+str(i)+'_beta'+str(k)+'_'+str(error)+'.csv', index=False)
