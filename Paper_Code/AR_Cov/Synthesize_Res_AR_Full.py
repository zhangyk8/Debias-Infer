#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Apr 6, 2026

Description: Synthesize the slurm array results (AR covariance).
"""

import pandas as pd
import pickle


## Homoscedastic case
d = 1000
n = 900


def load_debias_rows(mcar_template, mar_template, B, label):
    rows = []
    for b in range(1, B + 1):
        try:
            with open(mcar_template.format(b=b), "rb") as file:
                m_deb1, asym_se1, sigma_hat1 = pickle.load(file)
            with open(mar_template.format(b=b), "rb") as file:
                m_deb2, asym_se2, sigma_hat2 = pickle.load(file)
        except FileNotFoundError:
            print(f'{label}: missing result for b={b}')
            continue
        except EOFError:
            print(f'{label}: incomplete pickle for b={b}')
            continue
        except pickle.UnpicklingError as err:
            print(f'{label}: invalid pickle for b={b}: {err}')
            continue

        rows.append({
            'm_deb1': m_deb1,
            'asym_se1': asym_se1,
            'sigma_hat1': sigma_hat1,
            'm_deb2': m_deb2,
            'asym_se2': asym_se2,
            'sigma_hat2': sigma_hat2,
        })

    return pd.DataFrame(rows)


def load_csv_results(path_template, B, label, skip_missing=False):
    frames = []
    for b in range(1, B + 1):
        try:
            frames.append(pd.read_csv(path_template.format(b=b)))
        except FileNotFoundError:
            if skip_missing:
                print(f'{label}: missing result for b={b}')
                continue
            raise

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_triplet_rows(mcar_template, mar_template, B, label,
                      point_col='m_hat', spread_col='asym_var', scale_col='sigma_hat'):
    rows = []
    for b in range(1, B + 1):
        try:
            with open(mcar_template.format(b=b), "rb") as file:
                point1, spread1, scale1 = pickle.load(file)
            with open(mar_template.format(b=b), "rb") as file:
                point2, spread2, scale2 = pickle.load(file)
        except FileNotFoundError:
            print(f'{label}: missing result for b={b}')
            continue
        except EOFError:
            print(f'{label}: incomplete pickle for b={b}')
            continue
        except pickle.UnpicklingError as err:
            print(f'{label}: invalid pickle for b={b}: {err}')
            continue

        rows.append({
            f'{point_col}1': point1,
            f'{spread_col}1': spread1,
            f'{scale_col}1': scale1,
            f'{point_col}2': point2,
            f'{spread_col}2': spread2,
            f'{scale_col}2': scale2,
        })

    return pd.DataFrame(rows)


def load_paired_csv_rows(mcar_template, mar_template, B, label):
    rows = []
    for b in range(1, B + 1):
        try:
            mcar_df = pd.read_csv(mcar_template.format(b=b))
            mar_df = pd.read_csv(mar_template.format(b=b))
        except FileNotFoundError:
            print(f'{label}: missing result for b={b}')
            continue

        if mcar_df.empty or mar_df.empty:
            print(f'{label}: empty result for b={b}')
            continue

        mcar_row = mcar_df.iloc[0].to_dict()
        mar_row = mar_df.iloc[0].to_dict()

        row = {}
        for key, value in mcar_row.items():
            row[f'{key}1'] = value
        for key, value in mar_row.items():
            row[f'{key}2'] = value
        rows.append(row)

    return pd.DataFrame(rows)

## Proposed debiased program (gaussian noise)
for i in range(6):
    for k in range(3):
        # Different rules for choosing the tuning parameter
        para_rule = ['1se', 'mincv', 'minfeas']
        for rule in para_rule:
            B = 1000
            deb_prog_1se = load_debias_rows(
                './debias_res/DebiasProg_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR{b}_x'+str(i)+'_beta'+str(k)+'_'+str(rule)+'.dat',
                './debias_res/DebiasProg_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'_'+str(rule)+'.dat',
                B,
                'Gaussian'
            )
            deb_prog_1se.to_csv('./Results/DebiasProg_AR_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                                '_x'+str(i)+'_beta'+str(k)+'_'+str(rule)+'.csv', 
                                index=False)
                

# ## Proposed debiased program (other types of noises)
for i in range(6):
    for k in range(3):
        # Different rules for choosing the tuning parameter
        para_rule = ['1se', 'mincv', 'minfeas']
        for rule in para_rule:
            # Different types of noises
            # noises_type = ['laperr', 'uniferr', 'terr']
            noises_type = ['laperr', 'terr']
            for noise in noises_type:
                B = 1000
                deb_prog_1se = load_debias_rows(
                    './debias_res/DebiasProg_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR{b}_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'_'+str(rule)+'.dat',
                    './debias_res/DebiasProg_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'_'+str(rule)+'.dat',
                    B,
                    noise
                )
                deb_prog_1se.to_csv('./Results/DebiasProg_AR_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                                    '_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'_'+str(rule)+'.csv', 
                                    index=False)
        


# ## Debiased Lasso (Javanmard and Montarani, 2014)
for i in range(6):
    for k in range(3):
        B = 1000
        debl_res1 = load_csv_results(
            './debl_res/debl_AR_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'.csv',
            B,
            'debl'
        )
        debl_res1.to_csv('./Results/debl_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
                          +str(k)+'.csv', index=False)
        
        ## t-dist noise
        debl_res1 = load_csv_results(
            './debl_res/debl_AR_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_terr.csv',
            B,
            'debl-terr'
        )
        debl_res1.to_csv('./Results/debl_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
                          +str(k)+'_terr.csv', index=False)
        
        # ## uniform-dist noise
        # debl_res1 = pd.DataFrame()
        # B = 1000
        # for b in range(1, B+1):
        #     debl = pd.read_csv('./debl_res/debl_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'\
        #                         +str(i)+'_beta'+str(k)+'_uniferr.csv')
        #     debl_res1 = pd.concat([debl_res1, debl])
        # debl_res1.to_csv('./Results/debl_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
        #                   +str(k)+'_uniferr.csv', index=False)
            
        ## laplace-dist noise
        debl_res1 = load_csv_results(
            './debl_res/debl_AR_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_laperr.csv',
            B,
            'debl-laperr'
        )
        debl_res1.to_csv('./Results/debl_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
                          +str(k)+'_laperr.csv', index=False)


## Debiased Lasso (van de geer et al., 2014)
for i in range(6):
    for k in range(3):
        B = 1000
        lproj_res1 = load_csv_results(
            './lproj_res/lproj_AR_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'.csv',
            B,
            'DL-Van',
            skip_missing=True
        )
        lproj_res1.to_csv('./Results/lproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
                          +str(k)+'.csv', index=False)
            
        ## t-dist noise
        lproj_res1 = load_csv_results(
            './lproj_res/lproj_AR_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_terr.csv',
            B,
            'DL-Van-terr'
        )
        lproj_res1.to_csv('./Results/lproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
                          +str(k)+'_terr.csv', index=False)
            
        ## laplace noise
        lproj_res1 = load_csv_results(
            './lproj_res/lproj_AR_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_laperr.csv',
            B,
            'DL-Van-laperr'
        )
        lproj_res1.to_csv('./Results/lproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'\
                          +str(k)+'_laperr.csv', index=False)



## Ridge projection (Buhlmann, 2013)
# for i in range(5):
#     for k in range(3):
#         if (i == 0) or (i == 2):
#             B = 1000
#             rproj_res1 = load_csv_results(
#                 './rproj_res/rproj_AR_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'.csv',
#                 B,
#                 'rproj'
#             )
#             rproj_res1.to_csv('./Results/rproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'.csv', 
#                               index=False)
            
#             ## t-dist noise
#             rproj_res1 = load_csv_results(
#                 './rproj_res/rproj_AR_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_terr.csv',
#                 B,
#                 'rproj-terr'
#             )
#             rproj_res1.to_csv('./Results/rproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_terr.csv', 
#                               index=False)
            
#             # ## uniform-dist noise
#             # rproj_res1 = pd.DataFrame()
#             # B = 1000
#             # for b in range(1, B+1):
#             #     rproj = pd.read_csv('./rproj_res/rproj_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'+str(i)+'_beta'+str(k)+'_uniferr.csv')
#             #     rproj_res1 = pd.concat([rproj_res1, rproj])
#             # rproj_res1.to_csv('./Results/rproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_uniferr.csv', 
#             #                   index=False)
            
#             ## laplace-dist noise
#             rproj_res1 = load_csv_results(
#                 './rproj_res/rproj_AR_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_laperr.csv',
#                 B,
#                 'rproj-laperr'
#             )
#             rproj_res1.to_csv('./Results/rproj_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_laperr.csv', 
#                               index=False)



## Lasso refit
for i in range(6):
    for k in range(3):
        B = 1000
        refit_res0 = load_csv_results(
            './refit_res/refit_AR_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'.csv',
            B,
            'refit'
        )
        refit_res0.to_csv('./Results/refit_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'.csv', 
                          index=False)
        
        ## t-dist noise
        refit_res0 = load_csv_results(
            './refit_res/refit_AR_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_terr.csv',
            B,
            'refit-terr'
        )
        refit_res0.to_csv('./Results/refit_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_terr.csv', 
                          index=False)
        
        # ## uniform-dist noise
        # refit_res0 = pd.DataFrame()
        # B = 1000
        # for b in range(1, B+1):
        #     refit0 = pd.read_csv('./refit_res/refit_AR_d'+str(d)+'_n'+str(n)+'_'+str(b)+'_x'+str(i)+'_beta'+str(k)+'_uniferr.csv')
        #     refit_res0 = pd.concat([refit_res0, refit0])
        # refit_res0.to_csv('./Results/refit_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_uniferr.csv', 
        #                   index=False)
        
        ## laplace-dist noise
        refit_res0 = load_csv_results(
            './refit_res/refit_AR_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_laperr.csv',
            B,
            'refit-laperr'
        )
        refit_res0.to_csv('./Results/refit_AR_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_laperr.csv', 
                          index=False)


## Lasso Pilot Estimates
for i in range(6):
    for k in range(3):
        for error in ['gauss', 'laperr', 'terr']:
            B = 1000
            lasso_pilot_res = load_csv_results(
                './pilot_res/lasso_pilot_AR_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_'+str(error)+'.csv',
                B,
                'lasso-pilot-'+str(error)
            )
            lasso_pilot_res.to_csv('./Results/lasso_pilot_AR_d'+str(d)+'_n'+str(n)+\
                                   '_x'+str(i)+'_beta'+str(k)+'_'+str(error)+'.csv', index=False)


## Tian et al. (2024) AIPW
for i in range(6):
    for k in range(3):
        B = 1000
        tian_res = load_triplet_rows(
            './Tian2024_AIPW_res/Tian2024_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR{b}_x'+str(i)+'_beta'+str(k)+'.dat',
            './Tian2024_AIPW_res/Tian2024_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'.dat',
            B,
            'Tian2024'
        )
        tian_res.to_csv('./Results/Tian2024_AR_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                         '_x'+str(i)+'_beta'+str(k)+'.csv', index=False)

        for noise in ['terr', 'laperr']:
            tian_res = load_triplet_rows(
                './Tian2024_AIPW_res/Tian2024_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR_'+str(noise)+'{b}_x'+str(i)+'_beta'+str(k)+'.dat',
                './Tian2024_AIPW_res/Tian2024_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR_'+str(noise)+'{b}_x'+str(i)+'_beta'+str(k)+'.dat',
                B,
                'Tian2024-'+str(noise)
            )
            tian_res.to_csv('./Results/Tian2024_AR_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                             '_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv', index=False)


## Chakrabortty et al. (2019) DDR
for i in range(6):
    for k in range(3):
        B = 1000
        ddr_res = load_paired_csv_rows(
            './HDM_DDR_res/DDR_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR{b}_x'+str(i)+'_beta'+str(k)+'.csv',
            './HDM_DDR_res/DDR_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'.csv',
            B,
            'DDR'
        )
        ddr_res.to_csv('./Results/DDR_AR_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                        '_x'+str(i)+'_beta'+str(k)+'.csv', index=False)

        for noise in ['terr', 'laperr']:
            ddr_res = load_paired_csv_rows(
                './HDM_DDR_res/DDR_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR{b}_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv',
                './HDM_DDR_res/DDR_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv',
                B,
                'DDR-'+str(noise)
            )
            ddr_res.to_csv('./Results/DDR_AR_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                            '_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv', index=False)


## Hou et al. (2023) debiased SAS
for i in range(6):
    for k in range(3):
        B = 1000
        hou_sas_res = load_triplet_rows(
            './Hou2023_SAS_res/Hou2023_SAS_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR{b}_x'+str(i)+'_beta'+str(k)+'.dat',
            './Hou2023_SAS_res/Hou2023_SAS_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'.dat',
            B,
            'Hou2023-SAS'
        )
        hou_sas_res.to_csv('./Results/Hou2023_SAS_AR_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                            '_x'+str(i)+'_beta'+str(k)+'.csv', index=False)

        for noise in ['terr', 'laperr']:
            hou_sas_res = load_triplet_rows(
                './Hou2023_SAS_res/Hou2023_SAS_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR_'+str(noise)+'{b}_x'+str(i)+'_beta'+str(k)+'.dat',
                './Hou2023_SAS_res/Hou2023_SAS_AR_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR_'+str(noise)+'{b}_x'+str(i)+'_beta'+str(k)+'.dat',
                B,
                'Hou2023-SAS-'+str(noise)
            )
            hou_sas_res.to_csv('./Results/Hou2023_SAS_AR_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                                '_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv', index=False)
