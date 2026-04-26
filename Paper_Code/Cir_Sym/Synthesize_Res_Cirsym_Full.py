#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: April 17, 2026

Description: Synthesize the slurm array results (circulant symmetric covariance).
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


def load_mar_quadruplet_rows(path_template, B, label):
    rows = []
    for b in range(1, B + 1):
        try:
            with open(path_template.format(b=b), "rb") as file:
                m_deb2, asym_se2, sigma_hat2, mae_prop = pickle.load(file)
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
            'm_deb2': m_deb2,
            'asym_se2': asym_se2,
            'sigma_hat2': sigma_hat2,
            'mae_prop': mae_prop,
        })

    return pd.DataFrame(rows)

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
        for rule in ['1se', 'mincv', 'minfeas']:
            B = 1000
            deb_prog = load_debias_rows(
                './debias_res/DebiasProg_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR{b}_x'+str(i)+'_beta'+str(k)+'_'+str(rule)+'.dat',
                './debias_res/DebiasProg_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'_'+str(rule)+'.dat',
                B,
                'Gaussian'
            )
            deb_prog.to_csv(
                './Results/DebiasProg_Cirsym_cov_homoerr_d'+str(d)+'_n'+str(n)+
                '_x'+str(i)+'_beta'+str(k)+'_'+str(rule)+'.csv',
                index=False
            )


## Proposed debiased program (other types of noises)
for i in range(6):
    for k in range(3):
        for rule in ['1se', 'mincv', 'minfeas']:
            for noise in ['laperr', 'terr']:
                B = 1000
                deb_prog = load_debias_rows(
                    './debias_res/DebiasProg_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR{b}_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'_'+str(rule)+'.dat',
                    './debias_res/DebiasProg_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'_'+str(rule)+'.dat',
                    B,
                    noise
                )
                deb_prog.to_csv(
                    './Results/DebiasProg_Cirsym_cov_homoerr_d'+str(d)+'_n'+str(n)+
                    '_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'_'+str(rule)+'.csv',
                    index=False
                )


## Proposed debiased program (nonparametric propensity score estimation)
for i in [0, 1, 2, 4]:
    for k in [0, 2]:
        for non_met in ['NB', 'NBcal', 'RF', 'RFcal', 'SVM', 'SVMcal', 'NN', 'NNcal']:
            for rule in ['1se', 'mincv', 'minfeas']:
                B = 1000
                deb_prog = load_debias_rows(
                    './debias_res/DebiasProg_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR{b}_x'+str(i)+'_beta'+str(k)+'_prop_'+str(non_met)+'_'+str(rule)+'.dat',
                    './debias_res/DebiasProg_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'_prop_'+str(non_met)+'_'+str(rule)+'.dat',
                    B,
                    'Nonpar-'+str(non_met)
                )
                deb_prog.to_csv(
                    './Results/DebiasProg_Cirsym_cov_homoerr_d'+str(d)+'_n'+str(n)+
                    '_x'+str(i)+'_beta'+str(k)+'_prop_'+str(non_met)+'_'+str(rule)+'.csv',
                    index=False
                )


## Proposed debiased program (nonparametric propensity score estimation with misspecified propensity score)
for i in [0, 1, 2, 4]:
    for k in [0, 1, 2]:
        for non_met in ['Oracle', 'LR', 'NB', 'NBcal', 'RF', 'RFcal', 'SVM', 'SVMcal', 'NN', 'NNcal']:
            for rule in ['1se', 'mincv', 'minfeas']:
                B = 1000
                deb_prog = load_mar_quadruplet_rows(
                    './debias_res/DebiasProg_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'_prop_'+str(non_met)+'_'+str(rule)+'_mis.dat',
                    B,
                    'Nonpar-'+str(non_met)+'-mis'
                )
                deb_prog.to_csv(
                    './Results/DebiasProg_Cirsym_cov_homoerr_d'+str(d)+'_n'+str(n)+
                    '_x'+str(i)+'_beta'+str(k)+'_prop_'+str(non_met)+'_'+str(rule)+'_mis.csv',
                    index=False
                )


## Proposed debiasing program (R implementation)
for i in range(6):
    for k in range(3):
        for rule in ['1se', 'mincv', 'minfeas']:
            B = 1000
            debias_res = load_csv_results(
                './debias_res/DebiasProg_CirSym_cov_homoerr_d'+str(d)+'_n'+str(n)+
                '_{b}_x'+str(i)+'_beta'+str(k)+'_rule'+str(rule)+'_gauss_R.csv',
                B,
                'Gaussian-R',
                skip_missing=True
            )
            debias_res.to_csv(
                './Results/DebiasProg_CirSym_cov_homoerr_d'+str(d)+'_n'+str(n)+'_x'+str(i)+
                '_beta'+str(k)+'_'+str(rule)+'_gauss_R.csv',
                index=False
            )


## Debiased Lasso (Javanmard and Montarani, 2014)
for i in range(6):
    for k in range(3):
        B = 1000
        debl_res = load_csv_results(
            './debl_res/debl_cirsym_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'.csv',
            B,
            'debl'
        )
        debl_res.to_csv(
            './Results/debl_cirsym_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'.csv',
            index=False
        )

        for noise in ['terr', 'laperr']:
            debl_res = load_csv_results(
                './debl_res/debl_cirsym_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv',
                B,
                'debl-'+str(noise)
            )
            debl_res.to_csv(
                './Results/debl_cirsym_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv',
                index=False
            )


## Debiased Lasso (van de geer et al., 2014)
for i in range(6):
    for k in range(3):
        B = 1000
        lproj_res = load_csv_results(
            './lproj_res/lproj_cirsym_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'.csv',
            B,
            'DL-Van',
            skip_missing=True
        )
        lproj_res.to_csv(
            './Results/lproj_cirsym_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'.csv',
            index=False
        )

        for noise in ['terr', 'laperr']:
            lproj_res = load_csv_results(
                './lproj_res/lproj_cirsym_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv',
                B,
                'DL-Van-'+str(noise),
                skip_missing=True
            )
            lproj_res.to_csv(
                './Results/lproj_cirsym_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv',
                index=False
            )


## Ridge projection (Buhlmann, 2013)
# for i in range(6):
#     for k in range(3):
#         if (i == 0) or (i == 2):
#             B = 1000
#             rproj_res = load_csv_results(
#                 './rproj_res/rproj_cirsym_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'.csv',
#                 B,
#                 'rproj'
#             )
#             rproj_res.to_csv(
#                 './Results/rproj_cirsym_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'.csv',
#                 index=False
#             )

#             for noise in ['terr', 'laperr']:
#                 rproj_res = load_csv_results(
#                     './rproj_res/rproj_cirsym_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv',
#                     B,
#                     'rproj-'+str(noise)
#                 )
#                 rproj_res.to_csv(
#                     './Results/rproj_cirsym_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv',
#                     index=False
#                 )


## Lasso refit
for i in range(6):
    for k in range(3):
        B = 1000
        refit_res = load_csv_results(
            './refit_res/refit_Cirsym_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'.csv',
            B,
            'refit'
        )
        refit_res.to_csv(
            './Results/refit_Cirsym_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'.csv',
            index=False
        )

        for noise in ['terr', 'laperr']:
            refit_res = load_csv_results(
                './refit_res/refit_Cirsym_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv',
                B,
                'refit-'+str(noise)
            )
            refit_res.to_csv(
                './Results/refit_Cirsym_d'+str(d)+'_n'+str(n)+'_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv',
                index=False
            )


## Lasso Pilot Estimates
for i in range(6):
    for k in range(3):
        for error in ['gauss', 'laperr', 'terr']:
            B = 1000
            lasso_pilot_res = load_csv_results(
                './pilot_res/lasso_pilot_Cirsym_d'+str(d)+'_n'+str(n)+'_{b}_x'+str(i)+'_beta'+str(k)+'_'+str(error)+'.csv',
                B,
                'lasso-pilot-'+str(error)
            )
            lasso_pilot_res.to_csv(
                './Results/lasso_pilot_Cirsym_d'+str(d)+'_n'+str(n)+
                '_x'+str(i)+'_beta'+str(k)+'_'+str(error)+'.csv',
                index=False
            )


## Tian et al. (2024) AIPW
for i in range(6):
    for k in range(3):
        B = 1000
        tian_res = load_triplet_rows(
            './Tian2024_AIPW_res/Tian2024_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR{b}_x'+str(i)+'_beta'+str(k)+'.dat',
            './Tian2024_AIPW_res/Tian2024_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'.dat',
            B,
            'Tian2024'
        )
        tian_res.to_csv('./Results/Tian2024_CirSym_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                         '_x'+str(i)+'_beta'+str(k)+'.csv', index=False)

        for noise in ['terr', 'laperr']:
            tian_res = load_triplet_rows(
                './Tian2024_AIPW_res/Tian2024_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR_'+str(noise)+'{b}_x'+str(i)+'_beta'+str(k)+'.dat',
                './Tian2024_AIPW_res/Tian2024_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR_'+str(noise)+'{b}_x'+str(i)+'_beta'+str(k)+'.dat',
                B,
                'Tian2024-'+str(noise)
            )
            tian_res.to_csv('./Results/Tian2024_CirSym_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                             '_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv', index=False)


## Chakrabortty et al. (2019) DDR
for i in range(6):
    for k in range(3):
        B = 1000
        ddr_res = load_paired_csv_rows(
            './HDM_DDR_res/DDR_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR{b}_x'+str(i)+'_beta'+str(k)+'.csv',
            './HDM_DDR_res/DDR_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'.csv',
            B,
            'DDR'
        )
        ddr_res.to_csv('./Results/DDR_CirSym_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                        '_x'+str(i)+'_beta'+str(k)+'.csv', index=False)

        for noise in ['terr', 'laperr']:
            ddr_res = load_paired_csv_rows(
                './HDM_DDR_res/DDR_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR{b}_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv',
                './HDM_DDR_res/DDR_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv',
                B,
                'DDR-'+str(noise)
            )
            ddr_res.to_csv('./Results/DDR_CirSym_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                            '_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv', index=False)


## Hou et al. (2023) debiased SAS
for i in range(6):
    for k in range(3):
        B = 1000
        hou_sas_res = load_triplet_rows(
            './Hou2023_SAS_res/Hou2023_SAS_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR{b}_x'+str(i)+'_beta'+str(k)+'.dat',
            './Hou2023_SAS_res/Hou2023_SAS_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR{b}_x'+str(i)+'_beta'+str(k)+'.dat',
            B,
            'Hou2023-SAS'
        )
        hou_sas_res.to_csv('./Results/Hou2023_SAS_CirSym_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                            '_x'+str(i)+'_beta'+str(k)+'.csv', index=False)

        for noise in ['terr', 'laperr']:
            hou_sas_res = load_triplet_rows(
                './Hou2023_SAS_res/Hou2023_SAS_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MCAR_'+str(noise)+'{b}_x'+str(i)+'_beta'+str(k)+'.dat',
                './Hou2023_SAS_res/Hou2023_SAS_CirSym_cov_homoerr_d'+str(d)+'n'+str(n)+'_MAR_'+str(noise)+'{b}_x'+str(i)+'_beta'+str(k)+'.dat',
                B,
                'Hou2023-SAS-'+str(noise)
            )
            hou_sas_res.to_csv('./Results/Hou2023_SAS_CirSym_cov_homoerr_d'+str(d)+'_n'+str(n)+\
                                '_x'+str(i)+'_beta'+str(k)+'_'+str(noise)+'.csv', index=False)
