#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Mar 30, 2026

Description: Simulations on the debised SAS estimator in Hou et al. (2023) 
(https://www.jmlr.org/papers/volume24/21-1075/21-1075.pdf) (circulant symmetric covariance with Gaussian noises)
"""

import numpy as np
import cvxpy as cp
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
import sys
import pickle
from pathlib import Path


def make_query_x(d: int, i: int) -> np.ndarray:
    x = np.zeros((d,))
    if i == 0:
        x[0] = 1.0
    elif i == 1:
        x[0] = 1.0
        x[1] = 1 / 2
        x[2] = 1 / 4
        x[6] = 1 / 2
        x[7] = 1 / 8
    elif i == 2:
        x[99] = 1.0
    elif i == 3:
        x = 1 / np.linspace(1, d, d)
    elif i == 4:
        x = 1 / np.linspace(1, d, d) ** 2
    elif i == 5:
        x = np.ones((d,)) / np.sqrt(d)
    else:
        raise ValueError(f"Unknown x setting: {i}")
    return x


def make_beta0(d: int, k: int) -> np.ndarray:
    if k == 0:
        beta_0 = np.zeros((d,))
        beta_0[:5] = np.sqrt(5)
    elif k == 1:
        beta_0 = 1 / np.sqrt(np.linspace(1, d, d))
        beta_0 = 5 * beta_0 / np.linalg.norm(beta_0)
    elif k == 2:
        beta_0 = 1 / np.linspace(1, d, d)
        beta_0 = 5 * beta_0 / np.linalg.norm(beta_0)
    else:
        raise ValueError(f"Unknown beta setting: {k}")
    return beta_0


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def fit_lasso_linear(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    # sklearn objective: 1/(2n)||y - Xb||_2^2 + alpha ||b||_1, intercept not penalized.
    model = Lasso(alpha=float(lam), fit_intercept=True, max_iter=20000)
    model.fit(X, y)
    beta = np.zeros(X.shape[1] + 1)
    beta[0] = model.intercept_
    beta[1:] = model.coef_
    return beta


def predict_from_beta(X1: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return X1 @ beta


def solve_u_identity(X1: np.ndarray, x_std1: np.ndarray, lam_u: float) -> np.ndarray:
    # Implements (20) under identity link g(u)=u, so g'(u)=1.
    # Objective: mean[ 1/2 (X^T u)^2 - u^T x_std ] + lam_u ||u||_1
    p1 = X1.shape[1]
    u = cp.Variable(p1)
    obj = cp.sum_squares(X1 @ u) / (2.0 * X1.shape[0]) - x_std1 @ u + lam_u * cp.norm1(u[1:])
    prob = cp.Problem(cp.Minimize(obj))
    solved = False
    for solver, kwargs in [
        (cp.ECOS, {}),
        (cp.SCS, {"max_iters": 20000, "eps": 1e-5}),
        (cp.CVXOPT, {"max_iters": 30000}),
    ]:
        try:
            prob.solve(solver=solver, **kwargs)
            if prob.status in ("optimal", "optimal_inaccurate") and u.value is not None:
                solved = True
                break
        except Exception:
            continue
    if not solved:
        raise RuntimeError(f"u-solver failed; status={prob.status}")
    return np.asarray(u.value).reshape(-1)


def split_indices(indices: np.ndarray, K: int, rng: np.random.Generator):
    perm = rng.permutation(indices)
    return [arr.astype(int) for arr in np.array_split(perm, K)]


def fit_sas_linear_identity(X: np.ndarray,
                            Y: np.ndarray,
                            R: np.ndarray,
                            x_query: np.ndarray,
                            K: int = 5,
                            c_gamma: float = 1.0,
                            c_beta: float = 1.0,
                            c_u: float = 1.0):
    """
    Faithful implementation of Hou-Guo-Cai (2023), Section 3, specialized to:
      - linear regression / identity link g(u)=u,
      - no surrogates S, so W = X,
      - target x^T beta0 via their debiased SAS estimator for x_std^T beta0.

    IMPORTANT:
      - The paper's theory is for SSL/MCAR. Applying this to MAR is a mechanical extension,
        not a faithful use of their theorem.
      - We use W = X because the current simulation has no additional surrogates S.
    """
    n_total, d = X.shape
    labeled = np.where(R == 1)[0]
    unlabeled = np.where(R == 0)[0]
    n_lab = len(labeled)
    if n_lab < max(10, K):
        raise RuntimeError("Too few labeled observations for SAS cross-fitting.")

    rho = n_lab / n_total
    X1 = add_intercept(X)
    p1 = X1.shape[1]

    x1 = np.concatenate([[0.0], x_query])
    xnorm = np.linalg.norm(x1)
    if xnorm == 0:
        raise RuntimeError("Query vector has zero norm.")
    x_std1 = x1 / xnorm

    rng = np.random.default_rng(123)
    I_folds = split_indices(labeled, K, rng)
    J_folds = split_indices(unlabeled, K, rng) if len(unlabeled) > 0 else [np.array([], dtype=int) for _ in range(K)]
    if len(J_folds) < K:
        J_folds += [np.array([], dtype=int) for _ in range(K - len(J_folds))]

    beta_k = []
    gamma_k = []
    u_k = []

    for k in range(K):
        I_k = I_folds[k]
        J_k = J_folds[k]
        train_lab = np.setdiff1d(labeled, I_k, assume_unique=False)
        train_all = np.setdiff1d(np.arange(n_total), np.concatenate([I_k, J_k]), assume_unique=False)

        lam_gamma = c_gamma * np.sqrt(np.log(p1) / max(len(train_lab), 1))
        lam_beta = c_beta * np.sqrt(np.log(p1) / max(len(train_all), 1))
        lam_u = c_u * np.sqrt(np.log(p1) / max(len(train_all), 1))

        # (17): imputation model on out-of-fold labeled data.
        gamma_hat = fit_lasso_linear(X[train_lab, :], Y[train_lab], lam_gamma)
        gamma_k.append(gamma_hat)

        # (18): prediction model on out-of-fold full data with imputed outcomes for unlabeled.
        y_train = Y[train_all].copy()
        unlab_mask = (R[train_all] == 0)
        if np.any(unlab_mask):
            y_train[unlab_mask] = predict_from_beta(X1[train_all[unlab_mask], :], gamma_hat)
        beta_hat = fit_lasso_linear(X[train_all, :], y_train, lam_beta)
        beta_k.append(beta_hat)

        # (20) specialized to identity link. In this case g'(.) = 1, so the objective no longer
        # depends on beta^(k,k0); it uses all out-of-fold data excluding fold k.
        u_hat = solve_u_identity(X1[train_all, :], x_std1, lam_u)
        u_k.append(u_hat)

    # (22): cross-fitted debiased estimator for x_std^T beta0.
    term1 = np.mean([x_std1 @ bk for bk in beta_k])

    term2 = 0.0
    for k in range(K):
        J_k = J_folds[k]
        if len(J_k) == 0:
            continue
        diff = predict_from_beta(X1[J_k, :], beta_k[k]) - predict_from_beta(X1[J_k, :], gamma_k[k])
        term2 += np.sum((X1[J_k, :] @ u_k[k]) * diff)
    term2 /= n_total

    term3 = 0.0
    for k in range(K):
        I_k = I_folds[k]
        pred_gamma = predict_from_beta(X1[I_k, :], gamma_k[k])
        pred_beta = predict_from_beta(X1[I_k, :], beta_k[k])
        resid = (1.0 - rho) * pred_gamma + rho * pred_beta - Y[I_k]
        term3 += np.sum((X1[I_k, :] @ u_k[k]) * resid)
    term3 /= n_lab

    xstd_beta_deb = term1 - term2 - term3
    m_hat = float(xnorm * xstd_beta_deb)

    # (23): cross-fitted variance estimator.
    Vhat = 0.0
    for k in range(K):
        I_k = I_folds[k]
        pred_gamma = predict_from_beta(X1[I_k, :], gamma_k[k])
        pred_beta = predict_from_beta(X1[I_k, :], beta_k[k])
        resid_lab = (1.0 - rho) * pred_gamma + rho * pred_beta - Y[I_k]
        Vhat += np.sum(((X1[I_k, :] @ u_k[k]) ** 2) * (resid_lab ** 2))

        J_k = J_folds[k]
        if len(J_k) > 0:
            diff = predict_from_beta(X1[J_k, :], beta_k[k]) - predict_from_beta(X1[J_k, :], gamma_k[k])
            Vhat += (rho ** 2) * np.sum(((X1[J_k, :] @ u_k[k]) ** 2) * (diff ** 2))
    Vhat /= n_lab

    # To match your current CI template est ± z * asym_var,
    # we store sigma_hat = sqrt(Vhat) and asym_var = ||x|| * sqrt(Vhat) / sqrt(n_lab).
    sigma_hat = float(np.sqrt(max(Vhat, 0.0)))
    asym_var = float(xnorm * sigma_hat / np.sqrt(n_lab))

    return {
        "m_hat": m_hat,
        "asym_var": asym_var,
        "sigma_hat": sigma_hat,
        "rho": rho,
        "Vhat": float(Vhat),
        "xstd_beta_deb": float(xstd_beta_deb),
    }


def save_same_format(out_dir: Path, prefix: str, job_id: int, x_id: int, beta_id: int, result: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{prefix}{job_id}_x{x_id}_beta{beta_id}.dat"
    with open(fname, "wb") as file:
        pickle.dump([result["m_hat"], result["asym_var"], result["sigma_hat"]], file)


job_id = int(sys.argv[1])
print(job_id)

#==========================================================================================#

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
sig = 1

out_dir = Path('./Hou2023_SAS_res')
out_dir.mkdir(parents=True, exist_ok=True)

## Consider different simulation settings
for i in range(6):
    x = make_query_x(d, i)
    for k in range(3):
        beta_0 = make_beta0(d, k)
        np.random.seed(job_id)

        X_sim = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
        eps_err_sim = sig*np.random.randn(n)
        Y_sim = np.dot(X_sim, beta_0) + eps_err_sim

        ## MCAR
        obs_prob1 = 0.7
        R1 = np.random.choice([0,1], size=n, replace=True, p=[1-obs_prob1, obs_prob1])

        ## MAR
        obs_prob2 = 1/(1 + np.exp(-1+X_sim[:,6]-X_sim[:,7]))
        R2 = np.ones((n,))
        R2[np.random.rand(n) >= obs_prob2] = 0

        # Faithful to the paper for MCAR/SSL. For MAR, this is only a mechanical extension.
        res1 = fit_sas_linear_identity(X_sim, Y_sim, R1, x)
        res2 = fit_sas_linear_identity(X_sim, Y_sim, R2, x)

        save_same_format(out_dir, f'Hou2023_SAS_CirSym_cov_homoerr_d{d}n{n}_MCAR', job_id, i, k, res1)
        save_same_format(out_dir, f'Hou2023_SAS_CirSym_cov_homoerr_d{d}n{n}_MAR', job_id, i, k, res2)
