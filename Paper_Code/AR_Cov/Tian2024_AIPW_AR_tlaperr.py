#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Mar 28, 2026

Description: Simulations on the AIPW estimator in Tian et al. (2024) 
(https://arxiv.org/pdf/2406.13906) (Toeplitz covariance AR(1) with t2 and Laplace noises)
"""

import numpy as np
import cvxpy as cp
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
import sys
import pickle
from pathlib import Path


def make_query_x(d: int, i: int) -> np.ndarray:
    x = np.zeros(d)
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
        x = np.ones(d) / np.sqrt(d)
    else:
        raise ValueError(f"Unknown x setting: {i}")
    return x


def make_beta0(d: int, k: int) -> np.ndarray:
    if k == 0:
        beta_0 = np.zeros(d)
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


def make_sigma_ar1(d: int, rho: float) -> np.ndarray:
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def generate_data(job_id: int, d: int, n: int, rho: float, sig: float, beta_0: np.ndarray):
    rng = np.random.default_rng(job_id)
    Sigma = make_sigma_ar1(d, rho)
    X = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
    eps = sig * rng.standard_normal(n)
    Y = X @ beta_0 + eps

    obs_prob1 = 0.7
    R1 = rng.choice([0, 1], size=n, replace=True, p=[1 - obs_prob1, obs_prob1])

    obs_prob2 = expit(-1 + X[:, 6] - X[:, 7])
    R2 = (rng.random(n) < obs_prob2).astype(int)
    return X, Y, R1, R2


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def calibrated_logistic_l1(X: np.ndarray, R: np.ndarray, lam: float = None, clip=1e-30):
    n, p = X.shape
    F = add_intercept(X)
    p1 = F.shape[1]
    if lam is None:
        lam = 0.5 * np.sqrt(np.log(p1) / n)

    gamma = cp.Variable(p1)
    linear = F @ gamma
    obj = cp.sum(cp.multiply(R, cp.exp(-linear)) + cp.multiply(1 - R, linear)) / n
    obj += lam * cp.norm1(gamma[1:])
    problem = cp.Problem(cp.Minimize(obj))

    solvers = [
        (cp.ECOS, {}),
        (cp.SCS, {"max_iters": 20000, "eps": 1e-5}),
        (cp.CVXOPT, {"max_iters": 30000}),
    ]
    solved = False
    for solver, kwargs in solvers:
        try:
            problem.solve(solver=solver, **kwargs)
            if problem.status in ("optimal", "optimal_inaccurate") and gamma.value is not None:
                solved = True
                break
        except Exception:
            continue
    if not solved:
        raise RuntimeError(f"Calibrated logistic solve failed; status={problem.status}")

    gamma_hat = np.asarray(gamma.value).reshape(-1)
    pi_hat = expit(F @ gamma_hat)
    pi_hat = np.clip(pi_hat, clip, 1 - clip)
    return gamma_hat, pi_hat


def weighted_lasso_or(X: np.ndarray, Y: np.ndarray, R: np.ndarray, pi_hat: np.ndarray):
    labeled = R == 1
    X_l = X[labeled]
    Y_l = Y[labeled]
    w = (1 - pi_hat[labeled]) / pi_hat[labeled]
    w = np.maximum(w, 1e-6)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_l)
    model = LassoCV(cv=5, fit_intercept=True, max_iter=20000)
    model.fit(Xs, Y_l, sample_weight=w)
    m_hat_all = model.predict(scaler.transform(X))
    return model, scaler, m_hat_all


def final_aipw_estimating_eq(X, Y, R, pi_hat, m_hat, ridge=1e-8):
    """
    Solve the sample AIPW estimating equation (15) under the identity link psi(u)=u,
    with Z = X.

    The estimating equation is
        mean[ R/pi * (Y - m_hat) * X + (m_hat - X beta) * X ] = 0.

    This implies
        Gamma_hat beta = rhs,
    where
        Gamma_hat = mean[X X^T],
        rhs = mean[(R/pi * (Y - m_hat) + m_hat) * X].

    Because d > n in the current simulation, Gamma_hat may be singular. We therefore
    return the minimum-norm solution based on the Moore-Penrose pseudoinverse.
    This is the natural numerical extension of the estimating equation, but its
    asymptotic theory is not covered by Theorem 1 when dim(Z)=d grows with n.
    """
    n, d = X.shape

    aipw_signal = (R / pi_hat) * (Y - m_hat) + m_hat
    Gamma_hat = (X.T @ X) / n
    rhs = (X.T @ aipw_signal) / n

    # Solve Gamma_hat beta = rhs.
    # If Gamma_hat is singular (which is common when d > n), use the
    # Moore-Penrose minimum-norm solution.
    try:
        beta_hat = np.linalg.solve(Gamma_hat + ridge * np.eye(d), rhs)
    except np.linalg.LinAlgError:
        beta_hat = np.linalg.pinv(Gamma_hat) @ rhs

    # Estimating-function contributions tau_i(beta_hat)
    xb = X @ beta_hat
    tau = ((R / pi_hat) * (Y - m_hat))[:, None] * X + ((m_hat - xb)[:, None] * X)

    # Heuristic sandwich quantities with Z = X, identity link:
    # Gamma_hat = mean[X X^T], Lambda_hat = mean[tau tau^T]
    # This matches the paper's formula algebraically, but the theory does NOT
    # apply when dim(Z)=d grows with n.
    Lambda_hat = (tau.T @ tau) / n

    return {
        "beta_hat": beta_hat,
        "Gamma_hat": Gamma_hat,
        "Lambda_hat": Lambda_hat,
        "tau": tau,
    }


def fit_one(X: np.ndarray, Y: np.ndarray, R: np.ndarray, x_query: np.ndarray):
    gamma_hat, pi_hat = calibrated_logistic_l1(X, R)
    or_model, or_scaler, m_hat_or = weighted_lasso_or(X, Y, R, pi_hat)

    est = final_aipw_estimating_eq(X, Y, R, pi_hat, m_hat_or)
    beta_hat = est["beta_hat"]

    m_hat = float(np.dot(x_query, beta_hat))

    # Heuristic variance for x^T beta_hat based on the sandwich form in Theorem 1:
    # Sigma_hat = Gamma_hat^{-1} Lambda_hat Gamma_hat^{-1},
    # var(x^T beta_hat) approx x^T Sigma_hat x / n.
    #
    # IMPORTANT: this is NOT justified by Theorem 1 when Z = X and d grows with n.
    Gamma_hat = est["Gamma_hat"]
    Lambda_hat = est["Lambda_hat"]

    try:
        Gamma_inv = np.linalg.inv(Gamma_hat)
    except np.linalg.LinAlgError:
        Gamma_inv = np.linalg.pinv(Gamma_hat)

    Sigma_hat = Gamma_inv @ Lambda_hat @ Gamma_inv
    var_xbeta = float(x_query @ Sigma_hat @ x_query / X.shape[0])
    asym_var = float(np.sqrt(max(var_xbeta, 0.0)))

    # Keep sigma_hat only for compatibility. It is NOT the paper's noise SD.
    tau_proj = est["tau"] @ x_query
    sigma_hat = float(np.sqrt(np.mean(tau_proj ** 2)))

    return {
        "m_hat": m_hat,
        "asym_var": asym_var,
        "sigma_hat": sigma_hat,
        "beta_hat": beta_hat,
        "pi_hat": pi_hat,
        "m_hat_or": m_hat_or,
        "Gamma_hat": Gamma_hat,
        "Lambda_hat": Lambda_hat,
        "tau": est["tau"],
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

Sigma = np.zeros((d, d)) + np.eye(d)
rho = 0.9
for i in range(d):
    for j in range(i + 1, d):
        Sigma[i, j] = rho ** (abs(i - j))
        Sigma[j, i] = rho ** (abs(i - j))
sig = 1

out_dir = Path("./Tian2024_AIPW_res")
out_dir.mkdir(parents=True, exist_ok=True)

## Consider different simulation settings
for i in range(6):
    x = make_query_x(d, i)

    for k in range(3):
        for err in ['terr', 'laperr']:
            beta_0 = make_beta0(d, k)

            # True regression function
            m_true = np.dot(x, beta_0)

            np.random.seed(job_id)

            X_sim = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
            if err == 'terr':
                eps_err_sim = np.random.standard_t(df=2, size=n)
            elif err == 'laperr':
                eps_err_sim = np.random.laplace(loc=0, scale=1/np.sqrt(2), size=n)
            else:
                raise ValueError(f"Unknown error type: {err}")
            Y_sim = np.dot(X_sim, beta_0) + eps_err_sim

            ## MCAR
            obs_prob1 = 0.7
            R1 = np.random.choice([0, 1], size=n, replace=True, p=[1 - obs_prob1, obs_prob1])

            ## MAR
            obs_prob2 = 1 / (1 + np.exp(-1 + X_sim[:, 6] - X_sim[:, 7]))
            R2 = np.ones((n,))
            R2[np.random.rand(n) >= obs_prob2] = 0

            ## Tian et al. method
            res1 = fit_one(X_sim, Y_sim, R1, x)
            res2 = fit_one(X_sim, Y_sim, R2, x)

            save_same_format(out_dir, f"Tian2024_AR_cov_homoerr_d{d}n{n}_MCAR_{err}", job_id, i, k, res1)

            save_same_format(out_dir, f"Tian2024_AR_cov_homoerr_d{d}n{n}_MAR_{err}", job_id, i, k, res2)