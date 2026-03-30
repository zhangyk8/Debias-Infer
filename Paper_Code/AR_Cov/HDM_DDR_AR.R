# Apply Chakrabortty et al. (2019) DDR code to our simulation design 
# on high-dimensional inference with missing outcomes.
#
# Inputs:
#   1) HD_M_est_functions.R   (from Chakrabortty et al.)
#   2) DDR_est_apply.R        (or source file containing DDR_est)
#
# Usage:
#   Rscript apply_ddr_to_your_simulation.R <jobid>
#
# This script reproduces the same data-generating mechanism as the Python script
# for the proposed method, then applies DDR_est to the same simulated datasets.
#
# Output:
#   One .rds result file per (missingness setting, x setting, beta setting)
#   containing the DDR estimate of m(x)=x^T beta, its standard error, CI, and
#   intermediate objects.

jobid = commandArgs(TRUE)
# jobid = b
jobid = as.integer(jobid)
print(jobid)

source("HD_M_est_functions.R")
source("DDR_est_apply.R")

suppressPackageStartupMessages({
  library(MASS)
})

# -----------------------------
# Common simulation settings
# -----------------------------
d <- 1000
n <- 900
rho <- 0.9
sig <- 1

Sigma <- matrix(0, d, d) + diag(d)
for (i in 1:(d - 1)) {
  for (j in (i + 1):d) {
    Sigma[i, j] <- rho^(abs(i - j))
    Sigma[j, i] <- Sigma[i, j]
  }
}

dir.create("DDR_res", showWarnings = FALSE)

make_x <- function(i, d) {
  x <- rep(0, d)
  if (i == 0) x[1] <- 1
  if (i == 1) {
    x[1] <- 1
    x[2] <- 1/2
    x[3] <- 1/4
    x[7] <- 1/2
    x[8] <- 1/8
  }
  if (i == 2) x[100] <- 1
  if (i == 3) x <- 1 / seq_len(d)
  if (i == 4) x <- 1 / (seq_len(d)^2)
  if (i == 5) x <- rep(1/sqrt(d), d)
  x
}

make_beta <- function(k, d) {
  beta <- rep(0, d)
  if (k == 0) {
    beta[1:5] <- sqrt(5)
  }
  if (k == 1) {
    beta <- 1 / sqrt(seq_len(d))
    beta <- 5 * beta / sqrt(sum(beta^2))
  }
  if (k == 2) {
    beta <- 1 / seq_len(d)
    beta <- 5 * beta / sqrt(sum(beta^2))
  }
  beta
}

run_one_setting <- function(X_sim, Y_sim, R, x, m_true,
                            model_T = "logit", tuning_T = "BIC",
                            model_m = "linear", tuning_m = "CV-MSE",
                            tuning_theta = "BIC",
                            split = TRUE, K = 2,
                            cov_est = "node") {

  # Chakrabortty et al. use T as the missingness/observation indicator.
  fit <- DDR_est(
    x = X_sim,
    y = as.numeric(Y_sim),
    T = as.numeric(R),
    model_T = model_T,
    tuning_T = tuning_T,
    model_m = model_m,
    tuning_m = tuning_m,
    tuning_theta = tuning_theta,
    split = split,
    K = K,
    infer = TRUE,
    cov_est = cov_est
  )

  # theta has intercept; theta_de does not include intercept in the supplied code.
  beta_hat_lasso <- as.numeric(fit$theta[-1])
  beta_hat_de <- as.numeric(fit$theta_de)

  # Point estimate of m(x) = x^T beta
  m_hat_lasso <- sum(x * beta_hat_lasso)
  m_hat_de <- sum(x * beta_hat_de)

  # Influence-function-based SE for the linear functional x^T theta_de
  # IF is n x p for theta_de in the supplied code.
  IF_fun <- as.matrix(fit$IF)
  infl_m <- as.numeric(IF_fun %*% x)
  se_m <- sqrt(mean(infl_m^2) / n)

  ci_lower <- m_hat_de - qnorm(0.975) * se_m
  ci_upper <- m_hat_de + qnorm(0.975) * se_m

  list(
    m_true = m_true,
    m_hat_lasso = m_hat_lasso,
    m_hat_de = m_hat_de,
    se_m = se_m,
    ci = c(lower = ci_lower, upper = ci_upper),
    covered = as.numeric(ci_lower <= m_true && m_true <= ci_upper),
    ci_length = ci_upper - ci_lower,
    beta_hat_lasso = beta_hat_lasso,
    beta_hat_de = beta_hat_de,
    raw_fit = fit
  )
}

for (i in 0:5) {
  x <- make_x(i, d)

  for (k in 0:2) {
    beta_0 <- make_beta(k, d)
    m_true <- sum(x * beta_0)

    set.seed(jobid)
    X_sim <- MASS::mvrnorm(n = n, mu = rep(0, d), Sigma = Sigma)
    eps_err_sim <- rnorm(n, mean = 0, sd = sig)
    Y_sim <- as.numeric(X_sim %*% beta_0 + eps_err_sim)

    # MCAR
    obs_prob1 <- 0.7
    R1 <- rbinom(n, size = 1, prob = obs_prob1)

    # MAR
    obs_prob2 <- 1 / (1 + exp(-1 + X_sim[, 7] - X_sim[, 8]))
    R2 <- rbinom(n, size = 1, prob = obs_prob2)

    # Apply DDR to the SAME datasets
    res_mcar <- run_one_setting(X_sim, Y_sim, R1, x, m_true)
    saveRDS(
      res_mcar,
      file = sprintf("HDM_DDR_res/DDR_AR_cov_homoerr_d%dn%d_MCAR%d_x%d_beta%d.rds", d, n, jobid, i, k)
    )

    res_mar <- run_one_setting(X_sim, Y_sim, R2, x, m_true)
    saveRDS(
      res_mar,
      file = sprintf("HDM_DDR_res/DDR_AR_cov_homoerr_d%dn%d_MAR%d_x%d_beta%d.rds", d, n, jobid, i, k)
    )
  }
}
