# Apply Chakrabortty et al. (2019) DDR code to our simulation design 
# on high-dimensional inference with missing outcomes.
#
# Inputs:
#   1) HD_M_est_functions.R   (from Chakrabortty et al.)
#   2) DDR_est_apply.R        (or source file containing DDR_est)
#
# This script reproduces the same data-generating mechanism as the Python script
# for the proposed method, then applies DDR_est to the same simulated datasets.
#
# Output:
#   One .csv summary file per (missingness setting, x setting, beta setting)
#   containing the DDR estimate of m(x)=x^T beta, its standard error, CI, and
#   coverage summary. The csv format is convenient for downstream Python
#   synthesis scripts.

jobid = commandArgs(TRUE)
# jobid = b
jobid = as.integer(jobid)
print(jobid)

source("HD_M_est_functions.R")
source("DDR_est_apply.R")

suppressPackageStartupMessages({
  library(MASS)
  library(rmutil)  # for rlaplace
})

# -----------------------------
# Common simulation settings
# -----------------------------
d <- 1000
n <- 900
sig <- 1

Sigma <- matrix(0, d, d) + diag(d)
rho = 0.1
for(i in 1:(d-1)){
  for(j in (i+1):d){
    if ((j < i+6) | (j > i+d-6)){
      Sigma[i,j] = rho
      Sigma[j,i] = rho
    }
  }
}

dir.create("HDM_DDR_res", showWarnings = FALSE)

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

save_result_summary <- function(result, file) {
  summary_df <- data.frame(
    m_true = as.numeric(result$m_true),
    m_hat_lasso = as.numeric(result$m_hat_lasso),
    m_hat_de = as.numeric(result$m_hat_de),
    se_m = as.numeric(result$se_m),
    ci_lower = as.numeric(result$ci["lower"]),
    ci_upper = as.numeric(result$ci["upper"]),
    covered = as.numeric(result$covered),
    ci_length = as.numeric(result$ci_length)
  )
  write.csv(summary_df, file = file, row.names = FALSE)
}

for (i in 0:5) {
  x <- make_x(i, d)

  for (k in 0:2) {
    for (err in c("terr", "laperr")) { 
      beta_0 <- make_beta(k, d)
      m_true <- sum(x * beta_0)

      set.seed(jobid)
      X_sim <- MASS::mvrnorm(n = n, mu = rep(0, d), Sigma = Sigma)
      if (err == "terr") {
        eps_err_sim <- rt(n, df = 2)  # t2 noise
      } else if (err == "laperr") {
        eps_err_sim <- rlaplace(n, m = 0, s = 1 / sqrt(2))  # Laplace noise
      }
      Y_sim <- as.numeric(X_sim %*% beta_0 + eps_err_sim)

      # MCAR
      obs_prob1 <- 0.7
      R1 <- rbinom(n, size = 1, prob = obs_prob1)

      # MAR
      obs_prob2 <- 1 / (1 + exp(-1 + X_sim[, 7] - X_sim[, 8]))
      R2 <- rbinom(n, size = 1, prob = obs_prob2)

      # Apply DDR to the SAME datasets
      mcar_file <- sprintf("HDM_DDR_res/DDR_CirSym_cov_homoerr_d%dn%d_MCAR%d_x%d_beta%d_%s.csv", d, n, jobid, i, k, err)
      if (file.exists(mcar_file)) {
        print(sprintf("MCAR result already exists for i=%d, k=%d, err=%s, skipping MCAR...", i, k, err))
      } else {
        print(sprintf("Running DDR MCAR for i=%d, k=%d, err=%s...", i, k, err))
        res_mcar <- run_one_setting(X_sim, Y_sim, R1, x, m_true)
        save_result_summary(res_mcar, file = mcar_file)
      }

      mar_file <- sprintf("HDM_DDR_res/DDR_CirSym_cov_homoerr_d%dn%d_MAR%d_x%d_beta%d_%s.csv", d, n, jobid, i, k, err)
      if (file.exists(mar_file)) {
        print(sprintf("MAR result already exists for i=%d, k=%d, err=%s, skipping MAR...", i, k, err))
      } else {
        print(sprintf("Running DDR MAR for i=%d, k=%d, err=%s...", i, k, err))
        res_mar <- run_one_setting(X_sim, Y_sim, R2, x, m_true)
        save_result_summary(res_mar, file = mar_file)
      }
    }
  }
}
