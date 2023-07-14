library(scalreg)
library(glmnet)
library(pracma)
library(MASS)
library(caret)
source("debias_prog.R")

jobid = commandArgs(TRUE)
# jobid = 1
print(jobid)

d = 1000
n = 900

Sigma = array(0, dim = c(d,d)) + diag(d)
rho = 0.1
for(i in 1:(d-1)){
  for(j in (i+1):d){
    if ((j < i+6) | (j > i+d-6)){
      Sigma[i,j] = rho
      Sigma[j,i] = rho
    }
  }
}
sig = 1

# Consider different simulation settings
for (i in 0:4) {
  if (i == 0) {
    ## x0
    x = rep(0, d)
    x[1] = 1
  }
  if (i == 1) {
    ## x1
    x = rep(0, d)
    x[c(1, 2, 3, 7, 8)] = c(1, 1/2, 1/4, 1/2, 1/8)
  }
  if (i == 2) {
    ## x2
    x = rep(0, d)
    x[100] = 1
  }
  if (i == 3) {
    ## x3
    x = 1 / seq(1, d, 1)
  }
  if (i == 4) {
    ## x4
    x = 1 / (seq(1, d, 1)^2)
  }
  
  for (k in 0:2) {
    if (k == 0) {
      s_beta = 5
      beta_0 = rep(0, d)
      beta_0[1:s_beta] = sqrt(5)
    }
    if (k == 1) {
      beta_0 = 1 / sqrt(seq(1, d, 1))
      beta_0 = 5*beta_0 / sqrt(sum(beta_0^2))
    }
    if (k == 2) {
      beta_0 = 1 / seq(1, d, 1)
      beta_0 = 5*beta_0 / sqrt(sum(beta_0^2))
    }
    
    # True regression function
    m_true = sum(x * beta_0)
    set.seed(jobid)
    x = array(x, dim = c(1,d))
    
    X_sim = mvrnorm(n, mu = rep(0, d), Sigma)
    eps_err_sim = sig * rnorm(n)
    Y_sim = drop(X_sim %*% beta_0) + eps_err_sim
    
    ## MCAR
    obs_prob1 = 0.7
    R1 = rbinom(n, 1, obs_prob1)
    
    ## MAR
    obs_prob2 = 1 / (1 + exp(-1 + X_sim[, 7] - X_sim[, 8]))
    R2 = rep(1, n)
    R2[runif(n) >= obs_prob2] = 0
    
    ## Lasso pilot estimator
    lasso_pilot1 = scalreg(X_sim[R1 == 1,], Y_sim[R1 == 1], lam0 = "univ", LSE = FALSE)
    beta_pilot1 = lasso_pilot1$coefficients
    sigma_pilot1 = lasso_pilot1$hsigma
    
    lasso_pilot2 = scalreg(X_sim[R2 == 1,], Y_sim[R2 == 1], lam0 = "univ", LSE = FALSE)
    beta_pilot2 = lasso_pilot2$coefficients
    sigma_pilot2 = lasso_pilot2$hsigma
    
    ## Propensity score estimation (logistic regression CV)
    zeta1 = 10^seq(-1, log10(300), length.out = 40) * sqrt(log(d) / n)
    lr1 = cv.glmnet(X_sim, R1, family = 'binomial', alpha = 1, type.measure = 'deviance', 
                     lambda = zeta1, nfolds = 5, parallel = FALSE)
    lr1 = glmnet(X_sim, R1, family = "binomial", alpha = 1, lambda = lr1$lambda.min, 
                 standardize = TRUE, thresh=1e-6)
    prop_score1 = drop(predict(lr1, newx = X_sim, type = 'response'))
    
    zeta2 = 10^seq(-1, log10(300), length.out = 40) * sqrt(log(d) / n)
    lr2 = cv.glmnet(X_sim, R2, family = 'binomial', alpha = 1, type.measure = 'deviance', 
                    lambda = zeta2, nfolds = 5, parallel = FALSE)
    lr2 = glmnet(X_sim, R2, family = "binomial", alpha = 1, lambda = lr2$lambda.min, 
                 standardize = TRUE, thresh=1e-6)
    prop_score2 = drop(predict(lr2, newx = X_sim, type = 'response'))
    
    gamma_n_lst = seq(0.001, max(abs(x)), length.out = 41)
    cv_fold = 5
    kf = createFolds(1:n, cv_fold, list = FALSE, returnTrain = TRUE)
    dual_loss1 = matrix(0, nrow = cv_fold, ncol = length(gamma_n_lst))
    dual_loss2 = matrix(0, nrow = cv_fold, ncol = length(gamma_n_lst))
    f_ind = 1
    for (fold in 1:cv_fold) {
      train_ind <- (kf != fold)
      test_ind <- (kf == fold)
      X_train <- X_sim[train_ind, ]
      X_test <- X_sim[test_ind, ]
      prop_score1_train <- prop_score1[train_ind]
      prop_score1_test <- prop_score1[test_ind]
      prop_score2_train <- prop_score2[train_ind]
      prop_score2_test <- prop_score2[test_ind]
      
      for (j in 1:length(gamma_n_lst)) {
        w_train1 <- DebiasProg(X_train, x, diag(prop_score1_train), gamma_n = gamma_n_lst[j])
        if (any(is.na(w_train1))) {
          cat("The primal debiasing program for this fold of the data is not feasible when gamma/n =",
              round(gamma_n_lst[j], 4), "!\n")
          dual_loss1[f_ind, j] = NA
        } else {
          ll_train1 <- DualCD(X_train, x, diag(prop_score1_train), gamma_n_lst[j], ll_init = NULL, 
                              eps = 1e-8, max_iter = 5000)
          if (sum(abs(w_train1 + drop(X_train %*% ll_train1) / (2 * sqrt(dim(X_train)[1]))) > 1e-3) > 0) {
            cat("The strong duality between primal and dual programs does not satisfy when gamma/n =",
                round(gamma_n_lst[j], 4), "!\n")
            dual_loss1[f_ind, j] = NA
          } else {
            dual_loss1[f_ind, j] = DualObj(X_test, x, diag(prop_score1_test), ll_cur = ll_train1, 
                                            gamma_n = gamma_n_lst[j])
          }
        }
        w_train2 <- DebiasProg(X_train, x, diag(prop_score2_train), gamma_n_lst[j])
        if (any(is.na(w_train2))) {
          cat("The primal debiasing program for this fold of the data is not feasible when gamma/n =",
              round(gamma_n_lst[j], 4), "!\n")
          dual_loss2[f_ind, j] = NA
        } else {
          ll_train2 = DualCD(X_train, x, diag(prop_score2_train), gamma_n_lst[j], ll_init = NULL, 
                              eps = 1e-8, max_iter = 5000)
          if (sum(abs(w_train2 + drop(X_train %*% ll_train2) / (2 * sqrt(dim(X_train)[1]))) > 1e-3) > 0) {
            cat("The strong duality between primal and dual programs does not satisfy when gamma/n =",
                round(gamma_n_lst[j], 4), "!\n")
            dual_loss2[f_ind, j] = NA
          } else {
            dual_loss2[f_ind, j] = DualObj(X_test, x, diag(prop_score2_test), ll_cur = ll_train2, 
                                            gamma_n = gamma_n_lst[j])
          }
        }
      }
      f_ind = f_ind + 1
    }
    mean_dual_loss1 = apply(dual_loss1, 2, mean, na.rm = FALSE)
    mean_dual_loss2 = apply(dual_loss2, 2, mean, na.rm = FALSE)
    std_dual_loss1 = apply(dual_loss1, 2, function(x) sd(x, na.rm = FALSE)) / sqrt(cv_fold)
    std_dual_loss2 = apply(dual_loss2, 2, function(x) sd(x, na.rm = FALSE)) / sqrt(cv_fold)
    
    # Different rules for choosing the tuning parameter
    para_rule = c('1se', 'mincv', 'minfeas')
    for (rule in para_rule) {
      if (rule == 'mincv') {
        gamma_n1_opt <- gamma_n_lst[which.min(mean_dual_loss1)]
        gamma_n2_opt <- gamma_n_lst[which.min(mean_dual_loss2)]
      }
      if (rule == '1se') {
        One_SE1 = (mean_dual_loss1 > min(mean_dual_loss1, na.rm = TRUE) +
                      std_dual_loss1[which.min(mean_dual_loss1)]) &
          (gamma_n_lst < gamma_n_lst[which.min(mean_dual_loss1)])
        if (sum(One_SE1, na.rm = TRUE) == 0) {
          One_SE1 = rep(TRUE, length(gamma_n_lst))
        }
        gamma_n_lst1 = gamma_n_lst[One_SE1]
        gamma_n1_opt = gamma_n_lst1[which.min(mean_dual_loss1[One_SE1])]
        
        One_SE2 <- (mean_dual_loss2 > min(mean_dual_loss2, na.rm = TRUE) +
                      std_dual_loss2[which.min(mean_dual_loss2)]) &
          (gamma_n_lst < gamma_n_lst[which.min(mean_dual_loss2)])
        if (sum(One_SE2, na.rm = TRUE) == 0) {
          One_SE2 <- rep(TRUE, length(gamma_n_lst))
        }
        gamma_n_lst2 <- gamma_n_lst[One_SE2]
        gamma_n2_opt <- gamma_n_lst2[which.min(mean_dual_loss2[One_SE2])]
      }
      if (rule == 'minfeas') {
        gamma_n1_opt <- min(gamma_n_lst[!is.na(mean_dual_loss1)])
        gamma_n2_opt <- min(gamma_n_lst[!is.na(mean_dual_loss2)])
      }
      
      # Solve the primal and dual on the original dataset
      w_obs1 <- DebiasProg(X_sim, x, diag(prop_score1), gamma_n1_opt)
      ll_obs1 <- DualCD(X_sim, x, diag(prop_score1), gamma_n1_opt, ll_init = NULL, 
                        eps = 1e-8, max_iter = 5000)
      
      w_obs2 <- DebiasProg(X_sim, x, diag(prop_score2), gamma_n2_opt)
      ll_obs2 <- DualCD(X_sim, x, diag(prop_score2), gamma_n2_opt, ll_init = NULL, 
                        eps = 1e-8, max_iter = 5000)
      
      # Store the results
      m_deb1 <- sum(x * beta_pilot1) + sum(w_obs1 * R1 * (Y_sim - X_sim %*% beta_pilot1)) / sqrt(n)
      asym_var1 <- sqrt(sum(prop_score1 * w_obs1^2) / n)
      sigma_hat1 <- sigma_pilot1
      
      m_deb2 <- sum(x * beta_pilot2) + sum(w_obs2 * R2 * (Y_sim - X_sim %*% beta_pilot2)) / sqrt(n)
      asym_var2 <- sqrt(sum(prop_score2 * w_obs2^2) / n)
      sigma_hat2 <- sigma_pilot2
      
      debias_res = data.frame(m_deb1 = m_deb1, asym_var1 = asym_var1, sigma_hat1 = sigma_hat1, 
                              m_deb2 = m_deb2, asym_var2 = asym_var2, sigma_hat2 = sigma_hat2)
      write.csv(debias_res, paste0("./debias_res/DebiasProg_CirSym_cov_homoerr_d", d, "_n", n, "_", jobid, "_x", i, "_beta", k, "_rule", rule, "_gauss_R.csv"), 
                row.names=FALSE)
    }
  }
}
