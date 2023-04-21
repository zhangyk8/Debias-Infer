library(mvtnorm)
library(hdi)

for(b in 1:500){
  # jobid = commandArgs(TRUE)
  jobid = b
print(jobid)

d = 600
n = 500

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

for(i in 0:3){
  if(i == 0){
    ## x0
    x = array(0, dim = c(d,1))
    x[1,1] = 1
  }
  if(i == 1){
    ## x1
    x = array(0, dim = c(d,1))
    x[1,1] = 1
    x[2,1] = 1/2
    x[3,1] = 1/4
    x[7,1] = 1/2
    x[8,1] = 1/8
  }
  if(i == 2){
    ## x2
    x = array(0, dim = c(d,1))
    x[100,1] = 1
  }
  if(i == 3){
    x = array(0, dim = c(d,1))
    x[,1] = 1/seq(1, d, by = 1)^2
  }
  
  for(k in 0:2){
    if(k == 0){
      s_beta = 5
      beta_0 = array(0, dim = c(d,1))
      beta_0[(1:s_beta),1] = 1
    }
    if(k == 1){
      beta_0 = array(0, dim = c(d,1))
      beta_0[,1] = 1/sqrt(seq(1, d, length.out=d))
      beta_0[,1] = 5*beta_0[,1]/sqrt(sum(beta_0[,1]^2))
    }
    if(k == 2){
      beta_0 = array(0, dim = c(d,1))
      beta_0[,1] = 1/seq(1, d, length.out=d)
      beta_0[,1] = 5*beta_0[,1]/sqrt(sum(beta_0[,1]^2))
    }
    
    set.seed(jobid)
    
    start_time = Sys.time()
    flag = 1
    while(flag == 1){
      flag = 0
      X_sim = rmvnorm(n, mean = rep(0, d), sigma = Sigma, method = "chol")
      eps_err_sim = rnorm(n, mean = 0, sd = sig)
      Y_sim = X_sim %*% beta_0 + eps_err_sim
      ## MCAR
      obs_prob1 = 0.7
      R1 = sample(c(0,1), size = n, replace = TRUE, prob = c(1 - obs_prob1, obs_prob1))
      ## MAR
      obs_prob2 = 1/(1 + exp(-1+X_sim[,6]-X_sim[,7]))
      unif = runif(n, min = 0, max = 1)
      R2 = as.numeric(unif <= obs_prob2)
      
      ## Debiased Lasso (van de geer et al., 2014)
      ### Complete-case data
      X = X_sim[R1 == 1,]
      Y = Y_sim[R1 == 1]
      delasso_obs = lasso.proj(X, Y, family = "gaussian", standardize = FALSE, 
                               parallel = TRUE, ncores = 10, betainit = "scaled lasso", 
                               return.Z = TRUE, suppress.grouptesting = TRUE, robust = FALSE)
      if(sum(abs(delasso_obs$bhat * x)) > 200){
        flag = 1
        next;
      }
      m_lproj_obs1 = sum(delasso_obs$bhat * x)
      # asym_var_lproj_obs1 = delasso_obs$se[1]/delasso_obs$sigmahat
      Z = delasso_obs$Z
      asym_var_lproj_obs1 = (t(x) %*% diag(1/diag(t(X) %*% Z)) %*% (t(Z) %*% Z) %*% diag(1/diag(t(X) %*% Z)) %*% x)[1,1]
      sigma_hat_lproj_obs1 = delasso_obs$sigmahat
      ci_len_debl_obs1 = 2*qnorm(0.975)*sigma_hat_lproj_obs1*sqrt(asym_var_lproj_obs1)
      
      X = X_sim[R2 == 1,]
      Y = Y_sim[R2 == 1]
      delasso_obs = lasso.proj(X, Y, family = "gaussian", standardize = FALSE, 
                               parallel = TRUE, ncores = 10, betainit = "scaled lasso", 
                               return.Z = TRUE, suppress.grouptesting = TRUE, robust = FALSE)
      if(sum(abs(delasso_obs$bhat * x)) > 200){
        flag = 1
        next;
      }
      m_lproj_obs2 = sum(delasso_obs$bhat * x)
      # asym_var_lproj_obs2 = delasso_obs$se[1]/delasso_obs$sigmahat
      Z = delasso_obs$Z
      asym_var_lproj_obs2 = (t(x) %*% diag(1/diag(t(X) %*% Z)) %*% (t(Z) %*% Z) %*% diag(1/diag(t(X) %*% Z)) %*% x)[1,1]
      sigma_hat_lproj_obs2 = delasso_obs$sigmahat
      ci_len_debl_obs2 = 2*qnorm(0.975)*sigma_hat_lproj_obs2*sqrt(asym_var_lproj_obs2)
      
      ### IPW
      X = (diag(R1/sqrt(obs_prob1)) %*% X_sim)[R1 == 1,]
      Y = (Y_sim * (R1/sqrt(obs_prob1)))[R1 == 1]
      delasso_ipw = lasso.proj(X, Y, family = "gaussian", standardize = FALSE, 
                               parallel = TRUE, ncores = 10, betainit = "scaled lasso", 
                               return.Z = TRUE, suppress.grouptesting = TRUE, robust = FALSE)
      if(sum(abs(delasso_ipw$bhat * x)) > 200){
        flag = 1
        next;
      }
      m_lproj_ipw1 = sum(delasso_ipw$bhat * x)
      # asym_var_lproj_ipw1 = delasso_ipw$se[1]/delasso_ipw$sigmahat
      Z = delasso_ipw$Z
      asym_var_lproj_ipw1 = (t(x) %*% diag(1/diag(t(X) %*% Z)) %*% (t(Z) %*% Z) %*% diag(1/diag(t(X) %*% Z)) %*% x)[1,1]
      sigma_hat_lproj_ipw1 = delasso_ipw$sigmahat
      ci_len_debl_ipw1 = 2*qnorm(0.975)*sigma_hat_lproj_ipw1*sqrt(asym_var_lproj_ipw1)
      
      X = (diag(1/sqrt(obs_prob2)) %*% X_sim)[R2 == 1,]
      Y = (Y_sim * (1/sqrt(obs_prob2)))[R2 == 1]
      delasso_ipw = lasso.proj(X, Y, family = "gaussian", standardize = FALSE, 
                               parallel = TRUE, ncores = 10, betainit = "scaled lasso", 
                               return.Z = TRUE, suppress.grouptesting = TRUE, robust = FALSE)
      if(sum(abs(delasso_ipw$bhat * x)) > 200){
        flag = 1
        next;
      }
      m_lproj_ipw2 = sum(delasso_ipw$bhat * x)
      # asym_var_lproj_ipw2 = delasso_ipw$se[1]/delasso_ipw$sigmahat
      Z = delasso_ipw$Z
      asym_var_lproj_ipw2 = (t(x) %*% diag(1/diag(t(X) %*% Z)) %*% (t(Z) %*% Z) %*% diag(1/diag(t(X) %*% Z)) %*% x)[1,1]
      sigma_hat_lproj_ipw2 = delasso_ipw$sigmahat
      ci_len_debl_ipw2 = 2*qnorm(0.975)*sigma_hat_lproj_ipw2*sqrt(asym_var_lproj_ipw2)
      
      ### Full (oracle) data
      X = X_sim
      Y = Y_sim
      delasso_full = lasso.proj(X, Y, family = "gaussian", standardize = FALSE, 
                                parallel = TRUE, ncores = 10, betainit = "scaled lasso", 
                                return.Z = TRUE, suppress.grouptesting = TRUE, robust = FALSE)
      if(sum(abs(delasso_full$bhat * x)) > 200){
        flag = 1
        next
      }
      m_lproj_full = sum(delasso_full$bhat * x)
      # asym_var_lproj_full = delasso_full$se[1]/delasso_full$sigmahat
      Z = delasso_full$Z
      asym_var_lproj_full = (t(x) %*% diag(1/diag(t(X) %*% Z)) %*% (t(Z) %*% Z) %*% diag(1/diag(t(X) %*% Z)) %*% x)[1,1]
      sigma_hat_lproj_full = delasso_full$sigmahat
      ci_len_debl_full = 2*qnorm(0.975)*sigma_hat_lproj_full*sqrt(asym_var_lproj_full)
    }
    print(Sys.time() - start_time)
    
    lproj_res = data.frame(m_obs1 = m_lproj_obs1, asym_se_obs1 = asym_var_lproj_obs1, 
                           sigma_hat_obs1 = sigma_hat_lproj_obs1, 
                           ci_len_obs1 = ci_len_debl_obs1,
                           m_obs2 = m_lproj_obs2, asym_se_obs2 = asym_var_lproj_obs2, 
                           sigma_hat_obs2 = sigma_hat_lproj_obs2, 
                           ci_len_obs2 = ci_len_debl_obs2,
                           m_ipw1 = m_lproj_ipw1, asym_se_ipw1 = asym_var_lproj_ipw1, 
                           sigma_hat_ipw1 = sigma_hat_lproj_ipw1,
                           ci_len_ipw1 = ci_len_debl_ipw1,
                           m_ipw2 = m_lproj_ipw2, asym_se_ipw2 = asym_var_lproj_ipw2, 
                           sigma_hat_ipw2 = sigma_hat_lproj_ipw2,
                           ci_len_ipw2 = ci_len_debl_ipw2,
                           m_full = m_lproj_full, asym_se_full = asym_var_lproj_full, 
                           sigma_hat_full = sigma_hat_lproj_full, 
                           ci_len_full = ci_len_debl_full)
    write.csv(lproj_res, paste0("./lproj_res/lproj_cirsym_d", d, "_n", n, "_", jobid, "_x", i, "_beta", k, ".csv"), 
              row.names=FALSE)
  }
}
}
