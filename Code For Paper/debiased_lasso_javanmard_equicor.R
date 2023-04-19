library(mvtnorm)
library(hdi)
source('sslasso_code/lasso_inference.R')

jobid = commandArgs(TRUE)
print(jobid)

d = 600
n = 500

rho = 0.8
Sigma = rho*array(1, dim = c(d,d)) + (1-rho)*diag(d)
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
    
    # dat = read.csv(paste0('./Data/dat_sim_', i, '_d', d, 'n', n, '.csv'), header = TRUE)
    # X_sim = dat[,1:d]
    # Y_sim = dat[,d+1]
    # R1 = dat[,d+2]
    # R2 = dat[,d+3]
    
    ## Debiased Lasso (Javanmard and Montarani, 2014)
    ### Complete-case data
    X = X_sim[R1 == 1,]
    Y = Y_sim[R1 == 1]
    debl_obs = SSLasso(X, Y, alpha = 0.05, lambda = NULL, mu = 2*sqrt(log(d)/length(Y)), 
                       intercept = FALSE, resol=1.3, maxiter=1000, threshold=1e-3, 
                       verbose = FALSE)
    m_debl_obs1 = sum(debl_obs$unb.coef * x)
    asym_var_debl_obs1 = sqrt(t(x) %*% debl_obs$cov.mat %*% x/n)
    sigma_hat_debl_obs1 = debl_obs$noise.sd
    if(i == 0){
      ci_len_debl_obs1 = (debl_obs$up.lim - debl_obs$low.lim)[1]
    } else if (i == 2){
      ci_len_debl_obs1 = (debl_obs$up.lim - debl_obs$low.lim)[100]
    } else{
      ci_len_debl_obs1 = 2*qnorm(0.975)*sigma_hat_debl_obs1*asym_var_debl_obs1 + 2*t(x) %*% debl_obs$addlength
    }
    
    X = X_sim[R2 == 1,]
    Y = Y_sim[R2 == 1]
    debl_obs = SSLasso(X, Y, alpha = 0.05, lambda = NULL, mu = 2*sqrt(log(d)/length(Y)), 
                       intercept = FALSE, resol=1.3, maxiter=1000, threshold=1e-3, 
                       verbose = FALSE)
    m_debl_obs2 = sum(debl_obs$unb.coef * x)
    asym_var_debl_obs2 = sqrt(t(x) %*% debl_obs$cov.mat %*% x/n)
    sigma_hat_debl_obs2 = debl_obs$noise.sd
    if(i == 0){
      ci_len_debl_obs2 = (debl_obs$up.lim - debl_obs$low.lim)[1]
    } else if (i == 2){
      ci_len_debl_obs2 = (debl_obs$up.lim - debl_obs$low.lim)[100]
    } else{
      ci_len_debl_obs2 = 2*qnorm(0.975)*sigma_hat_debl_obs2*asym_var_debl_obs2 + 2*t(x) %*% debl_obs$addlength
    }
    
    ### IPW
    X = (diag(R1/sqrt(obs_prob1)) %*% X_sim)[R1 == 1,]
    Y = (Y_sim * (R1/sqrt(obs_prob1)))[R1 == 1]
    debl_ipw = SSLasso(X, Y, alpha = 0.05, lambda = NULL, mu = 2*sqrt(log(d)/length(Y)), 
                       intercept = FALSE, resol=1.3, maxiter=1000, threshold=1e-3, 
                       verbose = FALSE)
    m_debl_ipw1 = sum(debl_ipw$unb.coef * x)
    asym_var_debl_ipw1 = sqrt(t(x) %*% debl_ipw$cov.mat %*% x/n)
    sigma_hat_debl_ipw1 = debl_ipw$noise.sd
    if(i == 0){
      ci_len_debl_ipw1 = (debl_ipw$up.lim - debl_ipw$low.lim)[1]
    } else if (i == 2){
      ci_len_debl_ipw1 = (debl_ipw$up.lim - debl_ipw$low.lim)[100]
    } else{
      ci_len_debl_ipw1 = 2*qnorm(0.975)*sigma_hat_debl_ipw1*asym_var_debl_ipw1 + 2*t(x) %*% debl_ipw$addlength
    }
    
    X = (diag(1/sqrt(obs_prob2)) %*% X_sim)[R2 == 1,]
    Y = (Y_sim * (1/sqrt(obs_prob2)))[R2 == 1]
    debl_ipw = SSLasso(X, Y, alpha = 0.05, lambda = NULL, 
                       mu = 2*sqrt(log(d)/length(Y)), intercept = FALSE, 
                       resol=1.3, maxiter=1000, threshold=1e-3, verbose = FALSE)
    m_debl_ipw2 = sum(debl_ipw$unb.coef * x)
    asym_var_debl_ipw2 = sqrt(t(x) %*% debl_ipw$cov.mat %*% x/n)
    sigma_hat_debl_ipw2 = debl_ipw$noise.sd
    if(i == 0){
      ci_len_debl_ipw2 = (debl_ipw$up.lim - debl_ipw$low.lim)[1]
    } else if (i == 2){
      ci_len_debl_ipw2 = (debl_ipw$up.lim - debl_ipw$low.lim)[100]
    } else{
      ci_len_debl_ipw2 = 2*qnorm(0.975)*sigma_hat_debl_ipw2*asym_var_debl_ipw2 + 2*t(x) %*% debl_ipw$addlength
    }
    
    X = X_sim
    Y = Y_sim
    debl_full = SSLasso(X, Y, alpha = 0.05, lambda = NULL, mu = 2*sqrt(log(d)/length(Y)), 
                        intercept = FALSE, resol=1.3, maxiter=1000, 
                        threshold=1e-3, verbose = FALSE)
    m_debl_full = sum(debl_full$unb.coef * x)
    asym_var_debl_full = sqrt(t(x) %*% debl_full$cov.mat %*% x/n)
    sigma_hat_debl_full = debl_full$noise.sd
    if(i == 0){
      ci_len_debl_full = (debl_full$up.lim - debl_full$low.lim)[1]
    } else if (i == 2){
      ci_len_debl_full = (debl_full$up.lim - debl_full$low.lim)[100]
    } else{
      ci_len_debl_full = 2*qnorm(0.975)*sigma_hat_debl_full*asym_var_debl_full + 2*t(x) %*% debl_full$addlength
    }
    print(Sys.time() - start_time)
    
    debl_res = data.frame(m_obs1 = m_debl_obs1, asym_se_obs1 = asym_var_debl_obs1, 
                          sigma_hat_obs1 = sigma_hat_debl_obs1, 
                          ci_len_obs1 = ci_len_debl_obs1, 
                          m_obs2 = m_debl_obs2, asym_se_obs2 = asym_var_debl_obs2, 
                          sigma_hat_obs2 = sigma_hat_debl_obs2, 
                          ci_len_obs2 = ci_len_debl_obs2, 
                          m_ipw1 = m_debl_ipw1, asym_se_ipw1 = asym_var_debl_ipw1, 
                          sigma_hat_ipw1 = sigma_hat_debl_ipw1,
                          ci_len_ipw1 = ci_len_debl_ipw1, 
                          m_ipw2 = m_debl_ipw2, asym_se_ipw2 = asym_var_debl_ipw2, 
                          sigma_hat_ipw2 = sigma_hat_debl_ipw2,
                          ci_len_ipw2 = ci_len_debl_ipw2, 
                          m_full = m_debl_full, asym_se_full = asym_var_debl_full, 
                          sigma_hat_full = sigma_hat_debl_full, 
                          ci_len_full = ci_len_debl_full)
    write.csv(debl_res, paste0("./debl_res/debl_equicor_d", d, "_n", n, "_", jobid, "_x", i, "_beta", k, ".csv"), 
              row.names=FALSE)
  }
}
