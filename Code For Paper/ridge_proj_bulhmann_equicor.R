library(mvtnorm)
library(hdi)

jobid = commandArgs(TRUE)
print(jobid)

d = 600
n = 500

rho = 0.5
Sigma = rho*array(1, dim = c(d,d)) + (1-rho)*diag(d)
sig = 1

for(i in c(0, 2)){
  if(i == 0){
    ## x0
    x = array(0, dim = c(d,1))
    x[1,1] = 1
  }
  if(i == 2){
    ## x2
    x = array(0, dim = c(d,1))
    x[100,1] = 1
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
      
      
      ## Ridge projection (Buhlmann, 2013)
      ### Complete-case data
      X = X_sim[R1 == 1,]
      Y = Y_sim[R1 == 1]
      deridge_obs = ridge.proj(X, Y, family = "gaussian", standardize = FALSE, 
                               lambda = 1, betainit = "scaled lasso", 
                               suppress.grouptesting = TRUE)
      if(sum(abs(deridge_obs$bhat * x)) > 200){
        flag = 1
        next;
      }
      m_rproj_obs1 = sum(deridge_obs$bhat * x)
      ci_deridge = confint(deridge_obs, level = 0.95)
      if(i == 0){
        ci_len_rproj_obs1 = (ci_deridge[,2] - ci_deridge[,1])[1]
      }else{
        ci_len_rproj_obs1 = (ci_deridge[,2] - ci_deridge[,1])[100]
      }
      
      X = X_sim[R2 == 1,]
      Y = Y_sim[R2 == 1]
      deridge_obs = ridge.proj(X, Y, family = "gaussian", standardize = FALSE, 
                               lambda = 1, betainit = "scaled lasso", 
                               suppress.grouptesting = TRUE)
      if(sum(abs(deridge_obs$bhat * x)) > 200){
        flag = 1
        next;
      }
      m_rproj_obs2 = sum(deridge_obs$bhat * x)
      ci_deridge = confint(deridge_obs, level = 0.95)
      if(i == 0){
        ci_len_rproj_obs2 = (ci_deridge[,2] - ci_deridge[,1])[1]
      }else{
        ci_len_rproj_obs2 = (ci_deridge[,2] - ci_deridge[,1])[100]
      }
      
      ### IPW
      X = (diag(R1/sqrt(obs_prob1)) %*% X_sim)[R1 == 1,]
      Y = (Y_sim * (R1/sqrt(obs_prob1)))[R1 == 1]
      deridge_ipw = ridge.proj(X, Y, family = "gaussian", standardize = FALSE, 
                               lambda = 1, betainit = "scaled lasso", 
                               suppress.grouptesting = TRUE)
      if(sum(abs(deridge_ipw$bhat * x)) > 200){
        flag = 1
        next;
      }
      m_rproj_ipw1 = sum(deridge_ipw$bhat * x)
      ci_deridge = confint(deridge_ipw, level = 0.95)
      if(i == 0){
        ci_len_rproj_ipw1 = (ci_deridge[,2] - ci_deridge[,1])[1]
      }else{
        ci_len_rproj_ipw1 = (ci_deridge[,2] - ci_deridge[,1])[100]
      }
      
      X = (diag(1/sqrt(obs_prob2)) %*% X_sim)[R2 == 1,]
      Y = (Y_sim * (1/sqrt(obs_prob2)))[R2 == 1]
      deridge_ipw = ridge.proj(X, Y, family = "gaussian", standardize = FALSE, 
                               lambda = 1, betainit = "scaled lasso", 
                               suppress.grouptesting = TRUE)
      if(sum(abs(deridge_ipw$bhat * x)) > 200){
        flag = 1
        next;
      }
      m_rproj_ipw2 = sum(deridge_ipw$bhat * x)
      ci_deridge = confint(deridge_ipw, level = 0.95)
      if(i == 0){
        ci_len_rproj_ipw2 = (ci_deridge[,2] - ci_deridge[,1])[1]
      }else{
        ci_len_rproj_ipw2 = (ci_deridge[,2] - ci_deridge[,1])[100]
      }
      
      ### Full (oracle) data
      X = X_sim
      Y = Y_sim
      deridge_full = ridge.proj(X, Y, family = "gaussian", standardize = FALSE, 
                                lambda = 1, betainit = "scaled lasso", 
                                suppress.grouptesting = TRUE)
      if(sum(abs(deridge_full$bhat * x)) > 200){
        flag = 1
        next;
      }
      m_rproj_full = sum(deridge_full$bhat * x)
      ci_deridge = confint(deridge_full, level = 0.95)
      if(i == 0){
        ci_len_rproj_full = (ci_deridge[,2] - ci_deridge[,1])[1]
      }else{
        ci_len_rproj_full = (ci_deridge[,2] - ci_deridge[,1])[100]
      }
    }
    print(Sys.time() - start_time)
    
    rproj_res = data.frame(m_obs1 = m_rproj_obs1, ci_len_obs1 = ci_len_rproj_obs1, 
                           m_obs2 = m_rproj_obs2, ci_len_obs2 = ci_len_rproj_obs2, 
                           m_ipw1 = m_rproj_ipw1, ci_len_ipw1 = ci_len_rproj_ipw1, 
                           m_ipw2 = m_rproj_ipw2, ci_len_ipw2 = ci_len_rproj_ipw2, 
                           m_full = m_rproj_full, ci_len_full = ci_len_rproj_full)
    write.csv(rproj_res, paste0("./rproj_res/rproj_equicor_d", d, "_n", n, "_", jobid, "_x", i, "_beta", k, ".csv"), 
              row.names=FALSE)
  }
}
