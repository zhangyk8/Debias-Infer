
### Codes and functions for implementing the DDR estimation and inference method of Chakrabortty et al. (2019, arXiv:1911.11345) for High-dimensional M-Estimation with Missing Outcomes. ###

### The code can be used for replicating the simulation results for different settings in the paper. The "DDR_est" function (somewhere in the middle of this code) is the main function for implementing
### the estimator (plus inference). But it requires several other functions as well which are defined elsewhere in the document. The function "DDR_data_gen" generates different data settings used for 
### the simulations in the paper. The other file "DDR_est_apply.R" contains code for 1 implementation of the DDR estimator (plus inference) on any given dataset. 
  

#load required package
require(MASS)
require(glmnet)
require(np)
require(stats)
require(mvtnorm)
require(biglm)

#data generation, one copy of data for given n,p and models
DDR_data_gen = function(n,p,s_T,s_Y,model_T,model_y,cov,rho){
  #generate covariance matrix of x, cov is the structure, rho is the diag elements for diag, ratio for auto-regressive and the other
  #elements in the off-diag, s_T, s_Y controls the sparsity of the true coeffcients of T and Y.
  if (cov == "diag"){
    sig <- diag(rep(rho,p))
  }else if(cov == "auto-regress"){
    sig <- matrix(0, p, p)
    sig <- rho^abs(row(sig) - col(sig))
  }else if(cov == "symmetric"){
    sig <- rho*rep(1,p) %*% t(rep(1,p)) + (1 - rho)*diag(rep(1,p))
  }else{
    print("Error: covariance structure not correct")
  }
  x <- mvrnorm(n, rep(0, p), sig)
  
  #model of T (one of "logit", "quad" or "sim", which stands for linear logistic model, logistic model with
  #quadratic terms and single index model)
  if (model_T == "logit"){
    beta_T <- rep(0, p)
    if (s_T == 5){
      beta_T[seq(1,5)] = c(1,-1,0.5,-0.5,0.5)/sqrt(s_T)
    }else if(s_T == 10){
      #beta_T[seq(1,15)] = c(1,1,1,-1,-1,0.5,0.5,-0.5,-0.5,-0.5,0.25,0.25,-0.25,-0.25,-0.25)/sqrt(s_T)
      beta_T[seq(1,10)] = c(1,1,1,-1,-1,0.5,0.5,-0.5,-0.5,-0.5)/sqrt(s_T)
    }else{
      print("Error: S_T not correctly specified")
    }
    int_beta_T = 0.5
    nu_exp <- exp(x%*%beta_T + int_beta_T)
    p_T_ori <- nu_exp/(1+nu_exp)
    p_T = p_T_ori
    p_T[p_T<=0.1]=0.1
    p_T[p_T>=0.9]=0.9
    pi_trunc = sum(p_T_ori <=0.1) + sum(p_T_ori >=0.9)

    
    T <- rbinom(n, 1, p_T)

  }else if (model_T == "quad"){
    beta_T <- rep(0, 2*p)
    if (s_T == 5){
      beta_T[seq(1,5)] = c(1,-1,0.5,-0.5,0.5)/sqrt(s_T)
      beta_T[seq(1 + p,2 +p)] = c(0.25,-0.25)
      
    }else if(s_T == 10){
      #beta_T[seq(1,15)] = c(1,1,1,-1,-1,0.5,0.5,-0.5,-0.5,-0.5,0.25,0.25,-0.25,-0.25,-0.25)/sqrt(s_T)
      beta_T[seq(1,10)] = c(1,1,1,-1,-1,0.5,0.5,-0.5,-0.5,-0.5)/sqrt(s_T)
      beta_T[seq(1+p,4+p)] = c(0.25,0.25,-0.25,-0.25)
    }else{
      print("Error: S_T not correctly specified")
    }
    int_beta_T = 0.5
    
    x_use = cbind(x,x*x)
    nu_exp <- exp(x_use%*%beta_T + int_beta_T)
    p_T_ori <- nu_exp/(1+nu_exp)
    
    p_T = p_T_ori
    p_T[p_T<=0.1]=0.1
    p_T[p_T>=0.9]=0.9
    pi_trunc = sum(p_T_ori <=0.1) + sum(p_T_ori >=0.9)
    
  
    T <- rbinom(n, 1, p_T)
  }else if (model_T == "sim"){
    beta_T <- rep(0,p)
    if (s_T == 5){
      beta_T[seq(1,5)] = c(1,-1,0.5,-0.5,0.5)/sqrt(s_T)
    }else if(s_T == 10){
      #beta_T[seq(1,15)] = c(1,1,1,-1,-1,0.5,0.5,-0.5,-0.5,-0.5,0.25,0.25,-0.25,-0.25,-0.25)/sqrt(s_T)
      beta_T[seq(1,10)] = c(1,1,1,-1,-1,0.5,0.5,-0.5,-0.5,-0.5)/sqrt(s_T)
    }else{
      print("Error: S_T not correctly specified")
    }
    int_beta_T = 0.5
    
    nu_exp <- exp((x%*%beta_T + int_beta_T) + 0.2*(x%*%beta_T)^2)
    p_T_ori <- nu_exp/(1+nu_exp)
    
    p_T = p_T_ori
    p_T[p_T<=0.1]=0.1
    p_T[p_T>=0.9]=0.9
    pi_trunc = sum(p_T_ori <=0.1) + sum(p_T_ori >=0.9)
    
    #summary(p_T_ori)
    T <- rbinom(n, 1, p_T)
    
  }else{
    print("Error: Model for T is not correctly specified, should be one of 'logit', 'quad' or 'sim'")
  }
  #model of y (one of "linear", "quad" or "sim", which stands for linear model, linear model with
  #quadratic terms and single index model)
  if (model_y == "linear"){
    beta_y <- rep(0, p)
    if (s_Y == 10){
      beta_y[seq(1,10)] = c(1,1,1,-1,-1,0.5,0.5,-0.5,-0.5,-0.5)
    }else if(s_Y == 20){
      #beta_y[seq(1,25)] = c(1,1,1,-1,-1,rep(0.5,5),rep(-0.5,5),rep(0.25,5),rep(-0.25,5))
      beta_y[seq(1,20)] = c(1,1,1,-1,-1,rep(0.5,5),rep(-0.5,5),-0.25,-0.25,0.25,0.25,0.25)
      
    }else{
      print("Error: S_T not correctly specified")
    }
    int_beta_y = 1
    e <- rnorm(n,0,1)
    m <- x%*%beta_y + int_beta_y
    y <- m + e
    
  }else if (model_y == "quad"){
    beta_y <- rep(0, 2*p)
    if (s_Y == 10){
      beta_y[seq(1,10)] = c(1,1,1,-1,-1,0.5,0.5,-0.5,-0.5,-0.5)
      #beta_y[seq(1+p,5+p)] = c(0.5,-0.5,0.25,0.25,-0.25) 9.16 version
      beta_y[seq(1+p,5+p)] = c(1,-1,0.5,0.5,-0.5)
    }else if(s_Y == 20){
      #beta_y[seq(1,25)] = c(1,1,1,-1,-1,rep(0.5,5),rep(-0.5,5),rep(0.25,5),rep(-0.25,5))
      beta_y[seq(1,20)] = c(1,1,1,-1,-1,rep(0.5,5),rep(-0.5,5),-0.25,-0.25,0.25,0.25,0.25)
      #beta_y[seq(1+p,5+p)] = c(0.5,-0.5,0.25,0.25,-0.25) 9.16 version
      beta_y[seq(1+p,5+p)] = c(1,-1,0.5,0.5,-0.5) 
    }else{
      print("Error: S_T not correctly specified")
    }
    
    x_use = cbind(x,x*x)
    
    int_beta_y = 1
    e <- rnorm(n,0,1)
    m <- x_use%*%beta_y + int_beta_y
    y <- m + e
    
  }else if (model_y == "sim"){
    beta_y <- rep(0, p)
    if (s_Y == 10){
      beta_y[seq(1,10)] = c(1,1,1,-1,-1,0.5,0.5,-0.5,-0.5,-0.5)
    }else if(s_Y == 20){
      #beta_y[seq(1,25)] = c(1,1,1,-1,-1,rep(0.5,5),rep(-0.5,5),rep(0.25,5),rep(-0.25,5))
      beta_y[seq(1,20)] = c(1,1,1,-1,-1,rep(0.5,5),rep(-0.5,5),-0.25,-0.25,0.25,0.25,0.25)
    }else{
      print("Error: S_T not correctly specified")
    }
    int_beta_y = 1
    e <- rnorm(n,0,1)
    temp = eigen(sig)
    #m <- (x%*%beta_y + int_beta_y) +  0.3 * (x%*%beta_y)^2 
    
    ### latest version updated on 4/12/2019
    m <- (x%*%beta_y + int_beta_y) +  0.3/sqrt(max(temp$values)) * (x%*%beta_y)^2 
    
    
    ### latest version updated on 4/12/2019
    # try a new version of sim model updated on 1/15/2019
    #m <- 0.3 * (x%*%beta_y + int_beta_y + e)^2
    # try a new version of sim model updated on 2/6/2019
    #m <- 0.2 * (x%*%beta_y + int_beta_y + e)^3
    
    y <- m + e
    
    
  }else{
    print("Error: Model for y is not correctly specified, should be one of 'linear', 'quad' or 'sim'")
  }
  
  summary_pi_ori = c(max(p_T_ori),min(p_T_ori),mean(p_T_ori),median(p_T_ori))
  #output as a list
  list(x=x, y=y, T=T,pi = p_T,m = m,beta_T = c(int_beta_T,beta_T),beta_y = c(int_beta_y,beta_y),summary_pi_ori = summary_pi_ori,pi_trunc = pi_trunc)
}

#data generation, multiple copies for given n,p and models
#model_T should be one of one of "logit", "quad" or "sim", model_y should be one of "linear", "quad" or "sim"
DDR_data = function(nsim,seed = 1234,n,p,s_T,s_Y,model_T,model_y,cov,rho,save){
  #generate data 
  set.seed(seed) 
  data <- vector("list", nsim)
  pi_trunc = rep(0,nsim)
  for (i in 1:nsim) {
    data[[i]] <- DDR_data_gen(n,p,s_T,s_Y,model_T,model_y,cov,rho)
    pi_trunc[i] <- data[[i]]$pi_trunc
    print(i)
  }
  
  beta_T = data[[1]]$beta_T
  beta_y = data[[1]]$beta_y 
  
  #compute the oracle parameter  #this is the target parameter that we estimate
  #linear
  
  if (p <=50){
    N = 500000
    data_temp= DDR_data_gen(N,p,s_T,s_Y,model_T,model_y,cov,rho)
    lm_temp = lm(data_temp$y~data_temp$x)
    oracle_linear = lm_temp$coefficients  #this is a p+1 dimensional vector including intercept
  }else{
    N = 200000
    data_temp= DDR_data_gen(N,p,s_T,s_Y,model_T,model_y,cov,rho)
    lm_temp = biglm(y~x, data_temp)
    oracle_linear = coefficients(lm_temp) #this is a p+1 dimensional vector including intercept
  }
 
  
  
  #mean of y
  oracle_mean = mean(data_temp$y)
  
  basic_infor = list(model = paste(paste("model of T:", model_T),paste(";model of y:", model_y),paste(";covariance of x:", cov),paste(";rho,n and p:", rho,n,p,sep=" ")))
  
  if (save == "T"){
    file_name <- paste("data_T_",model_T,"_y_",model_y,"_cov_",cov,"_rho_",rho,"_",paste(n, p,sep="_"), ".RData", sep="")
    save(data, oracle_linear, oracle_mean,beta_T,beta_y,basic_infor =basic_infor,file=paste("data/", file_name, sep=""))  
  }else{
    return(list(data = data,oracle_linear = oracle_linear,
                oracle_mean = oracle_mean,beta_T = beta_T,beta_y = beta_y,basic_infor = basic_infor,pi_trunc = pi_trunc))
  }
 
}

#function for finding model with smallest BIC
bic.opt <- function(obj,n){ 
  bic = deviance(obj) + log(n)*obj$df; ind = which.min(bic); 
  bic.list <- list() 
  bic.list$lam.opt = obj$lambda[ind]; 
  bic.list$coeff = coef(obj)[,ind]
  return(bic.list)
}

#function for finding model with smallest AIC
aic.opt <- function(obj){ 
  aic = deviance(obj) + 2*obj$df; ind = which.min(aic); 
  aic.list <- list() 
  aic.list$lam.opt = obj$lambda[ind]; 
  aic.list$coeff = coef(obj)[,ind]
  return(aic.list)
}

#estimation of propensity score
#method should be one of "logit", "quad" or "sim"
#tuning should be one of "CV-AUC","CV-deviance","AIC","BIC"
#if method is sim, tuning is the tuning parameter selection method for SIM
DDR_pi = function(x,T,n,method,tuning,x_new){
  if (method == "logit"){
    if (tuning == "CV-AUC"){
      cvfit = cv.glmnet(x, T, family = "binomial", type.measure = "auc")
      beta_T = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
    }else if (tuning == "CV-deviance"){
      cvfit = cv.glmnet(x, T, family = "binomial", type.measure = "deviance") 
      beta_T = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
    }else if (tuning == "AIC"){
      fit = glmnet(x, T, family = "binomial")
      AIC = aic.opt(fit)
      beta_T = AIC$coeff
    }else if (tuning == "BIC"){
      fit = glmnet(x, T, family = "binomial")
      BIC = bic.opt(fit,n)
      beta_T = BIC$coeff
    }else{
      print("Error: method for tuning parameter selection is not avaliable")
    }
    nu_hat <- exp(x_new%*%beta_T[-1] + beta_T[1])
    pi_hat <- nu_hat/(1+nu_hat)
  }else if (method == "quad"){
    x_use = cbind(x,x*x)
    if (tuning == "CV-AUC"){
      cvfit = cv.glmnet(x_use, T, family = "binomial", type.measure = "auc")
      beta_T = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
    }else if (tuning == "CV-deviance"){
      cvfit = cv.glmnet(x_use, T, family = "binomial", type.measure = "deviance") 
      beta_T = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
    }else if (tuning == "AIC"){
      fit = glmnet(x_use, T, family = "binomial")
      AIC = aic.opt(fit)
      beta_T = AIC$coeff
    }else if (tuning == "BIC"){
      fit = glmnet(x_use, T, family = "binomial")
      BIC = bic.opt(fit,n)
      beta_T = BIC$coeff
    }else{
      print("Error: method for tuning parameter selection is not avaliable")
    }
    x_new_use <- cbind(x_new,x_new*x_new)
    nu_hat <- exp(x_new_use%*%beta_T[-1] + beta_T[1])
    pi_hat <- nu_hat/(1+nu_hat)
  }else if (method == "sim"){
    #linear part in single index model
    if (tuning == "CV-AUC"){
      cvfit = cv.glmnet(x, T, family = "binomial", type.measure = "auc")
      beta_T = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
    }else if (tuning == "CV-deviance"){
      cvfit = cv.glmnet(x, T, family = "binomial", type.measure = "deviance") 
      beta_T = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
    }else if (tuning == "AIC"){
      fit = glmnet(x, T, family = "binomial")
      AIC = aic.opt(fit)
      beta_T = AIC$coeff
    }else if (tuning == "BIC"){
      fit = glmnet(x, T, family = "binomial")
      BIC = bic.opt(fit,n)
      beta_T = BIC$coeff
    }else{
      print("Error: method for tuning parameter selection of the SIM is not avaliable")
    }
    #funcional part in single index model
    nmulti =1
    
    if (sum(abs(beta_T[-1]))<= 0.1){
      nu_hat <- exp(beta_T[1])
      pi_hat <- nu_hat/(1+nu_hat)
    }else{
    
    
    if (n>1000){
      temp_id = sample(n,1000)
      nu_hat = x[temp_id,]%*%beta_T[-1]
      np.bw = npcdensbw(xdat=nu_hat,ydat=as.vector(T[temp_id]),bwmethod="cv.ml",bwtype="fixed",cxkertype="gaussian",cxkerorder=2,uykertype="aitchisonaitken",nmulti=nmulti)
      bw.opt = np.bw$xbw
      bw.opt = (bw.opt/(1000)^-0.2)*(n)^0.2
    }else{
      nu_hat = x%*%beta_T[-1]
      np.bw = npcdensbw(xdat=nu_hat,ydat=as.vector(T),bwmethod="cv.ml",bwtype="fixed",cxkertype="gaussian",cxkerorder=2,uykertype="aitchisonaitken",nmulti=nmulti)
      bw.opt = np.bw$xbw
    }
    
    nu_hat = x%*%beta_T[-1]
    nu_hat_new = x_new%*%beta_T[-1]
    g_fun = npreg(txdat =nu_hat,tydat = as.vector(T),bws=bw.opt,exdat = nu_hat_new)

    pi_hat = g_fun$mean
    }
  }else{
    print("Error: Working model is not avaliable ")
  }
  return(pi_hat)
}


#estimation of conditional mean 
#method should be one of "linear", "quad" or "sim"
#tuning should be one of "CV-MSE","AIC","BIC"
#if method is sim, tuning is the tuning parameter selection method for SIM
#if method is sim, need special argument for estimating the linear part: "Lasso" for using Lasso on complete data,
#"ora_w_Lasso" for weighted lasso using oracle propensity score and "est_w_Lasso" for using estimated prepensity score
#if using the later two method, additional input pi is the oracle ps or estimated ps.
DDR_m = function(x,T,y,method,tuning,sim_method,pi,x_new,beta_ora){  #beta_ora should include intercept
  #observed data
  x.ob = x[T==1,]
  y.ob = y[T==1]
  pi = as.vector(pi)
  n = length(y.ob)
  if (method == "linear"){
    if (tuning == "CV-MSE"){
      cvfit = cv.glmnet(x.ob, y.ob, family = "gaussian", type.measure = "mse")
      beta_m = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
    }else if (tuning == "AIC"){
      fit = glmnet(x.ob, y.ob, family = "gaussian")
      AIC = aic.opt(fit)
      beta_m = AIC$coeff
    }else if (tuning == "BIC"){
      fit = glmnet(x.ob, y.ob, family = "gaussian")
      BIC = bic.opt(fit,n)
      beta_m = BIC$coeff
    }else{
      print("Error: method for tuning parameter selection is not avaliable")
    }
   m_hat = x_new%*%beta_m[-1] + beta_m[1]
    
  }else if (method == "quad"){
     x.ob_use = cbind(x.ob,x.ob*x.ob)  
    if (tuning == "CV-MSE"){
      cvfit = cv.glmnet(x.ob_use, y.ob, family = "gaussian", type.measure = "mse")
      beta_m = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
    }else if (tuning == "AIC"){
      fit = glmnet(x.ob_use, y.ob, family = "gaussian")
      AIC = aic.opt(fit)
      beta_m = AIC$coeff
    }else if (tuning == "BIC"){
      fit = glmnet(x.ob_use, y.ob, family = "gaussian")
      BIC = bic.opt(fit,n)
      beta_m = BIC$coeff
    }else{
      print("Error: method for tuning parameter selection is not avaliable")
    }
    x_new_use <- cbind(x_new,x_new*x_new)
    m_hat = x_new_use%*%beta_m[-1] + beta_m[1]
  }else if (method == "sim"){
    #linear part in single index model
    if (sim_method == "Lasso"){
      cvfit = cv.glmnet(x.ob, y.ob, family = "gaussian",type.measure = "mse")
      beta_m = as.vector(coef(cvfit, s = "lambda.min"))
    }else if (sim_method == "ora_w_Lasso"){
      pi_use = pi[T==1]
      cvfit = cv.glmnet(x.ob, y.ob, family = "gaussian",weights = pi_use^-1, type.measure = "mse")
      beta_m = as.vector(coef(cvfit, s = "lambda.min"))
    }else if (sim_method == "est_w_Lasso"){
      pi_use = pi[T==1]
      cvfit = cv.glmnet(x.ob, y.ob, family = "gaussian",weights = pi_use^-1, type.measure = "mse")
      beta_m = as.vector(coef(cvfit, s = "lambda.min"))
    }else if (sim_method == "beta_ori"){
      beta_m = beta_ora
    }
    
    #funcional part in single index model
    if (sum(abs(beta_m[-1]))<= 0.1){
      m_hat=beta_m[1]
    }else{ 
  
    start_time = Sys.time()
    nmulti =1
    n_temp = length(y.ob)
    if (n>1000){
      temp_id = sample(n_temp,1000)
      nu_hat = x.ob[temp_id,]%*%beta_m[-1]
      np.bw = npregbw(xdat=nu_hat,ydat=as.vector(y.ob[temp_id]),bwmethod="cv.ls",nmulti=nmulti)  
      bw.opt = np.bw$bw;
      bw.opt = (bw.opt/(1000)^-0.2)*(n)^0.2
    }else{
      nu_hat = x.ob%*%beta_m[-1]
      np.bw = npregbw(xdat=nu_hat,ydat=as.vector(y.ob),bwmethod="cv.ls",nmulti=nmulti)  
      bw.opt = np.bw$bw;
    }
    
    nu_hat = x.ob%*%beta_m[-1]
    nu_hat_new = x_new%*%beta_m[-1]
    g_fun = npreg(txdat=nu_hat,tydat=as.vector(y.ob),bws=bw.opt,exdat=nu_hat_new)
    m_hat = g_fun$mean
    
    end_time = Sys.time()
    print(paste("Time for single index model:",end_time - start_time))
    }
    
  }else{
    print("Error: Working model not avaliable ")
  }
  return(m_hat)
}


#Diased doubly robust estimation for given data x,y,T
#function for calculating the DDR estimator
#model_T,model_y specify the working model for y and T given x
#model_theta specifies the model of the target parameter, should be one of "linear", "mean"
#tuning_T, tuning_T are the methods of tuning parameter in function DDR_pi
#tuning_m, tuning_m are the methods of tuning parameter in function DDR_m
#tuning_theta is the method of tuning parameter for outcome; if model_theta is linear, one of "CV-MSE","AIC","BIC"
#split: argument for using sample spliting in m. Input should be "True" or "False"
#infer: "True" or "False" for providing inference results
#cov_est: method for estimating the precision matrix:  "node" for node-wise lasso "inv" for taking inver directly
DDR_est = function(x,y,T,model_T,tuning_T,model_m,tuning_m,sim_method_m,pi_ora,model_theta,tuning_theta,split,K,infer,cov_est,beta_ora){
  set.seed(2222)
  if (split == "True"){
    #creat split samples:
    id = seq(1:nrow(y))
    id<-id[sample(nrow(y))]
    y_pseudo = rep(0,length(y))
    #Create K equally size folds
    folds <- cut(seq(1,nrow(y)),breaks=K,labels=FALSE)
    pi_hat = DDR_pi(x,T,nrow(y),model_T,tuning_T,x)
    
    if (length(pi_hat) ==1){
      print("Error: Single index model for pi is not working")
    }else{
      
    m_hat_full = pi_hat
    check = TRUE
    for (i in 1:K){
      id_test = which(folds == i,arr.ind = TRUE)
      id_test = id[id_test]
      train_x = x[-id_test,]
      train_y = y[-id_test]
      train_T = T[-id_test]
      test_x = x[id_test,]
      test_y = y[id_test]
      test_T = T[id_test]
      
      #cross-fitted method
      #pi_hat
      # pi using all folds, no spliting
      #m_hat
      if (model_m == "sim"){
        if (sim_method_m == "ora_w_Lasso"){
          pi_use = pi_ora[-id_test]
          m_hat = DDR_m(train_x,train_T,train_y,model_m,tuning_m,sim_method_m,pi_use,test_x,beta_ora)
          if (length(m_hat) ==1){
            check = FALSE
          }
        }else if (sim_method_m== "est_w_Lasso"){
          pi_use = pi_hat[-id_test]
          m_hat = DDR_m(train_x,train_T,train_y,model_m,tuning_m,sim_method_m,pi_use,test_x,beta_ora)
          if (length(m_hat) ==1){
            check = FALSE
          }
        }else if (sim_method_m== "beta_ori"){
          m_hat = DDR_m(train_x,train_T,train_y,model_m,tuning_m,sim_method_m,pi=NA,test_x,beta_ora)
          if (length(m_hat) ==1){
            check = FALSE
          }
        }
      }else{
        m_hat = DDR_m(train_x,train_T,train_y,model_m,tuning_m,sim_method = NA,pi=NA,test_x,beta_ora)
        if (length(m_hat) ==1){
          check = FALSE
        }
      }
      #creat pseudo outcome in the i-th fold
      if (check == TRUE){
      y_pseudo[id_test] = m_hat + (T[id_test]/pi_hat[id_test]) * (y[id_test] - m_hat)
      m_hat_full[id_test] = m_hat
      }
    }
    }
    
  }else if (split == "False"){
    n = length(y)   # y has to be one dimensional
    pi_hat = DDR_pi(x,T,n,model_T,tuning_T,x)
    y_pseudo = rep(0,length(y))
    if (length(pi_hat) ==1){
      print("Error: Single index model for pi is not working")
    }else{
      
    check = TRUE
    
    if (model_m == "sim"){
      if (sim_method_m == "ora_w_Lasso"){
        m_hat = DDR_m(x,T,y,model_m,tuning_m,sim_method_m,pi_ora,x,beta_ora)
        if (length(m_hat) ==1){
          check = FALSE
        }
      }else if (sim_method_m== "est_w_Lasso"){
        m_hat = DDR_m(x,T,y,model_m,tuning_m,sim_method_m,pi_hat,x,beta_ora)
        if (length(m_hat) ==1){
          check = FALSE
        }
      }else if (sim_method_m== "beta_ori"){
        m_hat = DDR_m(x,T,y,model_m,tuning_m,sim_method_m,pi_hat,x,beta_ora)
        if (length(m_hat) ==1){
          check = FALSE
        }
      }else{
        m_hat = DDR_m(x,T,y,model_m,tuning_m,sim_method_m,pi = NA,x,beta_ora)
        if (length(m_hat) ==1){
          check = FALSE
        }
      }
    }else{
      m_hat = DDR_m(x,T,y,model_m,tuning_m,sim_method = NA,pi= NA,x,beta_ora)
      if (length(m_hat) ==1){
        check = FALSE
      }
    }
    if (check == TRUE){
      m_hat_full = m_hat
      y_pseudo = m_hat + (T/pi_hat) * (y - m_hat)  
    }
    
    }
  }else{
    print("Error: Wrong argument for split. Should be either 'True' or 'False'")
  }
  
  if (check == TRUE){
    n = length(y_pseudo)
    
    if (model_theta == "linear"){
      if (tuning_theta == "CV-MSE"){
        cvfit = cv.glmnet(x, y_pseudo, family = "gaussian", type.measure = "mse")
        theta = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
      }else if (tuning_theta == "AIC"){
        fit = glmnet(x, y_pseudo, family = "gaussian")
        AIC = aic.opt(fit)
        theta = AIC$coeff
      }else if (tuning_theta == "BIC"){
        fit = glmnet(x, y_pseudo, family = "gaussian")
        BIC = bic.opt(fit,n)
        theta = BIC$coeff
      }else{
        print("Error: method for tuning parameter selection is not avaliable")
      }
      
    }else if (model_theta == "mean"){
      
    }else{
      
    }
  }
  
  
  if (infer == "False"){
    #return theta if not providing confidence intervals
    if (check == TRUE){
      return(list(theta = theta,pi_hat = pi_hat,m_hat_full = m_hat_full,y_pseudo = y_pseudo))   # p+1 dim vector with intercept
    }else{
      return(list(theta = 0))  
    }
    
    
  }else if (infer == "True"){
    if (check == TRUE){
      start_time = Sys.time()
      x_one = cbind(rep(1,length(y)),x)
      
      if (cov_est == "inv"){
        #using matrix inverse
        Omega_hat = solve(1/length(y)*t(x_one)%*%x_one)
        #construct debiased estimates
        temp = (y_pseudo - x_one %*% theta)
        res = 0
        for (i in 1:length(y)){
          res = res + temp[i] * x_one[i,]
        }
        
        theta_de = theta + 1/length(y)*Omega_hat%*%res
        
        # variance estimator
        theta_l = rep(0,length(theta))
        theta_u = rep(0,length(theta))
        IF = matrix(0,length(y),length(theta))
        for (i in 1:length(y)){
          temp = Omega_hat%*%x_one[i,]*(m_hat_full[i] - sum(x_one[i,] %*% theta) + T[i]/pi_hat[i]*(y[i] - m_hat_full[i]))
          IF[i,] = temp 
        }
        
        for (j in 1:length(theta)){
          theta_l[j] = theta_de[j] - 1/sqrt(length(y))*1.96*sqrt(1/length(y)*t(IF[,j])%*%IF[,j]) 
          theta_u[j] = theta_de[j] + 1/sqrt(length(y))*1.96*sqrt(1/length(y)*t(IF[,j])%*%IF[,j]) 
        }  
        
        #return a p+1 dimensional vector
      }else if (cov_est == "node"){
        y_center = y_pseudo - mean(y_pseudo)
        x_center = scale(x,scale = FALSE)
        
        #using node-wise lasso
        Omega_hat = matrix(0,dim(x_center)[2],dim(x_center)[2])
        C_hat = matrix(0,dim(x_center)[2],dim(x_center)[2])
        T2_hat = matrix(0,dim(x_center)[2],dim(x_center)[2])
        for (j in 1:dim(x_center)[2]){
          #obtain each column of Omega_hat
          x_j = x_center[,j]
          x_other = x_center[,-j]
          cvfit = cv.glmnet(x_other, x_j, family = "gaussian", type.measure = "mse")
          gamma = as.vector(coef(cvfit, s = "lambda.min"))
          lam_opt = cvfit$lambda.min 
          tau = x_j - x_other %*% gamma[-1] - gamma[1]
          tau = 1/length(y)*sum(tau^2) + lam_opt*sum(abs(gamma))
          
          T2_hat[j,j] = tau
          
          C_hat[j,-j] = -gamma[-1] 
          C_hat[j,j] = 1
          #print(j)
        }
        Omega_hat = solve(T2_hat) %*% C_hat 
        
        temp = (y_center - x_center %*% theta[-1])
        res = 0
        for (i in 1:length(y)){
          res = res + temp[i] * x_center[i,]
        }
        
        theta_de = theta[-1] + 1/n*Omega_hat%*%res
        
        # variance estimator
        theta_l = rep(0,length(theta)-1)
        theta_u = rep(0,length(theta)-1)
        IF = matrix(0,length(y),length(theta) - 1)
        for (i in 1:length(y)){
          temp = Omega_hat%*%x_center[i,]*(m_hat_full[i] - sum(x_one[i,] %*% theta) + T[i]/pi_hat[i]*(y[i] - m_hat_full[i]))
          IF[i,] = temp 
        }
        
        for (j in 1:(length(theta)-1)){
          theta_l[j] = theta_de[j] - 1/sqrt(length(y))*1.96*sqrt(1/length(y)*t(IF[,j])%*%IF[,j]) 
          theta_u[j] = theta_de[j] + 1/sqrt(length(y))*1.96*sqrt(1/length(y)*t(IF[,j])%*%IF[,j]) 
        }  
        
        
      }else if(cov_est == "ind"){ 
        y_center = y_pseudo - mean(y_pseudo)
        x_center = scale(x,scale = FALSE)
        
        Omega_hat = diag(rep(rho,p))
        temp = (y_center - x_center %*% theta[-1])
        res = 0
        for (i in 1:length(y)){
          res = res + temp[i] * x_center[i,]
        }
        
        theta_de = theta[-1] + 1/n*Omega_hat%*%res
        
        # variance estimator
        theta_l = rep(0,length(theta)-1)
        theta_u = rep(0,length(theta)-1)
        IF = matrix(0,length(y),length(theta) - 1)
        for (i in 1:length(y)){
          temp = Omega_hat%*%x_center[i,]*(m_hat_full[i] - sum(x_one[i,] %*% theta) + T[i]/pi_hat[i]*(y[i] - m_hat_full[i]))
          IF[i,] = temp 
        }
        
        for (j in 1:(length(theta)-1)){
          theta_l[j] = theta_de[j] - 1/sqrt(length(y))*1.96*sqrt(1/length(y)*t(IF[,j])%*%IF[,j]) 
          theta_u[j] = theta_de[j] + 1/sqrt(length(y))*1.96*sqrt(1/length(y)*t(IF[,j])%*%IF[,j]) 
        }  
      }else if(cov_est == "auto"){
        y_center = y_pseudo - mean(y_pseudo)
        x_center = scale(x,scale = FALSE)
        
        #using node-wise lasso
      
        sig <- matrix(0, p, p)
        sig <- rho^abs(row(sig) - col(sig))
        
        
        Omega_hat = solve(sig)
        
        temp = (y_center - x_center %*% theta[-1])
        res = 0
        for (i in 1:length(y)){
          res = res + temp[i] * x_center[i,]
        }
        
        theta_de = theta[-1] + 1/n*Omega_hat%*%res
        
        # variance estimator
        theta_l = rep(0,length(theta)-1)
        theta_u = rep(0,length(theta)-1)
        IF = matrix(0,length(y),length(theta) - 1)
        for (i in 1:length(y)){
          temp = Omega_hat%*%x_center[i,]*(m_hat_full[i] - sum(x_one[i,] %*% theta) + T[i]/pi_hat[i]*(y[i] - m_hat_full[i]))
          IF[i,] = temp 
        }
        
        for (j in 1:(length(theta)-1)){
          theta_l[j] = theta_de[j] - 1/sqrt(length(y))*1.96*sqrt(1/length(y)*t(IF[,j])%*%IF[,j]) 
          theta_u[j] = theta_de[j] + 1/sqrt(length(y))*1.96*sqrt(1/length(y)*t(IF[,j])%*%IF[,j]) 
        }  
      }else if(cov_est == "symm"){
        y_center = y_pseudo - mean(y_pseudo)
        x_center = scale(x,scale = FALSE)
        
        
       
        sig <- rho*rep(1,p) %*% t(rep(1,p)) + (1 - rho)*diag(rep(1,p))
        Omega_hat = solve(sig) 
        
        temp = (y_center - x_center %*% theta[-1])
        res = 0
        for (i in 1:length(y)){
          res = res + temp[i] * x_center[i,]
        }
        
        theta_de = theta[-1] + 1/n*Omega_hat%*%res
        
        # variance estimator
        theta_l = rep(0,length(theta)-1)
        theta_u = rep(0,length(theta)-1)
        IF = matrix(0,length(y),length(theta) - 1)
        for (i in 1:length(y)){
          temp = Omega_hat%*%x_center[i,]*(m_hat_full[i] - sum(x_one[i,] %*% theta) + T[i]/pi_hat[i]*(y[i] - m_hat_full[i]))
          IF[i,] = temp 
        }
        
        for (j in 1:(length(theta)-1)){
          theta_l[j] = theta_de[j] - 1/sqrt(length(y))*1.96*sqrt(1/length(y)*t(IF[,j])%*%IF[,j]) 
          theta_u[j] = theta_de[j] + 1/sqrt(length(y))*1.96*sqrt(1/length(y)*t(IF[,j])%*%IF[,j]) 
        }  
      }
      else{
        print("Error: Wrong method for estimating the precision matrix")
      }
      
      end_time = Sys.time()
      print(paste("Time for inference:",end_time - start_time))
      
      # return theta together with its upper and lower confidence intervals and the estimated precision matrix. 
      return(list(theta = theta,theta_de = theta_de, theta_l = theta_l,
                  theta_u = theta_u, Omega_hat = Omega_hat,IF = IF,m_hat_full = m_hat_full))
    }else{
      return(list(theta = 0))  
    }
    
  }else{
    print("Error: Wrong input for 'infer'")
  }

}

#model assuming pi is known for all subjects
DDR_est_pi_oracle = function(x,y,T,model_m,tuning_m,sim_method_m,pi_ora,model_theta,tuning_theta,split,K){
  set.seed(2222)
  if (split == "True"){
    #creat split samples:
    id = seq(1:nrow(y))
    id<-id[sample(nrow(y))]
    y_pseudo = rep(0,length(y))
    #Create 10 equally size folds
    folds <- cut(seq(1,nrow(y)),breaks=K,labels=FALSE)
    for (i in 1:K){
      id_test = which(folds == i,arr.ind = TRUE)
      id_test = id[id_test]
      train_x = x[-id_test,]
      train_y = y[-id_test]
      train_T = T[-id_test]
      test_x = x[id_test,]
      test_y = y[id_test]
      test_T = T[id_test]
      n_temp = length(train_y)   # y has to be one dimensional
      
      #cross-fitted method
      #m_hat
      if (model_m == "sim"){
        if (sim_method_m == "ora_w_Lasso"){
          pi_use = pi_ora[-id_test]
          m_hat = DDR_m(train_x,train_T,train_y,model_m,tuning_m,sim_method_m,pi_use,test_x,beta_ora)
        }else if (sim_method_m == "beta_ori"){
          m_hat = DDR_m(train_x,train_T,train_y,model_m,tuning_m,sim_method_m,pi= NA,test_x,beta_ora)
        }else{
          m_hat = DDR_m(train_x,train_T,train_y,model_m,tuning_m,sim_method_m,pi = NA,test_x,beta_ora)
        }
      }else{
        m_hat = DDR_m(train_x,train_T,train_y,model_m,tuning_m,sim_method = NA,pi =NA ,test_x,beta_ora)
      }
      #creat pseudo outcome in the i-th fold
      y_pseudo[id_test] = m_hat + (T[id_test]/pi_ora[id_test]) * (y[id_test] - m_hat)
      
    }
    
    
    
  }else if (split == "False"){
    n = length(y)   # y has to be one dimensional
    if (model_m == "sim"){
    if (sim_method_m == "ora_w_Lasso"){
      m_hat = DDR_m(x,T,y,model_m,tuning_m,sim_method_m,pi_ora,x)
    }else{
      m_hat = DDR_m(x,T,y,model_m,tuning_m,sim_method_m,pi= NA,x)
    }
    }else{
      m_hat = DDR_m(x,T,y,model_m,tuning_m,sim_method = NA,pi = NA,x)
    }
    y_pseudo = m_hat + (T/pi_ora) * (y - m_hat)
  }else{
    print("Error: Wrong argument for split. Should be either 'True' or 'False'")
  }
  
  n = length(y_pseudo)
  
  if (model_theta == "linear"){
    if (tuning_theta == "CV-MSE"){
      cvfit = cv.glmnet(x, y_pseudo, family = "gaussian", type.measure = "mse")
      theta = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
    }else if (tuning_theta == "AIC"){
      fit = glmnet(x, y_pseudo, family = "gaussian")
      AIC = aic.opt(fit)
      theta = AIC$coeff
    }else if (tuning_theta == "BIC"){
      fit = glmnet(x, y_pseudo, family = "gaussian")
      BIC = bic.opt(fit,n)
      theta = BIC$coeff
    }else{
      print("Error: method for tuning parameter selection is not avaliable")
    }
    
  }else if (model_theta == "mean"){
    
  }else{
    
  }
  
  return(theta)   # p+1 dim vector with intercept
}

#model assuming pi and m is known for all subjects
DDR_est_oracle = function(x,y,T,pi_ora,m_ora,model_theta,tuning_theta){
  set.seed(2222)
  y_pseudo = m_ora + (T/pi_ora) * (y - m_ora)
  
  n = length(y_pseudo)
  
  if (model_theta == "linear"){
    if (tuning_theta == "CV-MSE"){
      cvfit = cv.glmnet(x, y_pseudo, family = "gaussian", type.measure = "mse")
      theta = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
    }else if (tuning_theta == "AIC"){
      fit = glmnet(x, y_pseudo, family = "gaussian")
      AIC = aic.opt(fit)
      theta = AIC$coeff
    }else if (tuning_theta == "BIC"){
      fit = glmnet(x, y_pseudo, family = "gaussian")
      BIC = bic.opt(fit,n)
      theta = BIC$coeff
    }else{
      print("Error: method for tuning parameter selection is not avaliable")
    }
    
  }else if (model_theta == "mean"){
    
  }else{
    
  }
  return(theta)   # p+1 dim vector with intercept
}

#model assuming no missing data
DDR_est_super_oracle = function(x,y,model_theta,tuning_theta){
  set.seed(2222)
  n = length(y)
  
  if (model_theta == "linear"){
    if (tuning_theta == "CV-MSE"){
      cvfit = cv.glmnet(x, y, family = "gaussian", type.measure = "mse")
      theta = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
    }else if (tuning_theta == "AIC"){
      fit = glmnet(x, y, family = "gaussian")
      AIC = aic.opt(fit)
      theta = AIC$coeff
    }else if (tuning_theta == "BIC"){
      fit = glmnet(x, y, family = "gaussian")
      BIC = bic.opt(fit,n)
      theta = BIC$coeff
    }else{
      print("Error: method for tuning parameter selection is not avaliable")
    }
    
  }else if (model_theta == "mean"){
    
  }else{
    
  }
  return(theta)   # p+1 dim vector with intercept
}

#model only using the observed data
DDR_complete = function(x,y,T,model_theta,tuning_theta){
  set.seed(2222)
  x.ob = x[T==1,]
  y.ob = y[T==1]
  n = length(y.ob)
  
  if (model_theta == "linear"){
    if (tuning_theta == "CV-MSE"){
      cvfit = cv.glmnet(x.ob, y.ob, family = "gaussian", type.measure = "mse")
      theta = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
    }else if (tuning_theta == "AIC"){
      fit = glmnet(x.ob, y.ob, family = "gaussian")
      AIC = aic.opt(fit)
      theta = AIC$coeff
    }else if (tuning_theta == "BIC"){
      fit = glmnet(x.ob, y.ob, family = "gaussian")
      BIC = bic.opt(fit,n)
      theta = BIC$coeff
    }else{
      print("Error: method for tuning parameter selection is not avaliable")
    }
    
  }else if (model_theta == "mean"){
    
  }else{
    
  }
  return(theta)   # p+1 dim vector with intercept
  
}

#


DDR_competing1 = function(x,y,T,tuning_T,tuning_m,tuning_theta){
  n = length(y)
  #subset selection for T
  if (tuning_T == "CV-AUC"){
    cvfit = cv.glmnet(x, T, family = "binomial", type.measure = "auc")
    beta_T = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
  }else if (tuning_T == "CV-deviance"){
    cvfit = cv.glmnet(x, T, family = "binomial", type.measure = "deviance") 
    beta_T = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
  }else if (tuning_T == "AIC"){
    fit = glmnet(x, T, family = "binomial")
    AIC = aic.opt(fit)
    beta_T = AIC$coeff
  }else if (tuning_T == "BIC"){
    fit = glmnet(x, T, family = "binomial")
    BIC = bic.opt(fit,n)
    beta_T = BIC$coeff
  }else{
    print("Error: method for tuning parameter selection is not avaliable")
  }
  set_T = which(abs(beta_T[-1])>=0.01)
 
  #subset selection for m 
  x.ob = x[T==1,]
  y.ob = y[T==1]
  n = length(y.ob)

  if (tuning_m == "CV-MSE"){
    cvfit = cv.glmnet(x.ob, y.ob, family = "gaussian", type.measure = "mse")
    beta_m = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
  }else if (tuning_m == "AIC"){
    fit = glmnet(x.ob, y.ob, family = "gaussian")
    AIC = aic.opt(fit)
    beta_m = AIC$coeff
  }else if (tuning_m == "BIC"){
    fit = glmnet(x.ob, y.ob, family = "gaussian")
    BIC = bic.opt(fit,n)
    beta_m = BIC$coeff
  }else{
    print("Error: method for tuning parameter selection is not avaliable")
  }
  
  set_m = which(abs(beta_m[-1])>=0.01)
  
  set = union(set_T,set_m)
  x_use = x[,set]
  
  
  glm_T = glm(T~x_use,family = binomial())
  pi_hat = glm_T$fitted.values
  
  lm_m = lm(y.ob ~ x_use[T==1,])
  beta_m = lm_m$coefficients
  m_hat = x_use %*% beta_m[-1] + beta_m[1]
  
  #pseudo outcome
  y_pseudo = m_hat + (T/pi_hat) * (y - m_hat)
  
  if (tuning_theta == "CV-MSE"){
    cvfit = cv.glmnet(x, y_pseudo, family = "gaussian", type.measure = "mse")
    theta = as.vector(coef(cvfit, s = "lambda.min"))   #using min not 1se #p+1 dim vector
  }else if (tuning_theta == "AIC"){
    fit = glmnet(x, y_pseudo, family = "gaussian")
    AIC = aic.opt(fit)
    theta = AIC$coeff
  }else if (tuning_theta == "BIC"){
    fit = glmnet(x, y_pseudo, family = "gaussian")
    BIC = bic.opt(fit,n)
    theta = BIC$coeff
  }else{
    print("Error: method for tuning parameter selection is not avaliable")
  }
  
  return(theta)
  
  }


#Debiased doubly robust estimation for simulation dataset
#save: "on" or "off", would save the result if on.
DDR_sim = function(nsim,n,p,gen_T,gen_y,cov,rho,model_T, tuning_T,model_m,tuning_m,sim_method_m,model_theta,
                   tuning_theta,split,K,save){
  #load data
  file_name <- paste("data_T_",gen_T,"_y_",gen_y,"_cov_",cov,"_rho_",rho,"_",paste(n, p,sep="_"), ".RData", sep="")
  file=paste("data/", file_name, sep="")
  load(file)
  #compute theta_hat for each copy of dataset
  result <- vector("list", nsim)
  l_inf <- vector("list", nsim)
  l_1 <- vector("list", nsim)
  l_2 <-vector("list", nsim)
  
  for (i in 1:nsim){
    x = data[[i]]$x
    y = data[[i]]$y
    T = data[[i]]$T
    pi_ora = data[[i]]$pi
    m_ora = data[[i]]$m
    # estimator using our method
    theta_hat = DDR_est(x,y,T,model_T, tuning_T,model_m,tuning_m,sim_method_m,
                        pi_ora,model_theta,tuning_theta,split,K)
    # estimator assuming ps is known
    if (model_m == "sim"){
      theta_hat_pi_oracle = DDR_est_pi_oracle(x,y,T,model_m,tuning_m,
                                              sim_method_m = "ora_w_Lasso",pi_ora,model_theta,tuning_theta,split,K)
    }else{
      theta_hat_pi_oracle = DDR_est_pi_oracle(x,y,T,model_m,tuning_m,
                                              sim_method_m,pi_ora,model_theta,tuning_theta,split,K)
    }
    # estimator assuming ps and conditional mean is known, only deal with missingness
    theta_hat_oracle = DDR_est_oracle(x,y,T,pi_ora,m_ora,model_theta,tuning_theta)
    # estimator with fully observed data
    theta_hat_super_oracle = DDR_est_super_oracle(x,y,model_theta,tuning_theta)
    
    #save result
    result[[i]]$theta_hat = theta_hat
    result[[i]]$theta_hat_pi_oracle  = theta_hat_pi_oracle 
    result[[i]]$theta_hat_oracle = theta_hat_oracle
    result[[i]]$theta_hat_super_oracle = theta_hat_super_oracle
    l_inf[[i]]$theta_hat = max(abs(theta_hat[-1] - oracle_linear[-1]))
    l_inf[[i]]$theta_hat_pi_oracle= max(abs(theta_hat_pi_oracle[-1] - oracle_linear[-1]))
    l_inf[[i]]$theta_hat_oracle= max(abs(theta_hat_oracle[-1] - oracle_linear[-1]))
    l_inf[[i]]$theta_hat_super_oracle = max(abs(theta_hat_super_oracle[-1] - oracle_linear[-1]))
    
    
    l_1[[i]]$theta_hat = sum(abs(theta_hat[-1] - oracle_linear[-1])) 
    l_1[[i]]$theta_hat_pi_oracle = sum(abs(theta_hat_pi_oracle[-1] - oracle_linear[-1])) 
    l_1[[i]]$theta_hat_oracle = sum(abs(theta_hat_oracle[-1] - oracle_linear[-1])) 
    l_1[[i]]$theta_hat_super_oracle = sum(abs(theta_hat_super_oracle[-1] - oracle_linear[-1])) 
    
    l_2[[i]]$theta_hat = sqrt(sum((theta_hat[-1] - oracle_linear[-1])^2))
    l_2[[i]]$theta_hat_pi_oracle = sqrt(sum((theta_hat_pi_oracle [-1] - oracle_linear[-1])^2))
    l_2[[i]]$theta_hat_oracle = sqrt(sum((theta_hat_oracle[-1] - oracle_linear[-1])^2))
    l_2[[i]]$theta_hat_super_oracle = sqrt(sum((theta_hat_super_oracle[-1] - oracle_linear[-1])^2))
    
    print(i)
  }
  
  
  if (save =="on"){
    file_name <- paste(paste("T",gen_T,"y",gen_y,"theta",model_theta,"Tw",model_T,"mw",model_m,split,K,"T",tuning_T,"m",tuning_m,"sim",sim_method_m,
                       "theta",tuning_theta,sep= "_"),"_",paste(n, p,sep="_"), ".Rdata", sep="")
    file=paste("results/", file_name, sep="")
    save(result,l_inf,l_1,l_2, file=file) 
  }else{
    return(list(result = result, l_inf = l_inf, l_1 = l_1, l_2 = l_2))
  }
  
}

DDR_sim_data = function(data_input,nsim,n,p,gen_T,gen_y,cov,rho,model_T, tuning_T,model_m,tuning_m,sim_method_m,model_theta,
                   tuning_theta,split,K,save,infer,cov_est){
  
  #compute theta_hat for each copy of dataset
  result <- vector("list", nsim)
  l_inf <- vector("list", nsim)
  l_1 <- vector("list", nsim)
  l_2 <-vector("list", nsim)
  pi_summary <- vector("list", nsim)
  

  data = data_input$data
  pi_trunc = data_input$pi_trunc
  
  oracle_linear = data_input$oracle_linear
  oracle_mean = data_input$oracle_mean
  beta_T = data_input$beta_T
  beta_y = data_input$beta_y
  
  theta_hat_t = matrix(0,p+1,nsim)
  theta_hat_pi_oracle_t = matrix(0,p+1,nsim)
  theta_hat_oracle_t = matrix(0,p+1,nsim)
  theta_hat_super_oracle_t = matrix(0,p+1,nsim)
  theta_hat_c_t = matrix(0,p+1,nsim)
  theta_hat_compet_t = matrix(0,p+1,nsim)
  
  l1_theta_hat_t = matrix(0,1,nsim)
  l1_theta_hat_pi_oracle_t = matrix(0,1,nsim)
  l1_theta_hat_oracle_t = matrix(0,1,nsim)
  l1_theta_hat_super_oracle_t = matrix(0,1,nsim)
  l1_theta_hat_c_t = matrix(0,1,nsim)
  l1_theta_hat_compet_t = matrix(0,1,nsim)
  
  l2_theta_hat_t = matrix(0,1,nsim)
  l2_theta_hat_pi_oracle_t = matrix(0,1,nsim)
  l2_theta_hat_oracle_t = matrix(0,1,nsim)
  l2_theta_hat_super_oracle_t = matrix(0,1,nsim)
  l2_theta_hat_c_t = matrix(0,1,nsim)
  l2_theta_hat_compet_t = matrix(0,1,nsim)
  
  linf_theta_hat_t = matrix(0,1,nsim)
  linf_theta_hat_pi_oracle_t = matrix(0,1,nsim)
  linf_theta_hat_oracle_t = matrix(0,1,nsim)
  linf_theta_hat_super_oracle_t = matrix(0,1,nsim)
  linf_theta_hat_c_t = matrix(0,1,nsim)
  linf_theta_hat_compet_t = matrix(0,1,nsim)
  
  theta_de_t = matrix(0,p,nsim)
  theta_l_t = matrix(0,p,nsim)
  theta_u_t = matrix(0,p,nsim)
  
  
  theta_cov_prob = matrix(0,p,1)
  theta_length = matrix(0,p,1)
  
  
  pi_all= matrix(0,n,nsim)
  
  fail = 0
  fail_list = c()
  for (i in 1:nsim){
    x = data[[i]]$x
    y = data[[i]]$y
    T = data[[i]]$T
    pi_ora = data[[i]]$pi
    m_ora = data[[i]]$m
    
    pi_all[,i] = pi_ora
    pi_summary[[i]] = data[[i]]$summary_pi_ori
    
    # estimator using our method
    theta_hat = DDR_est(x,y,T,model_T, tuning_T,model_m,tuning_m,sim_method_m,
                        pi_ora,model_theta,tuning_theta,split,K,infer,cov_est,beta_y)
    
    if ( length(theta_hat$theta) == 1){
      fail = fail +1
      fail_list = c(fail_list,i)
    }else{
      
    
    # estimator assuming ps is known
    if (model_m == "sim"){
      theta_hat_pi_oracle = DDR_est_pi_oracle(x,y,T,model_m,tuning_m,
                                              sim_method_m = "ora_w_Lasso",pi_ora,model_theta,tuning_theta,split,K)
    }else{
      theta_hat_pi_oracle = DDR_est_pi_oracle(x,y,T,model_m,tuning_m,
                                              sim_method_m,pi_ora,model_theta,tuning_theta,split,K)
    }
    
    # estimator assuming ps and conditional mean is known, only deal with missingness
    theta_hat_oracle = DDR_est_oracle(x,y,T,pi_ora,m_ora,model_theta,tuning_theta)
    # estimator with fully observed data (assuming T = 1 for all subjects)
    theta_hat_super_oracle = DDR_est_super_oracle(x,y,model_theta,tuning_theta)
    
    #estimator based on the observed data wit T =1
    theta_hat_c = DDR_complete(x,y,T,model_theta,tuning_theta)
    
    #competing estimator
    theta_hat_compet = DDR_competing1(x,y,T,tuning_T,tuning_m,tuning_theta)
    
    
    
    #save result
    theta_hat_t[,i] = theta_hat$theta
    theta_hat_pi_oracle_t[,i] = theta_hat_pi_oracle
    theta_hat_oracle_t[,i] = theta_hat_oracle
    theta_hat_super_oracle_t[,i] = theta_hat_super_oracle
    theta_hat_c_t[,i] = theta_hat_c
    theta_hat_compet_t[,i] = theta_hat_compet
    
    l1_theta_hat_t[i] =  sum(abs(theta_hat$theta - oracle_linear)) 
    l1_theta_hat_pi_oracle_t[i] =  sum(abs(theta_hat_pi_oracle - oracle_linear)) 
    l1_theta_hat_oracle_t[i] = sum(abs(theta_hat_oracle - oracle_linear)) 
    l1_theta_hat_super_oracle_t[i] = sum(abs(theta_hat_super_oracle - oracle_linear))
    l1_theta_hat_c_t[i] = sum(abs(theta_hat_c - oracle_linear))
    l1_theta_hat_compet_t[i] = sum(abs(theta_hat_compet - oracle_linear)) 
    
    l2_theta_hat_t[i] =  sqrt(sum((theta_hat$theta - oracle_linear)^2))
    l2_theta_hat_pi_oracle_t[i] =sqrt(sum((theta_hat_pi_oracle - oracle_linear)^2))
    l2_theta_hat_oracle_t[i] = sqrt(sum((theta_hat_oracle - oracle_linear)^2))
    l2_theta_hat_super_oracle_t[i] = sqrt(sum((theta_hat_super_oracle - oracle_linear)^2))
    l2_theta_hat_c_t[i] = sqrt(sum((theta_hat_c - oracle_linear)^2))
    l2_theta_hat_compet_t[i] = sqrt(sum((theta_hat_compet - oracle_linear)^2))
    
    linf_theta_hat_t[i] = max(abs(theta_hat$theta - oracle_linear))
    linf_theta_hat_pi_oracle_t[i] =  max(abs(theta_hat_pi_oracle - oracle_linear))
    linf_theta_hat_oracle_t[i] = max(abs(theta_hat_oracle - oracle_linear))
    linf_theta_hat_super_oracle_t[i] = max(abs(theta_hat_super_oracle - oracle_linear))
    linf_theta_hat_c_t[i] =max(abs(theta_hat_c - oracle_linear))
    linf_theta_hat_compet_t[i] = max(abs(theta_hat_compet - oracle_linear))
    

    if (infer == "True"){
      if (cov_est == "inv"){
        theta_de_t[,i] =  theta_hat$theta_de[2:(p+1)]
        theta_l_t[,i] = theta_hat$theta_l[2:(p+1)]
        theta_u_t[,i] = theta_hat$theta_u[2:(p+1)]
      }else{
        theta_de_t[,i] =  theta_hat$theta_de
        theta_l_t[,i] = theta_hat$theta_l
        theta_u_t[,i] = theta_hat$theta_u
      }
    }
    }
    print(i)
    #if (i %% 10 == 0){
    #  status_name <- paste(paste("status_T",gen_T,"y",gen_y,"cov",cov,"Tw",model_T,"mw",model_m,sep= "_"),"_",paste(n, p,sep="_"), ".txt", sep="")
    #  status = paste("Current iteration is:",i)
    #  write.table(status,file = status_name)
    #}
    
  }
  
  
  
  theta_cov_prob = matrix(0,p)
  theta_length = matrix(0,p)
  
  if (length(theta_hat$theta)>1){
  if (infer == "True"){
    for (i in 1:nsim){
        theta_length = theta_length + (theta_u_t[,i] - theta_l_t[,i])
        theta_cov_prob = theta_cov_prob + (oracle_linear[2:(p+1)] <= theta_u_t[,i] & oracle_linear[2:(p+1)] >= theta_l_t[,i] )
      }
    theta_length = theta_length/nsim
    theta_cov_prob = theta_cov_prob/nsim
  }
  }
  
  #save result
  theta_hat_bias = rowMeans(theta_hat_t) - oracle_linear
  theta_hat_pi_oracle_bias = rowMeans(theta_hat_pi_oracle_t) - oracle_linear
  theta_hat_oracle_bias = rowMeans( theta_hat_oracle_t) - oracle_linear
  theta_hat_super_oracle_bias = rowMeans(theta_hat_super_oracle_t) - oracle_linear
  theta_hat_c_bias = rowMeans(theta_hat_c_t) - oracle_linear
  theta_hat_compet_bias = rowMeans(theta_hat_compet_t) - oracle_linear
  
  theta_hat_var = apply(theta_hat_t, 1, var)
  theta_hat_pi_oracle_var = apply(theta_hat_pi_oracle_t,1,var)
  theta_hat_oracle_var = apply( theta_hat_oracle_t,1,var)
  theta_hat_super_oracle_var = apply(theta_hat_super_oracle_t,1,var)
  theta_hat_c_var = apply(theta_hat_c_t,1,var)
  theta_hat_compet_var = apply(theta_hat_compet_t,1,var)
  
  theta_summary = list(theta_hat_bias = theta_hat_bias,theta_hat_pi_oracle_bias = theta_hat_pi_oracle_bias,theta_hat_oracle_bias=theta_hat_oracle_bias ,
                       theta_hat_super_oracle_bias = theta_hat_super_oracle_bias,theta_hat_c_bias = theta_hat_c_bias,theta_hat_compet_bias =theta_hat_compet_bias,
                       theta_hat_var = theta_hat_var,theta_hat_pi_oracle_var = theta_hat_pi_oracle_var,theta_hat_oracle_var = theta_hat_oracle_var,
                       theta_hat_super_oracle_var = theta_hat_super_oracle_var,theta_hat_c_var = theta_hat_c_var, theta_hat_compet_var  =  theta_hat_compet_var)
  
  #
  basic_infor = list(model = paste(paste("model of T:", gen_T),paste(";model of y:", gen_y),paste(";covariance of x:", cov),paste(";rho,n and p:", rho,n,p,sep=" ")),
                     working_model =paste(paste("working model of T:", model_T),paste(";model of y:", model_m),paste(";model of theta:", model_theta)),
                     tuning_parameter =paste(paste("Tuning paramter of T:", tuning_T),paste(";Tuning paramter of m:", tuning_m),paste(";Tuning paramter of theta:", tuning_theta)),
                     sim = paste("Single index model:",sim_method_m),
                     split = paste("Split:",split,"; K:",K))
  
  if (save =="on"){
    file_name <- paste(paste("T",gen_T,"y",gen_y,"cov",cov,"rho",rho,"theta",model_theta,"Tw",model_T,"mw",model_m,split,K,"T",tuning_T,"m",tuning_m,"sim",sim_method_m,
                             "theta",tuning_theta,sep= "_"),"_",paste(n, p,sep="_"), ".Rdata", sep="")
    file=paste("results/", file_name, sep="")
    
    if (infer == "True"){
      save(theta_hat_t,theta_hat_pi_oracle_t,theta_hat_oracle_t, theta_hat_super_oracle_t, theta_hat_c_t,theta_hat_compet_t,
           l1_theta_hat_t,l1_theta_hat_pi_oracle_t,l1_theta_hat_oracle_t,l1_theta_hat_super_oracle_t, l1_theta_hat_c_t,l1_theta_hat_compet_t,
           l2_theta_hat_t, l2_theta_hat_pi_oracle_t,l2_theta_hat_oracle_t,l2_theta_hat_super_oracle_t,l2_theta_hat_c_t,l2_theta_hat_compet_t,
           linf_theta_hat_t,linf_theta_hat_pi_oracle_t,linf_theta_hat_oracle_t, linf_theta_hat_super_oracle_t,linf_theta_hat_c_t,linf_theta_hat_compet_t,
           basic_infor,oracle_linear,oracle_mean,beta_T, beta_y,pi_summary,pi_all = pi_all, pi_trunc,
           theta_summary = theta_summary, theta_de_t,theta_l_t,theta_u_t,theta_length,theta_cov_prob,fail=fail,fail_list = fail_list,file=file)
    }else{
      save(theta_hat_t,theta_hat_pi_oracle_t,theta_hat_oracle_t, theta_hat_super_oracle_t, theta_hat_c_t,theta_hat_compet_t,
           l1_theta_hat_t,l1_theta_hat_pi_oracle_t,l1_theta_hat_oracle_t,l1_theta_hat_super_oracle_t, l1_theta_hat_c_t,l1_theta_hat_compet_t,
           l2_theta_hat_t, l2_theta_hat_pi_oracle_t,l2_theta_hat_oracle_t,l2_theta_hat_super_oracle_t,l2_theta_hat_c_t,l2_theta_hat_compet_t,
           linf_theta_hat_t,linf_theta_hat_pi_oracle_t,linf_theta_hat_oracle_t, linf_theta_hat_super_oracle_t,linf_theta_hat_c_t,linf_theta_hat_compet_t,
           basic_infor,oracle_linear,oracle_mean,beta_T, beta_y,pi_summary,pi_all = pi_all, pi_trunc,
           theta_summary = theta_summary,fail=fail,fail_list = fail_list,file=file)
    }
  }else{
    if (infer == "True"){
      return(list(theta_hat_t,theta_hat_pi_oracle_t,theta_hat_oracle_t, theta_hat_super_oracle_t, theta_hat_c_t,theta_hat_compet_t,
           l1_theta_hat_t,l1_theta_hat_pi_oracle_t,l1_theta_hat_oracle_t,l1_theta_hat_super_oracle_t, l1_theta_hat_c_t,l1_theta_hat_compet_t,
           l2_theta_hat_t, l2_theta_hat_pi_oracle_t,l2_theta_hat_oracle_t,l2_theta_hat_super_oracle_t,l2_theta_hat_c_t,l2_theta_hat_compet_t,
           linf_theta_hat_t,linf_theta_hat_pi_oracle_t,linf_theta_hat_oracle_t, linf_theta_hat_super_oracle_t,linf_theta_hat_c_t,linf_theta_hat_compet_t,
           basic_infor,oracle_linear,oracle_mean,beta_T, beta_y,pi_summary,pi_all = pi_all, pi_trunc,
           theta_summary = theta_summary, theta_de_t,theta_l_t,theta_u_t,theta_length,theta_cov_prob,fail=fail,fail_list = fail_list))
    }else{
      return(list(theta_hat_t,theta_hat_pi_oracle_t,theta_hat_oracle_t, theta_hat_super_oracle_t, theta_hat_c_t,theta_hat_compet_t,
           l1_theta_hat_t,l1_theta_hat_pi_oracle_t,l1_theta_hat_oracle_t,l1_theta_hat_super_oracle_t, l1_theta_hat_c_t,l1_theta_hat_compet_t,
           l2_theta_hat_t, l2_theta_hat_pi_oracle_t,l2_theta_hat_oracle_t,l2_theta_hat_super_oracle_t,l2_theta_hat_c_t,l2_theta_hat_compet_t,
           linf_theta_hat_t,linf_theta_hat_pi_oracle_t,linf_theta_hat_oracle_t, linf_theta_hat_super_oracle_t,linf_theta_hat_c_t,linf_theta_hat_compet_t,
           basic_infor,oracle_linear,oracle_mean,beta_T, beta_y,pi_summary,pi_all = pi_all, pi_trunc,
           theta_summary = theta_summary,fail=fail,fail_list = fail_list))
    }
   
  }
  
}

DDR_sim_compet1 = function(data_input,nsim,n,p,gen_T,gen_y,model_T, tuning_T,model_m,tuning_m,model_theta,
                        tuning_theta,save){
  
  #compute theta_hat for each copy of dataset
  result_cp <- vector("list", nsim)
  l_inf_cp <- vector("list", nsim)
  l_1_cp <- vector("list", nsim)
  l_2_cp <-vector("list", nsim)
  
  data = data_input$data
  oracle_linear = data_input$oracle_linear
  oracle_mean = data_input$oracle_mean
  
  
  for (i in 1:nsim){
    x = data[[i]]$x
    y = data[[i]]$y
    T = data[[i]]$T
    pi_ora = data[[i]]$pi
    m_ora = data[[i]]$m
    # estimator using our method
    theta_hat_compet = DDR_competing1(x,y,T,tuning_T,tuning_m,tuning_theta)
    #save result
    result_cp[[i]]$theta_hat_compet = theta_hat_compet

    l_inf_cp[[i]]$theta_hat_compet = max(abs(theta_hat_compet[-1] - oracle_linear[-1]))
    l_1_cp[[i]]$theta_hat_compet = sum(abs(theta_hat_compet[-1] - oracle_linear[-1])) 
    l_2_cp[[i]]$theta_hat_compet = sqrt(sum((theta_hat_compet[-1] - oracle_linear[-1])^2))
    
    print(i)
  }
  
  
  if (save =="on"){
    file_name <- paste(paste("cp","T",gen_T,"y",gen_y,"theta",model_theta,"Tw",model_T,"mw",model_m,split,K,"T",tuning_T,"m",tuning_m,"sim",sim_method_m,
                             "theta",tuning_theta,sep= "_"),"_",paste(n, p,sep="_"), ".Rdata", sep="")
    file=paste("results/", file_name, sep="")
    save(result_cp,l_inf_cp,l_1_cp,l_2_cp, file=file) 
  }else{
    return(list(result_cp = result_cp, l_inf_cp = l_inf_cp, l_1_cp = l_1_cp, l_2_cp = l_2_cp))
  }
}



#make latex tables
require(xtable)
#function for examing tuning parameter selections
DDR_table_tuning = function(nsim ,n,p,gen_T,gen_y,model_theta,model_T,model_m,split,K,tuning_T_list,tuning_m_list,sim_method_m,
                            tuning_theta){
  output = matrix(0,length(tuning_T_list)*length(tuning_m_list),4)
  
  for (i in 1:length(tuning_T_list)){
    for (j in 1: length(tuning_m_list)){
      file_name <- paste(paste("T",gen_T,"y",gen_y,"theta",model_theta,"Tw",model_T,"mw",model_m,split,K,"T",tuning_T_list[i],"m",tuning_m_list,"sim",sim_method_m,
                               "theta",tuning_theta,sep= "_"),"_",paste(n, p,sep="_"), ".Rdata", sep="")
      file=paste("results/", file_name, sep="")
      load(file)
      output[(i-1)*length(tuning_m_list)+j,1] = paste("T:",tuning_T_list[i],",m:",tuning_m_list[j],sep = "")
      output[(i-1)*length(tuning_m_list)+j,2] = paste(round(mean(l_inf),digits= 3),"(",round(sd(l_inf),digits=2),")")
      output[(i-1)*length(tuning_m_list)+j,3] = paste(round(mean(l_1),digits= 3),"(",round(sd(l_1),digits=2),")")
      output[(i-1)*length(tuning_m_list)+j,4] = paste(round(mean(l_2),digits= 3),"(",round(sd(l_2),digits=2),")")
    }
  }
  
  temp = xtable(output,align="ccccc")
  names(temp) = c("Tuning Choice","l_inf","l_1","l_2")
  print(temp,include.rownames=FALSE)
}


DDR_table = function(file1){
  load(file1)

  table = matrix(0,1,6)

    
  table[1,1] = paste(round(mean(l2_theta_hat_t[l2_theta_hat_t!=0]),3),"(",round(sd(l2_theta_hat_t[l2_theta_hat_t!=0]),3),")")  
  table[1,2] = paste(round(mean(l2_theta_hat_pi_oracle_t[l2_theta_hat_pi_oracle_t!=0]),3),"(",round(sd(l2_theta_hat_pi_oracle_t[l2_theta_hat_pi_oracle_t!=0]),3),")")  
  table[1,3] = paste(round(mean(l2_theta_hat_oracle_t[l2_theta_hat_oracle_t!=0]),3),"(",round(sd(l2_theta_hat_oracle_t[l2_theta_hat_oracle_t!=0]),3),")")  
  table[1,4] = paste(round(mean(l2_theta_hat_super_oracle_t[l2_theta_hat_super_oracle_t!=0]),3),"(",round(sd(l2_theta_hat_super_oracle_t[l2_theta_hat_super_oracle_t!=0]),3),")")
  table[1,5] = paste(round(mean(l2_theta_hat_c_t[l2_theta_hat_c_t!=0]),3),"(",round(sd(l2_theta_hat_c_t[l2_theta_hat_c_t!=0]),3),")")
  table[1,6] = paste(round(mean(l2_theta_hat_compet_t[l2_theta_hat_compet_t!=0]),3),"(",round(sd(l2_theta_hat_compet_t[l2_theta_hat_compet_t!=0]),3),")")
  
  print(length(l2_theta_hat_t))
  print(sum(l2_theta_hat_t==0))
  
  return(table)
  
}

DDR_table_infer = function(file1){
  load(file1)
  print(length(theta_cov_prob))
  result = c(paste(round(mean(theta_cov_prob),3),"(",round(sd(theta_cov_prob),3),")"),paste(round(mean(theta_length),3),"(",round(sd(theta_length),3),")"))
  return(result)
}

DDR_table_infer_new = function(file1){
  load(file1)
  print(length(theta_cov_prob))
  if (length(theta_cov_prob) ==50){
    result = c(paste(round(mean(theta_cov_prob),3),"(",round(mean(theta_cov_prob[1:10]),3),round(mean(theta_cov_prob[11:50]),3),")"),paste(round(mean(theta_length),3),"(",round(sd(theta_length),3),")"))  
  }else if (length(theta_cov_prob) ==500){
    result = c(paste(round(mean(theta_cov_prob),3),"(",round(mean(theta_cov_prob[1:20]),3),round(mean(theta_cov_prob[21:500]),3),")"),paste(round(mean(theta_length),3),"(",round(sd(theta_length),3),")"))
  }
  return(result)
}

DDR_table_infer_prob = function(file1){
  load(file1)
  print(length(theta_cov_prob))
  temp = sum(abs(oracle_linear[-1])>0.05)
  print(temp)
  result = c(paste(round(mean(theta_cov_prob),2),"(",round(sd(theta_cov_prob),2),"),", round(median(theta_cov_prob),2),sep=""),paste(round(mean(theta_cov_prob[1:temp]),2),"(",round(sd(theta_cov_prob[1:temp]),2),"),",round(median(theta_cov_prob[1:temp]),2),sep=""),paste(round(mean(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"(",round(sd(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"),",round(median(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),sep="")) 
  return(result)
}

DDR_table_infer_length = function(file1){
  load(file1)
  print(length(theta_length))
  temp = sum(abs(oracle_linear[-1])>0.05)
  result = c(paste(round(mean(theta_length),2),"(",round(sd(theta_length),2),"),", round(median(theta_length),2),sep=""),paste(round(mean(theta_length[1:temp]),2),"(",round(sd(theta_length[1:temp]),2),"),",round(median(theta_length[1:temp]),2),sep=""),paste(round(mean(theta_length[(temp+1):length(theta_length)]),2),"(",round(sd(theta_length[(temp+1):length(theta_length)]),2),"),",round(median(theta_length[(temp+1):length(theta_length)]),2),sep="")) 
  return(result)
}

DDR_table_infer_mad = function(file1){
  load(file1)
  print(length(theta_cov_prob))
  temp = sum(abs(oracle_linear[-1])>0.05)
  print(temp)
  result = paste(round(mad(theta_cov_prob),2),",",round(mad(theta_cov_prob[1:temp]),2),",",round(mad(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),sep = "") 
  return(result)
}

DDR_table_summary = function(file1,n,p,nsim){
  load(file1)
  print(length(theta_cov_prob))
  
  theta_cov_prob = matrix(0,p)
  theta_length = matrix(0,p)
  count =0
  
  for (i in 1:nsim){
    if (sum(abs(theta_hat_t[,i]))>0){
        theta_length = theta_length + (theta_u_t[,i] - theta_l_t[,i])
        theta_cov_prob = theta_cov_prob + (oracle_linear[2:(p+1)] <= theta_u_t[,i] & oracle_linear[2:(p+1)] >= theta_l_t[,i] )
        count = count+1      
        }
  }
  theta_length = theta_length/count
  theta_cov_prob = theta_cov_prob/count
  
  print(count)
  #print(fail_list)
  temp = sum(abs(oracle_linear[-1])>0.05)
  #print(temp)
  result = c(paste("$",round(mean(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"_{",round(sd(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"}$",sep = ""),paste("$(",round(median(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"_{",round(mad(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"})$",sep = ""),paste("$",round(mean(theta_length[(temp+1):length(theta_length)]),2),"_{",round(sd(theta_length[(temp+1):length(theta_length)]),2),"}$",sep = ""),paste("$",round(mean(theta_cov_prob[1:temp]),2),"_{",round(sd(theta_cov_prob[1:temp]),2),"}$",sep = ""),paste("$(",round(median(theta_cov_prob[1:temp]),2),"_{",round(mad(theta_cov_prob[1:temp]),2),"})$",sep = ""),paste("$",round(mean(theta_length[1:temp]),2),"_{",round(sd(theta_length[1:temp]),2),"}$",sep = "")) 
  return(result)
}

DDR_table_summary_correct = function(file1){
  load(file1)
  print(length(theta_cov_prob_correct))
  #print(fail_list)
  #temp = sum(abs(oracle_linear[-1])>0.05)
  #print(temp)
  temp=20
  result = c(paste("$",round(mean(theta_cov_prob_correct[(temp+1):length(theta_cov_prob_correct)]),2),"_{",round(sd(theta_cov_prob_correct[(temp+1):length(theta_cov_prob_correct)]),2),"}$",sep = ""),paste("$(",round(median(theta_cov_prob_correct[(temp+1):length(theta_cov_prob_correct)]),2),"_{",round(mad(theta_cov_prob_correct[(temp+1):length(theta_cov_prob_correct)]),2),"})$",sep = ""),paste("$",round(mean(theta_length_correct[(temp+1):length(theta_length_correct)]),2),"_{",round(sd(theta_length_correct[(temp+1):length(theta_length_correct)]),2),"}$",sep = ""),paste("$",round(mean(theta_cov_prob_correct[1:temp]),2),"_{",round(sd(theta_cov_prob_correct[1:temp]),2),"}$",sep = ""),paste("$(",round(median(theta_cov_prob_correct[1:temp]),2),"_{",round(mad(theta_cov_prob_correct[1:temp]),2),"})$",sep = ""),paste("$",round(mean(theta_length_correct[1:temp]),2),"_{",round(sd(theta_length_correct[1:temp]),2),"}$",sep = "")) 
  return(result)
}

DDR_table_summary_orac = function(file1){
  load(file1)
  print(length(theta_cov_prob_orac))
  #print(fail_list)
  #temp = sum(abs(oracle_linear[-1])>0.05)
  #print(temp)
  temp=20
  result = c(paste("$",round(mean(theta_cov_prob_orac[(temp+1):length(theta_cov_prob_orac)]),2),"_{",round(sd(theta_cov_prob_orac[(temp+1):length(theta_cov_prob_orac)]),2),"}$",sep = ""),paste("$(",round(median(theta_cov_prob_orac[(temp+1):length(theta_cov_prob_orac)]),2),"_{",round(mad(theta_cov_prob_orac[(temp+1):length(theta_cov_prob_orac)]),2),"})$",sep = ""),paste("$",round(mean(theta_length_orac[(temp+1):length(theta_length_orac)]),2),"_{",round(sd(theta_length_orac[(temp+1):length(theta_length_orac)]),2),"}$",sep = ""),paste("$",round(mean(theta_cov_prob_orac[1:temp]),2),"_{",round(sd(theta_cov_prob_orac[1:temp]),2),"}$",sep = ""),paste("$(",round(median(theta_cov_prob_orac[1:temp]),2),"_{",round(mad(theta_cov_prob_orac[1:temp]),2),"})$",sep = ""),paste("$",round(mean(theta_length_orac[1:temp]),2),"_{",round(sd(theta_length_orac[1:temp]),2),"}$",sep = "")) 
  return(result)
}


DDR_table_summary_beta_y = function(file1,p,nsim,beta_y){
  load(file1)
  print(length(theta_cov_prob_correct))
  theta_cov_prob = matrix(0,p)
  for (i in 1:nsim){
    theta_cov_prob = theta_cov_prob + (beta_y[2:(p+1)] <= theta_u_t_correct[,i] & beta_y[2:(p+1)] >= theta_l_t_correct[,i] )
  }
  theta_cov_prob = theta_cov_prob/nsim
  #print(fail_list)
  temp = sum(abs(beta_y[-1])>0.05)
  print(temp)
  result = c(paste("$",round(mean(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"_{",round(sd(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"}$",sep = ""),paste("$(",round(median(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"_{",round(mad(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"})$",sep = ""),paste("$",round(mean(theta_length[(temp+1):length(theta_length)]),2),"_{",round(sd(theta_length[(temp+1):length(theta_length)]),2),"}$",sep = ""),paste("$",round(mean(theta_cov_prob[1:temp]),2),"_{",round(sd(theta_cov_prob[1:temp]),2),"}$",sep = ""),paste("$(",round(median(theta_cov_prob[1:temp]),2),"_{",round(mad(theta_cov_prob[1:temp]),2),"})$",sep = ""),paste("$",round(mean(theta_length[1:temp]),2),"_{",round(sd(theta_length[1:temp]),2),"}$",sep = "")) 
  return(result)
}



DDR_table_summary_beta_y = function(file1,p,nsim){
  load(file1)
  print(length(theta_cov_prob))
  theta_cov_prob = matrix(0,p)
  for (i in 1:nsim){
         theta_cov_prob = theta_cov_prob + (beta_y[2:(p+1)] <= theta_u_t[,i] & beta_y[2:(p+1)] >= theta_l_t[,i] )
  }
  theta_cov_prob = theta_cov_prob/nsim
  #print(fail_list)
  temp = sum(abs(oracle_linear[-1])>0.05)
  print(temp)
  result = c(paste("$",round(mean(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"_{",round(sd(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"}$",sep = ""),paste("$(",round(median(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"_{",round(mad(theta_cov_prob[(temp+1):length(theta_cov_prob)]),2),"})$",sep = ""),paste("$",round(mean(theta_length[(temp+1):length(theta_length)]),2),"_{",round(sd(theta_length[(temp+1):length(theta_length)]),2),"}$",sep = ""),paste("$",round(mean(theta_cov_prob[1:temp]),2),"_{",round(sd(theta_cov_prob[1:temp]),2),"}$",sep = ""),paste("$(",round(median(theta_cov_prob[1:temp]),2),"_{",round(mad(theta_cov_prob[1:temp]),2),"})$",sep = ""),paste("$",round(mean(theta_length[1:temp]),2),"_{",round(sd(theta_length[1:temp]),2),"}$",sep = "")) 
  return(result)
}
