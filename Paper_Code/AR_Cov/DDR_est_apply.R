
### Code for implementing (on a given dataset) the L1-regularized debiased and doubly robust estimator and inference of Chakrabortty et al. (2019, arXiv:1911.11345) for High-dimensional M-Estimation 
### with Missing Outcomes. Note: Need to source "HD_M_est_functions.R" for calling several functions internally involved. The other file contains more details on functions plus simulation settings. 

source("HD_M_est_functions.R")


DDR_est = function(x,y,T,model_T = c("logit","quad","sim"),tuning_T = "BIC",model_m = c("linear","quad","sim"),tuning_m = "CV-MSE",tuning_theta = c("CV-MSE","AIC","BIC"),split = TRUE,K = 2,infer = FALSE,cov_est = NULL){
  set.seed(2222)
  if (split == TRUE){
    #creat split samples:
    id = seq(1:length(y))
    id<-id[sample(length(y))]
    y_pseudo = rep(0,length(y))
    #Create K equally size folds
    folds <- cut(seq(1,length(y)),breaks=K,labels=FALSE)
    pi_hat = DDR_pi(x,T,length(y),model_T,tuning_T,x)
    
    m_hat_full = pi_hat
    
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
        pi_use = pi_hat[-id_test]
        m_hat = DDR_m(train_x,train_T,train_y,model_m,tuning_m,"est_w_Lasso",pi_use,test_x)
      }else{
        m_hat = DDR_m(train_x,train_T,train_y,model_m,tuning_m,sim_method = NA,pi=NA,test_x)
      }
      #creat pseudo outcome in the i-th fold
      y_pseudo[id_test] = m_hat + (T[id_test]/pi_hat[id_test]) * (y[id_test] - m_hat)
      m_hat_full[id_test] = m_hat
    }
    
  }else if (split == FALSE){
    n = length(y)   # y has to be one dimensional
    pi_hat = DDR_pi(x,T,n,model_T,tuning_T,x)
    y_pseudo = rep(0,length(y))
    if (model_m == "sim"){
      m_hat = DDR_m(x,T,y,model_m,tuning_m,"est_w_Lasso",pi_hat,x)
    }else{
      m_hat = DDR_m(x,T,y,model_m,tuning_m,sim_method = NA,pi = NA,x)
    }
    m_hat_full = m_hat
    y_pseudo = m_hat + (T/pi_hat) * (y - m_hat)  
    
    
  }else{
    print("Error: Wrong argument for split. Should be either 'True' or 'False'")
  }
  
  n = length(y_pseudo)
  #print(y_pseudo)
  #print(m_hat_full)
  
  if (tuning_theta == "CV-MSE"){
    cvfit = cv.glmnet(x, y_pseudo, family = "gaussian", type.measure = "mse",nfolds = length(y_pseudo))
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
  
  
  
  
  
  if (infer == FALSE){
    #return theta if not providing confidence intervals
    
    return(list(theta = theta,pi_hat = pi_hat,m_hat = m_hat_full,y_pseudo = y_pseudo))   # p+1 dim vector with intercept
    
  }else if (infer == TRUE){
    
    x_one = cbind(rep(1,length(y)),x)
    
    if (cov_est == "inv"){
      y_center = y_pseudo - mean(y_pseudo)
      x_center = scale(x,scale = FALSE)
      
     
      #using matrix inverse
      #Omega_hat = solve(1/length(y)*t(x_one)%*%x_one)
      Omega_hat = solve(1/length(y)*t(x_center)%*%x_center)
      #construct debiased estimates
      #temp = (y_pseudo - x_one %*% theta)
      temp = (y_center - x_center %*% theta[-1])
      
      res = 0
      #for (i in 1:length(y)){
      #  res = res + temp[i] * x_one[i,]
      #}
      
      #theta_de = theta + 1/length(y)*Omega_hat%*%res
      
      for (i in 1:length(y)){
        res = res + temp[i] * x_center[i,]
      }
      
      theta_de = theta[-1] + 1/n*Omega_hat%*%res
      
      # variance estimator
      #theta_l = rep(0,length(theta))
      #theta_u = rep(0,length(theta))
      #IF = matrix(0,length(y),length(theta))
      #for (i in 1:length(y)){
      #  temp = Omega_hat%*%x_one[i,]*(m_hat_full[i] - sum(x_one[i,] %*% theta) + T[i]/pi_hat[i]*(y[i] - m_hat_full[i]))
      #  IF[i,] = temp 
      #}
      
      #for (j in 1:length(theta)){
      #  theta_l[j] = theta_de[j] - 1/sqrt(length(y))*1.96*sqrt(1/length(y)*t(IF[,j])%*%IF[,j]) 
      #  theta_u[j] = theta_de[j] + 1/sqrt(length(y))*1.96*sqrt(1/length(y)*t(IF[,j])%*%IF[,j]) 
      #}
      
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
      
      
      
    }else{
      print("Error: Wrong method for estimating the precision matrix")
    }
    
    
    theta_CI = cbind(theta_l,theta_u)
    # return theta together with its upper and lower confidence intervals and the estimated precision matrix. 
    return(list(theta = theta,theta_de = theta_de, theta_CI = theta_CI,Omega_hat = Omega_hat,IF = IF,m_hat_full = m_hat_full,pi_hat = pi_hat,y_pseudo = y_pseudo))
    
  }else{
    print("Error: Wrong input for 'infer'")
  }
  
}

result_summary = function(result){
  theta = result$theta
  select_Lasso = which(theta !=0)[-1]-1
  theta_de = result$theta_CI
  select_Lasso_de = which(theta_de[,1] >0 | theta_de[,2] < 0)
  print(paste("select by Lasso", paste(select_Lasso,collapse =",")))
  print(sign(theta[select_Lasso+1]))
  print(paste("select by debaised Lasso", paste(select_Lasso_de, collapse =",")))
  print(sign(result$theta_de)[select_Lasso_de])
}

