library(scalreg)
library(CVXR)
# library(Rmosek)

DebiasProg = function(X, x, Pi, gamma_n = 0.1) {
  # Debiasing (primal) program.
  
  # Parameters
  #   gamma_n: float
  #     The regularization parameter "\gamma/n". (Default: gamma_n=0.1.)
  
  n = dim(X)[1]
  w = Variable(rows = n, cols = 1)
  debias_obj = Minimize(quad_form(w, Pi))
  constraints = list(x - (1/sqrt(n))*(t(w) %*% Pi %*% X) <= gamma_n, 
                     x - (1/sqrt(n))*(t(w) %*% Pi %*% X) >= -gamma_n)
  debias_prog = Problem(debias_obj, constraints)
  
  tryCatch({
    res = psolve(debias_prog)
  }, error = function(e) {
    # res = psolve(debias_prog, solver = "MOSEK", max_iters = 30000)
    return(matrix(NA, nrow = n, ncol = 1))
  })
  
  tryCatch({
    if(res$value == Inf) {
      print("The primal debiasing program is infeasible!")
      return(matrix(NA, nrow = n, ncol = 1))
    } else if (sum(res[[1]] == "solver_error") > 0){
      return(matrix(NA, nrow = n, ncol = 1))
    }
    else {
      return(res$getValue(w))
    }
  }, error = function(e){
      print("The 'CVXR' fails to solve this program!")
      return(matrix(NA, nrow = n, ncol = 1))
  })
}


SoftThres = function(theta, lamb) {
  # Thresholding function.
  
  # Parameters:
  #   theta: numeric vector
  #     The input vector.
  #   lamb: numeric
  #     The thresholding parameter.
  
  if (is.vector(theta)) {
    if (length(theta) > 1) {
      res <- sign(theta) * pmax(abs(theta) - lamb, 0)
    } else {
      res <- sign(theta) * max(abs(theta) - lamb, 0)
    }
  } else {
    res <- matrix(0, nrow = nrow(theta), ncol = 2)
    res[, 1] <- abs(theta) - lamb
    res <- sign(theta) * apply(res, 1, max)
  }
  
  return(res)
}


DualObj <- function(X, x, Pi, ll_cur, gamma_n = 0.05) {
  # Objective function of the dual form of our debiasing program.
  
  # Parameters:
  #   X: numeric matrix
  #     The input matrix.
  #   x: numeric vector
  #     The input vector.
  #   Pi: numeric matrix
  #     The input matrix.
  #   ll_cur: numeric vector
  #     The input vector.
  #   gamma_n: numeric
  #     The regularization parameter "\gamma/n". (Default: gamma_n = 0.05)
  
  n = nrow(X)
  A = t(X) %*% Pi %*% X
  obj = t(ll_cur) %*% A %*% ll_cur / (4 * n) + sum(x * ll_cur) + gamma_n * sum(abs(ll_cur))
  
  return(obj)
}


DualCD = function(X, x, Pi = NULL, gamma_n = 0.05, ll_init = NULL, eps = 1e-9, 
                  max_iter = 5000) {
  # Coordinate descent algorithm for solving the dual form of our debiasing program.
  
  # Parameters:
  #   X: numeric matrix
  #     The input matrix.
  #   x: numeric vector
  #     The input vector.
  #   Pi: numeric matrix
  #     The input matrix. (Default: Pi = NULL)
  #   gamma_n: numeric
  #     The regularization parameter "\gamma/n". (Default: gamma_n = 0.05)
  #   ll_init: numeric vector
  #     The initial vector. (Default: ll_init = NULL)
  #   eps: numeric
  #     The tolerance value for convergence. (Default: eps = 1e-9)
  #   max_iter: integer
  #     Maximum number of coordinate descent iterations. (Default: max_iter = 5000)
  
  n <- nrow(X)
  d <- ncol(X)
  
  if (is.null(Pi)) {
    Pi <- diag(n)
  }
  
  A <- t(X) %*% Pi %*% X
  
  if (is.null(ll_init)) {
    ll_new <- rep(1, d)
  } else {
    ll_new <- ll_init
  }
  
  ll_old <- 100 * rep(1, d)
  cnt <- 0
  flag <- 0
  
  while ((norm(ll_old - ll_new, type = "2") > eps) && ((cnt <= max_iter) || (flag == 0))) {
    ll_old = ll_new
    cnt = cnt + 1
    
    # Coordinate descent
    for (j in 1:d) {
      ll_cur = ll_new
      mask = rep(TRUE, d)
      mask[j] = FALSE
      A_kj = A[mask, j]
      ll_cur = ll_cur[mask]
      ll_new[j] = SoftThres(-(A_kj %*% ll_cur) / (2 * n) - x[j], lamb = gamma_n) / (A[j, j] / (2 * n))
    }
    
    if ((cnt > max_iter) && (flag == 0)) {
      print(paste0("The coordinate descent algorithm has reached its maximum number of iterations: ", 
                   max_iter, "!"))
      A <- A + 1e-9 * diag(d)
      cnt <- 0
      flag <- 1
    }
  }
  
  return(ll_new)
}


DebiasProgCV = function(X, x, prop_score, gamma_lst = NULL, cv_fold = 5, cv_rule = '1se') {
  require(caret)
  if (is.null(gamma_lst)) {
    gamma_lst = seq(0.001, max(abs(x)), length.out = 41)
  }
  
  kf = createFolds(1:n, cv_fold, list = FALSE, returnTrain = TRUE)
  dual_loss = matrix(0, nrow = cv_fold, ncol = length(gamma_lst))
  f_ind = 1
  
  for (fold in 1:cv_fold) {
    train_ind <- (kf != fold)
    test_ind <- (kf == fold)
    X_train <- X[train_ind, ]
    X_test <- X[test_ind, ]
    prop_score_train <- prop_score[train_ind]
    prop_score_test <- prop_score[test_ind]
    
    for (j in 1:length(gamma_lst)) {
      w_train = DebiasProg(X = X_train, x = x, Pi = diag(prop_score_train), gamma_n = gamma_lst[j])
      
      if (any(is.na(w_train))) {
        cat(paste("The primal debiasing program for this fold of the data is not feasible when gamma/n=", round(gamma_lst[j], 4), "!\n"))
        dual_loss[f_ind, j] = NA
      } else {
        ll_train = DualCD(X = X_train, x = x, Pi = diag(prop_score_train), gamma_n = gamma_lst[j], ll_init = NULL, eps = 1e-8, max_iter = 5000)
        
        if (sum(abs(w_train + drop(X_train %*% ll_train) / (2 * sqrt(dim(X_train)[1]))) > 1e-3) > 0) {
          cat(paste("The strong duality between primal and dual programs does not satisfy when gamma/n=", round(gamma_lst[j], 4), "!\n"))
          dual_loss[f_ind, j] = NA
        } else {
          dual_loss[f_ind, j] = DualObj(X_test, x = x, Pi = diag(prop_score_test), ll_cur = ll_train, gamma_n = gamma_lst[j])
        }
      }
    }
    
    f_ind = f_ind + 1
  }
  
  mean_dual_loss = apply(dual_loss, 2, mean, na.rm = FALSE)
  std_dual_loss = apply(dual_loss, 2, function(x) sd(x, na.rm = FALSE)) / sqrt(cv_fold)
  
  if (cv_rule == 'mincv') {
    gamma_n_opt = gamma_lst[which.min(mean_dual_loss)]
  }
  if (cv_rule == '1se') {
    One_SE = (mean_dual_loss > min(mean_dual_loss, na.rm = TRUE) + std_dual_loss[which.min(mean_dual_loss)]) &
      (gamma_lst < gamma_lst[which.min(mean_dual_loss)])
    if (sum(One_SE, na.rm = TRUE) == 0) {
      One_SE = rep(TRUE, length(gamma_lst))
    }
    gamma_lst = gamma_lst[One_SE]
    gamma_n_opt = gamma_lst[which.min(mean_dual_loss[One_SE])]
  }
  if (cv_rule == 'minfeas') {
    gamma_n_opt = min(gamma_lst[!is.na(mean_dual_loss)])
  }
  
  w_obs = DebiasProg(X = X, x = x, Pi = diag(prop_score), gamma_n = gamma_n_opt)
  ll_obs = DualCD(X = X, x = x, Pi = diag(prop_score), gamma_n = gamma_n_opt, ll_init = NULL, eps = 1e-9)
  
  return(list(w_obs = w_obs, ll_obs = ll_obs, gamma_n_opt = gamma_n_opt))
}

