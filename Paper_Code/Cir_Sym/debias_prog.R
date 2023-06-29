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
    FALSE
  })
  
  if(res$value == Inf) {
    print("The primal debiasing program is infeasible!")
    return(matrix(NA, nrow = n, ncol = 1))
  } else if (sum(res[[1]] == "solver_error") > 0){
    return(matrix(NA, nrow = n, ncol = 1))
  }
  else {
    return(res$getValue(w))
  }
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

