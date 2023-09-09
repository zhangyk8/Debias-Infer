#' The proposed debiasing (primal) program.
#'
#' This function implements our proposed debiasing (primal) program that solves for
#' the weights for correcting the Lasso pilot estimate.
#'
#' @param X The input design n*d matrix.
#' @param x The current query point, which is a 1*d array.
#' @param Pi An n*n diagonal matrix with (estimated) propensity scores as its diagonal entries.
#' @param gamma_n The regularization parameter "\eqn{\gamma/n}". (Default: gamma_n=0.1.)
#'
#' @return The estimated weights by our debiasing program, which is a n-dim vector.
#'
#' @author Yikun Zhang, \email{yikunzhang@@foxmail.com}
#' @keywords debiasing primal program
#'
#' @examples
#' \donttest{
#'   require(MASS)
#'   require(glmnet)
#'   d = 1000
#'   n = 900
#'  
#'   Sigma = array(0, dim = c(d,d)) + diag(d)
#'   rho = 0.1
#'   for(i in 1:(d-1)){
#'     for(j in (i+1):d){
#'       if ((j < i+6) | (j > i+d-6)){
#'         Sigma[i,j] = rho
#'         Sigma[j,i] = rho
#'       }
#'     }
#'   }
#'   sig = 1
#'  
#'   ## Current query point
#'   x_cur = rep(0, d)
#'   x_cur[c(1, 2, 3, 7, 8)] = c(1, 1/2, 1/4, 1/2, 1/8)
#'   x_cur = array(x_cur, dim = c(1,d))
#'  
#'   ## True regression coefficient
#'   s_beta = 5
#'   beta_0 = rep(0, d)
#'   beta_0[1:s_beta] = sqrt(5)
#'  
#'   ## Generate the design matrix and outcomes
#'   X_sim = mvrnorm(n, mu = rep(0, d), Sigma)
#'   eps_err_sim = sig * rnorm(n)
#'   Y_sim = drop(X_sim %*% beta_0) + eps_err_sim
#'  
#'   obs_prob = 1 / (1 + exp(-1 + X_sim[, 7] - X_sim[, 8]))
#'   R_sim = rep(1, n)
#'   R_sim[runif(n) >= obs_prob] = 0
#'  
#'   ## Estimate the propensity scores via the Lasso-type generalized linear model
#'   zeta = 5*sqrt(log(d)/n)/n
#'   lr1 = glmnet(X_sim, R_sim, family = "binomial", alpha = 1, lambda = zeta,
#'                standardize = TRUE, thresh=1e-6)
#'   prop_score = drop(predict(lr1, newx = X_sim, type = "response"))
#'  
#'   ## Estimate the debiasing weights
#'   w_obs = DebiasProg(X_sim, x_cur, Pi=diag(prop_score), gamma_n = 0.1)
#' }
#' 
#' @export
#' @importFrom CVXR Variable Minimize quad_form Problem psolve
#'
DebiasProg = function(X, x, Pi, gamma_n = 0.1) {
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
