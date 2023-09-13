#' The objective function of the debiasing dual program.
#'
#' This function computes the objective function value of the debiasing dual program.
#'
#' @param X The input design n*d matrix.
#' @param x The current query point, which is a 1*d array.
#' @param Pi An n*n diagonal matrix with (estimated) propensity scores as its diagonal entries.
#' @param ll_cur The current value of the dual solution vector.
#' @param gamma_n The regularization parameter "\eqn{\gamma/n}". (Default: gamma_n=0.1.)
#'
#' @return The value of the objective function of our dual debiasing program.
#'
#' @author Yikun Zhang, \email{yikunzhang@@foxmail.com}
#' @references Zhang, Y., Giessing, A. and Chen, Y.-C. (2023)
#' \emph{Efficient Inference on High-Dimensional Linear Model with Missing Outcomes.}
#' \url{https://arxiv.org/abs/2309.06429}.
#' @keywords utility
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
#'   ## Solve the debiasing dual program and estimate the dual objective function value
#'   ll_cur = DualCD(X_sim, x_cur, Pi = diag(prop_score), gamma_n = 0.1, ll_init = NULL,
#'                   eps=1e-9, max_iter = 5000)
#'   dual_val = DualObj(X_sim, x_cur, Pi=diag(prop_score), ll_cur=ll_cur, gamma_n=0.1)
#' }
#'
#' @export
#'
DualObj <- function(X, x, Pi, ll_cur, gamma_n = 0.05) {
  n = nrow(X)
  A = t(X) %*% Pi %*% X
  obj = t(ll_cur) %*% A %*% ll_cur / (4 * n) + sum(x * ll_cur) + gamma_n * sum(abs(ll_cur))

  return(obj)
}
