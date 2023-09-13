#' Coordinate descent algorithm for solving the dual form of our debiasing program.
#'
#' This function implements the coordinate descent algorithm for the debiasing
#' dual program. More details can be found in Appendix A of our paper.
#'
#' @param X The input design n*d matrix.
#' @param x The current query point, which is a 1*d array.
#' @param Pi An n*n diagonal matrix with (estimated) propensity scores as its diagonal entries.
#' @param gamma_n The regularization parameter "\eqn{\gamma/n}". (Default: gamma_n=0.05.)
#' @param ll_init The initial value of the dual solution vector. (Default: ll_init=NULL. Then, the vector with all-one entries is used.)
#' @param eps The tolerance value for convergence. (Default: eps=1e-9.)
#' @param max_iter The maximum number of coordinate descent iterations. (Default: max_iter=5000.)
#'
#' @return The solution vector to our dual debiasing program.
#'
#' @author Yikun Zhang, \email{yikunzhang@@foxmail.com}
#' @references Zhang, Y., Giessing, A. and Chen, Y.-C. (2023)
#' \emph{Efficient Inference on High-Dimensional Linear Model with Missing Outcomes.}
#' \url{https://arxiv.org/abs/2309.06429}.
#' @keywords program dual debiasing
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
#'   ## Solve the debiasing dual program
#'   ll_cur = DualCD(X_sim, x_cur, Pi = diag(prop_score), gamma_n = 0.1, ll_init = NULL,
#'                   eps=1e-9, max_iter = 5000)
#' }
#'
#' @export
#'
DualCD = function(X, x, Pi=NULL, gamma_n=0.05, ll_init=NULL, eps=1e-9,
                  max_iter=5000) {
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
      warning(paste0("The coordinate descent algorithm has reached its maximum number of iterations: ",
                   max_iter, "! Reiterate one more times without small perturbations to the scaled design matrix..."))
      A <- A + 1e-9 * diag(d)
      cnt <- 0
      flag <- 1
    }
  }

  return(ll_new)
}
