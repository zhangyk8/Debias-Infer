# ``DebiasInfer``: A R Package for Efficient Inference on High-Dimensional Linear Models With Missing Outcomes

<!-- badges: start -->
[![CRAN status](https://www.r-pkg.org/badges/version/DebiasInfer)](https://CRAN.R-project.org/package=DebiasInfer)
<!-- badges: end -->

## Installation

The latest release of the R package can be installed through CRAN:

```R
install.packages("DebiasInfer")
```

The development version can be installed from github:

```R
devtools::install_github("zhangyk8/Debias-Infer", subdir = "R_Package")
```

## Toy Example

```R
require(MASS)
require(glmnet)
d = 1000
n = 900

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

## Current query point
x_cur = rep(0, d)
x_cur[c(1, 2, 3, 7, 8)] = c(1, 1/2, 1/4, 1/2, 1/8)
x_cur = array(x_cur, dim = c(1,d))

## True regression coefficient
s_beta = 5
beta_0 = rep(0, d)
beta_0[1:s_beta] = sqrt(5)

## Generate the design matrix and outcomes
X_sim = mvrnorm(n, mu = rep(0, d), Sigma)
eps_err_sim = sig * rnorm(n)
Y_sim = drop(X_sim %*% beta_0) + eps_err_sim

obs_prob = 1 / (1 + exp(-1 + X_sim[, 7] - X_sim[, 8]))
R_sim = rep(1, n)
R_sim[runif(n) >= obs_prob] = 0

## Estimate the propensity scores via the Lasso-type generalized linear model
zeta = 5*sqrt(log(d)/n)/n
lr1 = glmnet(X_sim, R_sim, family = "binomial", alpha = 1, lambda = zeta,
             standardize = TRUE, thresh=1e-6)
prop_score = drop(predict(lr1, newx = X_sim, type = "response"))

## Estimate the debiasing weights with the tuning parameter selected by cross-validations.
deb_res = DebiasProgCV(X_sim, x_cur, prop_score, gamma_lst = c(0.1, 0.5, 1),
                       cv_fold = 5, cv_rule = '1se')
```

