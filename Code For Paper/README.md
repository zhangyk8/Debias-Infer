# Simulation Studies

The data sample <img src="https://latex.codecogs.com/svg.latex?&space;\left\{(X_i,Y_i)\right\}_{i=1}^n"/> is generated from the following linear model:

<img src="https://latex.codecogs.com/svg.latex?\large&space;Y_i=\beta_0^TX_i+\epsilon_i,\quad\,i=1,...,n,"/>

where <img src="https://latex.codecogs.com/svg.latex?&space;X_i\sim\mathcal{N}_d(0,\Sigma)"/>, 
<img src="https://latex.codecogs.com/svg.latex?&space;\epsilon_i\sim\mathcal{N}(0,\sigma^2)"/> with 
<img src="https://latex.codecogs.com/svg.latex?&space;\sigma^2=1"/>, and <img src="https://latex.codecogs.com/svg.latex?&space;d=600,n=500"/>.

Here, we consider three different covariance matrices <img src="https://latex.codecogs.com/svg.latex?&space;\Sigma"/>:
- <img src="https://latex.codecogs.com/svg.latex?&space;\Sigma"/> is toeplitz, in which <img src="https://latex.codecogs.com/svg.latex?&space;\Sigma_{ij}=\rho^{|i-j|}"/> with <img src="https://latex.codecogs.com/svg.latex?&space;\rho=0.9\,"/>; see Section 4 of van de Geer et al. (2014).
- <img src="https://latex.codecogs.com/svg.latex?&space;\Sigma"/> is equi-correlated, in which <img src="https://latex.codecogs.com/svg.latex?&space;\Sigma_{ij}=0.8"/> when <img src="https://latex.codecogs.com/svg.latex?&space;i\neq\,j"/> and <img src="https://latex.codecogs.com/svg.latex?&space;\Sigma_{ii}=1\,"/>; see Section 4 of van de Geer et al. (2014).
- <img src="https://latex.codecogs.com/svg.latex?&space;\Sigma"/> is circulant symmetric; see Section 5.1 in Javavanmard and Montarani (2014).


We also specify three different choices for <img src="https://latex.codecogs.com/svg.latex?&space;\beta_0"/>:
- <img src="https://latex.codecogs.com/svg.latex?&space;\beta_0=(\underbrace{1,...,1}_{5},0,...,0)^T\in\mathbb{R}^d"/> is sparse.
- <img src="https://latex.codecogs.com/svg.latex?&space;\beta_0\propto\left(1,\frac{1}{\sqrt{2}},...,\frac{1}{\sqrt{d}}\right)^T\in\mathbb{R}^d"/> is dense.
- <img src="https://latex.codecogs.com/svg.latex?&space;\beta_0\propto\left(1,\frac{1}{2},...,\frac{1}{d}\right)^T\in\mathbb{R}^d"/> is pseudo-dense.

In the latter two cases, we normalize <img src="https://latex.codecogs.com/svg.latex?&space;\beta_0\in\mathbb{R}^d"/> so that it has the norm <img src="https://latex.codecogs.com/svg.latex?&space;\|\beta_0\|_2=5"/>.

To infer the regression function <img src="https://latex.codecogs.com/svg.latex?&space;m(x)=x^T\beta_0"/>, we experiment on four different choices of the query point <img src="https://latex.codecogs.com/svg.latex?&space;x"/>:
- <img src="https://latex.codecogs.com/svg.latex?&space;x=(1,0,...,0)^T\in\mathbb{R}^d"/> so that <img src="https://latex.codecogs.com/svg.latex?&space;x^T\beta_0\in\mathbb{R}"/> becomes the first coordinate of <img src="https://latex.codecogs.com/svg.latex?&space;\beta_0\in\mathbb{R}^d"/>.
- <img src="https://latex.codecogs.com/svg.latex?&space;x=\left(1,\frac{1}{2},\frac{1}{4},0,0,0,\frac{1}{2},\frac{1}{8},0,...,0\right)\in\mathbb{R}^d"/>.
- <img src="https://latex.codecogs.com/svg.latex?&space;x=(0,...,0,\underbrace{1}_{100^{th}},0,...,0)\in\mathbb{R}^d"/> so that <img src="https://latex.codecogs.com/svg.latex?&space;x^T\beta_0\in\mathbb{R}"/> becomes the 100-th coordinate of <img src="https://latex.codecogs.com/svg.latex?&space;\beta_0\in\mathbb{R}^d"/>.
- <img src="https://latex.codecogs.com/svg.latex?&space;x=\left(1,\frac{1}{2^2},...,\frac{1}{d^2}\right)\in\mathbb{R}^d"/>.


For the comparative studies, we compare our debiased framework with the following four existing methods:
1. `lasso.proj`: the debiased lasso proposed by van de Geer et al. (2014).
2. `ridge.proj`: the projected ridge regression proposed by Buhlmann (2013).
3. `sslasso`: the debiased lasso proposed by Javavanmard and Montarani (2014).
4. `refit`: solve a Lasso regression for <img src="https://latex.codecogs.com/svg.latex?&space;\hat{\beta}\in\mathbb{R}^d"/> and fit an ordinary least-square regression on the covariates in the support set of <img src="https://latex.codecogs.com/svg.latex?&space;\hat{\beta}\in\mathbb{R}^d"/>.

