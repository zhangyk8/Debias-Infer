Methodology
===========

Proposed Debiasing Inference Procedure
------------

Consider the observed data :math:`\{(Y_i,R_i,X_i)\}_{i=1}^n \subset \mathbb{R}\times \{0,1\} \times \mathbb{R}^d`, where :math:`Y_i` is the outcome variable that could be missing, :math:`R_i` is the binary variable indicating the missingness of :math:`Y_i`, and :math:`X_i` is the fully observed high-dimensional covariate vector for :math:`i=1,...,n`. 

Our proposed debiasing method aims at conducting valid statistical inference on the linear regression function :math:`m_0(x)=\mathrm{E}(Y|X=x)=x^T\beta_0` under the "missing at random (MAR)" assumption. Specifically, the method consists of the following procedures.

* **Step 1:** Compute the Lasso pilot estimate :math:`\hat{\beta}` of the regression coefficient using the complete-case data as:

.. math::

    \hat{\beta}=\mathrm{argmin}_{\beta \in \mathbb{R}^d} \left[\frac{1}{n}\sum\limits_{i=1}^n R_i (Y_i-X_i^T \beta)^2+ \lambda \|\beta\|_1 \right], 

where :math:`\lambda >0` is a regularization parameter.

* **Step 2:** Obtain consistent estimates :math:`\hat{\pi}_i, i=1,...,n` of the propensity scores :math:`\pi_i = \mathrm{P}(R_i=1|X_i)` by any machine learning method (not necessarily a parametric model) applied on the data :math:`\{(X_i,R_i)\}_{i=1}^n \subset \mathbb{R}^d \times \{0,1\}`. See the `scikit-learn <https://scikit-learn.org/stable/>`_ package and the related `probability calibration <https://scikit-learn.org/stable/modules/calibration.html>`_ for potential propensity score estimation methods.

* **Step 3:** Solve for the debiasing weight vector :math:`\hat{\mathbf{w}}\equiv \hat{\mathbf{w}}(x) = \left(\hat{w}_1(x),...,\hat{w}_n(x)\right)^T \in \mathbb{R}^n` through a debiasing program defined as:

 .. math::
 
     \min_{\mathbf{w}\equiv \mathbf{w}(x) \in \mathbb{R}^n} \left\{\sum_{i=1}^n \hat{\pi}_iw_i(x)^2: \left\|x- \frac{1}{\sqrt{n}}\sum_{i=1}^n w_i(x)\cdot \hat{\pi}_i\cdot X_i \right\|_{\infty} \leq \frac{\gamma}{n} \right\},

where :math:`\gamma >0` is a tuning parameter.

* **Step 4:** Define the debiasing estimator for :math:`m_0(x)` as:

.. math::

    \hat{m}^{\text{debias}}(x;\hat{\mathbf{w}}) = x^T \hat{\beta} + \frac{1}{\sqrt{n}} \sum_{i=1}^n \hat{w}_i(x)R_i \left(Y_i-X_i^T \hat{\beta} \right).

* **Step 5:** Construct the asymptotic :math:`(1-\alpha)`-level confidence interval for :math:`m_0(x)` as:

.. math::

    \left[\hat{m}^{\text{debias}}(x;\hat{\mathbf{w}}) - \Phi^{-1}\left(1-\frac{\tau}{2}\right) \cdot \sigma_{\epsilon}\cdot \sqrt{\frac{1}{n}\sum_{i=1}^n \hat{\pi}_i \hat{w}_i(x)^2},\; \hat{m}^{\text{debias}}(x;\hat{\mathbf{w}}) + \Phi^{-1}\left(1-\frac{\tau}{2}\right) \cdot \sigma_{\epsilon}\cdot \sqrt{\frac{1}{n}\sum_{i=1}^n \hat{\pi}_i \hat{w}_i(x)^2} \right],

where :math:`\Phi(\cdot)` denotes the cumulative distribution function (CDF) of :math:`\mathcal{N}(0,1)`. If :math:`\sigma_{\epsilon}^2` is unknown, then it can be replaced by any consistent estimator :math:`\hat{\sigma}_{\epsilon}^2`.

For the implementation of this package ``Debias-Infer``, we fit the Lasso pilot estimate :math:`\hat{\beta}` in **Step 1** by the scaled Lasso [2]_ so as to automatically select the regularization parameter :math:`\lambda >0` and simultaneously produce a consistent estimator of the noise level :math:`\sigma_{\epsilon}^2`.

As for solving the debiasing program in **Step 3**, we leverage the `CVXPY <https://www.cvxpy.org/>`_ package. To select the tuning parameter :math:`\gamma >0`, we use the cross-validation on the dual formulation of the debiasing program as:

.. math::

    \min_{\ell(x) \in \mathbb{R}^d} \left\{\frac{1}{4n} \sum_{i=1}^n \hat{\pi}_i \left[X_i^T \ell(x)\right]^2 + x^T \ell(x) +\frac{\gamma}{n}\|\ell(x)\|_1 \right\}.
    
This dual program is solved via the coordinate descent algorithm [3]_ in our package.

In the reference paper [1]_, we prove that the confidence interval in **Step 5** is asymptotically valid and our debiased estimator in **Step 4** is semi-parametrically efficient among all asymptotically linear estimators with MAR outcomes; see Figure 1 for how our debiased estimator performs (under two different rules for the cross-validation) when compared with the Lasso estimate.

.. image:: cirsym_lasso_bias_expl_x4_beta1.png
  :alt: Debiasing method illustration
  :class: with-shadow float-left

**Figure 1**: Comparison of our debiased estimators under two different choices of the tuning parameters ("1SE" and "min-feas") with the conventional Lasso estimates based on complete-case or oracle data.

References
----------

.. [1] Yikun Zhang, Alexander Giessing, Yen-Chi Chen (2023+). Efficient Inference on High-Dimensional Linear Models with Missing Outcomes.
.. [2] Tingni Sun and Cun-Hui Zhang (2012). Scaled Sparse Linear Regression." *Biometrika*, **99**, no.4: 879-898.
.. [3] Stephen J. Wright (2015). Coordinate Descent Algorithms. *Mathematical Programming* **151**, no.1: 3-34.
