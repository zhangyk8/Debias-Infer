Welcome to the documentation of "Debias-Infer"!
===================================

**Debias-Infer** is a Python library for conducting valid and efficient inference on high-dimensional linear models with missing outcomes.

A Preview into the Proposed Debiasing Inference Method
------------

The proposed debiasing method introduces a novel debiased estimator for inferring the linear regression function with "missing at random (MAR)" outcomes. The key idea is to correct the bias of the Lasso solution [2]_ with complete-case data through a quadratic debiasing program with box constraints and construct the confidence interval based on the asymptotic normality of the debiased estimator.

More details can be found in :doc:`Methodology <method>` and the reference paper [1]_.

.. note::

   This project is under active development.
   

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   method
   Example_Debiasing
   api_reference
   
   
References
----------
.. [1] Yikun Zhang, Alexander Giessing, Yen-Chi Chen (2023+) Efficient Inference on High-Dimensional Linear Models with Missing Outcomes.
.. [2] Robert Tibshirani (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society Series B: Statistical Methodology* **58**, no.1: 267-288.

