[![PyPI pyversions](https://img.shields.io/pypi/pyversions/Debias-Infer.svg)](https://pypi.python.org/pypi/Debias-Infer/)
[![PyPI version](https://badge.fury.io/py/Debias-Infer.svg)](https://badge.fury.io/py/Debias-Infer)
[![Downloads](https://static.pepy.tech/badge/Debias-Infer)](https://pepy.tech/project/Debias-Infer)
[![Documentation Status](https://readthedocs.org/projects/sconce-scms/badge/?version=latest)](http://debias-infer.readthedocs.io/?badge=latest)

# Efficient Inference on High-Dimensional Linear Models With Missing Outcomes

This package implement the proposed debiasing method for conducting valid inference on the high-dimensional linear regression function with missing outcomes. We also document all the code for the simulations and real-world applications in our paper [here](https://github.com/zhangyk8/Debias-Infer/tree/main/Paper_Code).

* Free software: MIT license
* Python Package Documentation: [https://debias-infer.readthedocs.io](https://debias-infer.readthedocs.io).


Installation guide
--------

```Debias-Infer``` requires Python 3.8+ (earlier version might be applicable), [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), [scikit-learn](https://scikit-learn.org/stable/), [CVXPY](https://www.cvxpy.org/), [statsmodels](https://www.statsmodels.org/). To install the latest version of ```Debias-Infer``` from this repository, run:

```
python setup.py install
```

To pip install a stable release, run:
```
pip install Debias-Infer
```

References
--------

<a name="debias">[1]</a> Y. Zhang, A. Giessing, Y.-C. Chen (2023+) Efficient Inference on High-Dimensional Linear Models with Missing Outcomes.

<a name="scaledlasso">[2]</a> T. Sun and C.-H. Zhang (2012). Scaled Sparse Linear Regression." *Biometrika*, **99**, no.4: 879-898.


