#!/usr/bin/env python

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Debias-Infer",
    version="0.0.6",
    author="Yikun Zhang",
    author_email="yikunzhang@foxmail.com",
    description="Efficient Inference on High-Dimensional Linear Models With Missing Outcomes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhangyk8/Debias-Infer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=["debiasing"]),
    install_requires=["numpy >= 1.16", "scipy >= 1.1.0", "cvxpy[CVXOPT,MOSEK]", "scikit-learn", "statsmodels"],
    python_requires=">=3.8",
)
