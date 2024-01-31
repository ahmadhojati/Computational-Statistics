#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy import stats

def kernel_smooth(data, n):
    """
    Perform kernel smoothing (kernel density estimation) on the given data.

    Parameters:
    - data: Array or list of data points.
    - n: Number of points for evaluating the smoothed kernel density.

    Returns:
    - lambda_ks: Array of points for kernel density estimation.
    - f_lambda_ks: Smoothed kernel density estimates at points in lambda_ks.
    """
    # Create a kernel density estimator using Gaussian kernel
    kde = stats.gaussian_kde(data)

    # Generate points for kernel density estimation
    lambda_ks = np.linspace(data.min(), data.max(), n)

    # Evaluate the kernel density estimator at specified points
    f_lambda_ks = kde(lambda_ks)

    return lambda_ks, f_lambda_ks

