#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import norm

def reject_sampling(g, n):
    """
    Perform rejection sampling to generate samples from a distribution using a proposal distribution.

    Parameters:
    - g: Probability density function (PDF) of the target distribution.
    - n: Number of samples to generate.

    Returns:
    - s: List of accepted samples.
    - var_s: Variance of the accepted samples.
    """
    s = []  # List to store accepted samples
    for i in range(n):
        z = np.random.normal(0, 1, 1)  # Sample from the proposal distribution (normal distribution)
        u = np.random.uniform(0, 1)  # Sample from a uniform distribution
        # Accept the sample if it satisfies the rejection criterion
        if u <= g(z) / (3 * norm.pdf(z)):
            s.append(z)

    var_s = np.var(s)  # Calculate the variance of the accepted samples
    return s, var_s

