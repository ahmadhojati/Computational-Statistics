#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import poisson

def Likelihood(sample, X):
    """
    Compute the likelihood values for each sample.

    Parameters:
    - sample: Array of samples.
    - X: Array of observed values.

    Returns:
    - L: Array of likelihood values corresponding to each sample.
    """
    
    # Initialize an array to store likelihood values
    L = np.zeros(len(sample))
    
    # Loop over each sample in the array
    for i in range(0, len(sample)):
        # Calculate the product of Poisson probability mass functions for each observed value
        L[i] = np.prod(poisson.pmf(X, sample[i]))
    
    return L

