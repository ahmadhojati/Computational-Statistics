#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def loglikelihood(func, x0, theta0, n):
    """
    Compute the log-likelihood values for a given function and parameter values.

    Parameters:
    - func: Function for which the log-likelihood is calculated.
    - x0: Data or observations used in the likelihood calculation.
    - theta0: Array or list of parameter values.
    - n: Number of parameter sets.

    Returns:
    - L: Array of log-likelihood values corresponding to each parameter set.
    """
    
    # Initialize an array to store log-likelihood values
    L = np.zeros(n)
    
    # Loop over the parameter sets
    for i in range(0, n):
        # Calculate the log-likelihood for each parameter set
        L[i] = np.log(np.prod(func(x0, theta0[i])))
    
    return L

