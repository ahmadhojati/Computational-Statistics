#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import norm
import vgam  # Assuming vgam is a package you've imported

def w1(x, mu, sigma):
    """
    Compute weights for the first function.

    Parameters:
    - x: Array of values.
    - mu: Mean of the normal distribution.
    - sigma: Standard deviation of the normal distribution.

    Returns:
    - out: Computed weights.
    """
    
    # Compute the numerator using vgam.dslash(x)
    numerator = vgam.dslash(x)
    
    # Divide by the probability density function of the normal distribution
    out = numerator / norm.pdf(x, mu, sigma)
    
    # Normalize the weights
    out = out / sum(out)
    
    return out

def w2(x, mu, sigma):
    """
    Compute weights for the second function.

    Parameters:
    - x: Array of values.
    - mu: Mean of the normal distribution.
    - sigma: Standard deviation of the normal distribution.

    Returns:
    - out: Computed weights.
    """
    
    # Compute the numerator using the probability density function of the normal distribution
    numerator = norm.pdf(x, mu, sigma)
    
    # Divide by vgam.dslash(x)
    out = numerator / vgam.dslash(x)
    
    # Normalize the weights
    out = out / sum(out)
    
    return out

