#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def func(Xs, r, sigma):
    """
    Calculate a mathematical expression involving exponential and square root operations.

    Parameters:
    - Xs: Input values
    - r: Interest rate
    - sigma: Volatility

    Returns:
    - Result of the mathematical expression
    """
    f = np.exp(((r - (sigma**2)/2) / 365) + sigma * Xs / np.sqrt(365))
    return f

