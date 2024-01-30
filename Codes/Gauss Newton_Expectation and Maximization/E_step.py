#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def E_step(x, p):
    """
    Calculate the expectation (E-step) for given parameters.

    Parameters:
    - x: Input values
    - p: Parameter values

    Returns:
    - Array representing the calculated expectations
    """

    # Calculate expectations based on the given formulas
    cc = x[0] * (p[0]**2) / (p[0]**2 + 2*p[0]*p[1] + 2*p[0]*p[2])
    ci = 2 * x[0] * p[0] * p[1] / (p[0]**2 + 2*p[0]*p[1] + 2*p[0]*p[2])
    ct = 2 * x[0] * p[0] * p[2] / (p[0]**2 + 2*p[0]*p[1] + 2*p[0]*p[2])
    ii = x[1] * (p[1]**2) / (p[1]**2 + 2*p[1]*p[2])
    it = 2 * x[1] * p[1] * p[2] / (p[1]**2 + 2*p[1]*p[2])

    # Create an array representing the calculated expectations
    n = np.array([cc, ci, ct, ii, it, x[2]])

    return n

