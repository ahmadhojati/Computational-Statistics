#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def M_step(x, n):
    """
    Update parameter values using the maximization (M-step) based on given expectations.

    Parameters:
    - x: Input values
    - n: Expectation values

    Returns:
    - Array representing the updated parameter values
    """

    # Update parameters based on the given formulas
    c = (2 * n[0] + n[1] + n[2]) / (2 * sum(x))
    i = (2 * n[3] + n[4] + n[1]) / (2 * sum(x))
    t = (2 * n[5] + n[2] + n[4]) / (2 * sum(x))

    # Create an array representing the updated parameter values
    p = np.array([c, i, t])

    return p

