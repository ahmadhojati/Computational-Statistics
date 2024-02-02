#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sympy as sym
import numpy as np

def lglklihd(x0):
    """
    Define the likelihood function for a distribution with a given formula.

    Parameters:
    - x0: Array or list of parameter values.

    Returns:
    - log_likelihood: Symbolic expression representing the logarithm of the likelihood function.
    """
    
    # Initialize the likelihood to 1
    l = 1
    
    # Define a symbolic variable 't' using SymPy
    t = sym.Symbol('t')
    
    # Loop over the parameter values in x0
    for i in range(0, len(x0)):
        # Multiply the likelihood by the distribution probability density function
        l *= (1 - sym.cos(x0[i] - t)) / (2 * np.pi)
    
    # Return the logarithm of the likelihood as a symbolic expression
    log_likelihood = sym.log(l)
    return log_likelihood

