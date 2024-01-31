#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy.linalg import inv
from scipy.stats import logistic
from tabulate import tabulate

def IRLS(y, z, n, Beta0):
    """
    Perform Iterative Reweighted Least Squares (IRLS) for logistic regression.

    Parameters:
    - y: Array or list of binary response values (0 or 1).
    - z: Array or list of predictor variable values.
    - n: Number of iterations.
    - Beta0: Initial guess for the coefficients.

    Returns:
    - Beta: Coefficient estimates at each iteration.
    - l_zegond: Inverse of the second derivative of the log-likelihood function.
    """
    
    # Prepare the design matrix Z
    Z = np.vstack((np.ones(len(z)), z))
    
    # Define table headers for displaying iteration details
    header = ['Iteration, t', 'βᵗ', '-l″(βᵗ)⁻¹']
    
    # Initialize Beta matrix to store coefficient estimates at each iteration
    Beta = np.zeros((2, n))
    Beta[0, 0] = Beta0  # Set initial guess for coefficients
    
    # Iterative Reweighted Least Squares (IRLS) loop
    for i in range(1, n):
        # Calculate predicted probabilities using the logistic function
        p = logistic.cdf(np.dot(Z.transpose(), Beta[:, i-1]))
        
        # Calculate the gradient (first derivative) of the log-likelihood function
        l_prime = np.dot(Z, (y - p))
        
        # Calculate the weight matrix W based on predicted probabilities
        W = np.diag(p * (1 - p))
        
        # Calculate the second derivative of the log-likelihood function
        l_zegond = -Z.dot(W).dot(Z.transpose())
        
        # Update coefficients using the IRLS update formula
        Beta[:, i] = Beta[:, i-1] - np.dot(inv(l_zegond), l_prime)

        # Display iteration details in a table
        table = [[i-1, Beta[:, i-1], -inv(l_zegond)]]
        print(tabulate(table, headers=header))

    # Return the final coefficient estimates and the inverse of the second derivative
    return Beta, -inv(l_zegond)

