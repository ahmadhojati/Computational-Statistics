#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def loglklihd(y, z, Beta0, Beta1):
    """
    Compute the log-likelihood function for logistic regression over a grid of Beta0 and Beta1 values.

    Parameters:
    - y: Array or list of binary response values (0 or 1).
    - z: Array or list of predictor variable values.
    - Beta0: Array or list of intercept values for the grid.
    - Beta1: Array or list of slope values for the grid.

    Returns:
    - l: 2D array representing the log-likelihood values over the grid.
    """
    
    # Prepare the design matrix Z
    Z = np.vstack((np.ones(len(z)), z))
    
    # Create a grid of Beta0 and Beta1 values using meshgrid
    X, Y = np.meshgrid(Beta0, Beta1)
    
    # Flatten the grid arrays to create a matrix of Beta values
    n = len(Beta0)
    x0 = np.reshape(X, n**2)
    y0 = np.reshape(Y, n**2)
    Beta = np.vstack((x0, y0))
    
    # Initialize an array to store log-likelihood values
    l = np.zeros(n**2)
    
    # Loop over the flattened Beta values to compute log-likelihood
    for i in range(1, n**2):
        # Calculate predicted probabilities using the logistic function
        p = 1 - 1 / (1 + np.exp(np.dot(Z.transpose(), Beta[:, i-1])))
        
        # Calculate log-likelihood values based on the predicted probabilities
        b = -np.log(1 - p)
        l[i] = y.transpose().dot(Z.transpose()).dot(Beta[:, i-1]) - sum(b)
    
    # Reshape the log-likelihood values to a 2D array
    return np.reshape(l, (n, n))

