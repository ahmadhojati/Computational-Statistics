#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def lglklihd_N(x0, b1, b2, alpha0, alpha1):
    """
    Numerical loglikelihood function for Poisson distribution.

    Parameters:
    - x0: Array of observed counts.
    - b1: Coefficients for the first predictor variable.
    - b2: Coefficients for the second predictor variable.
    - alpha0: Array of values for the first parameter.
    - alpha1: Array of values for the second parameter.

    Returns:
    - Tuple containing meshgrid matrices for alpha0 and alpha1, and reshaped loglikelihood values.

    Notes:
    - The function calculates the loglikelihood for a Poisson distribution using numerical methods.
    - It creates a meshgrid of alpha0 and alpha1 values.
    - It iterates through the flattened meshgrid, computing the loglikelihood for each combination.
    - The loglikelihood values are reshaped to a 2D array and returned along with the meshgrid matrices.

    """
    X, Y = np.meshgrid(alpha0, alpha1)  # Create a meshgrid of alpha0 and alpha1 values
    n = len(alpha1)  # Number of values for the second parameter
    x = np.reshape(X, n**2)  # Flatten the meshgrid for alpha0
    y = np.reshape(Y, n**2)  # Flatten the meshgrid for alpha1
    l = np.zeros(n**2)  # Initialize an array for loglikelihood values

    # Iterate through flattened meshgrid, computing loglikelihood for each combination
    for i in range(n**2 - 1):
        l[i] = sum(x0 * np.log(x[i] * b1 + y[i] * b2)) - sum((x[i] * b1 + y[i] * b2)) - sum(np.log(factorial(x0)))

    # Reshape the loglikelihood values to a 2D array
    return X, Y, np.reshape(l, (n, n))

