#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def secant(func, x0, x1, eps):
    """
    Perform the Secant Method to find the root of a function.

    Parameters:
    - func: Function for which the root is sought.
    - x0: Initial guess for the root.
    - x1: Second initial guess for the root.
    - eps: Tolerance for stopping criteria.

    Returns:
    - x2: Estimated value of the root.
    """
    
    # Initialize iteration count
    i = 1
    
    # Iterate until the function value at the current estimate is below the tolerance
    while abs(func(x1)) > eps:
        # Update the estimate using the Secant Method formula
        x2 = x1 - func(x1) * (x1 - x0) / (func(x1) - func(x0))
        
        # Update previous estimates
        x0 = x1
        x1 = x2
        
        i += 1
    
    # Print the number of iterations and the estimated root
    print('Number of iterations = {}'.format(i))
    print('The root of f(θ) is achieved at {}'.format(x2))
    
    return x2

