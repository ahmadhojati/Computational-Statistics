#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def Newton(f, g, a, eps):
    """
    Perform Newton's Method to find the maximum of a function.

    Parameters:
    - f: Function for which the maximum is sought.
    - g: Derivative of the function f.
    - a: Initial guess for the maximum.
    - eps: Tolerance for stopping criteria.

    Returns:
    - a: Estimated value of the maximum.
    """
    
    # Initialize iteration count
    i = 0
    
    # Iterate until the derivative at the current estimate is below the tolerance
    while abs(g(a)) > eps:
        # Update the estimate using Newton's method
        a = a + f(a)
        i += 1
    
    # Print the number of iterations and the estimated maximum
    print('Number of iterations = {}'.format(i))
    print('The maximum of f(θ) with respect to θ is achieved at {}'.format(a))
    
    return a

