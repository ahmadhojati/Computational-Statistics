#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def Newton(f, g, a, eps):
    """
    Perform Newton's Method to find the root of a function.

    Parameters:
    - f: Function for which the root is sought.
    - g: Derivative of the function f.
    - a: Initial guess for the root.
    - eps: Tolerance for stopping criteria.

    Returns:
    - a: Estimated value of the root.
    """
    
    # Initialize iteration count
    i = 0
    
    # Iterate until the derivative at the current estimate is below the tolerance
    while abs(g(a)) > eps:
        # Update the estimate using Newton's method
        a = a + f(a)
        i += 1
        
        # Print iteration details
        print('Number of iteration = {}'.format(i))
        print('X = {}'.format(a))
    
    # Print the final estimate of the root
    print('The root of f(x) is achieved at {}'.format(a))
    
    return a

