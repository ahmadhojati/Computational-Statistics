#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def fixed_point(func, x0, alpha, eps, max_iter):
    """
    Perform the Fixed-Point Iteration method to find the fixed point of a function.

    Parameters:
    - func: Function for which the fixed point is sought.
    - x0: Initial guess for the fixed point.
    - alpha: Step size or damping factor for the iteration.
    - eps: Tolerance for stopping criteria.
    - max_iter: Maximum number of iterations allowed.

    Returns:
    - x0: Estimated value of the fixed point.
    """
    
    # Initialize iteration count
    i = 0
    
    # Iterate until the function value at the current estimate is below the tolerance
    while abs(func(x0)) > eps:
        # Update the estimate using the Fixed-Point Iteration formula
        x0 = x0 + alpha * func(x0)
        i += 1
        
        # Check for maximum iteration limit
        if i > max_iter:
            print('Iteration exceeds {}, select new α'.format(max_iter))
            break
    
    # Print the number of iterations and the estimated fixed point
    print('Number of iterations = {}'.format(i))
    print('The fixed point of f(θ) is achieved at {}'.format(x0))
    
    return x0

