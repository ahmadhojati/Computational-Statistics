#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def bisection(f, a, b, eps):
    """
    Perform the Bisection Method to find the root of a function within a given interval.

    Parameters:
    - f: Function for which the root is sought.
    - a: Left endpoint of the interval.
    - b: Right endpoint of the interval.
    - eps: Tolerance for stopping criteria.

    Returns:
    - c: Estimated value of the root.
    """
    
    # Initialize iteration count
    i = 1
    
    # Iterate until the interval width is below the tolerance
    while abs(a - b) > eps:
        # Calculate the midpoint of the interval
        c = a + (b - a) / 2
        
        # Update the interval based on the sign of the function values at the endpoints
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        
        i += 1
    
    # Print the number of iterations and the estimated root
    print('Number of iterations = {}'.format(i))
    print('The root of f(θ) within the interval [{}, {}] is achieved at {}'.format(a, b, c))
    
    return c

