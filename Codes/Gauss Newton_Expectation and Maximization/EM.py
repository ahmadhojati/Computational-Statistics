#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tabulate import tabulate

def EM(x, p, p1, eps):
    """
    Perform Expectation-Maximization (EM) optimization to estimate parameters based on given input data.

    Parameters:
    - x: Input data
    - p: Initial parameter values
    - p1: Reference parameter values
    - eps: Convergence threshold

    Returns:
    - None (prints iteration details)
    """

    # Initialize variables
    R = 1
    itr = 0
    header = ['Iteration', 'P_C\u207D\u1d57\u207E', 'P_I\u207D\u1d57\u207E',
              'R\u207D\u1d57\u207E', "D_C\u207D\u1d57\u207E", 'D_I\u207D\u1d57\u207E']

    # Display initial iteration
    if itr == 0:
        table = [[itr, p[0], p[1], None, None, None]]
        print(tabulate(table, headers=header))

    # EM optimization loop
    while R > eps:
        p0 = p
        n = E_step(x, p)
        p = M_step(x, n)
        R = np.sqrt(sum((p[0:2] - p0[0:2])**2)) / np.sqrt(sum(p0[0:2]**2))
        D_c = (p[0] - p1[0]) / (p0[0] - p1[0])
        D_i = (p[1] - p1[1]) / (p0[1] - p1[1])
        table = [[itr, p[0], p[1], R, D_c, D_i]]
        print(tabulate(table, headers=header))
        itr += 1

# Note: The function does not return any value; it prints the iteration details.

