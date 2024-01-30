#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy.linalg import inv
from tabulate import tabulate

def Newton(r0, k0, t, N, eps):
    """
    Perform Newton-Raphson optimization to estimate parameters 'r' and 'k' for a given model.

    Parameters:
    - r0: Initial guess for parameter 'r'
    - k0: Initial guess for parameter 'k'
    - t: Independent variable values
    - N: Observed values
    - eps: Convergence threshold

    Returns:
    - Tuple containing the final estimates for 'r' and 'k'
    """

    # Initialize variables and arrays
    i = 0
    nitr = 1
    r1 = np.zeros(6)
    k1 = np.ones(6)
    SE_r = np.zeros(6)
    SE_k = np.zeros(6)
    corr = np.zeros(6)
    Hessian = np.zeros((10, 2))
    r1[0] = r0
    k1[0] = k0

    # Iterative optimization loop
    while sum(N - g(r1[i], k1[i], t, N))**2 > eps:
        nitr += 1 

        # Check for the end of the array
        if i == 5:
            r1[0] = r1[i]
            k1[0] = k1[i]
            Hessian[0:2, :] = H
            SE_r[0] = SE_r[i]
            SE_k[0] = SE_k[i]
            corr[0] = corr[i]
            i = 0

        # Construct the Hessian matrix H
        H = np.array([[sum(g_rr(r1[i], k1[i], t, N)), sum(g_rk(r1[i], k1[i], t, N))],
                      [sum(g_rk(r1[i], k1[i], t, N)), sum(g_kk(r1[i], k1[i], t, N))]])

        # Update parameter estimates using Newton-Raphson method
        [[r1[i+1]], [k1[i+1]]] = [[r1[i]], [k1[i]]] - inv(H).dot([[sum(g_r(r1[i], k1[i], t, N))],
                                                                   [sum(g_k(r1[i], k1[i], t, N))]])
        Hessian[2*i:2*i+2, :] = H
        SE_r[i+1] = np.sqrt(inv(H)[0][0])
        SE_k[i+1] = np.sqrt(inv(H)[1][1])
        corr[i+1] = (inv(H)[0][1]) / (np.sqrt(inv(H)[0][0]) * np.sqrt(inv(H)[1][1]))
        i = i + 1

        # Check for maximum number of iterations
        if nitr > 100:
            print('r and k estimates after 100 iterations:')
            break

    # Reorder the results for tabular display
    ID = np.concatenate((np.arange(nitr % 5, 5), np.arange(0, nitr % 5)), axis=None)

    # Display results in a table using the tabulate library
    print(tabulate({"Iteration": list(np.arange(nitr-4, nitr+1)),
                    '\u03B11': list(r1[ID]), '\u03B12': list(k1[ID]),
                    'Hessian Matrix': np.array(Hessian),
                    'SE for r': list(SE_r[ID]), 'SE for k': list(SE_k[ID]),
                    'Correlation': list(corr[ID])}, headers="keys"))

    # Return the final parameter estimates
    return r1[i], k1[i]

