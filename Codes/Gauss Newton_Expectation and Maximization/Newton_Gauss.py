#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy.linalg import inv
from tabulate import tabulate

def Newton_Gauss(r0, k0, N, t, eps):
    """
    Perform Gauss-Newton optimization to estimate parameters 'r' and 'k' for a given model.

    Parameters:
    - r0: Initial guess for parameter 'r'
    - k0: Initial guess for parameter 'k'
    - N: Observed values
    - t: Independent variable values
    - eps: Convergence threshold

    Returns:
    - Tuple containing the final estimates for 'r' and 'k'
    """

    # Initialize variables and arrays
    i = 0
    nitr = 1
    r1 = np.zeros(6)
    k1 = np.ones(6)
    SE_k = np.zeros(6)
    SE_r = np.zeros(6)
    corr = np.zeros(6)
    r1[0] = r0
    k1[0] = k0

    # Iterative optimization loop
    while sum(N - f(r1[i], k1[i], t))**2 > eps:
        nitr += 1 

        # Check for the end of the array
        if i == 5:
            r1[0] = r1[i]
            k1[0] = k1[i]
            SE_r[0] = SE_r[i]
            SE_k[0] = SE_k[i]
            corr[0] = corr[i]
            i = 0

        # Construct the Jacobian matrix A
        A = np.array([f_r(r1[i], k1[i], t), f_k(r1[i], k1[i], t)]).T    

        # Update parameter estimates using Gauss-Newton method
        [r1[i+1], k1[i+1]] = [r1[i], k1[i]] + inv(A.T.dot(A)).dot(A.T).dot(N - f(r1[i], k1[i], t))

        # Calculate Standard Errors (SE) and Correlation coefficient for parameters
        SE_r[i+1] = np.sqrt(inv(A.T.dot(A)).diagonal())[0] * np.sqrt(sum((N - f(r1[i], k1[i], t))**2) / (len(N) - 2))
        SE_k[i+1] = np.sqrt(inv(A.T.dot(A)).diagonal())[1] * np.sqrt(sum((N - f(r1[i], k1[i], t))**2) / (len(N) - 2))
        corr[i+1] = inv(A.T.dot(A))[0][1] / (np.sqrt(inv(A.T.dot(A)).diagonal())[0] * np.sqrt(inv(A.T.dot(A)).diagonal())[1])

        i += 1

        # Check for maximum number of iterations
        if nitr > 100:
            print('r and k estimates after 100 iterations:')
            break

    # Reorder the results for tabular display
    ID = np.concatenate((np.arange(nitr % 5, 5), np.arange(0, nitr % 5)), axis=None)

    # Display results in a table using the tabulate library
    print(tabulate({"Iteration": list(np.arange(nitr - 4, nitr + 1)),
                    'r': list(r1[ID]), 'k': list(k1[ID]),
                    'SE for r': list(SE_r[ID]), 'SE for k': list(SE_k[ID]),
                    'Correlation': list(corr[ID])}, headers="keys"))

    # Return the final parameter estimates
    return r1[i], k1[i]

