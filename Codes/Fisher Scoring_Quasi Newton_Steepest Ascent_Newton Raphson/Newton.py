#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def Newton(f1, f2, Hessian, a01, a02, eps):
    """
    Newton's method for finding the minimum of a multivariable function.

    Parameters:
    - f1: Function representing the first component of the vector-valued function.
    - f2: Function representing the second component of the vector-valued function.
    - Hessian: Function providing the Hessian matrix of the vector-valued function.
    - a01: Initial guess for the first variable.
    - a02: Initial guess for the second variable.
    - eps: Tolerance to determine convergence.

    Returns:
    - Tuple containing the optimized values for the first and second variables.

    Notes:
    - The function uses Newton's method to iteratively update the values of the variables until convergence.
    - The iteration stops when the sum of absolute values of both components of the vector-valued function is below 'eps'.
    - The algorithm employs a cyclic buffer to store the last 6 values of the variables for printing and analysis.

    """
    i = 0            # Initialize iteration counter
    nitr = 1         # Initialize total number of iterations
    alpha1 = np.zeros(6)  # Initialize an array to store the values of the first variable
    alpha2 = np.zeros(6)  # Initialize an array to store the values of the second variable
    alpha1[0] = a01  # Set the initial guess for the first variable
    alpha2[0] = a02  # Set the initial guess for the second variable

    # Iterate until convergence
    while abs(f1(alpha1[i], alpha2[i])) + abs(f2(alpha1[i], alpha2[i])) > eps:
        nitr += 1  # Increment iteration counter

        # Handle cyclic buffer for storing variable values
        if i == 5:
            alpha1[0] = alpha1[i]
            alpha2[0] = alpha2[i]
            i = 0

        # Update variable values using Newton's method
        [[alpha1[i + 1]], [alpha2[i + 1]]] = [[alpha1[i]], [alpha2[i]]] - inv(Hessian(alpha1[i], alpha2[i])).dot(
            [[f1(alpha1[i], alpha2[i])], [f2(alpha1[i], alpha2[i])]])
        i += 1

    # Create an index array for proper printing order
    ID = np.concatenate((np.arange(nitr % 5, 5), np.arange(0, nitr % 5)), axis=None)

    # Print the results in tabular format
    print(tabulate({"Iteration": list(np.arange(nitr - 4, nitr + 1)),
                    '\u03B11': list(alpha1[ID]),
                    '\u03B12': list(alpha2[ID]),
                    'Hessian Matrix': np.array(Hessian(alpha1[ID], alpha2[ID])).T},
                   headers="keys"))

    # Return the optimized variable values
    return alpha1[i], alpha2[i]

