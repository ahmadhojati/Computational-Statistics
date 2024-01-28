#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def Fisher_N(f1, f2, Fisher, a01, a02, eps):
    """
    Fisher Scoring function for parameter estimation.

    Parameters:
    - f1: Function representing the first component of the vector-valued function.
    - f2: Function representing the second component of the vector-valued function.
    - Fisher: Function providing the Fisher information matrix.
    - a01: Initial guess for the first parameter.
    - a02: Initial guess for the second parameter.
    - eps: Tolerance to determine convergence.

    Returns:
    - Tuple containing the optimized values for the first and second parameters.

    Notes:
    - The function uses Fisher Scoring, an iterative optimization method, for parameter estimation.
    - The iteration stops when the sum of absolute values of both components of the vector-valued function is below 'eps'.
    - The algorithm employs a cyclic buffer to store the last 6 values of the parameters, standard errors, and Fisher matrix for printing and analysis.
    - Standard errors are calculated from the inverse of the Fisher information matrix.

    """
    i = 0            # Initialize iteration counter
    nitr = 1         # Initialize total number of iterations
    alpha1 = np.zeros(6)  # Initialize an array to store the values of the first parameter
    alpha2 = np.zeros(6)  # Initialize an array to store the values of the second parameter
    std_err_alpha1 = np.zeros(6) + 100  # Initialize an array to store the standard errors of the first parameter
    std_err_alpha2 = np.zeros(6) + 100  # Initialize an array to store the standard errors of the second parameter
    alpha1[0] = a01  # Set the initial guess for the first parameter
    alpha2[0] = a02  # Set the initial guess for the second parameter

    # Iterate until convergence
    while abs(f1(alpha1[i], alpha2[i])) + abs(f2(alpha1[i], alpha2[i])) > eps:
        nitr += 1  # Increment iteration counter

        # Handle cyclic buffer for storing parameter values, standard errors, and Fisher matrix
        if i == 5:
            alpha1[0] = alpha1[i]
            alpha2[0] = alpha2[i]
            std_err_alpha1[0] = std_err_alpha1[i]
            std_err_alpha2[0] = std_err_alpha2[i]
            i = 0

        # Update parameter values using Fisher Scoring
        [[alpha1[i + 1]], [alpha2[i + 1]]] = [[alpha1[i]], [alpha2[i]]] + inv(Fisher(alpha1[i], alpha2[i])).dot(
            [[f1(alpha1[i], alpha2[i])], [f2(alpha1[i], alpha2[i])]])

        # Update standard errors using the inverse of the Fisher information matrix
        std_err_alpha1[i + 1] = inv(Fisher(alpha1[i], alpha2[i]))[0][0]
        std_err_alpha2[i + 1] = inv(Fisher(alpha1[i], alpha2[i]))[1][1]
        i += 1

    # Create an index array for proper printing order
    ID = np.concatenate((np.arange(nitr % 5, 5), np.arange(0, nitr % 5)), axis=None)

    # Print the results in tabular format
    print(tabulate({"Iteration": list(np.arange(nitr - 4, nitr + 1)),
                    '\u03B11': list(alpha1[ID]),
                    '\u03B12': list(alpha2[ID]),
                    'Fisher Matrix': np.array(Fisher(alpha1[ID], alpha2[ID])).T,
                    'Standard Error \u03B11': list(std_err_alpha1[ID]),
                    'Standard Error \u03B12': list(std_err_alpha2[ID])},
                   headers="keys"))

    # Return the optimized parameter values
    return alpha1[i], alpha2[i]

