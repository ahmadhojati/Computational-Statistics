#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def steepest_ascent_backtrack(f, f1, f2, a01, a02, eps, step):
    """
    Steepest Ascent function for optimization with backtracking.

    Parameters:
    - f: Function representing the objective function to be optimized.
    - f1: Function representing the first component of the gradient of the objective function.
    - f2: Function representing the second component of the gradient of the objective function.
    - a01: Initial guess for the first variable.
    - a02: Initial guess for the second variable.
    - eps: Tolerance to determine convergence.
    - step: Initial step size for updating variables.

    Returns:
    - Tuple containing the optimized values for the first and second variables.

    Notes:
    - The function uses Steepest Ascent with backtracking, an iterative optimization method, for variable optimization.
    - The iteration stops when the sum of absolute values of both components of the gradient is below 'eps'.
    - The algorithm employs a cyclic buffer to store the last 6 values of the variables and step sizes for printing and analysis.
    - Backtracking is used to ensure that the objective function is non-increasing in each iteration.
    - If the algorithm does not converge after 100 iterations, a warning message is printed.

    """
    i = 0            # Initialize iteration counter
    nitr = 1         # Initialize total number of iterations
    alpha1 = np.zeros(6)  # Initialize an array to store the values of the first variable
    alpha2 = np.zeros(6)  # Initialize an array to store the values of the second variable
    alpha1[0] = a01  # Set the initial guess for the first variable
    alpha2[0] = a02  # Set the initial guess for the second variable
    step_size = np.zeros(6)  # Initialize an array to store the step sizes
    step_size[0] = step  # Set the initial step size

    M = -np.eye(2)  # Negative identity matrix used for steepest ascent

    # Iterate until convergence or maximum iterations
    while abs(f1(alpha1[i], alpha2[i])) + abs(f2(alpha1[i], alpha2[i])) > eps:
        nitr += 1  # Increment iteration counter

        # Handle cyclic buffer for storing variable values and step sizes
        if i == 5:
            alpha1[0] = alpha1[i]
            alpha2[0] = alpha2[i]
            step_size[0] = step
            i = 0

        # Update variable values using steepest ascent with backtracking
        [[alpha1[i + 1]], [alpha2[i + 1]]] = [[alpha1[i]], [alpha2[i]]] - step * inv(M).dot(
            [[f1(alpha1[i], alpha2[i])], [f2(alpha1[i], alpha2[i])]])

        # Backtracking to ensure non-increasing objective function
        while f(alpha1[i + 1], alpha2[i + 1]) < f(alpha1[i], alpha2[i]):
            step = step / 2
            [[alpha1[i + 1]], [alpha2[i + 1]]] = [[alpha1[i]], [alpha2[i]]] - step * inv(M).dot(
                [[f1(alpha1[i], alpha2[i])], [f2(alpha1[i], alpha2[i])]])

        step_size[i + 1] = step  # Store the step size
        i += 1

        # Check for non-convergence after 100 iterations
        if nitr > 100:
            print('Steepest ascent backtracking does not converge after 100 iterations')
            break

    # Create an index array for proper printing order
    ID = np.concatenate((np.arange(nitr % 5, 5), np.arange(0, nitr % 5)), axis=None)

    # Print the results in tabular format
    print(tabulate({"Iteration": list(np.arange(nitr - 4, nitr + 1)),
                    '\u03B11': list(alpha1[ID]),
                    '\u03B12': list(alpha2[ID]),
                    'Step size': list(step_size[ID])},
                   headers="keys"))

    # Return the optimized variable values
    return alpha1[i], alpha2[i]

