#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def Q_Newton_backtrack(f, f1, f2, a01, a02, eps, step):
    """
    Quasi Newton function with backtracking for optimization.

    Parameters:
    - f: Objective function to be optimized.
    - f1: Function representing the first component of the vector-valued function.
    - f2: Function representing the second component of the vector-valued function.
    - a01: Initial guess for the first variable.
    - a02: Initial guess for the second variable.
    - eps: Tolerance to determine convergence.
    - step: Initial step size for updating variables.

    Returns:
    - Tuple containing the optimized values for the first and second variables.

    Notes:
    - The function uses Quasi-Newton with backtracking, an iterative optimization method, for variable optimization.
    - The iteration stops when the sum of absolute values of both components of the vector-valued function is below 'eps'.
    - The algorithm employs a cyclic buffer to store the last 6 values of the variables, step sizes, and M matrices for printing and analysis.
    - Backtracking is implemented to adjust the step size to ensure convergence.
    - The BFGS update formula is used to update the inverse Hessian matrix approximation.
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
    M = [-np.eye(2), -np.eye(2), -np.eye(2), -np.eye(2), -np.eye(2), -np.eye(2)]  # List to store the inverse Hessian matrix approximations

    # Iterate until convergence or maximum iterations
    while abs(f1(alpha1[i], alpha2[i])) + abs(f2(alpha1[i], alpha2[i])) > eps:
        nitr += 1  # Increment iteration counter

        # Handle cyclic buffer for storing variable values, step sizes, and M matrices
        if i == 5:
            alpha1[0] = alpha1[i]
            alpha2[0] = alpha2[i]
            step_size[0] = step_size[i]
            M[0] = M[i]
            i = 0

        # Update variable values using Quasi-Newton
        [[alpha1[i + 1]], [alpha2[i + 1]]] = [[alpha1[i]], [alpha2[i]]] - step_size[i] * inv(M[i]).dot(
            [[f1(alpha1[i], alpha2[i])], [f2(alpha1[i], alpha2[i])]])

        # Backtracking to adjust step size for convergence
        while f(alpha1[i + 1], alpha2[i + 1]) < f(alpha1[i], alpha2[i]):
            step_size[i] = step_size[i] / 2
            [[alpha1[i + 1]], [alpha2[i + 1]]] = [[alpha1[i]], [alpha2[i]]] - step_size[i] * inv(M[i]).dot(
                [[f1(alpha1[i], alpha2[i])], [f2(alpha1[i], alpha2[i])]])

        # Calculate vectors and matrices for BFGS update
        Z = np.array([[alpha1[i + 1] - alpha1[i]], [alpha2[i + 1] - alpha2[i]]])
        Y = np.array([[f1(alpha1[i + 1], alpha2[i + 1]) - f1(alpha1[i], alpha2[i])],
                      [f2(alpha1[i + 1], alpha2[i + 1]) - f2(alpha1[i], alpha2[i])]])
        V = Y - M[i].dot(Z)
        M[i + 1] = M[i] - (M[i].dot(Z).dot((M[i].dot(Z)).T)) / (Z.T.dot(M[i]).dot(Z)) + Y.dot(Y.T) / ((Z.T).dot(Y))

        # If the curvature condition is met, update M; otherwise, retain the previous M
        if abs(V.T.dot(Z)) < eps:
            M[i + 1] = M[i]

        i += 1

        # Check for non-convergence after 100 iterations
        if nitr > 100:
#             print('Quasi Newton backtracking does not converge after 100 iterations')
            break

    # Create an index array for proper printing order
    ID = np.concatenate((np.arange(nitr % 5, 5), np.arange(0, nitr % 5)), axis=None)

    # Print the results in tabular format
    print(tabulate({"Iteration": list(np.arange(nitr - 4, nitr + 1)),
                    '\u03B11': list(alpha1[ID]),
                    '\u03B12': list(alpha2[ID]),
                    'Step size': list(step_size[ID]),
                    'M Matrix': np.array(M)[ID]},
                   headers="keys"))

    # Return the optimized variable values
    return alpha1[i], alpha2[i]

