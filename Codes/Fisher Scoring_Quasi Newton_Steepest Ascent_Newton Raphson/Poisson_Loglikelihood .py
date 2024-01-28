#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def lglklihd_N(x0, b1, b2):
    """
    Compute the Parametric Loglikelihood for the Poisson distribution.

    Parameters:
    - x0: List of observed counts.
    - b1: List of coefficients for the first parameter.
    - b2: List of coefficients for the second parameter.

    Returns:
    - Loglikelihood value.

    Notes:
    - This function calculates the loglikelihood for a Poisson distribution with parameters alpha1 and alpha2.
    - Uses the formula for the Poisson probability mass function for each observation in the dataset.
    - Assumes the observed counts follow a Poisson distribution with parameters alpha1 and alpha2.
    - The loglikelihood is computed as the logarithm of the product of individual Poisson probabilities.

    """
    l = 1  # Initialize likelihood value
    alpha1 = sym.Symbol('a1')  # Symbolic representation for the first parameter
    alpha2 = sym.Symbol('a2')  # Symbolic representation for the second parameter

    # Iterate over each observation in the dataset
    for i in range(0, len(x0)):
        # Update likelihood with the Poisson probability mass function
        l *= (alpha1 * b1[i] + alpha2 * b2[i]) ** x0[i] * exp(-(alpha1 * b1[i] + alpha2 * b2[i])) / np.math.factorial(x0[i])

    # Return the loglikelihood value
    return log(l)

