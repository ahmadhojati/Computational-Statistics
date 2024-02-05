#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def option_pricing_simulation(m, n, f, S0, T, r, K):
    """
    Perform Monte Carlo simulation for option pricing.

    Parameters:
    - m: Number of simulations
    - n: Number of steps or paths
    - f: Multidimensional array representing the evolution of the underlying asset's value over time
    - S0: Initial value of the underlying asset
    - T: Time to maturity
    - r: Risk-free interest rate
    - K: Strike price of the option

    Returns:
    - mu_mc: Mean values of the option payoff over all simulations
    - theta_mc: Mean values of another related value over all simulations
    """

    # Initialize arrays to store results
    mu_mc = np.zeros(m)
    theta_mc = np.zeros(m)

    # Monte Carlo simulations
    for j in range(m):
        # Initialize array to store asset values over time
        ST = np.zeros((T+1, n))
        ST[0] = S0 + ST[0]

        # Generate asset values over time
        for k in range(1, T+1):
            ST[k] = ST[k-1] * f[j, :, k]

        # Calculate option payoff and another related value
        A = (np.exp(-r*T/365) * np.maximum(np.zeros(n), np.mean(ST[1:T+1], axis=0) - K)).T
        theta = np.exp(-r*T/365) * np.maximum(np.zeros(n), np.exp(np.mean(np.log(ST[1:T+1]), axis=0)) - K)

        # Store mean values in respective arrays
        mu_mc[j] = np.mean(A)
        theta_mc[j] = np.mean(theta)

    # Return results
    return mu_mc, theta_mc

