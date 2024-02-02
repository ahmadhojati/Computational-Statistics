#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt
import seaborn as sns

def alpha(x, y):
    """
    Compute the acceptance probability for the Metropolis algorithm.

    Parameters:
    - x: Current state.
    - y: Proposed state.

    Returns:
    - Probability of accepting the proposed state.
    """
    return min(1, (0.7 * norm.pdf(y, 7, 0.5) + 0.3 * norm.pdf(y, 10, 0.5)) / (0.7 * norm.pdf(x, 7, 0.5) + 0.3 * norm.pdf(x, 10, 0.5)))

def N_metro(mu, sigma, nitr):
    """
    Perform the Metropolis algorithm for a normal distribution.

    Parameters:
    - mu: Mean of the normal distribution.
    - sigma: Standard deviation of the normal distribution.
    - nitr: Number of iterations.

    Returns:
    - r: Array of sampled values.
    - Estimated Mean.
    - Estimated Standard Deviation.
    """
    
    # Initialize array to store sampled values
    r = np.zeros(nitr)
    
    # Initialize current state
    x = mu
    
    # Perform Metropolis algorithm iterations
    for k in range(0, nitr):
        # Generate a random value from a uniform distribution
        u = uniform.rvs(0, 1, 1)
        
        # Propose a new state from a normal distribution
        y = norm.rvs(x, sigma, 1)
        
        # Accept or reject the proposed state based on the acceptance probability
        if u < alpha(x, y):
            x = y
        else:
            x = mu
        
        # Record the current state
        r[k] = x
    
    # Create x-axis values for plotting
    xaxis = np.linspace(5, 11, 200)
    
    # Plot the histogram of sampled values and the true distribution
    plotf(r, xaxis, mu, sigma)
    
    # Print estimated mean and standard deviation
    return r, print('Estimated Mean = ', np.mean(r)), print('Estimated std = ', np.std(r))

def plotf(r, xaxis, mu, sigma):
    """
    Plot sample paths and the mixture of normal density.

    Parameters:
    - r: Array of sampled values.
    - xaxis: Array of x-axis values for plotting.
    - mu: Mean of the normal distribution.
    - sigma: Standard deviation of the normal distribution.
    """
    
    # Create a figure with two subplots
    plt.figure(figsize=(15, 10))
    
    # Plot sample paths for δ in the first subplot
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, len(r), len(r)), r)
    plt.title('Sample paths for δ', fontsize=15)
    plt.xlabel('t', fontsize=15)
    plt.ylabel('δ', fontsize=15)
    
    # Plot the mixture of normal density in the second subplot
    plt.subplot(2, 1, 2)
    sns.distplot(r, hist=True, bins=20, kde=False, color='green', hist_kws={'edgecolor': 'black'}, norm_hist=True)
    y = 0.7 * norm.pdf(xaxis, 7, 0.5) + 0.3 * norm.pdf(xaxis, 10, 0.5)
    plt.plot(xaxis, y, 'k-')
    plt.title('Mixture of Normal Density', fontsize=15)
    plt.xlabel('y', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    
    # Show the plot
    plt.show()


