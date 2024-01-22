# Expectation and Maximization - Gaussian Mixture Model

This repository contains Python code implementing the Expectation and Maximization (EM) algorithm for a Gaussian Mixture Model (GMM). The GMM is applied to both a univariate Poisson distribution and a multivariate normal distribution.

## Table of Contents
- [Description](#description)
- [Usage](#usage)
- [Files](#files)
- [Dependencies](#dependencies)
- [License](#license)

## Description

The code includes two main sections:

1. **EM Algorithm for Poisson Distribution**
   - The EM algorithm is applied to estimate parameters for a Poisson distribution.
   - It iteratively updates parameters such as weights, means, and standard deviations.
   - The algorithm terminates when the change in parameters falls below a specified threshold.

2. **Gaussian Mixture Model for Multivariate Normal Distribution**
   - The GMM is implemented for a multivariate normal distribution using the EM algorithm.
   - Initial values for weights, means, and covariances are provided.
   - The code outputs Maximum Likelihood Estimates (MLE) for the parameters.
   - It includes a log-likelihood plot to visualize the convergence of the algorithm.
   - The dataset is classified into two classes based on the GMM estimates, and the results are plotted.
