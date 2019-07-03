"""grainStats.py: utilities to generate grain size distributions."""


from scipy.stats import truncnorm 
from numpy import log, exp, sqrt

def generateTruncLogNormal(n, rmin, rmax, rmean, rstd):
    """Generates a realization of a truncated log normal distribution.

PARAMETERS
----------
n : int
    Number of values to generate.
rmin : float, int
    Minimum value of the distribution.
rmax : float, int
    Maximum value of the distribution.
rmean : float, int
    Mean value of the distribution.
restd : float, int
    Standard deviation of the dsitribution.

RETURNS
-------
vals : array
    Array of values based on a truncated log normal distribution with the given input parameters."""
    
    # Calculate statistics of logarithm
    log_rmin = log(rmin)
    log_rmax = log(rmax)
    log_rmean = log(rmean) - 0.5*log(1 + (rstd**2) / (rmean**2))
    log_rstd = sqrt(log(1 + rstd**2 / rmean**2))

    # Convert min and max for normal distribution to min and max for standard normal distribution
    a, b = (log_rmin - log_rmean) / log_rstd, (log_rmax - log_rmean) / log_rstd

    # Generate n values based on the truncated standard normal distribution
    vals = truncnorm.rvs(a, b, size=n)

    # Trandform the values back to a log normal distribution
    vals = exp(vals * log_rstd + log_rmean)

    return vals

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    n = 1000000
    rmin = 0.01
    rmax = 100
    rmean = 0.4
    rstd = 0.25

    dist = generateTruncLogNormal(n, rmin, rmax, rmean, rstd)
    print(np.mean(dist))

    fig, ax = plt.subplots()
    ax.hist(dist, bins=100)
    plt.show()