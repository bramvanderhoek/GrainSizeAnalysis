"""grainStats.py: utilities to generate grain size distributions."""

from scipy.stats import truncnorm, rv_continuous
import numpy as np


class TruncLogNormal(rv_continuous):

    def __init__(self, mean, std, a, b):
        super(TruncLogNormal, self).__init__()
        self._mean = mean
        self._std = std
        self._a = a
        self._b = b

    def _pdf(self, x):
        return 1 / (x * self._std * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - self._mean)**2 / (2 * self._std**2))


def generate_trunc_log_normal(n, rmin, rmax, rmean, rstd, seed=False):
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
rstd : float, int
    Standard deviation of the distribution.
seed : bool, int
    If provided, use seed to generate realization.

RETURNS
-------
vals : array
    Array of values based on a truncated log normal distribution with the given input parameters."""
    
    # Calculate statistics of logarithm
    log_rmin = np.log(rmin)
    log_rmax = np.log(rmax)
    log_rmean = np.log(rmean) - 0.5*np.log(1 + (rstd**2) / (rmean**2))
    log_rstd = np.sqrt(np.log(1 + rstd**2 / rmean**2))

    # Convert min and max for normal distribution to min and max for standard normal distribution
    a, b = (log_rmin - log_rmean) / log_rstd, (log_rmax - log_rmean) / log_rstd

    if seed is not False:
        np.random.seed(seed)

    # Generate n values based on the truncated standard normal distribution
    vals = truncnorm.rvs(a, b, size=n)

    # Transform the values back to a log normal distribution
    vals = np.exp(vals * log_rstd + log_rmean)

    return vals


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n = 1000000
    rmin = 0.01
    rmax = 100
    rmean = 0.4
    rstd = 0.25

    dist = generate_trunc_log_normal(n, rmin, rmax, rmean, rstd)
    print(np.mean(dist))

    fig, ax = plt.subplots()
    ax.hist(dist, bins=100)
    plt.show()