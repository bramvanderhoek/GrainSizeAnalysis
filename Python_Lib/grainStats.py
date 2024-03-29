"""grainStats.py: utilities to generate grain size distributions."""

from scipy.stats import truncnorm, rv_histogram
import numpy as np


class TruncLogNorm:
    """Class representing a truncated log normal distribution."""

    def __init__(self, a, b, loc=0, scale=1):

        self._scale = scale
        self._loc = np.log(loc)
        # TODO: transform mu and sigma
        self._a = (np.log(a) - self._loc) / scale
        self._b = (np.log(b) - self._loc) / scale

    def pdf(self, x):
        """Get the value of the distribution's probability density function at location x."""
        pdf = truncnorm.pdf((np.log(x) - self._loc) / self._scale, self._a, self._b) / self._scale / x
        return pdf

    def rvs(self, size):
        """Pull a number of random variables from the distribution."""
        n_vals = truncnorm.rvs(self._a, self._b, size=size)
        ln_vals = np.exp(n_vals * self._scale + self._loc)
        return ln_vals

    def transform_moment(self, mu, std, lognorm_to_norm=True):
        """Transform first two moments of log normal distribution to the moments of normal distribution or vice versa."""
        if lognorm_to_norm:
            m1 = np.exp(np.log(mu) - 0.5 * np.log(1 + std**2 / mu**2))
            m2 = np.sqrt(np.log(1 + std**2 / mu**2))
        else:
            m1 = np.exp(mu + 0.5*std)
            m2 = m1**2 * (np.exp(std**2) - 1)
        return m1, m2


class DataDistribution:
    """Class representing a distribution from input data."""

    def __init__(self, data_points, c_freq):
        self.data_points = np.array(data_points)
        self.c_freq = np.array(c_freq)
        self.freq = np.diff(self.c_freq)
        arg = np.log(self.data_points)
        self.hist = rv_histogram((100*self.freq, arg))

    def pdf(self, x):
        pdf = self.hist.pdf(x)
        return pdf

    def rvs(self, size):
        values = np.exp(self.hist.rvs(size=size))
        return values


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

    data = [0.00063,0.002,0.016,0.032,0.05,0.063,0.125,0.25,0.5,1,2,6.3]
    c_freq = [0,9.7,19,26,30,33,36,56,83,94,98,100]

    dist = DataDistribution(data, c_freq)
    vals = dist.rvs(1000)
    print(np.mean(vals))

    # x = np.arange(-5, 5, 0.01)
    # plt.plot(x, dist.pdf(x))
    # plt.show()