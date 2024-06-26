"""
Wrapper module to import distributions from pydelfi.priors
and make their configuration easier in the sbi interface.
"""

import numpy as np
from scipy.stats import norm
from pydelfi.priors import Uniform, TruncatedGaussian


class Uniform(Uniform):
    # Conform pydelfi's Uniform to sbi's BoxUniform
    def __init__(self, low, high, device='cpu'):
        low, high = map(np.ascontiguousarray, [low, high])
        self.low = low
        self.high = high
        super().__init__(lower=low, upper=high)


class IndependentNormal():
    def __init__(self, loc, scale, device='cpu'):
        loc, scale = map(np.ascontiguousarray, [loc, scale])
        self.loc = loc
        self.scale = scale

    def draw(self):
        return norm.rvs(loc=self.loc, scale=self.scale)

    def logpdf(self, x):
        return np.sum(norm.logpdf(x, loc=self.loc, scale=self.scale))

    def pdf(self, x):
        return np.prod(norm.pdf(x, loc=self.loc, scale=self.scale))


class MultivariateTruncatedNormal(TruncatedGaussian):
    """Note the pdf and logpdf as implemented in pydelfi are not normalized."""

    def __init__(self, loc, covariance_matrix, low, high, device='cpu'):
        loc, covariance_matrix, low, high = map(
            np.ascontiguousarray, [loc, covariance_matrix, low, high])
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        self.low = low
        self.high = high
        super().__init__(mean=loc, C=covariance_matrix,
                         lower=low, upper=high)


class IndependentTruncatedNormal(MultivariateTruncatedNormal):
    """Note the pdf and logpdf as implemented in pydelfi are not normalized."""

    def __init__(self, loc, scale, low, high, device='cpu'):
        loc, scale, low, high = map(
            np.ascontiguousarray, [loc, scale, low, high])
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        super().__init__(loc, np.diag(scale**2), low, high)
