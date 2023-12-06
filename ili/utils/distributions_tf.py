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
        self.low = low
        self.high = high
        super().__init__(lower=low, upper=high)


class IndependentNormal():
    def __init__(self, loc, scale, device='cpu'):
        self.loc = loc
        self.scale = scale

    def draw(self):
        return norm.rvs(loc=self.loc, scale=self.scale)

    def logpdf(self, x):
        return np.sum(norm.logpdf(x, loc=self.loc, scale=self.scale))

    def pdf(self, x):
        return np.prod(norm.pdf(x, loc=self.loc, scale=self.scale))
