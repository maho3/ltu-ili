"""
Wrapper module to import distributions from pydelfi.priors
and make their configuration easier in the sbi interface.
"""

from pydelfi.priors import Uniform, TruncatedGaussian


class Uniform(Uniform):
    # Conform pydelfi's Uniform to sbi's BoxUniform
    def __init__(self, low, high):
        self.low = low
        self.high = high
        super().__init__(lower=low, upper=high)
