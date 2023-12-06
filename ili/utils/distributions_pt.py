"""
Wrapper module to import distributions from torch.distributions
and make their configuration easier in the ltu-ili interface.

Specifically, if we're using a vector of parameters, we want to
be able to pass the vector to the log_prob method of the distribution
and return a scalar. This is not the default behavior of several
distributions in torch.distributions, so we wrap them here.
"""


import torch
from .import_utils import load_class

# sbi has nicely wrapped torch's Uniform distribution
from sbi.utils import BoxUniform as Uniform

# Load Independent base class
from torch.distributions import Independent

# These distributions will be loaded and wrapped
dist_names = [
    'Normal', 'Beta', 'Cauchy', 'Chi2', 'Exponential', 'FisherSnedecor',
    'Gamma', 'Gumbel', 'HalfCauchy', 'HalfNormal', 'Laplace',
    'LogNormal', 'Pareto', 'StudentT', 'VonMises', 'Weibull', 'Dirichlet'
]


class CustomIndependent(Independent):
    def __init__(self, *args, **kwargs):
        dist = self.Distribution(*args, **kwargs)
        return super().__init__(dist, 1)


# Load and wrap distributions
dist_dict = {}
for name in dist_names:
    dist = load_class('torch.distributions', name)
    dist_dict['Independent'+name] = \
        type('Independent'+name, (CustomIndependent,), {'Distribution': dist})
locals().update(dist_dict)
# Now, for all distributions in dist_names, we have a custom Independent
# version that can handle vectorized inputs. For example, if 'Normal' is in
# dist_names, then we have a 'IndependentNormal' class parameterized by a
# loc and scale vector


# load multivariate, continuous distributions
# this is done for API convenience, but we don't wrap them
from torch.distributions import (  # noqa
    MultivariateNormal, Dirichlet, LowRankMultivariateNormal
)
