"""
Wrapper module to import distributions from torch.distributions
and make their configuration easier in the ltu-ili interface.

Specifically, if we're using a vector of parameters, we want to
be able to pass the vector to the log_prob method of the distribution
and return a scalar. This is not the default behavior of several
distributions in torch.distributions, so we wrap them here.
"""


import torch
from torch.distributions import Independent
from .import_utils import load_class


# These distributions will be loaded and wrapped
dist_names = [
    'Uniform', 'Normal', 'Beta', 'Cauchy', 'Chi2', 'Exponential',
    'FisherSnedecor', 'Gamma', 'Gumbel', 'HalfCauchy', 'HalfNormal', 'Laplace',
    'LogNormal', 'Pareto', 'StudentT', 'VonMises', 'Weibull', 'Dirichlet'
]


class CustomIndependent(Independent):
    def __init__(self, device='cpu', *args, **kwargs):
        # Convert args and kwargs to torch tensors
        args = [torch.as_tensor(v, dtype=torch.float32, device=device)
                for v in args]
        kwargs = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
                  for k, v in kwargs.items()}

        self.device = device
        self.dist = self.Distribution(*args, **kwargs)
        return super().__init__(self.dist, 1)


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

Uniform = IndependentUniform  # Uniform is always independent

# load multivariate, continuous distributions
# this is done for API convenience, but we don't wrap them
from torch.distributions import (  # noqa
    MultivariateNormal, Dirichlet, LowRankMultivariateNormal
)
