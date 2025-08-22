"""
Wrapper module to import distributions from torch.distributions
and make their configuration easier in the ltu-ili interface.

Specifically, if we're using a vector of parameters, we want to
be able to pass the vector to the log_prob method of the distribution
and return a scalar. This is not the default behavior of several
distributions in torch.distributions, so we wrap them here.
"""


from torch.distributions.utils import broadcast_all
from torch.distributions import constraints, Distribution
from numbers import Number
import math
import torch
from torch.distributions import Independent
from .import_utils import load_class

# Not used directly, but raises error if tried loading with wrong backend
import sbi


# These distributions will be loaded and wrapped
dist_names = [
    'Uniform', 'Normal', 'Beta', 'Cauchy', 'Chi2', 'Exponential',
    'FisherSnedecor', 'Gamma', 'Gumbel', 'HalfCauchy', 'HalfNormal', 'Laplace',
    'LogNormal', 'Pareto', 'StudentT', 'VonMises', 'Weibull'
]


class CustomIndependent(Independent):
    def __init__(self, *args, device='cpu', **kwargs):
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
    MultivariateNormal, LowRankMultivariateNormal
)

# redefining these to not require torch tensors as inputs


class MultivariateNormal(MultivariateNormal):
    def __init__(self, device='cpu', *args, **kwargs):
        # Convert args and kwargs to torch tensors
        args = [torch.as_tensor(v, dtype=torch.float32, device=device)
                for v in args]
        kwargs = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
                  for k, v in kwargs.items()}

        self.device = device
        return super().__init__(*args, **kwargs)


class LowRankMultivariateNormal(LowRankMultivariateNormal):
    def __init__(self, device='cpu', *args, **kwargs):
        # Convert args and kwargs to torch tensors
        args = [torch.as_tensor(v, dtype=torch.float32, device=device)
                for v in args]
        kwargs = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
                  for k, v in kwargs.items()}

        self.device = device
        return super().__init__(*args, **kwargs)


# Define TruncatedIndependentNormal to mirror pydelfi distribution
# Adapted from: https://github.com/toshas/torch_truncnorm

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class _TruncatedStandardNormal(Distribution):
    """Truncated Standard Normal distribution.

    Source: https://github.com/toshas/torch_truncnorm
    Theory: https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True
    eps = 1e-6

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super().__init__(
            batch_shape, validate_args=validate_args
        )
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError("Incorrect truncation range")
        eps = self.eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp(eps, 1 - eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b
            - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = (
            1
            - self._lpbb_m_lpaa_d_Z
            - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        )
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI

    def _big_phi(self, x):
        phi = 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())
        return phi.clamp(self.eps, 1 - self.eps)

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        y = self._big_phi_a + value * self._Z
        y = y.clamp(self.eps, 1 - self.eps)
        return self._inv_big_phi(y)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        out = CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5
        out.masked_fill_((value < self.a) | (value > self.b), -float("inf"))
        return out.squeeze()

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(
            self._dtype_min_gt_0, self._dtype_max_lt_1
        )
        out = self.icdf(p)
        if len(out.shape) == 1:
            return out.unsqueeze(-1)
        return out  # this is a hack


class _UnivariateTruncatedNormal(_TruncatedStandardNormal):
    """Truncated Normal distribution.

    Source: https://github.com/toshas/torch_truncnorm
    Theory: https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, low, high, validate_args=None):
        # scale = scale.clamp_min(self.eps)
        self.loc, self.scale, a, b = broadcast_all(loc, scale, low, high)
        self._non_std_a = a
        self._non_std_b = b
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super().__init__(a, b, validate_args=validate_args)
        self.base = _TruncatedStandardNormal(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self._non_std_a, self._non_std_b)

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return self.base.cdf(self._to_std_rv(value))

    def icdf(self, value):
        sample = self._from_std_rv(super().icdf(value))

        # clamp data but keep gradients
        sample_clip = torch.stack(
            [sample.detach(), self._non_std_a.detach().expand_as(sample)], 0
        ).max(0)[0]
        sample_clip = torch.stack(
            [sample_clip, self._non_std_b.detach().expand_as(sample)], 0
        ).min(0)[0]
        sample.data.copy_(sample_clip)
        return sample

    def log_prob(self, value):
        value = self._to_std_rv(value)
        return self.base.log_prob(value) - self._log_scale


# Define IndependentTruncatedNormal as a class for multivariate priors
IndependentTruncatedNormal = \
    type('IndependentTruncatedNormal', (CustomIndependent,),
         {'Distribution': _UnivariateTruncatedNormal})


class CustomJointIndependent(Distribution):
    """A joint distribution over independent variables with user-provided distributions."""

    def __init__(self, distributions, validate_args=False):
        if not all(isinstance(d, Distribution) for d in distributions):
            raise ValueError(
                "All elements must be torch.distributions.Distribution instances.")

        self.distributions = distributions
        self._support = constraints.stack(
            [d.support for d in distributions], dim=-1)

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([len(distributions)]),
            validate_args=validate_args
        )

    @property
    def support(self):
        return self._support

    def sample(self, sample_shape=torch.Size()):
        return torch.concatenate(
            [d.sample(sample_shape) for d in self.distributions], dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        if not all(getattr(d, "has_rsample", False)
                   for d in self.distributions):
            raise NotImplementedError(
                "At least one component does not support rsample().")
        return torch.concatenate(
            [d.rsample(sample_shape) for d in self.distributions], dim=-1)

    def log_prob(self, value):
        if value.shape[-1] != len(self.distributions):
            raise ValueError(
                f"Expected last dim size {len(self.distributions)},"
                f" got {value.shape[-1]}")
        if self._validate_args and not self.support.check(value).all():
            raise ValueError("Value out of support.")
        return torch.stack(
            [d.log_prob(v)
             for d, v in zip(self.distributions, value.unbind(-1))],
            dim=-1
        ).sum(-1)

    @property
    def mean(self):
        return torch.stack([d.mean for d in self.distributions], dim=-1)

    @property
    def variance(self):
        return torch.stack([d.variance for d in self.distributions], dim=-1)

    def entropy(self):
        return sum(d.entropy() for d in self.distributions)
