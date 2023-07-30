import os
import numpy as np
import emcee
from abc import ABC, abstractmethod

try:
    import torch
except ModuleNotFoundError:
    pass


class BaseSampler(ABC):
    def __init__(
            self,
            posterior,
            num_chains=-1,
            thin=10,
            burn_in=100
    ) -> None:
        super().__init__()
        self.posterior = posterior
        self.num_chains = os.cpu_count()-1 if num_chains == -1 else num_chains
        self.thin = thin
        self.burn_in = burn_in

    @abstractmethod
    def sample(self, sample_shape):
        pass


class EmceeSampler(BaseSampler):
    def sample(self, nsteps, x, progress=False):
        theta0 = np.stack([self.posterior.prior.sample()
                          for i in range(self.num_chains)])

        def log_target(t, x):
            return np.array(self.posterior.potential(
                t.astype(np.float32), x.astype(np.float32)
            ))
        self.sampler = emcee.EnsembleSampler(
            self.num_chains,
            theta0.shape[-1],
            log_target,
            vectorize=False,
            args=(x,)
        )
        self.sampler.run_mcmc(
            theta0,
            self.burn_in + nsteps,
            thin_by=self.thin,
            progress=progress,
        )
        return self.sampler.get_chain(discard=self.burn_in, flat=True)


class PyroSampler(BaseSampler):
    def __init__(self, posterior, num_chains, thin, burn_in,
                 method='slice_np_vectorize') -> None:
        super().__init__(posterior, num_chains, thin, burn_in)
        self.method = method

    def sample(self, nsteps, x, progress=False):
        return self.posterior.sample(
            (nsteps,),
            x=torch.Tensor(x),
            method=self.method,
            num_chains=self.num_chains,
            thin=self.thin,
            warmup_steps=self.burn_in,
            show_progress_bars=progress
        )


class DirectSampler(ABC):
    def __init__(self, posterior):
        self.posterior = posterior

    def sample(self, nsteps, x, progress=False):
        return self.posterior.sample((nsteps,), x=x, progress_bar=progress)
