"""
Custom samplers for sampling posteriors for Likelihood Estimation and
Ratio Estimation models. Currently supports emcee samplers for both sbi
and pydelfi backends, and pyro samplers only for the sbi backend.
"""

import os
import warnings
import numpy as np
import emcee
from abc import ABC

try:
    import torch
    from sbi.inference.posteriors.base_posterior import NeuralPosterior
    ModelClass = NeuralPosterior
except ModuleNotFoundError:
    from ili.inference.pydelfi_wrappers import DelfiWrapper
    ModelClass = DelfiWrapper


class _BaseSampler(ABC):
    """Base sampler class demonstrating the sampler functionality

    Args:
        posterior (Posterior): posterior object to sample from, must have
            a .potential method specifiying the log posterior
        num_chains (int, optional): number of chains to sample from. Defaults
            to os.cpu_count()-1
        thin (int, optional): thinning factor for the chains. Defaults to 10
        burn_in (int, optional): number of steps to discard as burn-in.
            Defaults to 100
    """

    def __init__(
            self,
            posterior: ModelClass,
            num_chains: int = -1,
            thin: int = 10,
            burn_in: int = 100
    ) -> None:
        super().__init__()
        self.posterior = posterior
        self.num_chains = os.cpu_count()-1 if num_chains == -1 else num_chains
        self.thin = thin
        self.burn_in = burn_in


class EmceeSampler(_BaseSampler):
    """Sampler class for emcee's EnsembleSampler

    Args:
        posterior (Posterior): posterior object to sample from, must have
            a .potential method specifiying the log posterior
        num_chains (int, optional): number of chains to sample from. Defaults
            to os.cpu_count()-1
        thin (int, optional): thinning factor for the chains. Defaults to 10
        burn_in (int, optional): number of steps to discard as burn-in.
            Defaults to 100
    """

    def sample(self, nsteps: int, x: np.ndarray, progress: bool = False) -> np.ndarray:
        """
        Sample nsteps samples from the posterior, evaluated at data x.

        Args:
            nsteps (int): number of samples to draw
            x (np.ndarray): data to evaluate the posterior at
            progress (bool, optional): whether to show progress bar.
                Defaults to False.
        """
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


class PyroSampler(_BaseSampler):
    """Sampler class for pyro's samplers. Integrates with pyro through the sbi
    backend

    Args:
        posterior (Posterior): posterior object to sample from, must have
            a .potential method specifiying the log posterior
        num_chains (int, optional): number of chains to sample from. Defaults
            to os.cpu_count()-1
        thin (int, optional): thinning factor for the chains. Defaults to 10
        burn_in (int, optional): number of steps to discard as burn-in.
            Defaults to 100
        method (str, optional): method to use for sampling. Defaults to
            'slice_np_vectorize'. See sbi documentation for more details.
    """

    def __init__(
        self,
        posterior: ModelClass,
        num_chains: int = -1,
        thin: int = 10,
        burn_in: int = 100,
        method='slice_np_vectorize'
    ) -> None:
        super().__init__(posterior, num_chains, thin, burn_in)
        self.method = method

    def sample(self, nsteps: int, x: np.ndarray, progress: bool = False) -> np.ndarray:
        """
        Sample nsteps samples from the posterior, evaluated at data x.

        Args:
            nsteps (int): number of samples to draw
            x (np.ndarray): data to evaluate the posterior at
            progress (bool, optional): whether to show progress bar.
                Defaults to False.
        """
        return self.posterior.sample(
            (nsteps,),
            x=torch.Tensor(x).to(self.posterior._device),
            method=self.method,
            num_chains=self.num_chains,
            thin=self.thin,
            warmup_steps=self.burn_in,
            show_progress_bars=progress
        ).detach().cpu().numpy()


class DirectSampler(ABC):
    """Sampler class for posteriors with a direct sampling method, i.e.
    amortized posterior inference models.

    Args:
        posterior (Posterior): posterior object to sample from, must have
            a .sample method allowing for direct sampling.
    """

    def __init__(self, posterior: ModelClass) -> None:
        self.posterior = posterior

    def sample(self, nsteps: int, x: np.ndarray, progress: bool = False) -> np.ndarray:
        """
        Sample nsteps samples from the posterior, evaluated at data x.

        Args:
            nsteps (int): number of samples to draw
            x (np.ndarray): data to evaluate the posterior at
            progress (bool, optional): whether to show progress bar.
                Defaults to False.
        """
        with warnings.catch_warnings():  # catching a nflows-caused deprecation warning
            warnings.filterwarnings("ignore")
            return self.posterior.sample(
                (nsteps,), x=torch.Tensor(x).to(self.posterior._device),
                show_progress_bars=progress
            ).detach().cpu().numpy()
