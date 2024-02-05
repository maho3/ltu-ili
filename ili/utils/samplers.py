"""
Custom samplers for sampling posteriors for Likelihood Estimation and
Ratio Estimation models. Currently supports emcee samplers for both sbi
and pydelfi backends, and pyro samplers only for the sbi backend.
"""

import os
import numpy as np
import emcee
from abc import ABC
from collections.abc import Sequence
from typing import Any
from math import ceil

try:
    import torch
    from sbi.inference.posteriors.base_posterior import NeuralPosterior
    from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
    from sbi.inference.posteriors import (
        DirectPosterior, MCMCPosterior, VIPosterior)
    from sbi.inference.potentials.posterior_based_potential import (
        posterior_estimator_based_potential)
    ModelClass = NeuralPosterior
except ModuleNotFoundError:
    from ili.inference.pydelfi_wrappers import DelfiWrapper
    ModelClass = DelfiWrapper


class _MCMCSampler(ABC):
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
            burn_in: int = 100,
    ) -> None:
        super().__init__()
        self.posterior = posterior
        self.num_chains = os.cpu_count()-1 if num_chains == -1 else num_chains
        self.thin = thin
        self.burn_in = burn_in


class EmceeSampler(_MCMCSampler):
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

    def sample(self, nsteps: int, x: np.ndarray,
               progress: bool = False,
               skip_initial_state_check: bool = False) -> np.ndarray:
        """
        Sample nsteps samples from the posterior, evaluated at data x.

        Args:
            nsteps (int): number of samples to draw
            x (np.ndarray): data to evaluate the posterior at
            progress (bool, optional): whether to show progress bar.
                Defaults to False.
            skip_initial_state_check (bool, optional): If True, a check that 
                the initial_state can fully explore the space will be skipped. 
                Defaults to False.
        """
        # calculate number of samples per chain
        per_chain = ceil(nsteps / self.num_chains)

        # build posterior to sample
        def log_target(t, x):
            res = self.posterior.potential(
                t.astype(np.float32), x.astype(np.float32))
            if hasattr(res, 'cpu'):
                res = np.array(res.detach().cpu())
            return res

        # Initialize walkers
        theta0 = [self.posterior.prior.sample()
                  for _ in range(self.num_chains)]
        if isinstance(theta0[0], np.ndarray):
            theta0 = np.stack(theta0)
        else:
            theta0 = np.array(torch.stack(theta0).cpu())

        # Set up the sampler
        self.sampler = emcee.EnsembleSampler(
            self.num_chains,
            theta0.shape[-1],
            log_target,
            vectorize=False,
            args=(x,),
        )

        # Sample
        self.sampler.run_mcmc(
            theta0,
            self.burn_in + per_chain,
            thin_by=self.thin,
            progress=progress,
            skip_initial_state_check=skip_initial_state_check
        )
        return self.sampler.get_chain(discard=self.burn_in, flat=True)[:nsteps]


class PyroSampler(_MCMCSampler):
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
        # convert DirectPosteriors to MCMCPosteriors
        if isinstance(posterior, DirectPosterior):
            posterior = self._Direct_to_MCMC(posterior)
        elif isinstance(posterior, NeuralPosteriorEnsemble):
            posteriors = posterior.posteriors
            posterior = NeuralPosteriorEnsemble(
                [(self._Direct_to_MCMC(p) if isinstance(p, DirectPosterior)
                  else p)
                 for p in posteriors],
                weights=posterior.weights,
                theta_transform=posterior.theta_transform
            )
        super().__init__(posterior, num_chains, thin, burn_in)
        self.method = method

    def _Direct_to_MCMC(self, posterior: ModelClass) -> ModelClass:
        """Converts a DirectPosterior to an MCMCPosterior, which is required
        for sampling with pyro.

        Args:
            posterior (DirectPosterior): posterior object to convert

        Returns:
            MCMCPosterior: converted posterior object
        """
        potential_fn, theta_transform = posterior_estimator_based_potential(
            posterior.posterior_estimator,
            posterior.prior,
            x_o=None,
            enable_transform=True,
        )
        return MCMCPosterior(
            potential_fn=potential_fn,
            proposal=posterior.prior,
            theta_transform=theta_transform,
            device=posterior._device
        )

    def sample(self, nsteps: int, x: np.ndarray,
               progress: bool = False) -> np.ndarray:
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

    def sample(self, nsteps: int, x: Any, progress: bool = False) -> np.ndarray:
        """
        Sample nsteps samples from the posterior, evaluated at data x.

        Args:
            nsteps (int): number of samples to draw
            x (np.ndarray): data to evaluate the posterior at
            progress (bool, optional): whether to show progress bar.
                Defaults to False.
        """
        try:
            x = torch.as_tensor(x)
            if hasattr(self.posterior, '_device'):
                x = x.to(self.posterior._device)
        except ValueError:
            pass
        return self.posterior.sample(
            (nsteps,), x=x,
            show_progress_bars=progress
        ).detach().cpu().numpy()


class VISampler(ABC):
    """Sampler class for variational inference methods. See 
    https://sbi-dev.github.io/sbi/reference/#sbi.inference.posteriors.vi_posterior.VIPosterior
    for more details.

    Args:
        posterior (Posterior): posterior object to sample from, must have
            a .potential method specifiying the log posterior
        dist (str, optional): distribution to use for the variational
            inference. Defaults to 'maf'.
        train_kwargs (dict, optional): keyword arguments to pass to the
            posterior's train method. Defaults to {}.
    """

    def __init__(self, posterior: ModelClass,
                 dist: str = 'maf', **train_kwargs) -> None:
        if isinstance(posterior, DirectPosterior):
            posterior = self._Direct_to_VI(posterior)
        elif isinstance(posterior, NeuralPosteriorEnsemble):
            posterior = VIPosterior(
                potential_fn=posterior.potential_fn,
                prior=posterior.prior,
                theta_transform=posterior.theta_transform,
                device=posterior._device
            )
        super().__init__()
        self.posterior = posterior
        self.dist = dist
        self.train_kwargs = train_kwargs

    def _Direct_to_VI(self, posterior: ModelClass) -> ModelClass:
        """Converts a DirectPosterior to a VIPosterior, which is required
        for sampling with variational inference.

        Args:
            posterior (DirectPosterior): posterior object to convert

        Returns:
            VIPosterior: converted posterior object
        """
        potential_fn, theta_transform = posterior_estimator_based_potential(
            posterior.posterior_estimator,
            posterior.prior,
            x_o=None,
            enable_transform=True,
        )
        return VIPosterior(
            potential_fn=potential_fn,
            prior=posterior.prior,
            theta_transform=theta_transform,
            device=posterior._device
        )

    def sample(self, nsteps: int, x: np.ndarray,
               progress: bool = False) -> np.ndarray:
        """
        Sample nsteps samples from the posterior, evaluated at data x.

        Args:
            nsteps (int): number of samples to draw
            x (np.ndarray): data to evaluate the posterior at
            progress (bool, optional): whether to show progress bar.
                Defaults to False.
        """
        x = torch.Tensor(x).to(self.posterior._device)
        self.posterior.set_default_x(x)
        self.posterior.set_q(self.dist)
        self.posterior.train(
            show_progress_bar=progress,
            quality_control=False,
            **self.train_kwargs
        )
        return self.posterior.sample((nsteps,)).detach().cpu().numpy()
