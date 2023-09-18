"""
Metrics for evaluating the performance of inference engines.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import os
from typing import List, Optional
from abc import ABC
from pathlib import Path
import warnings
from ili.utils.samplers import (_BaseSampler, EmceeSampler, PyroSampler,
                                DirectSampler)

try:
    from sbi.inference.posteriors.base_posterior import NeuralPosterior
    from sbi.inference.posteriors import DirectPosterior
    from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
    ModelClass = NeuralPosterior
    import tarp  # doesn't yet work with pydelfi/python 3.6
except ModuleNotFoundError:
    from ili.inference.pydelfi_wrappers import DelfiWrapper
    ModelClass = DelfiWrapper


class _BaseMetric(ABC):
    """Base class for calculating validation metrics.

    Args:
        backend (str): the backend for the posterior models
            ('sbi' or 'pydelfi')
        output_path (Path): path where to store outputs
    """

    def __init__(
        self,
        backend: str,
        output_path: Path,
        labels: Optional[List[str]] = None,
    ):
        """Construct the base metric."""
        self.backend = backend
        self.output_path = output_path
        self.labels = labels


class _SampleBasedMetric(_BaseMetric):
    def __init__(
        self,
        backend: str,
        output_path: Path,
        num_samples: int,
        sample_method: str = 'emcee',
        sample_params: dict = {},
        labels: Optional[List[str]] = None,
    ):
        super().__init__(backend, output_path, labels)
        self.num_samples = num_samples
        self.sample_method = sample_method
        self.sample_params = sample_params

    def _build_sampler(self, posterior) -> _BaseSampler:
        if self.sample_method == 'emcee':
            return EmceeSampler(posterior, **self.sample_params)
        else:
            # check if pytorch backend is available
            if self.backend != 'sbi':
                raise ValueError(
                    'Pyro backend is only available for sbi posteriors')
            # check if DirectPosterior is available
            if isinstance(posterior, NeuralPosteriorEnsemble):
                if isinstance(posterior.posteriors[0], DirectPosterior):
                    warnings.warn(
                        'DirectPosterior detected. '
                        'Ignoring mcmc sampler parameters.')
                    return DirectSampler(posterior)
            return PyroSampler(posterior, method=self.sample_method,
                               **self.sample_params)


class PlotSinglePosterior(_SampleBasedMetric):
    """Perform inference sampling on a single test point and plot the
    posterior in a corner plot.

    Args:
        num_samples (int): number of posterior samples
        labels (List[str]): list of parameter names
        backend (str): the backend for the posterior models
            ('sbi' or 'pydelfi')
        output_path (Path): path where to store outputs
    """

    def __call__(
        self,
        posterior: ModelClass,
        x: np.array,
        theta: np.array,
        x_obs: Optional[np.array] = None,
        theta_obs: Optional[np.array] = None
    ):
        """Given a posterior and test data, plot the inferred posterior of a
        single test point and save to file.

        Args:
            posterior (ModelClass): trained sbi posterior inference engine
            x (np.array): tensor of test summaries
            theta (np.array): tensor of test parameters
            x_obs (np.array, optional): tensor of observed summaries
            theta_obs (np.array, optional): tensor of true parameters for x_obs
        """
        ndim = theta.shape[-1]

        # choose a random test datapoint if not supplied
        if x_obs is None or theta_obs is None:
            ind = np.random.choice(len(x))
            x_obs = x[ind]
            theta_obs = theta[ind]

        # sample from the posterior
        sampler = self._build_sampler(posterior)
        samples = sampler.sample(self.num_samples, x=x_obs, progress=True)

        # plot
        with warnings.catch_warnings():  # catching mpl-caused warning
            warnings.filterwarnings("ignore")
            g = sns.pairplot(
                pd.DataFrame(samples, columns=self.labels),
                kind=None,
                diag_kind="kde",
                corner=True,
            )
        g.map_lower(sns.kdeplot, levels=4, color=".2")

        for i in range(ndim):
            for j in range(i + 1):
                if i == j:
                    g.axes[i, i].axvline(theta_obs[i], color="r")
                else:
                    g.axes[i, j].axhline(theta_obs[i], color="r")
                    g.axes[i, j].axvline(theta_obs[j], color="r")
                    g.axes[i, j].plot(theta_obs[j], theta_obs[i], "ro")

        if self.output_path is None:
            return g
        g.savefig(self.output_path / "plot_single_posterior.jpg",
                  dpi=200, bbox_inches='tight')


class PlotRankStatistics(_SampleBasedMetric):
    """Plot rank histogram, posterior coverage, and true-pred diagnostics
    based on rank statistics inferred from posteriors. These are derived
    from sbi posterior metrics originally written by Chirag Modi.
    Reference: https://github.com/modichirag/contrastive_cosmology/blob/main/src/sbiplots.py

    Args:
        num_samples (int): number of posterior samples
        labels (List[str]): list of parameter names
        output_path (Path): path where to store outputs
    """
    # TODO: Clean these functions up

    def _get_ranks(
        self,
        posterior: ModelClass,
        x: np.array,
        theta: np.array
    ):
        """Samples inferred parameters from a trained posterior given observed
        data and calculates posterior metrics such as means, stdevs, and ranks.

        Args:
            posterior (ModelClass): trained sbi posterior inference engine
            x (np.array): tensor of test summaries
            y (np.array): tensor of test parameters

        Returns:
            trues (np.array): array of true parameter values
            mus (np.array): array of posterior prediction means
            stds (np.array): array of posterior prediction standard deviations
            ranks (np.array): array of posterior prediction ranks
        """
        sampler = self._build_sampler(posterior)

        ndim = theta.shape[1]
        ranks = []
        mus, stds = [], []
        trues = []
        for ii in tqdm.tqdm(range(x.shape[0])):
            try:
                posterior_samples = sampler.sample(
                    self.num_samples, x=x[ii], progress=False)
            except Warning as w:
                # except :
                print("WARNING\n", w)
                continue
            mu, std = posterior_samples.mean(
                axis=0)[:ndim], posterior_samples.std(axis=0)[:ndim]
            rank = [(posterior_samples[:, i] < theta[ii, i]).sum()
                    for i in range(ndim)]
            mus.append(mu)
            stds.append(std)
            ranks.append(rank)
            trues.append(theta[ii][:ndim])
        mus, stds, ranks = np.array(mus), np.array(stds), np.array(ranks)
        trues = np.array(trues)
        return trues, mus, stds, ranks

    def _plot_ranks_histogram(self, ranks, nbins=10):
        ncounts = ranks.shape[0] / nbins
        npars = ranks.shape[-1]

        fig, ax = plt.subplots(1, npars, figsize=(npars * 3, 4))

        for i in range(npars):
            ax[i].hist(np.array(ranks)[:, i], bins=nbins)
            ax[i].set_title(self.labels[i])
            ax[0].set_ylabel('counts')

        for axis in ax:
            axis.set_xlim(0, ranks.max())
            axis.set_xlabel('rank')
            axis.grid(visible=True)
            axis.axhline(ncounts, color='k')
            axis.axhline(ncounts - ncounts ** 0.5, color='k', ls="--")
            axis.axhline(ncounts + ncounts ** 0.5, color='k', ls="--")

        if self.output_path is None:
            return fig
        plt.savefig(self.output_path / 'rankplot.jpg',
                    dpi=300, bbox_inches='tight')

    def _plot_coverage(self, ranks, plotscatter=True):
        ncounts = ranks.shape[0]
        npars = ranks.shape[-1]

        unicov = [np.sort(np.random.uniform(0, 1, ncounts)) for j in range(20)]

        fig, ax = plt.subplots(1, npars, figsize=(npars * 4, 4))

        for i in range(npars):
            xr = np.sort(ranks[:, i])
            xr = xr / xr[-1]
            cdf = np.arange(xr.size) / xr.size
            if plotscatter:
                for j in range(len(unicov)):
                    ax[i].plot(unicov[j], cdf, lw=1, color='gray', alpha=0.2)
            ax[i].plot(xr, cdf, lw=2, label='posterior')
            ax[i].set(adjustable='box', aspect='equal')
            ax[i].set_title(self.labels[i])
            ax[i].set_xlabel('Predicted Percentile')
            ax[i].legend()

        ax[0].set_ylabel('Empirical Percentile')

        for axis in ax:
            axis.grid(visible=True)

        if self.output_path is None:
            return fig
        plt.savefig(self.output_path / 'coverage.jpg',
                    dpi=300, bbox_inches='tight')

    def _plot_predictions(self, trues, mus, stds):
        npars = trues.shape[-1]

        # plot predictions
        fig, axs = plt.subplots(1, npars, figsize=(npars * 4, 4))
        axs = axs.flatten()
        for j in range(npars):
            axs[j].errorbar(trues[:, j], mus[:, j], stds[:, j],
                            fmt="none", elinewidth=0.5, alpha=0.5)

            axs[j].plot(
                *(2 * [np.linspace(min(trues[:, j]), max(trues[:, j]), 10)]),
                'k--', ms=0.2, lw=0.5)
            axs[j].grid(which='both', lw=0.5)
            axs[j].set(adjustable='box', aspect='equal')
            axs[j].set_title(self.labels[j], fontsize=12)

            axs[j].set_xlabel('True')
            axs[j].set_ylabel('Predicted')

        if self.output_path is None:
            return fig
        plt.savefig(self.output_path / 'predictions.jpg',
                    dpi=300, bbox_inches='tight')

    def __call__(
        self,
        posterior: ModelClass,
        x: np.array,
        theta: np.array,
        x_obs: Optional[np.array] = None,
        theta_obs: Optional[np.array] = None
    ):
        """Given a posterior and test data, plot the rank statistics evaluated
        on the test set and save to file.

        Args:
            posterior (ModelClass): trained sbi posterior inference engine
            x (np.array): tensor of test summaries
            y (np.array): tensor of test parameters
            x_obs (None, optional): Not used
            theta_obs (None, optional): Not used
        """

        trues, mus, stds, ranks = self._get_ranks(posterior, x, theta)
        self._plot_ranks_histogram(ranks)
        self._plot_coverage(ranks)
        self._plot_predictions(trues, mus, stds)


class TARP(_SampleBasedMetric):
    """Compute the TARP validation metric
    Reference: https://arxiv.org/abs/2302.03026.

    Args:
        num_samples (int): number of posterior samples
        output_path (Path): path where to store outputs
    """

    def __call__(
        self,
        posterior: ModelClass,
        x: np.array,
        theta: np.array,
        x_obs: Optional[np.array] = None,
        theta_obs: Optional[np.array] = None,
        references: str = "random",
        metric: str = "euclidean"
    ):
        """Given a posterior and test data, compute the TARP metric and save
        to file.

        Args:
            posterior (ModelClass): trained sbi posterior inference engine
            x (np.array): tensor of test summaries
            theta (np.array): tensor of test parameters
            x_obs (np.array, optional): Not used
            theta_obs (np.array, optional): Not used
            references (str, optional): how to select the reference points.
                Defaults to "random".
            metric (str, optional): which metric to use.
                Defaults to "euclidean".
        """
        # sample from the posterior
        sampler = self._build_sampler(posterior)
        if self.sample_method == "emcee":
            P = sampler.num_chains*self.num_samples
        else:
            P = self.num_samples
        posterior_samples = np.zeros(
            (P, x.shape[0], theta.shape[1]))
        for ii in tqdm.tqdm(range(x.shape[0])):
            try:
                samp_i = sampler.sample(
                    self.num_samples, x=x[ii], progress=False)
                posterior_samples[:, ii] = samp_i
            except Warning as w:
                # except :
                print("WARNING\n", w)
                continue

        alpha, ecp = tarp.get_drp_coverage(posterior_samples,
                                           theta,
                                           references=references,
                                           metric=metric)

        # plot the TARP metric
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot([0, 1], [0, 1], ls='--', color='k')
        ax.plot(alpha, ecp, label='DRP')
        ax.legend()
        ax.set_ylabel("Expected Coverage")
        ax.set_xlabel("Credibility Level")

        if self.output_path is None:
            return fig
        plt.savefig(self.output_path / "plot_tarp.jpg",
                    dpi=300, bbox_inches='tight')
