"""
Metrics for evaluating the performance of inference engines.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from typing import List, Optional
from abc import ABC
from pathlib import Path
import logging
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
        # signature: Optional[str] = "",
    ):
        """Construct the base metric."""
        self.backend = backend
        self.output_path = output_path
        self.labels = labels
        # self.signature = signature


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

        # check if pytorch backend is available
        if self.backend != 'sbi':
            raise ValueError(
                'Pyro backend is only available for sbi posteriors')

        # check if DirectPosterior is available
        if self.sample_method == 'direct':
            # First case: we have a NeuralPosteriorEnsemble instance
            # We only need to check the first element
            if (isinstance(posterior, NeuralPosteriorEnsemble) and
                    isinstance(posterior.posteriors[0], DirectPosterior)):
                return DirectSampler(posterior)
            # Second case (when ValidationRunner.ensemble_mode = False)
            elif isinstance(posterior, DirectPosterior):
                return DirectSampler(posterior)
            else:
                raise ValueError(
                    'Direct sampling is only available for DirectPosteriors')

        return PyroSampler(posterior, method=self.sample_method,
                           **self.sample_params)


class PosteriorSamples(_SampleBasedMetric):
    """
    Class to save samples from posterior at x data (test data) for downstream
    tasks (e.g. nested sampling) or making custom plots
    """

    def __call__(
        self,
        posterior: ModelClass,
        x: np.array,
        theta: np.array,
        signature: Optional[str] = "",
        # here for debugging purpose, otherwise error in runner.py line 123
        x_obs: Optional[np.array] = None,
        theta_obs: Optional[np.array] = None
    ):
        """Given a posterior and test data, infer posterior samples of a
        single test point and save to file.

        Args:
            posterior (ModelClass): trained sbi posterior inference engine
            x (np.array): tensor of test summaries
            theta (np.array): tensor of test parameters
            x_obs (np.array, optional): tensor of observed summaries
            theta_obs (np.array, optional): tensor of true parameters for x_obs
        """

        # build for the posterior (required for both ranks and TARP)
        sampler = self._build_sampler(posterior)
        ndim = theta.shape[1]
        ntest = x.shape[0]

        # Initialize posterior samples array ; careful about backend
        if self.sample_method == "emcee":
            P = sampler.num_chains*self.num_samples
        else:
            P = self.num_samples

        # shape = (num_samples, ntest = x.shape[0], ndim)
        posterior_samples = np.zeros((P, ntest, ndim))

        # Next line equiv  "for each x realization of the test set:
        for ii in tqdm.tqdm(range(ntest)):
            try:
                # Sample posterior P(theta | x[ii])
                # shape (num_samples, dim of theta)
                samp_i = sampler.sample(
                    self.num_samples, x=x[ii], progress=False)
                # same as "posterior_samples[:, ii, :] = samp_i"
                posterior_samples[:, ii] = samp_i

            except Warning as w:
                logging.warn("WARNING\n", w)
                continue

        if self.output_path is None:
            return posterior_samples
        strFig = self.output_path / (signature + "single_samples.npy")
        np.save(strFig, posterior_samples)


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

    def __init__(self, save_samples: bool = False, seed: int = None, **kwargs):
        self.save_samples = save_samples
        self.seed = seed
        super().__init__(**kwargs)

    def __call__(
        self,
        posterior: ModelClass,
        x: np.array,
        theta: np.array,
        x_obs: Optional[np.array] = None,
        theta_obs: Optional[np.array] = None,
        signature: Optional[str] = ""
    ):
        """Given a posterior and test data, plot the inferred posterior of a
        single test point and save to file.

        Args:
            posterior (ModelClass): trained sbi posterior inference engine
            x (np.array): tensor of test summaries
            theta (np.array): tensor of test parameters
            x_obs (np.array, optional): tensor of observed summaries
            theta_obs (np.array, optional): tensor of true parameters for x_obs
            signature (str, optional): signature for the output file name
        """
        ndim = theta.shape[-1]

        # choose a random test datapoint if not supplied
        if x_obs is None or theta_obs is None:
            if self.seed:
                np.random.seed(self.seed)
            ind = np.random.choice(len(x))
            x_obs = x[ind]
            theta_obs = theta[ind]

        # sample from the posterior
        sampler = self._build_sampler(posterior)
        samples = sampler.sample(self.num_samples, x=x_obs, progress=True)

        # plot
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
        strFig = self.output_path / (signature + "plot_single_posterior.jpg")
        logging.info(f"Saving plot to {strFig}")
        g.savefig(strFig,
                  dpi=200, bbox_inches='tight')

        if self.save_samples:
            strFig = self.output_path / (signature + "single_samples.npy")
            logging.info(f"Saving samples to {strFig}")
            np.save(strFig, samples)


class PosteriorCoverage(_SampleBasedMetric):

    """Plot rank histogram, posterior coverage, and true-pred diagnostics
    based on rank statistics inferred from posteriors. These are derived
    from sbi posterior metrics originally written by Chirag Modi.
    Reference: https://github.com/modichirag/contrastive_cosmology/blob/main/src/sbiplots.py

    Also has the option to compute the TARP validation metric.
    Reference: https://arxiv.org/abs/2302.03026

    Args:
        plot_list (list): list of plot types to save
        num_samples (int): number of posterior samples
        labels (List[str]): list of parameter names
        output_path (Path): path where to store outputs
    """

    def __init__(self, plot_list: List[str], **kwargs):
        self.plot_list = plot_list
        super(PosteriorCoverage, self).__init__(**kwargs)

    # First, plot functions: histogram, coverage, predictions and TARP
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
                logging.warn("WARNING\n", w)
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

    def _plot_ranks_histogram(self, ranks, signature, nbins=10):

        ncounts = ranks.shape[0] / nbins
        npars = ranks.shape[-1]

        fig, ax = plt.subplots(1, npars, figsize=(npars * 3, 4))
        if npars == 1:
            ax = [ax]

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
        strFig = self.output_path / (signature + "rankplot.jpg")
        logging.info(f"Saving plot to {strFig}")
        plt.savefig(strFig, dpi=300, bbox_inches='tight')

    def _plot_coverage(self, ranks, signature, plotscatter=True):
        ncounts = ranks.shape[0]
        npars = ranks.shape[-1]

        unicov = [np.sort(np.random.uniform(0, 1, ncounts)) for j in range(20)]

        fig, ax = plt.subplots(1, npars, figsize=(npars * 4, 4))
        if npars == 1:
            ax = [ax]

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
        strFig = self.output_path / (signature + "coverage.jpg")
        logging.info(f"Saving plot to {strFig}")
        plt.savefig(strFig, dpi=300, bbox_inches='tight')

    def _plot_predictions(self, trues, mus, stds, signature):
        npars = trues.shape[-1]

        # plot predictions
        fig, axs = plt.subplots(1, npars, figsize=(npars * 4, 4))
        if npars == 1:
            axs = [axs]
        else:
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
        strFig = self.output_path / (signature + "predictions.jpg")
        logging.info(f"Saving plot to {strFig}")
        plt.savefig(strFig, dpi=300, bbox_inches='tight')

    def _plot_TARP(self, alpha, ecp, signature):
        # plot the TARP metric
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot([0, 1], [0, 1], ls='--', color='k')
        ax.plot(alpha, ecp, label='TARP')
        ax.legend()
        ax.set_ylabel("Expected Coverage")
        ax.set_xlabel("Credibility Level")

        if self.output_path is None:
            return fig
        strFig = self.output_path / (signature + "plot_tarp.jpg")
        logging.info(f"Saving plot to {strFig}")
        plt.savefig(strFig, dpi=300, bbox_inches='tight')

    def __call__(
        self,
        posterior: ModelClass,
        x: np.array,
        theta: np.array,
        x_obs: Optional[np.array] = None,
        theta_obs: Optional[np.array] = None,
        signature: Optional[str] = "",
        plot_list: Optional[list] = ["coverage", "histogram",
                                     "predictions", "tarp"],
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
            signature (str, optional): signature for the output file name
            plot_list (list, optional): list of plot types to save

        Args (TARP only):
            references (str, optional): how to select the reference points.
                Defaults to "random".
            metric (str, optional): which metric to use.
                Defaults to "euclidean".
        """

        ranks_b = False
        pred_b = False
        plot_list = self.plot_list

        # build for the posterior (required for both ranks and TARP)
        sampler = self._build_sampler(posterior)
        ndim = theta.shape[1]
        ntest = x.shape[0]

        # initializations for rank statistics
        if "coverage" in plot_list or "histogram" in plot_list:
            ranks = np.zeros((ntest, ndim), dtype=int)
            ranks_b = True
        if "predictions" in plot_list:
            mus, stds = np.zeros((ntest, ndim), dtype=float), np.zeros(
                (ntest, ndim), dtype=float)
            trues = np.zeros((ntest, ndim), dtype=float)
            pred_b = True

        # Initialize posterior samples array ; careful about backend
        if self.sample_method == "emcee":
            P = sampler.num_chains*self.num_samples
        else:
            P = self.num_samples

        posterior_samples = np.zeros((P, ntest, ndim))

        # Next line equiv  "for each x realization of the test set:
        for ii in tqdm.tqdm(range(ntest)):
            try:
                # Sample posterior P(theta | x[ii])
                # shape (num_samples, dim of theta)
                samp_i = sampler.sample(
                    self.num_samples, x=x[ii], progress=False)
                # same as "posterior_samples[:, ii, :] = samp_i"
                posterior_samples[:, ii] = samp_i

                if ranks_b:
                    rank = [(samp_i[:, k] < theta[ii, k]).sum()
                            for k in range(ndim)]
                    ranks[ii] = rank

                if pred_b:
                    mu, std = samp_i.mean(
                        axis=0)[:ndim], samp_i.std(axis=0)[:ndim]
                    mus[ii] = mu
                    stds[ii] = std
                    trues[ii] = theta[ii]

            except Warning as w:
                # except :
                logging.warn("WARNING\n", w)
                continue
        # Save the plots
        if "coverage" in plot_list:
            self._plot_coverage(ranks, signature)
        if "histogram" in plot_list:
            self._plot_ranks_histogram(ranks, signature)
        if "predictions" in plot_list:
            self._plot_predictions(trues, mus, stds, signature)

        # Specifically for TARP
        if "tarp" in plot_list:

            # check if if backend is sbi
            if self.backend != 'sbi':
                raise ValueError(
                    'TARP is not yet supported by pydelfi backend')

            # TARP Expected Coverage Probability
            alpha, ecp = tarp.get_drp_coverage(posterior_samples,
                                               theta,
                                               references=references,
                                               metric=metric)

            self._plot_TARP(alpha, ecp, signature)
