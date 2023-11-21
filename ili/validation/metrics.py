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
    ):
        """Construct the base metric."""
        self.backend = backend
        self.output_path = output_path
        self.labels = labels


class _SampleBasedMetric(_BaseMetric):
    """Base class for metrics that require sampling from the posterior.

    Args:
        backend (str): The backend used for sampling ('sbi' or 'pydelfi').
        output_path (Path): The path to save the output.
        num_samples (int): The number of samples to generate.
        sample_method (str, optional): The method used for sampling. Defaults to 'emcee'.
        sample_params (dict, optional): Additional parameters for the sampling method. Defaults to {}.
        labels (List[str], optional): The labels for the metric. Defaults to None.
    """

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

    def _build_sampler(self, posterior: ModelClass) -> _BaseSampler:
        """Builds the sampler based on the specified sample method.

        Args:
            posterior (ModelClass): The posterior object to sample from.

        Returns:
            _BaseSampler: The sampler object.

        Raises:
            ValueError: If the specified sample method is not supported.
        """
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
        fig = sns.pairplot(
            pd.DataFrame(samples, columns=self.labels),
            kind=None,
            diag_kind="kde",
            corner=True,
        )
        fig.map_lower(sns.kdeplot, levels=4, color=".2")

        for i in range(ndim):
            for j in range(i + 1):
                if i == j:
                    fig.axes[i, i].axvline(theta_obs[i], color="r")
                else:
                    fig.axes[i, j].axhline(theta_obs[i], color="r")
                    fig.axes[i, j].axvline(theta_obs[j], color="r")
                    fig.axes[i, j].plot(theta_obs[j], theta_obs[i], "ro")

        # save
        if self.output_path is None:
            return fig
        filepath = self.output_path / (signature + "plot_single_posterior.jpg")
        logging.info(f"Saving single posterior plot to {filepath}...")
        fig.savefig(filepath, bbox_inches='tight')

        # save single posterior samples if asked
        if self.save_samples:
            filepath = self.output_path / (signature + "single_samples.npy")
            logging.info(f"Saving single posterior samples to {filepath}...")
            np.save(filepath, samples)

        return fig


class PosteriorSamples(_SampleBasedMetric):
    """
    Class to save samples from posterior at x data (test data) for downstream
    tasks (e.g. nested sampling) or making custom plots.
    """

    def _sample_dataset(self, posterior, x):
        """Sample from posterior for all datapoints within a
        test dataset.

        Args:
            posterior (ModelClass): trained sbi posterior inference engine
            x (np.array): tensor of test summaries (ndata, nsummaries)

        Returns:
            np.array: posterior samples of shape (nsamples, ndata, npars)
        """
        # Build a sampler
        sampler = self._build_sampler(posterior)

        # Calculate shape of posterior samples
        _t = posterior.prior.sample()
        Ntest = x.shape[0]
        Nparams = _t.shape[0]
        Nsamps = self.num_samples
        if self.sample_method == "emcee":
            Nsamps *= sampler.num_chains

        posterior_samples = np.zeros((Nsamps, Ntest, Nparams))
        for ii in tqdm.tqdm(range(Ntest)):
            try:
                # Sample posterior P(theta | x[ii])
                posterior_samples[:, ii] = sampler.sample(
                    self.num_samples, x=x[ii], progress=False)
            except Warning as w:
                logging.warning("WARNING\n", w)
                continue
        return posterior_samples

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
        test dataset and save to file.

        Args:
            posterior (ModelClass): trained sbi posterior inference engine
            x (np.array): tensor of test summaries
            theta (np.array): tensor of test parameters
            x_obs (np.array, optional): tensor of observed summaries
            theta_obs (np.array, optional): tensor of true parameters for x_obs
        """
        # Sample the full dataset
        posterior_samples = self._sample_dataset(posterior, x)

        if self.output_path is None:
            return posterior_samples
        filepath = self.output_path / (signature + "posterior_samples.npy")
        logging.info(f"Saving posterior samples to {filepath}...")
        np.save(filepath, posterior_samples)
        return posterior_samples


class PosteriorCoverage(PosteriorSamples):
    """Plot rank histogram, posterior coverage, and true-pred diagnostics
    based on rank statistics inferred from posteriors. These are derived
    from sbi posterior metrics originally written by Chirag Modi.
    Reference: https://github.com/modichirag/contrastive_cosmology/blob/main/src/sbiplots.py

    Also has the option to compute the TARP validation metric.
    Reference: https://arxiv.org/abs/2302.03026

    Args:
        num_samples (int): number of posterior samples
        labels (List[str]): list of parameter names
        output_path (Path): path where to store outputs
        plot_list (list): list of plot types to save
        save_samples (bool): whether to save posterior samples
    """

    def __init__(self, plot_list: List[str], save_samples: bool = False, **kwargs):
        self.plot_list = plot_list
        self.save_samples = save_samples
        super().__init__(**kwargs)

    def _get_ranks(
        self,
        samples: np.array,
        trues: np.array,
    ) -> np.array:
        """Get the marginal ranks of the true parameters in the posterior samples.

        Args:
            samples (np.array): posterior samples of shape (nsamples, ndata, npars)
            trues (np.array): true parameters of shape (ndata, npars)

        Returns:
            np.array: ranks of the true parameters in the posterior samples 
                of shape (ndata, npars)
        """
        ranks = (samples < trues[None, ...]).sum(axis=0)
        return ranks

    def _plot_ranks_histogram(
        self, samples: np.ndarray, trues: np.ndarray,
        signature: str, nbins: int = 10
    ) -> plt.Figure:
        """
        Plot a histogram of ranks for each parameter.

        Args:
            samples (numpy.ndarray): List of samples.
            trues (numpy.ndarray): Array of true values.
            signature (str): Signature for the histogram file name.
            nbins (int, optional): Number of bins for the histogram. Defaults to 10.

        Returns:
            matplotlib.figure.Figure: The generated figure.

        """
        ndata, npars = trues.shape
        navg = ndata / nbins
        ranks = self._get_ranks(samples, trues)

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
            axis.axhline(navg, color='k')
            axis.axhline(navg - navg ** 0.5, color='k', ls="--")
            axis.axhline(navg + navg ** 0.5, color='k', ls="--")

        if self.output_path is None:
            return fig
        filepath = self.output_path / (signature + "ranks_histogram.jpg")
        logging.info(f"Saving ranks histogram to {filepath}...")
        fig.savefig(filepath, bbox_inches='tight')
        return fig

    def _plot_coverage(
        self, samples: np.ndarray, trues: np.ndarray,
        signature: str, plotscatter: bool = True
    ) -> plt.Figure:
        """
        Plot the coverage of predicted percentiles against empirical percentiles.

        Args:
            samples (numpy.ndarray): Array of predicted samples.
            trues (numpy.ndarray): Array of true values.
            signature (str): Signature for the plot file name.
            plotscatter (bool, optional): Whether to plot the scatter plot. Defaults to True.

        Returns:
            matplotlib.figure.Figure: The generated figure.

        """
        ndata, npars = trues.shape
        ranks = self._get_ranks(samples, trues)

        unicov = [np.sort(np.random.uniform(0, 1, ndata)) for j in range(200)]
        unip = np.percentile(unicov, [5, 16, 84, 95], axis=0)

        fig, ax = plt.subplots(1, npars, figsize=(npars * 4, 4))
        if npars == 1:
            ax = [ax]
        cdf = np.linspace(0, 1, len(ranks))
        for i in range(npars):
            xr = np.sort(ranks[:, i])
            xr = xr / xr[-1]
            ax[i].plot(cdf, cdf, 'k--')
            if plotscatter:
                ax[i].fill_between(cdf, unip[0], unip[-1],
                                   color='gray', alpha=0.2)
                ax[i].fill_between(cdf, unip[1], unip[-2],
                                   color='gray', alpha=0.4)
            ax[i].plot(xr, cdf, lw=2, label='posterior')
            ax[i].set(adjustable='box', aspect='equal')
            ax[i].set_title(self.labels[i])
            ax[i].set_xlabel('Predicted Percentile')
            ax[i].set_xlim(0, 1)
            ax[i].set_ylim(0, 1)

        ax[0].set_ylabel('Empirical Percentile')
        for axis in ax:
            axis.grid(visible=True)

        if self.output_path is None:
            return fig
        filepath = self.output_path / (signature + "plot_coverage.jpg")
        logging.info(f"Saving coverage plot to {filepath}...")
        fig.savefig(filepath, bbox_inches='tight')
        return fig

    def _plot_predictions(
        self, samples: np.ndarray, trues: np.ndarray,
        signature: str
    ) -> plt.Figure:
        """
        Plot the mean and standard deviation of the predicted samples against
        the true values.

        Args:
            samples (np.ndarray): Array of predicted samples.
            trues (np.ndarray): Array of true values.
            signature (str): Signature for the plot.

        Returns:
            plt.Figure: The plotted figure.
        """
        npars = trues.shape[-1]
        mus, stds = samples.mean(axis=0), samples.std(axis=0)

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
        axs[0].set_ylabel('Predicted')

        if self.output_path is None:
            return fig
        filepath = self.output_path / (signature + "plot_predictions.jpg")
        fig.savefig(filepath, bbox_inches='tight')
        return fig

    def _plot_TARP(
        self, posterior_samples: np.array, theta: np.array,
        signature: str,
        references: str = "random", metric: str = "euclidean",
        num_alpha_bins: None, num_bootstrap: int = 100,
        bootstrap: bool = True, norm: bool = True
    ) -> plt.Figure:
        """
        Plots the TARP credibility metric for the given posterior samples and theta values.

        Args:
            posterior_samples (np.array): Array of posterior samples.
            theta (np.array): Array of theta values.
            signature (str): Signature for the plot.
            references (str, optional): Reference type for TARP calculation. Defaults to "random".
            metric (str, optional): Distance metric for TARP calculation. Defaults to "euclidean".
            bootstrap (bool, optional): Whether to use bootstrapping for TARP calculation. Defaults to False.
            norm (bool, optional): Whether to normalize the TARP metric. Defaults to True.

        Returns:
            plt.Figure: The generated TARP plot.
        """

        ecp, alpha = tarp.get_tarp_coverage(
            posterior_samples,
            theta,
            references=references,
            metric=metric,
            num_alpha_bins=num_alpha_bins,
            num_bootstrap=num_bootstrap,
            norm=norm,
            bootstrap=bootstrap
        )

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot([0, 1], [0, 1], ls='--', color='k')
        if bootstrap:
            ecp_mean = np.mean(ecp, axis=0)
            ecp_std = np.std(ecp, axis=0)
            ax.plot(alpha, ecp_mean, label='TARP')
            ax.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std,
                            alpha=0.2)
            ax.fill_between(alpha, ecp_mean - 2 * ecp_std, ecp_mean + 2 * ecp_std,
                            alpha=0.2)
        else:
            ax.plot(alpha, ecp, label='TARP')
        ax.legend()
        ax.set_ylabel("Expected Coverage")
        ax.set_xlabel("Credibility Level")

        if self.output_path is None:
            return fig
        filepath = self.output_path / (signature + "plot_TARP.jpg")
        fig.savefig(filepath, bbox_inches='tight')
        return fig

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
        metric: str = "euclidean",
        num_alpha_bins: Union[int, None] = None,
        num_bootstrap: int = 100,
        norm: bool = True,
        bootstrap: bool = True
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
            num_alpha_bins (Union[int, None], optional): number of bins to use
                for the credibility values. If ``None``, then
                ``n_sims // 10`` bins are used. Defaults to None.
            num_bootstrap (int, optional): number of bootstrap iterations to
                perform. Defaults to 100.
            norm (bool, optional): whether to normalize the metric.
                Defaults to True.
            bootstrap (bool, optional): whether to use bootstrapping.
                Defaults to False.
        """
        # Sample the full dataset
        if self.save_samples:
            # Call PosteriorSamples to calculate and save samples
            posterior_samples = super().__call__(
                posterior, x, theta, signature)
        else:
            posterior_samples = self._sample_dataset(posterior, x)

        figs = []
        # Save the plots
        if "coverage" in plot_list:
            figs.append(self._plot_coverage(
                posterior_samples, theta, signature))
        if "histogram" in plot_list:
            figs.append(self._plot_ranks_histogram(
                posterior_samples, theta, signature))
        if "predictions" in plot_list:
            figs.append(self._plot_predictions(
                posterior_samples, theta, signature))

        # Specifically for TARP
        if "tarp" in plot_list:
            # check if if backend is sbi
            if self.backend != 'sbi':
                raise NotImplementedError(
                    'TARP is not yet supported by pydelfi backend')
            figs.append(self._plot_TARP(posterior_samples, theta, signature,
                                        references=references, metric=metric,
                                        num_alpha_bins=num_alpha_bins,
                                        num_bootstrap=num_bootstrap,
                                        norm=norm, bootstrap=bootstrap))

        return figs
