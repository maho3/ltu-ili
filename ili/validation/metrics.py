from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import tqdm
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from typing import List
from pathlib import Path


class BaseMetric(ABC):
    @abstractmethod
    def __call__(
        self, posterior: NeuralPosterior, x: torch.Tensor, theta: torch.Tensor
    ):
        """Given a posterior and test data, measure a validation metric and save to file.

        Args:
            posterior (NeuralPosterior): trained sbi posterior inference engine
            x (torch.Tensor): tensor of test summaries
            y (torch.Tensor): tensor of test parameters
        """


class PlotSinglePosterior(BaseMetric):
    def __init__(
        self,
        num_samples: int,
        labels: List[str],
        output_path: Path,
    ):
        """Perform inference sampling on a single test point and plot the posterior in a corner plot.

        Args:
            num_samples (int): number of posterior samples
            labels (List[str]): list of parameter names
            output_path (Path): path where to store outputs
        """
        self.num_samples = num_samples
        self.labels = labels
        self.output_path = output_path

    def __call__(
        self,
        posterior: NeuralPosterior,
        x: torch.Tensor,
        theta: torch.Tensor
    ):
        """Given a posterior and test data, plot the inferred posterior of a single point and save to file.

        Args:
            posterior (NeuralPosterior): trained sbi posterior inference engine
            x (torch.Tensor): tensor of test summaries
            y (torch.Tensor): tensor of test parameters
        """
        ndim = theta.shape[-1]

        # choose a random test datapoint
        ind = np.random.choice(len(x))
        x_obs = x[ind]
        theta_obs = theta[ind]

        # sample from the posterior
        samples = posterior.sample((self.num_samples,), x=x_obs)

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
        g.savefig(self.output_path / "plot_single_posterior.jpg")

class PlotRankStatistics(BaseMetric):
    def __init__(
        self,
        num_samples: int,
        labels: List[str],
        output_path: Path,
    ): # TODO: Clean these functions up
        """Plot rank histogram, posterior coverage, and true-pred diagnostics based on rank statistics
        inferred from posteriors. These are derived from sbi posterior metrics originally written by Chirag Modi.
        Reference: https://github.com/modichirag/contrastive_cosmology/blob/main/src/sbiplots.py

        Args:
            num_samples (int): number of posterior samples
            labels (List[str]): list of parameter names
            output_path (Path): path where to store outputs
        """
        self.num_samples = num_samples
        self.labels = labels
        self.output_path = output_path

    def _get_ranks(
        self,
        posterior: NeuralPosterior,
        x: torch.Tensor,
        theta: torch.Tensor
    ):
        """Samples inferred parameters from a trained posterior given observed data and calculates posterior metrics
        such as means, stdevs, and ranks.

        Args:
            posterior (NeuralPosterior): trained sbi posterior inference engine
            x (torch.Tensor): tensor of test summaries
            y (torch.Tensor): tensor of test parameters

        Returns:
            trues (np.array): array of true parameter values
            mus (np.array): array of posterior prediction means
            stds (np.array): array of posterior prediction standard deviations
            ranks (np.array): array of posterior prediction ranks
        """
        theta = theta.numpy()

        ndim = theta.shape[1]
        ranks = []
        mus, stds = [], []
        trues = []
        for ii in tqdm.tqdm(range(x.shape[0])):
            try:
                posterior_samples = posterior.sample((self.num_samples,),
                                                     x=x[ii],
                                                     show_progress_bars=False).detach().numpy()
            except Warning as w:
                # except :
                print("WARNING\n", w)
                continue
            mu, std = posterior_samples.mean(axis=0)[:ndim], posterior_samples.std(axis=0)[:ndim]
            rank = [(posterior_samples[:, i] < theta[ii, i]).sum() for i in range(ndim)]
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

        plt.savefig(self.output_path / 'rankplot.png', bbox_inches='tight')

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
                for j in range(len(unicov)): ax[i].plot(unicov[j], cdf, lw=1, color='gray', alpha=0.2)
            ax[i].plot(xr, cdf, lw=2, label='posterior')
            ax[i].set(adjustable='box', aspect='equal')
            ax[i].set_title(self.labels[i])
            ax[i].set_xlabel('Predicted Percentile')
            ax[i].legend()

        ax[0].set_ylabel('Empirical Percentile')

        for axis in ax:
            axis.grid(visible=True)

        plt.savefig(self.output_path / 'coverage.png', bbox_inches='tight')

    def _plot_predictions(self, trues, mus, stds):
        npars = trues.shape[-1]

        # plot predictions
        fig, axs = plt.subplots(1, npars, figsize=(npars * 4, 4))
        axs = axs.flatten()
        for j in range(npars):
            axs[j].errorbar(trues[:, j], mus[:, j], stds[:, j],
                                     fmt="none", elinewidth=0.5, alpha=0.5)

            axs[j].plot(*(2*[np.linspace(min(trues[:,j]), max(trues[:,j]), 10)]),
                        'k--', ms=0.2, lw=0.5)
            axs[j].grid(which='both', lw=0.5)
            axs[j].set(adjustable='box', aspect='equal')
            axs[j].set_title(self.labels[j], fontsize=12)

            axs[j].set_xlabel('True')
            axs[j].set_ylabel('Predicted')


        plt.savefig(self.output_path /  'predictions.png', bbox_inches='tight')

    def __call__(
        self,
        posterior: NeuralPosterior,
        x: torch.Tensor,
        theta: torch.Tensor
    ):
        """Plot rank histogram, posterior coverage, and true-pred diagnostics based on rank statistics
        inferred from posteriors. These are derived from sbi posterior metrics originally written by Chirag Modi.
        Reference: https://github.com/modichirag/contrastive_cosmology/blob/main/src/sbiplots.py

        Args:
            posterior (NeuralPosterior): trained sbi posterior inference engine
            x (torch.Tensor): tensor of test summaries
            y (torch.Tensor): tensor of test parameters
        """

        trues, mus, stds, ranks = self._get_ranks(posterior, x, theta)
        self._plot_ranks_histogram(ranks)
        self._plot_coverage(ranks)
        self._plot_predictions(trues,mus,stds)



