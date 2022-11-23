from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from typing import List
from pathlib import Path


class BaseMetric(ABC):
    @abstractmethod
    def __call__(
        self,
        posterior: NeuralPosterior,
        x: torch.Tensor,
        theta: torch.Tensor
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
        posterior,
        x,
        theta
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
            kind=None, diag_kind='kde',
            corner=True,
        )
        g.map_lower(sns.kdeplot, levels=4, color=".2")

        for i in range(ndim):
            for j in range(i+1):
                if i==j:
                    g.axes[i, i].axvline(theta_obs[i], color='r')
                else:
                    g.axes[i, j].axhline(theta_obs[i], color='r')
                    g.axes[i, j].axvline(theta_obs[j], color='r')
                    g.axes[i, j].plot(theta_obs[j], theta_obs[i], 'ro')

        if self.output_path is None:
            return g
        g.savefig(self.output_path / 'plot_single_posterior.jpg')