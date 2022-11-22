import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class PlotSinglePosterior():
    def __init__(
        self,
        num_samples,
        labels,
        output_path
    ):
        self.num_samples = num_samples
        self.labels = labels
        self.output_path = output_path

    def __call__(
        self,
        posterior,
        x,
        theta
    ):
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

        g.savefig(self.output_path / 'plot_single_posterior.jpg')