import warnings  # noqa
warnings.filterwarnings('ignore')  # noqa

import numpy as np
from numpy import testing
import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path
import unittest

import ili
from ili.dataloaders import (NumpyLoader, StaticNumpyLoader)
from ili.inference import LampeRunner, InferenceRunner
from ili.validation.metrics import (
    PlotSinglePosterior, PosteriorCoverage, PosteriorSamples)
from ili.validation.runner import ValidationRunner
from ili.embedding import FCN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)


def test_npe(monkeypatch):
    """Test the SNPE inference class with a simple toy model."""

    monkeypatch.setattr(plt, 'show', lambda: None)

    # construct a working directory
    if not os.path.isdir("toy"):
        os.mkdir("toy")

    # create synthetic catalog
    def simulator(params):
        # create toy simulations
        x = np.linspace(0, 10, 20)
        y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
        y += 1*np.random.randn(len(x))
        return y

    theta = np.random.rand(100, 3)  # 100 simulations, 3 parameters
    x = np.array([simulator(t) for t in theta])

    # make a dataloader
    loader = NumpyLoader(x=x, theta=theta)

    # define a prior
    prior = ili.utils.Uniform(low=[0, 0, 0], high=[1, 1, 1], device=device)

    # define an inference class (we are doing amortized posterior inference)
    engine = 'NPE'

    # test all of the ndes
    # instantiate one of each neural networks to be used as an ensemble
    nets = [
        ili.utils.load_nde_lampe(
            model='mdn', hidden_features=50, num_components=2)
    ]
    nets += [
        ili.utils.load_nde_lampe(
            model=name, hidden_features=50, num_transforms=5)
        for name in ['maf', 'nsf', 'nice', 'gf', 'sospf', 'naf', 'unaf']
    ]

    # define training arguments
    train_args = {
        'training_batch_size': 32,
        'learning_rate': 1e-4,
        'max_num_epochs': 5
    }

    # define an embedding network
    embedding_args = {
        'n_hidden': [x.shape[1], x.shape[1]],
        'act_fn': "SiLU"
    }
    embedding_net = FCN(**embedding_args)

    # initialize the trainer
    runner = LampeRunner(
        prior=prior,
        nets=nets,
        engine=engine,
        device=device,
        train_args=train_args,
        proposal=None,
        out_dir=None  # no output path, so nothing will be saved to file
    )

    # train the model
    posterior, summaries = runner(loader=loader)

    # retrain with two ndes (to make remaining tests faster)
    runner = LampeRunner(
        prior=prior,
        nets=nets[:2],
        engine=engine,
        device=device,
        train_args=train_args,
        proposal=None,
        out_dir=None  # no output path, so nothing will be saved to file
    )
    posterior, summaries = runner(loader=loader)

    # choose a random input
    ind = np.random.randint(len(theta))
    nsamples = 6

    # generate samples from the posterior using accept/reject sampling
    samples = posterior.sample((nsamples,), torch.Tensor(x[ind]).to(device))

    # calculate the log_prob for each sample
    log_prob = posterior.log_prob(samples, torch.Tensor(x[ind]).to(device))

    # use ltu-ili's built-in validation metrics to plot the posterior
    if os.path.isfile('./toy/single_samples.npy'):
        os.remove('./toy/single_samples.npy')

    metric = PlotSinglePosterior(
        out_dir='./toy', num_samples=nsamples,
        sample_method='direct', labels=[f'$\\theta_{i}$' for i in range(3)],
        seed=1, save_samples=True
    )
    fig = metric(
        posterior=posterior,
        x_obs=x[ind], theta_fid=theta[ind],
        x=x, theta=theta
    )
    # check that samples were saved
    unittest.TestCase().assertTrue(os.path.isfile('./toy/single_samples.npy'))

    # calculate and plot the rank statistics + TARP to describe univariate
    metric = PosteriorCoverage(
        out_dir=Path('./toy'), num_samples=nsamples,
        sample_method='direct', labels=[f'$\\theta_{i}$' for i in range(3)],
        plot_list=["tarp", "predictions", "coverage", "histogram", "logprob"],
        save_samples=True,
    )
    fig = metric(
        posterior=posterior,
        x_obs=x[ind], theta_fid=theta[ind],
        x=x, theta=theta, bootstrap=False,
    )
    # get samples using emcee with the PosteriorSamples class
    nsamp = 2
    nchain = 6
    ntest = 1
    metric = PosteriorSamples(
        out_dir=None, num_samples=nsamp,
        sample_method='emcee',
        labels=[f'$\\theta_{i}$' for i in range(3)],
        sample_params={'num_chains': nchain})
    samples = metric(
        posterior=posterior,
        x_obs=x[ind], theta_fid=theta[ind],
        x=x[:ntest], theta=theta[:ntest],
        skip_initial_state_check=True,
    )
    unittest.TestCase().assertIsInstance(samples, np.ndarray)
    unittest.TestCase().assertListEqual(
        list(samples.shape), [nsamp, ntest, 3])

    # get samples using direct method with PosteriorSamples class
    metric = PosteriorSamples(
        out_dir=None, num_samples=nsamp,
        sample_method='direct',
        labels=[f'$\\theta_{i}$' for i in range(3)])
    samples = metric(
        posterior=posterior,
        x_obs=x[ind], theta_fid=theta[ind],
        x=x[:ntest], theta=theta[:ntest, :],
    )
    unittest.TestCase().assertIsInstance(samples, np.ndarray)
    unittest.TestCase().assertListEqual(list(samples.shape), [nsamp, ntest, 3])

    return
