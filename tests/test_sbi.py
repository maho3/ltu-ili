import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sbi
import os

from ili.dataloaders import NumpyLoader, SBISimulator
from ili.inference.runner_sbi import SBIRunner, SBIRunnerSequential
from ili.validation.metrics import PlotSinglePosterior, PlotRankStatistics, TARP
from ili.validation.runner import ValidationRunner

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

def test_snpe(monkeypatch):
    
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # create synthetic catalog
    def simulator(params):
        # create toy simulations
        x = np.linspace(0, 10, 20)
        y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
        y += 1*np.random.randn(len(x))
        return y

    theta = np.random.rand(200, 3)  # 200 simulations, 3 parameters
    x = np.array([simulator(t) for t in theta])

    # make a dataloader
    loader = NumpyLoader(x=x, theta=theta)
    
    # define a prior
    prior = sbi.utils.BoxUniform(low=(0,0,0), high=(1,1,1), device=device)

    # define an inference class (here, we are doing amortized posterior inference)
    inference_class = sbi.inference.SNPE

    # instantiate your neural networks to be used as an ensemble
    nets = [
        sbi.utils.posterior_nn(model='maf', hidden_features=50, num_transforms=5),
        sbi.utils.posterior_nn(model='mdn', hidden_features=50, num_components=2)
    ]

    # define training arguments
    train_args = {
        'training_batch_size': 32,
        'learning_rate': 1e-4,
        'max_num_epochs':5
    }

    # initialize the trainer
    runner = SBIRunner(
        prior=prior,
        inference_class=inference_class,
        nets=nets,
        device=device,
        embedding_net=None,
        train_args=train_args,
        proposal=None,
        output_path=None  # no output path, so nothing will be saved to file
    )
    
    # train the model
    posterior, summaries = runner(loader=loader)
    
    # choose a random input
    ind = np.random.randint(len(theta))
    
    nsamples = 20

    # generate samples from the posterior using accept/reject sampling
    samples = posterior.sample((nsamples,), torch.Tensor(x[ind]).to(device))

    # calculate the log_prob for each sample
    log_prob = posterior.log_prob(samples, torch.Tensor(x[ind]).to(device))
    
    # use ltu-ili's built-in validation metrics to plot the posterior for this point
    metric = PlotSinglePosterior(
    backend='sbi', output_path=None, num_samples=nsamples, 
    sample_method='direct', labels=[f'$\\theta_{i}$' for i in range(3)]
    )
    fig = metric(
        posterior=posterior,
        x_obs = x[ind], theta_obs=theta[ind],
        x=x, theta=theta
    )
    
    # calculate and plot the rank statistics to describe univariate posterior coverage
    metric = PlotRankStatistics(
        backend='sbi', output_path=None, num_samples=nsamples, 
        sample_method='direct', labels=[f'$\\theta_{i}$' for i in range(3)]
    )
    fig = metric(
        posterior=posterior,
        x_obs = x[ind], theta_obs=theta[ind],
        x=x, theta=theta
    )
    
    # calculate and plot the TARP metric to describe multivariate posterior coverage
    metric = TARP(
        backend='sbi', output_path=None, num_samples=nsamples, 
        sample_method='direct', labels=[f'$\\theta_{i}$' for i in range(3)]
    )
    fig = metric(
        posterior=posterior,
        x_obs = x[ind], theta_obs=theta[ind],
        x=x, theta=theta
    )

    return


def test_snle(monkeypatch):
    
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # create the same synthetic catalog as the previous example
    def simulator(params):
        # create toy simulations
        x = np.linspace(0, 10, 20)
        y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
        y += 1*np.random.randn(len(x))
        return y

    theta = np.random.rand(200, 3)  # 200 simulations, 3 parameters
    x = np.array([simulator(t) for t in theta])

    # make a dataloader
    loader = NumpyLoader(x=x, theta=theta)
    
    # define a prior
    prior = sbi.utils.BoxUniform(low=(0,0,0), high=(1,1,1), device=device)

    # define an inference class (here, we are doing amortized likelihood inference)
    inference_class = sbi.inference.SNLE

    # instantiate your neural networks to be used as an ensemble
    nets = [
        sbi.utils.likelihood_nn(model='maf', hidden_features=50, num_transforms=5),
        sbi.utils.likelihood_nn(model='made', hidden_features=50, num_transforms=5)
    ]

    # define training arguments
    train_args = {
        'training_batch_size': 32,
        'learning_rate': 1e-4,
        'max_num_epochs':5
    }

    # initialize the trainer
    runner = SBIRunner(
        prior=prior,
        inference_class=inference_class,
        nets=nets,
        device=device,
        embedding_net=None,
        train_args=train_args,
        proposal=None,
        output_path=None  # no output path, so nothing will be saved to file
    )
    
    # train the model. this outputs a posterior model and training logs
    posterior, summaries = runner(loader=loader)
    
    # choose a random input
    ind = np.random.randint(len(theta))
    
    nsamples = 20

    # generate samples from the posterior using MCMC
    samples = posterior.sample(
        (nsamples,), x[ind], 
        method='slice_np_vectorized', num_chains=2
    ).detach().cpu().numpy()

    # calculate the potential (prop. to log_prob) for each sample
    log_prob = posterior.log_prob(
        nsamples, 
        x[ind]
    ).detach().cpu().numpy()
    
    # use ltu-ili's built-in validation metrics to plot the posterior for this point
    metric = PlotSinglePosterior(
        backend='sbi', output_path=None, num_samples=nsamples, 
        sample_method='slice_np_vectorized',
        sample_params={'num_chains': 2, 'burn_in':1, 'thin': 1},
        labels=[f'$\\theta_{i}$' for i in range(3)]
    )
    fig = metric(
        posterior=posterior, 
        x_obs = x[ind], theta_obs=theta[ind], 
        x=x, theta=theta
    )
    
    return


def test_multiround():
    
    device = 'cpu'
    
    def simulator(params):
        # create toy 'simulations'
        x = np.arange(10)
        y = params @ np.array([np.sin(x), x ** 2, x])
        y += np.random.randn(len(params), len(x))
        return y

    # construct a working directory
    if not os.path.isdir("toy"):
        os.mkdir("toy")

    # simulate a single test observation and save as numpy files
    theta0 = np.zeros((1, 3))+0.5
    x0 = simulator(theta0)
    np.save('toy/thetaobs.npy', theta0[0])
    np.save('toy/xobs.npy', x0[0])

    # setup a dataloader which can simulate
    all_loader = SBISimulator('./toy',
                            'xobs.npy',
                            'thetaobs.npy',
                            './toy',
                            'x.npy',
                            'theta.npy',
                            400,
                            simulator,
                             )

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    
    # define a prior
    prior = sbi.utils.BoxUniform(low=(0,0,0), high=(1,1,1), device=device)
    
    # define an inference class (here, we are doing amortized posterior inference)
    inference_class = sbi.inference.SNPE_C
    
    # instantiate your neural networks to be used as an ensemble
    nets = [
        sbi.utils.posterior_nn(model='maf', hidden_features=100, num_transforms=2),
        sbi.utils.posterior_nn(model='mdn', hidden_features=50, num_transforms=4)
    ]
    
    # define training arguments
    train_args = {
        'training_batch_size': 32,
        'learning_rate': 1e-3,
        'max_num_epochs':5,
        'num_round':2,
    }
    
    # initialize the trainer
    runner = SBIRunnerSequential(
        prior=prior,
        inference_class=inference_class,
        nets=nets,
        device=device,
        embedding_net=nn.Identity(),
        train_args=train_args,
        output_path='./toy',
    )

    # train the model
    runner(loader=all_loader)

    return
