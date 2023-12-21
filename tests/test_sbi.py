import warnings  # noqa
warnings.filterwarnings('ignore')  # noqa

import numpy as np
from numpy import testing
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sbi
import os
import yaml
import pickle
from pathlib import Path
import xarray as xr
import csv
import json
import unittest

import ili
from ili.dataloaders import (
    NumpyLoader, SBISimulator, StaticNumpyLoader, SummarizerDatasetLoader)
from ili.inference.runner_sbi import SBIRunner, SBIRunnerSequential, ABCRunner
from ili.validation.metrics import PlotSinglePosterior, PosteriorCoverage
from ili.validation.runner import ValidationRunner
from ili.embedding import FCN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)


def test_snpe(monkeypatch):
    """Test the SNPE inference class with a simple toy model."""

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
    prior = ili.utils.Uniform(low=[0, 0, 0], high=[1, 1, 1], device=device)

    # define an inference class (we are doing amortized posterior inference)
    inference_class = sbi.inference.SNPE

    # instantiate your neural networks to be used as an ensemble
    nets = [
        sbi.utils.posterior_nn(
            model='maf', hidden_features=50, num_transforms=5),
        sbi.utils.posterior_nn(
            model='mdn', hidden_features=50, num_components=2)
    ]

    # define training arguments
    train_args = {
        'training_batch_size': 32,
        'learning_rate': 1e-4,
        'max_num_epochs': 5
    }

    # define an embedding network
    embedding_args = {
        'n_data': x.shape[1],
        'n_hidden': [x.shape[1], x.shape[1], x.shape[1]],
        'act_fn': "SiLU"
    }
    embedding_net = FCN(**embedding_args)

    # initialize the trainer
    runner = SBIRunner(
        prior=prior,
        inference_class=inference_class,
        nets=nets,
        device=device,
        embedding_net=embedding_net,
        train_args=train_args,
        proposal=None,
        output_path=None  # no output path, so nothing will be saved to file
    )

    # train the model
    posterior, summaries = runner(loader=loader)

    signatures = posterior.signatures

    # choose a random input
    ind = np.random.randint(len(theta))

    nsamples = 20

    # generate samples from the posterior using accept/reject sampling
    samples = posterior.sample((nsamples,), torch.Tensor(x[ind]).to(device))

    # calculate the log_prob for each sample
    log_prob = posterior.log_prob(samples, torch.Tensor(x[ind]).to(device))

    # use ltu-ili's built-in validation metrics to plot the posterior
    metric = PlotSinglePosterior(
        backend='sbi', output_path=None, num_samples=nsamples,
        sample_method='direct', labels=[f'$\\theta_{i}$' for i in range(3)]
    )
    fig = metric(
        posterior=posterior,
        x_obs=x[ind], theta_obs=theta[ind],
        x=x, theta=theta
    )

    # calculate and plot the rank statistics + TARP to describe univariate
    # posterior coverage
    metric = PosteriorCoverage(
        backend='sbi', output_path=None, num_samples=nsamples,
        sample_method='direct', labels=[f'$\\theta_{i}$' for i in range(3)],
        plot_list=["tarp", "predictions", "coverage", "histogram", "logprob"]
    )
    fig = metric(
        posterior=posterior,
        x_obs=x[ind], theta_obs=theta[ind],
        x=x, theta=theta
    )

    return


def test_snle(monkeypatch):
    """Test the SNLE inference class with a simple toy model."""

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
    prior = ili.utils.IndependentNormal(
        loc=[0, 0, 0], scale=[1, 1, 1], device=device)

    # define an inference class (we are doing amortized likelihood inference)
    inference_class = sbi.inference.SNLE

    # instantiate your neural networks to be used as an ensemble
    nets = [
        sbi.utils.likelihood_nn(
            model='maf', hidden_features=50, num_transforms=5),
        sbi.utils.likelihood_nn(
            model='made', hidden_features=50, num_transforms=5)
    ]

    # define training arguments
    train_args = {
        'training_batch_size': 32,
        'learning_rate': 1e-4,
        'max_num_epochs': 5
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
    posterior, summaries = runner(loader=loader, seed=1)

    signatures = posterior.signatures

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

    # use ltu-ili's built-in validation metrics to plot the posterior
    metric = PlotSinglePosterior(
        backend='sbi', output_path=None, num_samples=nsamples,
        sample_method='slice_np_vectorized',
        sample_params={'num_chains': 2, 'burn_in': 1, 'thin': 1},
        labels=[f'$\\theta_{i}$' for i in range(3)]
    )
    fig = metric(
        posterior=posterior,
        x_obs=x[ind], theta_obs=theta[ind],
        x=x, theta=theta
    )

    metric = PlotSinglePosterior(
        backend='sbi', output_path=None, num_samples=nsamples,
        sample_method='vi',
        sample_params={'dist': 'maf',
                       'n_particles': 32, 'learning_rate': 0.01},
        labels=[f'$\\theta_{i}$' for i in range(3)]
    )
    fig = metric(
        posterior=posterior,
        x_obs=x[ind], theta_obs=theta[ind],
        x=x, theta=theta
    )

    return


def test_snre():
    """Test the SNRE inference class with a simple toy model."""

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
    prior = ili.utils.Uniform(low=[0, 0, 0], high=[1, 1, 1], device=device)

    # define an inference class (we are doing amortized likelihood inference)
    inference_class = sbi.inference.SNRE

    nets = [
        sbi.utils.classifier_nn(
            model='resnet', hidden_features=50, num_blocks=3),
        sbi.utils.classifier_nn(model='mlp', hidden_features=50),
    ]

    train_args = {'training_batch_size': 32,
                  'learning_rate': 0.001, 'max_num_epochs': 5}

    # initialize the trainer
    runner = SBIRunner(
        prior=prior,
        inference_class=inference_class,
        nets=nets,
        device=device,
        embedding_net=None,
        train_args=train_args,
        proposal=None,
        output_path=Path('./toy')
    )

    # train the model. this outputs a posterior model and training logs
    posterior, summaries = runner(loader=loader)

    return


def test_multiround():
    """Test the SNPE inference class with multiround training and a
    simple toy simulator."""

    device = 'cpu'

    def simulator(params):
        # create toy 'simulations'
        x = np.arange(10)
        y = params @ np.array([np.sin(x), x ** 2, x])
        y += np.random.randn(len(params), len(x))
        return y
    theta = np.random.rand(200, 3)  # 200 simulations, 3 parameters
    # x = np.array([simulator(t) for t in theta])
    x = simulator(theta)

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
    unittest.TestCase().assertEqual(len(all_loader), 400)
    np.testing.assert_almost_equal(
        np.squeeze(all_loader.get_obs_parameters()), 
        np.squeeze(theta0)
    )
    
    # train an SBI sequential model to infer x -> theta

    # define a prior
    prior = ili.utils.Uniform(low=[0, 0, 0], high=[1, 1, 1], device=device)

    # define an inference class (we are doing amortized posterior inference)
    inference_class = sbi.inference.SNPE_C

    # instantiate your neural networks to be used as an ensemble
    nets = [
        sbi.utils.posterior_nn(
            model='maf', hidden_features=100, num_transforms=2),
        sbi.utils.posterior_nn(
            model='mdn', hidden_features=50, num_transforms=4)
    ]

    # define training arguments
    train_args = {
        'training_batch_size': 32,
        'learning_rate': 1e-3,
        'max_num_epochs': 5,
        'num_round': 2,
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

    # sample an ABC model to infer x -> theta
    train_args = {
        'num_simulations': 1000,
        'quantile': 0.1,
    }

    # define an inference class (we are doing approximate bayesian computation)
    inference_class = sbi.inference.MCABC

    runner = ABCRunner(
        prior=prior,
        inference_class=inference_class,
        device=device,
        train_args=train_args,
        output_path='./toy',
    )
    runner(loader=all_loader)

    return


def test_prior():
    """Test the prior classes."""

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
    priors = [
        ili.utils.Uniform(low=[0, 0, 0], high=[1, 1, 1], device=device),
        ili.utils.IndependentNormal(
            loc=[0, 0, 0], scale=[1, 1, 1], device=device),
        ili.utils.MultivariateNormal(
            loc=[0, 0, 0], covariance_matrix=np.diag([1, 2, 3]), device=device),
        ili.utils.IndependentTruncatedNormal(
            loc=[0, 0, 0], scale=[1, 1, 1], low=[0, 0, 0], high=[1, 1, 1],
            device=device),
        ili.utils.LowRankMultivariateNormal(
            loc=[0, 0, 0], cov_factor=np.diag([1, 2, 3]), cov_diag=[1, 1, 1],
            device=device),
    ]

    for p in priors:
        # define an inference class (we are doing amortized posterior inference)
        inference_class = sbi.inference.SNPE

        # instantiate your neural networks to be used as an ensemble
        nets = [
            sbi.utils.posterior_nn(
                model='maf', hidden_features=50, num_transforms=5),
        ]

        train_args = {'training_batch_size': 32,
                      'learning_rate': 0.001, 'max_num_epochs': 5}

        # initialize the trainer
        runner = SBIRunner(
            prior=p,
            inference_class=inference_class,
            nets=nets,
            device=device,
            embedding_net=None,
            train_args=train_args,
            proposal=None,
            output_path=Path('./toy')
        )

        # train the model. this outputs a posterior model and training logs
        posterior, summaries = runner(loader=loader)

        # choose a random input
        ind = np.random.randint(len(theta))

        nsamples = 20

        # generate samples from the posterior using accept/reject sampling
        samples = posterior.sample(
            (nsamples,), torch.Tensor(x[ind]).to(device))

        # calculate the log_prob for each sample
        log_prob = posterior.log_prob(samples, torch.Tensor(x[ind]).to(device))

    return


def test_custom_priors():
    from ili.utils.distributions_pt import _UnivariateTruncatedNormal

    loc, scale, low, high, value = 0.0, 1.0, -1.0, 1.0, 0.5
    dist = _UnivariateTruncatedNormal(loc, scale, low, high)
    cdf = dist.cdf(value)
    testing.assert_almost_equal(cdf.item(), 0.780453, decimal=5)
    icdf = dist.icdf(value)
    np.testing.assert_almost_equal(icdf.item(), 0.0, decimal=5)
    log_prob = dist.log_prob(value)
    testing.assert_almost_equal(log_prob.item(), -0.66222, decimal=5)

    # Test IndependentTruncatedNormal
    loc, scale, low, high, value = \
        [0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [0.5, 0.1]
    dist = ili.utils.IndependentTruncatedNormal(loc, scale, low, high)
    log_prob = dist.log_prob(torch.Tensor(value))
    testing.assert_almost_equal(log_prob.item(), -1.20445, decimal=5)
    sample = dist.sample()[:, 0]
    testing.assert_array_less(sample.numpy(), high)
    testing.assert_array_less(low, sample.numpy())


def test_yaml():
    """Test SNPE/SNLE/SNRE/ABC inference classes instantiation
    with yaml config files."""

    if not os.path.isdir("toy"):
        os.mkdir("toy")

    # Yaml file for data - standard
    data = dict(
        in_dir='./toy',
        x_file='x.npy',
        theta_file='theta.npy'
    )
    with open('./toy/data.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Yaml file for data - multiround
    data = dict(
        in_dir='./toy',
        out_dir='./toy',
        xobs_file='xobs.npy',
        thetaobs_file='thetaobs.npy',
        x_file='x.npy',
        theta_file='theta.npy',
        num_simulations=400,
    )
    with open('./toy/data_multi.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Yaml file for infer - standard
    data = dict(
        prior={'module': 'ili.utils',
               'class': 'IndependentNormal',
               'args': dict(
                   loc=[0.5, 0.5, 0.5],
                   scale=[0.5, 0.5, 0.5],
               ),
               },
        model={'module':  'sbi.inference',
               'class': 'SNPE',
               'nets': [
                   dict(model='maf', hidden_features=50, num_transforms=5),
                   dict(model='mdn', hidden_features=50, num_transforms=2)],
               },
        train_args=dict(
            training_batch_size=32,
            learning_rate=0.001,
        ),
        device='cpu',
        output_path='./toy'
    )
    with open('./toy/infer_snpe.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    data['model']['class'] = 'SNLE'
    data['model']['nets'] = [
        dict(model='maf', hidden_features=50, num_transforms=5),
        dict(model='made', hidden_features=50, num_transforms=5)]
    with open('./toy/infer_snle.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    data['model']['class'] = 'SNRE'
    data['model']['nets'] = [
        dict(model='resnet', hidden_features=50, num_blocks=3),
        dict(model='mlp', hidden_features=50)]
    with open('./toy/infer_snre.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Yaml file for infer - multiround
    data = dict(
        prior={'module': 'ili.utils',
               'class': 'Uniform',
               'args': dict(
                   low=[0, 0, 0],
                   high=[1, 1, 1],
               ),
               },
        model={'module':  'sbi.inference',
               'class': 'SNPE_C',
               'nets': [
                   dict(model='maf', hidden_features=100, num_transforms=2),
                   dict(model='mdn', hidden_features=50, num_transforms=4)],
               },
        train_args=dict(
            training_batch_size=32,
            learning_rate=0.01,
            num_round=2,
        ),
        device='cpu',
        output_path='./toy'
    )
    with open('./toy/infer_multi.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
        
    # Yaml file for infer - ABC
    data = dict(
        prior={'module': 'ili.utils',
               'class': 'Uniform',
               'args': dict(
                   low=[0, 0, 0],
                   high=[1, 1, 1],
               ),
               },
        model={'module':  'sbi.inference',
               'class': 'MCABC',
               'name': 'toy_abc',
               'num_workers': 8,
              },
        train_args=dict(
            num_simulations=1000000,
            quantile=0.01,
        ),
        device='cpu',
        output_path='./toy',
    )
    with open('./toy/infer_abc.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Yaml file for validation
    data = dict(
        backend='sbi',
        posterior_path='./toy/posterior.pkl',
        output_path='./toy',
        labels=['t1', 't2', 't3'],
        metrics=dict(
            single_example={
                'module': 'ili.validation.metrics',
                'class': 'PlotSinglePosterior',
                'args': dict(
                    num_samples=1000,
                    sample_method='slice_np_vectorized',
                    sample_params=dict(
                        num_chains=1,
                        burn_in=100,
                        thin=10,
                    )
                )
            },
            coverage={
                'module': 'ili.validation.metrics',
                'class': 'PosteriorCoverage',
                'args': dict(
                    plot_list=["coverage", "histogram", "predictions", "tarp"],
                    num_samples=100,
                    sample_method='slice_np_vectorized',
                    sample_params=dict(
                        num_chains=1,
                        burn_in=100,
                        thin=1,
                    )
                )
            },
            save_samples={
                'module': 'ili.validation.metrics',
                'class': 'PosteriorSamples',
                'args': dict(
                    num_samples=10,
                    sample_method='slice_np_vectorized',
                    sample_params=dict(
                        num_chains=1,
                        burn_in=100,
                        thin=1
                    )
                )
            },
        )
    )
    with open('./toy/val.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # -------
    # Run for single round

    def simulator(params):
        # create toy simulations
        x = np.arange(10)
        y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
        y += np.random.randn(len(x))
        return y

    # simulate data and save as numpy files
    theta = np.random.rand(200, 3)  # 200 simulations, 3 parameters
    x = np.array([simulator(t) for t in theta])
    np.save("toy/theta.npy", theta)
    np.save("toy/x.npy", x)

    # Test objects
    StaticNumpyLoader.from_config("./toy/data.yml", stage='train')
    SBIRunner.from_config("./toy/infer_snpe.yml")
    SBIRunner.from_config("./toy/infer_snle.yml")
    SBIRunner.from_config("./toy/infer_snre.yml")

    # -------
    # Run for multi round

    def simulator(params):
        # create toy 'simulations'
        x = np.arange(10)
        y = params @ np.array([np.sin(x), x ** 2, x])
        y += np.random.randn(len(params), len(x))
        return y

    # simulate a single test observation and save as numpy files
    theta0 = np.zeros((1, 3))+0.5
    x0 = simulator(theta0)
    np.save('toy/thetaobs.npy', theta0[0])
    np.save('toy/xobs.npy', x0[0])

    loader = SBISimulator.from_config("./toy/data_multi.yml")
    loader.set_simulator(simulator)
    run_seq = SBIRunnerSequential.from_config("./toy/infer_multi.yml")
    run_seq(loader=loader)
    
    # -------
    # Run for ABC
    
    ABCRunner.from_config("./toy/infer_abc.yml")

    # -------
    # Run validation

    ValidationRunner.from_config("./toy/val.yml")

    return

def test_loaders():
    """Additional tests for data loaders."""
    
    from typing import Dict, Optional
    
    # -------
    # NumpyLoader
    
    # Exception if data and parameters of different size
    theta = np.random.rand(200, 3)  # 200 simulations, 3 parameters
    x = np.random.rand(190)  # 190 simulations
    try:
        NumpyLoader(x, theta)
    except Exception as e:
        pass
    
    # Check length attribute
    x = np.random.rand(theta.shape[0])
    loader = NumpyLoader(x, theta)
    unittest.TestCase().assertEqual(len(x), len(loader))
    
    # -------
    # SummarizerDatasetLoader
    
    if not os.path.isdir("toy"):
        os.mkdir("toy")

    # create the same synthetic catalog as the previous example
    def simulator(params):
        # create toy 'simulations'
        x = np.arange(10)
        y = params @ np.array([np.sin(x), x ** 2, x])
        y += np.random.randn(len(x))
        return y
    theta = np.random.rand(200, 3)  # 200 simulations, 3 parameters
    
    class Catalogue:
        def __init__(
            self,
            pos: np.array,
            vel: np.array,
            redshift: float,
            boxsize: float,
            cosmo_dict: Dict[str, float],
            name: str,
            mass: Optional[np.array] = None,
            mesh: bool = True,
            n_mesh: Optional[int] = 360,
        ):
            self.pos = pos % boxsize  
            self.vel = vel
            self.mass = mass
            self.redshift = redshift
            self.boxsize = boxsize
            self.cosmo_dict = cosmo_dict
            self.name = name
            
        def __str__(self,)->str:
            return self.name
            

    # make catalogues
    all_cat = [Catalogue(
        pos=np.ones((10, 3)),
        vel=np.ones((10, 3)),
        redshift=0.,
        boxsize=1000.,
        cosmo_dict={'t0': theta[i, 0], 't1': theta[i, 1], 't2': theta[i, 2]},
        name=f'cat_node{i}',
        mass=None,
        mesh=False,
        n_mesh=50
    ) for i in range(theta.shape[0])]

    # define the summary
    class SimpleSummary():

        def __init__(self, bins):
            self.bins = bins

        def __str__(self,):
            return 'simple_summary'

        def __call__(self, catalogue: Catalogue) -> np.array:
            """ Given a catalogue, compute our simple summary
            Args:
                catalogue (Catalogue):  catalogue to summarize
            Returns:
                np.array: the probability of finding N tracers inside random spheres
            """
            t = [catalogue.cosmo_dict[f't{i}'] for i in range(3)]
            return np.array([self.bins, simulator(t)])

        def to_dataset(self, summary: np.array) -> xr.DataArray:
            """ Convert a tpcf array into an xarray dataset
            with coordinates
            Args:
                summary (np.array): summary to convert
            Returns:
                xr.DataArray: dataset array
            """
            radii = [t[0] for t in summary]
            p_N = np.array([t[1] for t in summary])
            return xr.DataArray(
                p_N,
                dims=('r',),
                coords={
                    'r': radii,
                },
            )
        
        def store_summary(self, filename: str, summary: np.array):
            """Store summary as xarray dataset

            Args:
                filename (str): where to store 
                summary (np.array): summary to store
            """
            ds = self.to_dataset(summary)
            ds.to_netcdf(filename)

    # make and save summaries
    summary = SimpleSummary(np.arange(10))
    if not os.path.isdir(Path('./toy') / f"{str(summary)}"):
        os.mkdir(Path('./toy') / f"{str(summary)}")
    for cat in all_cat:
        s = summary(cat)
        summary.store_summary(
            Path('./toy') / f"{str(summary)}/{str(cat)}.nc", s
        )

    # save the parameters
    with open('./toy/summarizer_params.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(['t0', 't1', 't2'])
        for t in theta:
            writer.writerow(t)

    # save the test-train split
    i0 = theta.shape[0]//2
    i1 = 2 * theta.shape[0]//3
    split = {
        "train": list(np.arange(i0)),
        "val": list(np.arange(i0, i1)),
        "test": list(np.arange(i1, theta.shape[0]))
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
    with open('./toy/summarizer_train_test_split.json', 'w') as f:
        json.dump(split, f, cls=NpEncoder)

    # check calling the dataloader
    loader = SummarizerDatasetLoader(
        stage='train',
        data_dir='./toy',
        data_root_file=f'{str(summary)}/cat',
        param_file='summarizer_params.txt',
        train_test_split_file='summarizer_train_test_split.json',
        param_names=['t0', 't1', 't2'],
    )
    
    unittest.TestCase().assertEqual(i0, len(loader))
    d = loader.get_all_data()
    p = loader.get_all_parameters()
    unittest.TestCase().assertIsInstance(d, np.ndarray)
    unittest.TestCase().assertIsInstance(p, np.ndarray)
    np.testing.assert_almost_equal(theta[:i0,:], p, decimal=5)
    
    d = loader.get_nodes_for_stage('train', 'summarizer_train_test_split.json')
    unittest.TestCase().assertListEqual(d, list(np.arange(i0)))
    p = loader.load_parameters('summarizer_params.txt', d, ['t0', 't1', 't2'])
    unittest.TestCase().assertIsInstance(p, np.ndarray)
    np.testing.assert_almost_equal(theta[:i0,:], p, decimal=5)
        
    d = loader.get_nodes_for_stage('val', 'summarizer_train_test_split.json')
    unittest.TestCase().assertListEqual(d, list(np.arange(i0,i1)))
    p = loader.load_parameters('summarizer_params.txt', d, ['t0', 't1', 't2'])
    unittest.TestCase().assertIsInstance(p, np.ndarray)
    np.testing.assert_almost_equal(theta[i0:i1,:], p, decimal=5)
    
    d = loader.get_nodes_for_stage('test', 'summarizer_train_test_split.json')
    unittest.TestCase().assertListEqual(d, list(np.arange(i1,theta.shape[0])))
    p = loader.load_parameters('summarizer_params.txt', d, ['t0', 't1', 't2'])
    unittest.TestCase().assertIsInstance(p, np.ndarray)
    np.testing.assert_almost_equal(theta[i1:,:], p, decimal=5)    
    
    return
