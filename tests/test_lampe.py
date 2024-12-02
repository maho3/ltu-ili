import warnings  # noqa
warnings.filterwarnings('ignore')  # noqa

import numpy as np
from numpy import testing
import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path
import unittest
import yaml

import ili
from ili.dataloaders import (NumpyLoader, StaticNumpyLoader)
from ili.inference import LampeRunner, InferenceRunner
from ili.validation.metrics import (
    PlotSinglePosterior, PosteriorCoverage, PosteriorSamples)
from ili.validation.runner import ValidationRunner
from ili.embedding import FCN

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)


def test_npe(monkeypatch):
    """Test the NPE inference class with a simple toy model."""

    monkeypatch.setattr(plt, 'show', lambda: None)

    # construct a working directory
    os.makedirs("./toy_lampe", exist_ok=True)

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

    # ~~~ Test all of the NDEs ~~~

    # instantiate one of each neural networks to be used as an ensemble
    nets = [
        ili.utils.load_nde_lampe(
            model='mdn', hidden_features=50, num_components=2,
            embedding_net=embedding_net),
    ]
    nets += [
        ili.utils.load_nde_lampe(
            model=name, hidden_features=50, num_transforms=5)
        for name in ['maf', 'nsf', 'ncsf', 'nice', 'gf', 'sospf', 'naf', 'unaf', 'cnf']
    ]

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

    # ~~~ Test a full runthrough ~~~

    # retrain with two ndes (to make remaining tests faster)
    runner = LampeRunner(
        prior=prior,
        nets=nets[:2],
        engine=engine,
        device=device,
        train_args=train_args,
        proposal=None,
        out_dir="./toy_lampe"
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
    if os.path.isfile('./toy_lampe/single_samples.npy'):
        os.remove('./toy_lampe/single_samples.npy')

    metric = PlotSinglePosterior(
        out_dir='./toy_lampe', num_samples=nsamples,
        sample_method='direct', labels=[f'$\\theta_{i}$' for i in range(3)],
        seed=1, save_samples=True
    )
    fig = metric(
        posterior=posterior,
        x_obs=x[ind], theta_fid=theta[ind],
        x=x, theta=theta
    )
    # check that samples were saved
    unittest.TestCase().assertTrue(os.path.isfile('./toy_lampe/single_samples.npy'))

    # calculate and plot the rank statistics + TARP to describe univariate
    metric = PosteriorCoverage(
        out_dir=Path('./toy_lampe'), num_samples=nsamples,
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

    # ~~~ Test other configurations ~~~
    # separate proposal and prior
    proposal = prior
    prior = ili.utils.IndependentNormal(
        loc=torch.zeros(3), scale=0.1*torch.ones(3))
    runner = LampeRunner(
        prior=prior,
        nets=nets[:2],
        engine=engine,
        device=device,
        train_args=train_args,
        proposal=None,
        out_dir="./toy_lampe"
    )
    posterior, summaries = runner(loader=loader, seed=12345)  # test seed
    prior = proposal  # reset prior

    # test TorchLoader
    train_dataset = TensorDataset(
        torch.Tensor(x[:-20]), torch.Tensor(theta[:-20]))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(
        x[-20:]), torch.Tensor(theta[-20:]))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    loader = ili.dataloaders.TorchLoader(
        train_loader, val_loader)

    runner = LampeRunner(
        prior=prior,
        nets=nets[:2],
        engine=engine,
        device=device,
        train_args=train_args,
        proposal=None,
        out_dir=None
    )
    posterior, summaries = runner(loader=loader)


def test_zuko(monkeypatch):
    """Test implementation of zuko flow models in ltu-ili."""

    # Test that NCSF throws an error when theta not in [-pi, pi]
    nde = ili.utils.load_nde_lampe(
        model='ncsf', hidden_features=2, num_transforms=2,
        x_normalize=False, theta_normalize=False)
    prior = ili.utils.Uniform(low=[0, 0], high=[10, 10])

    theta = torch.ones(1, 2)*5
    x = torch.zeros(1, 5)

    model = nde(x, theta, prior)

    unittest.TestCase().assertRaises(
        ValueError, model, theta, x)


def test_yaml():
    """Test the LampeRunner integration with yaml config files."""

    if not os.path.isdir("toy_lampe"):
        os.mkdir("toy_lampe")

    config_ndes = [
        {'model': 'mdn', 'hidden_features': 50, 'num_components': 6},
        {'model': 'maf', 'hidden_features': 50, 'num_transforms': 5}
    ]

    def simulator(params):
        # create toy simulations
        x = np.arange(10)
        y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
        y += np.random.randn(len(x))
        return y

    # simulate data and save as numpy files
    theta = np.random.rand(200, 3)  # 200 simulations, 3 parameters
    x = np.array([simulator(t) for t in theta])
    np.save("toy_lampe/theta.npy", theta)
    np.save("toy_lampe/x.npy", x)

    # Yaml file for data
    data = dict(
        in_dir='./toy_lampe',
        x_file='x.npy',
        theta_file='theta.npy'
    )
    with open('./toy_lampe/data.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Yaml file for infer
    data = dict(
        proposal={
            'module': 'ili.utils',
            'class': 'Uniform',
            'args': {'low': [0, 0, 0], 'high': [1, 1, 1]},
        },
        prior={
            'module': 'ili.utils',
            'class': 'IndependentNormal',
            'args': {'loc': [0, 0, 0], 'scale': [0.1, 0.1, 0.1]},
        },
        model={
            'engine': 'NPE',
            'nets': config_ndes,
        },
        train_args={'batch_size': 32, 'epochs': 5},
        out_dir='toy_lampe',
        device='cpu',
    )
    with open('./toy_lampe/infer_noname.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    data['model']['name'] = 'test_lampe'
    with open('./toy_lampe/infer.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    data['embedding_net'] = {
        'module': 'ili.embedding',
        'class': 'FCN',
        'args': {'n_hidden': [10, 10], 'act_fn': 'SiLU'}
    }
    with open('./toy_lampe/infer_embed.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Yaml file for validation
    data = dict(
        posterior_file='posterior.pkl',
        out_dir='./toy_lampe/',
        labels=['t1', 't2', 't3'],
        metrics={
            'single_example': {
                'module': 'ili.validation.metrics',
                'class': 'PlotSinglePosterior',
                'args': dict(
                    num_samples=20,
                    sample_method='direct'
                )
            }
        }
    )
    with open('./toy_lampe/val.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    all_loader = StaticNumpyLoader.from_config("./toy_lampe/data.yml")
    runner = LampeRunner.from_config("./toy_lampe/infer.yml")
    runner(loader=all_loader)
    runner = LampeRunner.from_config("./toy_lampe/infer_noname.yml")
    runner(loader=all_loader)
    runner = LampeRunner.from_config("./toy_lampe/infer_embed.yml")
    runner(loader=all_loader)
    ValidationRunner.from_config("./toy_lampe/val.yml")

    return


def test_universal(monkeypatch):
    # construct a working directory
    os.makedirs("./toy_lampe", exist_ok=True)

    # create synthetic catalog
    def simulator(params):
        # create toy simulations
        x = np.linspace(0, 10, 20)
        y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
        y += 1*np.random.randn(len(x))
        return y

    theta = np.random.rand(100, 3)  # 100 simulations, 3 parameters
    x = np.array([simulator(t) for t in theta])

    # define a prior
    prior = ili.utils.Uniform(low=[0, 0, 0], high=[1, 1, 1], device=device)

    # define training arguments
    nets = [
        ili.utils.load_nde_lampe(
            model='mdn', hidden_features=50, num_components=2),
    ]

    # check that the correct trainers are loaded
    runner = InferenceRunner.load(
        backend='lampe',
        engine='NPE',
        prior=prior,
        nets=nets
    )
    assert isinstance(runner, LampeRunner)

    # test wrong engine
    unittest.TestCase().assertRaises(
        ValueError,
        InferenceRunner.load,
        backend='lampe',
        engine='ANDRE',
        prior=prior,
        nets=nets
    )

    # test wrong backend
    unittest.TestCase().assertRaises(
        ValueError,
        InferenceRunner.load,
        backend='pydelfi',
        engine='NPE',
        prior=prior,
        nets=nets
    )
