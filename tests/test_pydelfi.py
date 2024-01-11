import warnings  # noqa
warnings.filterwarnings('ignore')  # noqa

import tensorflow as tf
from pathlib import Path
import yaml
import ili
from ili.validation.metrics import PlotSinglePosterior, PosteriorCoverage
from ili.inference.pydelfi_wrappers import DelfiWrapper
from ili.validation.runner import ValidationRunner
from ili.inference import DelfiRunner, InferenceRunner
from ili.dataloaders import StaticNumpyLoader, NumpyLoader
from ili.utils import load_nde_pydelfi
import os
import numpy as np
from numpy import testing
import unittest


def test_toy():

    tf.keras.backend.clear_session()

    def simulator(params):
        # create toy simulations
        x = np.arange(10)
        y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
        y += np.random.randn(len(x))
        return y

    # construct a working directory
    if not os.path.isdir("toy_pydelfi"):
        os.mkdir("toy_pydelfi")
    if os.path.isfile('./toy_pydelfi/posterior.pkl'):
        os.remove('./toy_pydelfi/posterior.pkl')

    # simulate data and save as numpy files
    theta = np.random.rand(200, 3)  # 200 simulations, 3 parameters
    x = np.array([simulator(t) for t in theta])
    np.save("toy_pydelfi/theta.npy", theta)
    np.save("toy_pydelfi/x.npy", x)

    # reload all simulator examples as a dataloader
    all_loader = StaticNumpyLoader(
        in_dir='./toy_pydelfi',
        x_file='x.npy',
        theta_file='theta.npy',
    )

    # define a prior
    prior = ili.utils.Uniform(low=np.zeros(3), high=np.ones(3))

    # define training arguments
    train_args = {
        'batch_size': 32,
        'epochs': 5,
    }

    # instantiate your neural networks to be used as an ensemble
    config_ndes = [
        {'model': 'mdn', 'hidden_features': 50, 'num_components': 6},
        {'model': 'maf', 'hidden_features': 50, 'num_transforms': 5}
    ]
    inference_class = DelfiWrapper

    # train a model to infer x -> theta. save it as toy_pydelfi/posterior.pkl
    runner = DelfiRunner(
        config_ndes=config_ndes,
        prior=prior,
        inference_class=inference_class,
        train_args=train_args,
        out_dir=Path('toy_pydelfi')

    )
    runner(loader=all_loader)

    # use the trained posterior model to predict on a single example from
    # the test set
    args = {
        'out_dir': Path('toy_pydelfi'),
        'labels': ['t1', 't2', 't3'],
        'num_samples': 20,
        'sample_method': 'emcee',
        'sample_params': {
            'num_chains': 10,
            'burn_in': 100,
            'thin': 10,
        }
    }
    metrics = {'single_example': PlotSinglePosterior(**args)}
    posterior = DelfiWrapper.load_engine('./toy_pydelfi/posterior.pkl')
    val_runner = ValidationRunner(
        posterior=posterior,
        metrics=metrics,
        out_dir=Path('./toy_pydelfi'),
    )
    val_runner(loader=all_loader)

    # Check sampling of the DelfiWrapper
    theta0 = np.zeros(3)+0.5
    x0 = simulator(theta0)
    samples = val_runner.posterior.sample(
        sample_shape=100,
        x=x0,
        show_progress_bars=False,
        burn_in=20,
    )
    assert samples.shape[1] == len(theta0)

    # TARP not yet available with pydelfi
    metric = PosteriorCoverage(
        out_dir=Path('./toy_pydelfi'), num_samples=2,
        sample_method='emcee', labels=[f'$\\theta_{i}$' for i in range(3)],
        plot_list=["tarp"],
        sample_params={'num_chains': 6},
    )
    unittest.TestCase().assertRaises(
        NotImplementedError,
        metric,
        posterior=posterior,
        x=x0,
        theta=theta0
    )

    #  Cannot sample directly with pydelfi
    metric = PosteriorCoverage(
        out_dir=Path('./toy_pydelfi'), num_samples=2,
        sample_method='direct', labels=[f'$\\theta_{i}$' for i in range(3)],
        plot_list=["coverage"],
    )
    unittest.TestCase().assertRaises(
        ValueError,
        metric,
        posterior=posterior,
        x=x0,
        theta=theta0
    )

    return


def test_prior():

    tf.keras.backend.clear_session()

    def simulator(params):
        # create toy simulations
        x = np.arange(10)
        y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
        y += np.random.randn(len(x))
        return y

    # construct a working directory
    if not os.path.isdir("toy_pydelfi"):
        os.mkdir("toy_pydelfi")
    if os.path.isfile('./toy_pydelfi/prior_posterior.pkl'):
        os.remove('./toy_pydelfi/prior_posterior.pkl')

    # simulate data and save as numpy files
    theta = np.random.rand(200, 3)  # 200 simulations, 3 parameters
    x = np.array([simulator(t) for t in theta])
    np.save("toy_pydelfi/theta.npy", theta)
    np.save("toy_pydelfi/x.npy", x)

    # reload all simulator examples as a dataloader
    all_loader = NumpyLoader(x=x, theta=theta,)

    # define a set of priors to test
    priors = [
        ili.utils.Uniform(low=np.zeros(3), high=np.ones(3)),
        ili.utils.IndependentNormal(loc=np.zeros(3), scale=np.ones(3)),
        ili.utils.IndependentTruncatedNormal(
            loc=np.zeros(3), scale=np.ones(3), low=np.zeros(3), high=np.ones(3)
        )
    ]

    # define training arguments
    train_args = {
        'batch_size': 32,
        'epochs': 5,
    }

    # instantiate your neural networks to be used as an ensemble
    config_ndes = [
        {'model': 'mdn', 'hidden_features': 50, 'num_components': 6},
    ]
    inference_class = DelfiWrapper

    # for each prior to test
    for p in priors:

        # train a model to infer x -> theta. save it as toy_pydelfi/posterior.pkl
        runner = DelfiRunner(
            config_ndes=config_ndes,
            prior=p,
            inference_class=inference_class,
            train_args=train_args,
            out_dir=Path('toy_pydelfi'),
            name='prior_'
        )
        runner(loader=all_loader)

        posterior = DelfiWrapper.load_engine(
            './toy_pydelfi/prior_posterior.pkl')
        # Check sampling of the DelfiWrapper
        theta0 = np.zeros(3)+0.5
        x0 = simulator(theta0)
        samples = posterior.sample(
            sample_shape=100,
            x=x0,
            show_progress_bars=False,
            burn_in=20,
        )
        assert samples.shape[1] == len(theta0)
        tf.reset_default_graph()

    return


def test_custom_priors():
    from ili.utils import IndependentNormal, MultivariateTruncatedNormal, IndependentTruncatedNormal
    from scipy.stats import norm, multivariate_normal

    tf.keras.backend.clear_session()
    # IndependentNormal
    loc = np.zeros(3)
    scale = np.ones(3)
    prior = IndependentNormal(loc=loc, scale=scale)
    testing.assert_array_equal(prior.draw().shape, loc.shape)
    testing.assert_allclose(prior.logpdf(loc), np.sum(
        norm.logpdf(loc, loc=loc, scale=scale)))
    testing.assert_allclose(prior.pdf(loc), np.prod(
        norm.pdf(loc, loc=loc, scale=scale)))

    # MultivariateTruncatedNormal
    loc = np.zeros(3)
    covariance_matrix = np.diag(np.ones(3))
    low = np.zeros(3)
    high = np.ones(3)
    prior = MultivariateTruncatedNormal(
        loc=loc, covariance_matrix=covariance_matrix, low=low, high=high)
    testing.assert_array_equal(prior.draw().shape, loc.shape)

    # IndependentTruncatedNormal
    loc = np.zeros(3)
    scale = np.ones(3)
    low = np.zeros(3)
    high = np.ones(3)
    prior = IndependentTruncatedNormal(
        loc=loc, scale=scale, low=low, high=high)
    testing.assert_array_equal(prior.draw().shape, loc.shape)


def test_yaml():
    """Test the DelfiRunner integration with yaml config files."""

    tf.keras.backend.clear_session()

    if not os.path.isdir("toy_pydelfi"):
        os.mkdir("toy_pydelfi")

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
    np.save("toy_pydelfi/theta.npy", theta)
    np.save("toy_pydelfi/x.npy", x)

    # Yaml file for data
    data = dict(
        in_dir='./toy_pydelfi',
        x_file='x.npy',
        theta_file='theta.npy'
    )
    with open('./toy_pydelfi/data.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Yaml file for infer
    data = dict(
        prior={
            'module': 'ili.utils',
            'class': 'Uniform',
            'args': {'low': [0, 0, 0], 'high': [1, 1, 1]},
        },
        model={
            'module': 'ili.inference.pydelfi_wrappers',
            'class': 'DelfiWrapper',
            'nets': config_ndes,
        },
        train_args={'batch_size': 32, 'epochs': 5},
        out_dir='toy_pydelfi',
    )
    with open('./toy_pydelfi/infer_noname.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    data['model']['name'] = 'test_pydelfi'
    with open('./toy_pydelfi/infer.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Yaml file for validation
    data = dict(
        posterior_file='posterior.pkl',
        out_dir='./toy_pydelfi/',
        labels=['t1', 't2', 't3'],
        metrics={
            'single_example': {
                'module': 'ili.validation.metrics',
                'class': 'PlotSinglePosterior',
                'args': dict(
                    num_samples=20,
                    sample_method='emcee',
                    sample_params={'num_chains': 10,
                                   'burn_in': 100, 'thin': 10}
                )
            }
        }
    )
    with open('./toy_pydelfi/val.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    all_loader = StaticNumpyLoader.from_config("./toy_pydelfi/data.yml")
    runner = DelfiRunner.from_config("./toy_pydelfi/infer.yml")
    runner(loader=all_loader)
    runner = DelfiRunner.from_config("./toy_pydelfi/infer_noname.yml")
    runner(loader=all_loader)
    ValidationRunner.from_config("./toy_pydelfi/val.yml")

    return


def test_universal():
    # -------
    # Setup a toy problem

    # construct a working directory
    if not os.path.isdir("toy_pydelfi"):
        os.mkdir("toy_pydelfi")

    # create synthetic catalog
    def simulator(params):
        # create toy simulations
        x = np.linspace(0, 10, 20)
        y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
        y += 1*np.random.randn(len(x))
        return y

    theta = np.random.rand(20, 3)  # 200 simulations, 3 parameters
    x = np.array([simulator(t) for t in theta])

    # make a dataloader
    loader = NumpyLoader(x=x, theta=theta)

    # define a prior
    prior = ili.utils.Uniform(low=[0, 0, 0], high=[1, 1, 1])

    config_ndes = [
        dict(model='maf', hidden_features=50, num_transforms=5),
        dict(model='mdn', hidden_features=50, num_components=2)
    ]

    # -------
    # Tests of InferenceRunner

    # InferenceRunner isn't supposed to be initialized
    unittest.TestCase().assertRaises(
        NotImplementedError,
        InferenceRunner
    )

    # check that the correct trainers are loaded
    runner0 = InferenceRunner.load(
        backend='pydelfi',
        engine='NLE',
        prior=prior,
        config_ndes=config_ndes,
    )
    assert isinstance(runner0, DelfiRunner)

    # you can't call a pydelfi engine that doesn't exist
    unittest.TestCase().assertRaises(
        ValueError,
        InferenceRunner.load,
        backend='pydelfi',
        engine='ANDRE',
        prior=prior,
        config_ndes=config_ndes,
    )

    # you can't load an sbi backend in the tf interface
    unittest.TestCase().assertRaises(
        ValueError,
        InferenceRunner.load,
        backend='sbi',
        engine='NLE',
        prior=prior,
        config_ndes=config_ndes,
    )

    netcfg = [
        dict(model='maf', hidden_features=50, num_transforms=5),
        dict(model='mdn', hidden_features=50, num_components=2)
    ]
    priorcfg = dict(
        module='ili.utils',
        args=dict(
            low=[0, 0, 0],
            high=[1, 1, 1],
        ),
    )
    priorcfg['class'] = 'Uniform'
    modelcfg = dict(
        backend='pydelfi',
        engine='NLE',
        nets=netcfg,
    )
    cfg = dict(
        model=modelcfg,
        prior=priorcfg,
        device='cpu',
        out_dir='./toy_pydelfi',
        train_args={}
    )
    with open('./toy_pydelfi/inf_univ.yml', 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)
    runner = InferenceRunner.from_config('./toy_pydelfi/inf_univ.yml')

    # -------
    # Test ndes_pt

    # test that it works
    model = load_nde_pydelfi(
        n_params=theta.shape[1], n_data=x.shape[1],
        model='maf', hidden_features=50, num_transforms=5)

    # test that it breaks if you misspecify model configs
    unittest.TestCase().assertRaises(
        ValueError,
        load_nde_pydelfi,
        n_params=theta.shape[1], n_data=x.shape[1],
        model='maf', hidden_features=50, num_components=2
    )
    unittest.TestCase().assertRaises(
        ValueError,
        load_nde_pydelfi,
        n_params=theta.shape[1], n_data=x.shape[1],
        model='mdn', hidden_features=50, num_transforms=5
    )
    unittest.TestCase().assertRaises(
        NotImplementedError,
        load_nde_pydelfi,
        n_params=theta.shape[1], n_data=x.shape[1],
        model='nsf', hidden_features=50, num_components=2
    )

    # test that it works if you underspecify
    tf.reset_default_graph()
    model = load_nde_pydelfi(
        n_params=theta.shape[1], n_data=x.shape[1],
        model='maf', hidden_features=50)
