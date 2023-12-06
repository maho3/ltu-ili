import warnings  # noqa
warnings.filterwarnings('ignore')  # noqa

import tensorflow as tf
from pathlib import Path
import yaml
import pydelfi

import ili
from ili.validation.metrics import PlotSinglePosterior
from ili.inference.pydelfi_wrappers import DelfiWrapper
from ili.validation.runner import ValidationRunner
from ili.inference.runner_pydelfi import DelfiRunner
from ili.dataloaders import StaticNumpyLoader
import os
import numpy as np


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

    n_params = 3
    n_data = 10

    # define a prior
    prior = ili.utils.Uniform(np.zeros(3), np.ones(3))

    # define training arguments
    train_args = {
        'batch_size': 32,
        'epochs': 5,
    }

    # instantiate your neural networks to be used as an ensemble
    config_ndes = [
        {'module': 'pydelfi.ndes', 'class': 'MixtureDensityNetwork',
         'args': {'n_components': 12, 'n_hidden': [64, 64],
                  'activations': ['tanh', 'tanh']}
         },
        {'module': 'pydelfi.ndes',
         'class': 'ConditionalMaskedAutoregressiveFlow',
         'args': {'n_hiddens': [50, 50], 'n_mades': 2, 'act_fun': 'tanh'}
         }
    ]
    inference_class = DelfiWrapper
    nets = inference_class.load_ndes(
        n_params=n_params,
        n_data=n_data,
        config_ndes=config_ndes,
    )

    # train a model to infer x -> theta. save it as toy_pydelfi/posterior.pkl
    runner = DelfiRunner(
        n_params=n_params,
        n_data=n_data,
        config_ndes=config_ndes,
        prior=prior,
        inference_class=inference_class,
        nets=nets,
        engine_kwargs={'nwalkers': 20},
        train_args=train_args,
        output_path=Path('toy_pydelfi')

    )
    runner(loader=all_loader)

    # use the trained posterior model to predict on a single example from
    # the test set
    args = {
        'backend': 'pydelfi',
        'output_path': Path('toy_pydelfi'),
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
    val_runner = ValidationRunner(
        posterior=DelfiWrapper.load_engine('./toy_pydelfi/posterior.pkl'),
        metrics=metrics,
        backend='pydelfi',
        output_path=Path('./toy_pydelfi'),
    )
    val_runner(loader=all_loader)

    # Check sampling of the DelfiWrapper
    theta0 = np.zeros(3)+0.5
    x0 = simulator(theta0)
    samples = val_runner.posterior.sample(
        sample_shape=100,
        x=x0,
        show_progress_bars=False,
        burn_in_chain=20,
    )
    assert samples.shape[1] == len(theta0)

    return


def test_yaml():
    """Test the DelfiRunner integration with yaml config files."""

    tf.keras.backend.clear_session()

    if not os.path.isdir("toy_pydelfi"):
        os.mkdir("toy_pydelfi")

    config_ndes = [
        {'module': 'pydelfi.ndes', 'class': 'MixtureDensityNetwork',
         'args': {'n_components': 12, 'n_hidden': [64, 64],
                  'activations': ['tanh', 'tanh']}
         },
        {'module': 'pydelfi.ndes',
         'class': 'ConditionalMaskedAutoregressiveFlow',
         'args': {'n_hiddens': [50, 50], 'n_mades': 2, 'act_fun': 'tanh'}
         }
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
        n_params=3,
        n_data=10,
        prior={
            'module': 'ili.utils',
            'class': 'Uniform',
            'args': {'low': [0, 0, 0], 'high': [1, 1, 1]},
        },
        model={
            'module': 'ili.inference.pydelfi_wrappers',
            'class': 'DelfiWrapper',
            'kwargs': {'nwalkers': 20},
            'nets': config_ndes,
        },
        train_args={'batch_size': 32, 'epochs': 5},
        output_path='toy_pydelfi',
    )
    with open('./toy_pydelfi/infer.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Yaml file for validation
    data = dict(
        backend='pydelfi',
        meta_path='./toy_pydelfi/posterior.pkl',
        output_path='./toy_pydelfi/',
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
    ValidationRunner.from_config("./toy_pydelfi/val.yml")

    return
