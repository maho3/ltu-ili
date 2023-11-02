import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
import numpy as np
from ili.dataloaders import StaticNumpyLoader
from ili.inference.runner_pydelfi import DelfiRunner
from ili.validation.runner import ValidationRunner
from ili.inference.pydelfi_wrappers import DelfiWrapper
from ili.validation.metrics import PlotSinglePosterior
import pydelfi
import yaml
import pickle
from pathlib import Path
import tensorflow as tf

def test_toy():
    
    tf.keras.backend.clear_session()

    def simulator(params):
        # create toy simulations
        x = np.arange(10)
        y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
        y += np.random.randn(len(x))
        return y

    # construct a working directory
    if not os.path.isdir("toy"):
        os.mkdir("toy")
    if os.path.isfile('./toy/tempmeta.pkl'):
        os.remove('./toy/tempmeta.pkl')

    # simulate data and save as numpy files
    theta = np.random.rand(200, 3)  # 200 simulations, 3 parameters
    x = np.array([simulator(t) for t in theta])
    np.save("toy/theta.npy", theta)
    np.save("toy/x.npy", x)

    # reload all simulator examples as a dataloader
    all_loader = StaticNumpyLoader(
        in_dir = './toy',
        x_file = 'x.npy',
        theta_file = 'theta.npy',
    )
    
    n_params = 3
    n_data = 10
    
    # define a prior
    prior = pydelfi.priors.Uniform(np.zeros(3), np.ones(3))
    
    # define training arguments
    train_args = {
         'batch_size': 32,
         'epochs': 5,
    }
    
    # instantiate your neural networks to be used as an ensemble
    config_ndes = [
        {'module':'pydelfi.ndes', 'class': 'MixtureDensityNetwork', 
         'args':{'n_components':12, 'n_hidden':[64,64], 
                 'activations':['tanh', 'tanh']}
        },
        {'module':'pydelfi.ndes', 'class':'ConditionalMaskedAutoregressiveFlow',
        'args':{'n_hiddens':[50,50], 'n_mades':2, 'act_fun':'tanh'}
        }
    ]
    inference_class = DelfiWrapper
    nets = inference_class.load_ndes(
            n_params=n_params,
            n_data=n_data,
            config_ndes=config_ndes,
    )

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    runner = DelfiRunner(
        n_params = n_params,
        n_data = n_data,
        config_ndes = config_ndes,
        prior = prior,
        inference_class = inference_class,
        nets = nets,
        engine_kwargs = {'nwalkers':20},
        train_args = train_args,
        output_path = Path('toy')
        
    )
    runner(loader=all_loader)

    # use the trained posterior model to predict on a single example from
    # the test set
    args = {
        'backend' : 'pydelfi',
        'output_path' : Path('toy'),
        'labels' : ['t1', 't2', 't3'],
        'num_samples' : 20,
        'sample_method' : 'emcee',
        'sample_params' : {
            'num_chains' : 10,
            'burn_in': 100,
            'thin': 10,
        }   
    }
    metrics = {'single_example': PlotSinglePosterior(**args)}
    val_runner = ValidationRunner(
        posterior = DelfiWrapper.load_engine('./toy/tempmeta.pkl'),
        metrics = metrics,
        backend = 'pydelfi',
        output_path = Path('./toy'),
    )
    val_runner(loader=all_loader)
    
    # Check sampling of the DelfiWrapper
    theta0 = np.zeros(3)+0.5
    x0 = simulator(theta0)
    print(type(val_runner.posterior))
    samples = val_runner.posterior.sample(
        sample_shape = 100,
        x = x0,
        show_progress_bars = False,
        burn_in_chain = 20,
    )
    assert samples.shape[1] == len(theta0)
    
    return


def test_yaml():
    
    tf.keras.backend.clear_session()
    
    if not os.path.isdir("toy"):
        os.mkdir("toy")
        
    config_ndes = [
        {'module':'pydelfi.ndes', 'class': 'MixtureDensityNetwork', 
         'args':{'n_components':12, 'n_hidden':[64,64], 
         'activations':['tanh', 'tanh']}
        },
        {'module':'pydelfi.ndes', 'class':'ConditionalMaskedAutoregressiveFlow',
        'args':{'n_hiddens':[50,50], 'n_mades':2, 'act_fun':'tanh'}
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
    np.save("toy/theta.npy", theta)
    np.save("toy/x.npy", x)

    # Yaml file for data
    data = dict(
        in_dir = './toy',
        x_file = 'x.npy',
        theta_file = 'theta.npy'
      )
    with open('./toy/data.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Yaml file for infer
    data = dict(
        n_params = 3,
        n_data = 10,
        prior = {
            'module' : 'pydelfi.priors',
            'class': 'Uniform',
            'args' : {'lower':[0,0,0], 'upper':[1,1,1]},
        },
        model = {
            'module' : 'ili.inference.pydelfi_wrappers',
            'class' : 'DelfiWrapper',
            'kwargs': {'nwalkers':20},
            'nets' : config_ndes,
        },
        train_args = {'batch_size':32, 'epochs':5},
        output_path = 'toy',
    )
    with open('./toy/infer.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    
    # Yaml file for validation
    data = dict(
        backend = 'pydelfi',
        meta_path = './toy/tempmeta.pkl',
        output_path = './toy/',
        labels = ['t1', 't2', 't3'],
        metrics = {
            'single_example': {
                'module': 'ili.validation.metrics',
                'class' : 'PlotSinglePosterior',
                'args' : dict(
                    num_samples = 20,
                    sample_method = 'emcee',
                    sample_params = {'num_chains':10, 'burn_in':100, 'thin':10}
                )
            }
        }
    )
    with open('./toy/val.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    
    all_loader = StaticNumpyLoader.from_config("./toy/data.yml")
    runner = DelfiRunner.from_config("./toy/infer.yml")
    runner(loader=all_loader)
    ValidationRunner.from_config("./toy/val.yml")
    
    return