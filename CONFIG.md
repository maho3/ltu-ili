# Configuration

This tutorial provides examples on how to configure an ltu-ili inference pipeline from `yaml` configuration files. For an introduction to the iPython interface, see [tutorial.ipynb](notebooks/tutorial.ipynb).

There are three stages to the inference pipeline:
- [**Data Loading**](#data-loading): Loading various structured data into memory
- [**Training**](#training): Training neural networks from the loaded data and saving them to file.
- [**Validation**](#validation): Loading neural networks from file, sampling posteriors on the test set, and computing metrics.

In general, these steps are designed to be independent. For example, if you wanted to try out a different training model, all you would need to do is change the training configuration, and the data loading and validation stages should integrate (nearly) seamlessly with the new model. 

Once you have configuration files, running a pipeline is as easy as:
```python
from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.validation import ValidationRunner

# load data
loader = StaticNumpyLoader.from_config('path/to/data_config.yaml')

# train models
inference = InferenceRunner.from_config('path/to/training_config.yaml')
inference.train(loader)

# measure validation metrics
validation = ValidationRunner.from_config('path/to/validation_config.yaml')
validation.run(loader, inference)
```
We will now describe and provide examples of each of these configuration files.

## Data Loading
We provide four primary objects for dataloading:
- [`NumpyLoader`](./ili/dataloaders/loaders.py#L139): Loads data and parameters from `np.array`'s stored in memory.
- [`StaticNumpyLoader`](./ili/dataloaders/loaders.py#L212): Loads data and parameters from `.npy` files on disk.
- [`SBISimulator`](./ili/dataloaders/loaders.py#L254): Contains a `simulate` function which, when given new parameters, generates new data. This interface is then used to train multiround inference with e.g. `SBIRunnerSequential`. It also can separately load `.npy` files of test data and parameters from disk, for validation.
- [`SummarizerDatasetLoader`](./ili/dataloaders/loaders.py#L365): Loads data following the [`ili_summarizer.Dataset`](https://github.com/florpi/ili-summarizer/blob/3d9d4005cfbc187afdbfbed2a5a4414bc07902ef/summarizer/dataset.py#L6) convention, as `xarray.DataArray`'s stored in `.nc` files on disk. Also, it loads parameters from `.txt` files on disk.
- [`TorchLoader`](.ili/dataloaders/loaders.py#L480): Allows users to custom define PyTorch `Dataloader` objects for more complex data loading tasks. Useful for loading data dynamically from disk or exotic data formats (e.g. graphs).

`NumpyLoader` and `TorchLoader` objects are only built from inline initialization, but the remaining classes can take config files that look like:

```bash
# StaticNumpyLoader configuration

in_dir: './toy'  # contains x_file and theta_file

x_file: 'x.npy'
theta_file: 'theta.npy'
```


```bash
# SBISimulator configuration

in_dir: './toy'  # where to find initial data

xobs_file: 'xobs.npy'  # the observed data around which to center simulations
thetafid_file: 'thetafid.npy'  # only if true parameters are known

x_file: 'x.npy'  # file name of the initial data
theta_file: 'theta.npy'  # file name of the initial parameters
num_simulations: 400  # number of simulations to generate per round
save_simulated: False  # whether to concatenate the simulated data into x_file and theta_file
```


```bash
# SummarizerDatasetLoader configuration

in_dir: './ltu-ili-data/quijote'  # contains param_file and train_test_split file and a hierarchy of .nc data files

x_root: 'tpcf/z_0.50/quijote'  # prefix to each .nc file
theta_file: 'latin_hypercube_params.txt' 
train_test_split_file: 'quijote_train_test_val.json'  # specifies which indices in param_file are in train/test/val

param_names: ['Omega_m', 'h', 'sigma_8']
```

```bash
# SBISimulator configuration

in_dir: './toy'  # where to find initial data

xobs_file: 'xobs.npy'  # the observed data around which to center simulations
thetafid_file: 'thetafid.npy'  # only if true parameters are known

x_file: 'x.npy'  # file name of the initial data
theta_file: 'theta.npy'  # file name of the initial parameters
num_simulations: 400  # number of simulations to generate per round
save_simulated: False  # whether to concatenate the simulated data into x_file and theta_file
```

You are also welcome to design your own dataloading objects. They will work with NPE/NLE/NRE models so long as they contain the functions: `__len__`, `get_all_data`, and `get_all_parameters`. For SNPE/SNLE/SNRE models, they must also contain the `simulate` and `get_obs_data` functions. See the [_BaseLoader](./ili/dataloaders/loaders.py#L20) template for more details.

## Training
There are three available ILI backends for ltu-ili, including `sbi` (PyTorch), `pydelfi` (Tensorflow), and `lampe` (PyTorch). Each of these engines can be configured from a `yaml`-like configuration file or from iPython initialization as in [tutorial.ipynb](notebooks/tutorial.ipynb) or [lampe.ipynb](notebooks/lampe.ipynb).

Here's a detailed example of using the universal [`InferenceRunner`](ili/inference/runner.py#L18) class to train Neural Posterior Estimation (NPE) using the `sbi` backend and an ensemble of architectures: a mixture density network (`mdn`), a masked autoregressive flows (`maf`), and a neural spline flow (`nsf`).

```yaml
# Specify prior (same as initial proposal distribution)
prior:
  module: 'ili.utils'
  class: 'Uniform'  # Using a uniform prior over 3 parameters
  args:
    low: [0,0,0]
    high: [1,1,1]

# Specify the embedding network to prepend the NDE (optional)
embedding_net:
  module: 'ili.embedding'
  class: 'FCN'  # Fully-connected network
  args:
    n_hidden: [100,100,100]  # width of hidden layers
    act_fn: "SiLU"  # activation function

# Specify the inference model
model:
  backend: 'sbi'  # sbi or pydelfi
  engine: 'NPE'  # Sequential Neural Posterior Estimation
  name: "toy_NPE"  # name of the ensemble posterior
  nets:
    - model: 'mdn'  # Mixture Density Network
      hidden_features: 50  # width of hidden layers
      num_components: 4  # number of gaussian mixture components
      signature: "m1"  # name of this neural network in the ensemble
    - model: 'maf'  # Masked Autoregressive Flow
      hidden_features: 50  
      num_transforms: 5  # number of flow transformations
      signature: "m2"
    - model: 'nsf'  # Neural Autoregressive Flow
      hidden_features: 50 
      num_transforms: 5
      signature: "m3"
    
# Specify the neural training hyperparameters
train_args:
  # Adam optimizer
  training_batch_size: 32  # batch size for training
  learning_rate: 0.001  # learning rate
  clip_max_norm: 5.0  # maximum gradient norm for gradient clipping

  # Early stopping
  validation_fraction: 0.1  # fraction of training data to use for validation
  stop_after_epochs: 20  # stop after this many epochs without improvement

  # Sequential learning
  num_round: 5  # number of rounds of simulations

device: 'cpu'  # Run on CPU
out_dir: './toy'  # Where to save the posterior
```
Now, let's break down each of these configuration parameters.

The **prior** distribution specifies our prior belief on the true distribution of the inference parameters. We include a variety of pre-implemented distributions, listed for each backend in [this file](./ili/utils/__init__.py).

The **embedding_net** configuration allows one to specify additional neural layers which will prepend the input layer of the above neural density estimators. This can be used to extract more complex features from the data. We provide an example of a fully-connected network (`ili.embedding.FCN`), but we also have an example of a CNN-like embedding network in [tutorial.ipynb](notebooks/tutorial.ipynb) and a graph neural embedding network in [lampe.ipynb](notebooks/lampe.ipynb).

The **model** configuration specifies how we train our neural networks to produce posterior inference. There are three available backends (`sbi`, `pydelfi`, `lampe`) and six available training engines: Neural Posterior Estimation (`NPE`), Neural Likelihood Estimation (`NLE`), and Neural Ratio Estimation (`NRE`). We have provided links (and references therein) to the implementation of each of these methods, but for a general review of the differences between these models, see Section 3B in [Cranmer et al. 2020](https://arxiv.org/abs/1911.01429).

The **model** configuration allows you to specify an ensemble of independently-trained neural density estimators. The configuration options for each NDE is given in [this file](./ili/utils/ndes.py).

Only certain NDEs and training engines are compatible with either the sbi or pydelfi backends. See the table below for a summary of the available options:

| Training Engine | sbi | pydelfi | lampe |
|----------|----------|----------|----------|
|  `NPE`,`SNPE`   |   mdn, maf, nsf, made   |      | maf, nsf, cnf, nice, gf, sospf, naf, unaf |
|  `NLE`, `SNLE`   |   mdn, maf, nsf, made   |   mdn, maf   | |
|  `NRE`, `SNRE`   |   linear, mlp, resnet   |      | | |

Lastly, the **train_args** are used to configure the training optimizer, the early stopping criterion, and the number of rounds of inference (for Sequential models). All engines use the Adam optimizer. Lastly, **device** specifies whether to use Pytorch's `cpu` or `cuda` backend, and **out_dir** specifies where to save your models after they are done training.

**Notes**: 
 * Newly added `sbi` training engines, such as `MNLE`, don't yet work in our framework. Also, the `nsf` and `made` architectures unfortunately don't work if you're using `cuda` for GPU acceleration (yet!).
 * Nicolas Chartier did extensive tests of model design choices for the `sbi` package and their impact on inference, including effects of model architectures, prior vs. proposal distribution, and data vector length. See [this notebook](https://github.com/maho3/ltu-ili/blob/backup_2023-20-11/notebooks/example_quijote_sbi_appendDummy.ipynb) for advice in choosing these hyperparameters.


## Validation
Here's an example configuration for a `ValidationRunner` object.
```bash
out_dir: './toy'  # Directory to load posterior from and to save the metrics
posterior_file: 'toy_NPE_posterior.pkl'  # Filename of posterior model
style_path: './style.mcstyle'  # [Optional] matplotlib style file
labels: ['t1', 't2', 't3']  # Names of the parameters

# If True, run validation for all networks as one ensemble posterior
# If False, run validation for each network separately
ensemble_mode: True

metrics:
  # plots a well-sampled posterior for a single test example
  single_example:  
    module: 'ili.validation.metrics'
    class: 'PlotSinglePosterior'
    args:
      num_samples: 1000
      sample_method: 'slice_np_vectorized'
      sample_params:
        num_chains: 7
        burn_in: 100
        thin: 10

  # samples posterior for all test examples sparsely, and computes univariate posterior coverage
  rank_stats:  
    module: 'ili.validation.metrics'
    class: 'PlotRankStatistics'
    args:
      num_samples: 100
      sample_method: 'emcee'
      sample_params:
        num_chains: 7
        burn_in: 100
        thin: 1

  # samples posterior for all test examples sparsely, and computes multivariate posterior coverage
  tarp:
    module: 'ili.validation.metrics'
    class: 'TARP'
    args:
      num_samples: 10
      sample_method: 'slice_np_vectorized'
      sample_params:
        num_chains: 7
        burn_in: 100
        thin: 1
```
All current implemented validation metrics follow the general formula of:
1. Pass the test inputs through the neural networks and estimate a posterior model.
2. Generate `num_sample` samples from the posterior for each input.
3. Compute a comparison metric between the posterior samples and the true values. Plot it and save the figure to `output_path`.

There are two available sampler backends in [`ili.utils.samplers`](ili/utils/samplers.py) to generate posterior samples, one that uses PyTorch's [`pyro`](https://github.com/pyro-ppl/pyro) and one that uses [`emcee`](https://github.com/dfm/emcee). The choice of training engine will constrain which sampler you can use.
- `pydelfi` models can only use the `emcee` sampler.
- `sbi`'s `NLE` or `NRE` models can use either the `emcee` or `pyro` samplers. The `pyro` samplers include several MCMC methods like slice sampling (`'slice_np'`, `'slice_np_vectorized'`), Hamiltonian Monte Carlo (`'hmc'`), and the NUTS sampler (`'nuts'`). From my experience, `slice_np_vectorized` works the fastest on CPU architectures for simple posteriors.
- `sbi`'s and `lampe`'s `NPE` models can use any of the `emcee` or `pyro` samplers. However, as they are amortized posterior estimators, they can also do fast direct estimation of the `log_prob` of samples, thus allowing for super fast Rejection Sampling. It is recommended to use this with the `'direct'` sample method for `NPE` models.

The `ensemble_mode` parameter allows you to specify whether you want to sample jointly from the ensemble of neural networks trained in your inference stage (`True`) or from each one individually (`False`). This can be useful for analyzing multiple trained architectures individually or for debugging for issues in training.

The `sampler_params` interface for specifying the number, length, and thinning of MCMC chains has been made identical for all implemented samplers.

## Overloading configuration files
Lastly, we provide the ability to overload configuration files within the `.from_config` function of each stage. This is very useful for running multipl experiments with the same base configuration file, but with slight changes to the hyperparameters. For example, if you wanted to run the same experiment with different learning rates, you could do:
```python
# experiment 1
exp1 = InferenceRunner.from_config('path/to/config.yaml', train_args={'learning_rate': 1e-3}, out_dir='./exp1')
exp1(loader)

# experiment 2
exp2 = InferenceRunner.from_config('path/to/config.yaml', train_args={'learning_rate': 1e-4}, out_dir='./exp2')
exp2(loader)
```
For an example on how this can be used for hyperparameter search, see [examples/README.md](./examples/README.md).

## Notes
All of the above configuration details are subject to change with the constant evolution of `ltu-ili`. If you notice anything to be not as documented, please write us an issue!

Happy Inference!
