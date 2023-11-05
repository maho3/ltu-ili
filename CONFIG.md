Configuration
---------------------------------

This document provides guidance to configuring an inference pipeline with ltu-ili and includes links and references to the various functionality options. There are three stages to the inference pipeline:
- [**Data Loading**](#data-loading): Loading various structured data into memory
- [**Training**](#training): Training neural networks from the loaded data and saving them to file.
- [**Validation**](#validation): Loading neural networks from file, sampling posteriors on the test set, and computing metrics.

In general, these steps are designed to be independent. For example, if you wanted to try out a different training model, all you would need to do is change the training configuration, and the data loading and validation stages should integrate (nearly) seamlessly with the new model. Each pipeline stage specified below can be instantiated from `json`-like configuration files or from iPython initialization as in [tutorial.ipynb](notebooks/tutorial.ipynb).

## Data Loading
We have four primary objects for dataloading:
- [`NumpyLoader`](./ili/dataloaders/loaders.py#L47): Loads summaries and parameters from `np.array`'s stored in memory.
- [`StaticNumpyLoader`](./ili/dataloaders/loaders.py#L81): Loads summaries and parameters from `.npy` files on disk.
- [`SummarizerDatasetLoader`](./ili/dataloaders/loaders.py#L101): Loads summaries following the [`ili_summarizer.Dataset`](https://github.com/florpi/ili-summarizer/blob/3d9d4005cfbc187afdbfbed2a5a4414bc07902ef/summarizer/dataset.py#L6) convention, as `xarray.DataArray`'s stored in `.nc` files on disk. Also, it loads parameters from `.txt` files on disk.
- [`SBISimulator`](./ili/dataloaders/loaders.py#L208): Contains a `simulate` function which, when given new parameters, generates new summaries. This interface is then used to train multiround inference with e.g. `SBIRunnerSequential`. It also can seperately load `.npy` files of test data and parameters from disk, for validation.

`NumpyLoader`s are only built from inline initialization, but the remaining classes can take config files that look like:

```bash
# StaticNumpyLoader configuration

in_dir: './toy'  # contains x_file and theta_file

x_file: 'x.npy'
theta_file: 'theta.npy'
```

```bash
# SummarizerDatasetLoader configuration

data_dir: './ltu-ili-data/quijote'  # contains param_file and train_test_split file and a hierarchy of .nc data files

summary_root_file: 'tpcf/z_0.50/quijote'  # prefix to each .nc file
param_file: 'latin_hypercube_params.txt' 
train_test_split_file: 'quijote_train_test_val.json'  # specifies which indices in param_file are in train/test/val

param_names: ['Omega_m', 'h', 'sigma_8']
```

```bash
# SBISimulator configuration

in_dir: './toy'  # where to load xobs_file and thetaobs_file from
out_dir: './toy'  # where to save simulated data to

xobs_file: 'xobs.npy'  # test data
thetaobs_file: 'thetaobs.npy'  # test parameters

x_file: 'x.npy'  # filename to save simulated data
theta_file: 'theta.npy'  # filename to save simulation parameters
num_simulations: 400  # how many simulations to generate for each training round
```

You are also welcome to design your own dataloading objects, so long as they contain the functions: `__len__`, `get_all_data`, and `get_all_parameters`.

## Training
There are three available training engines in ltu-ili, SBIRunner and SBIRunnerSequential for `sbi` models, and PydelfiRunner for `pydelfi` models. Each of these engines can be configured from a `yaml`-like configuration file or from iPython initialization as in [tutorial.ipynb](notebooks/tutorial.ipynb).

### SBIRunner
Here's an example configuration for `SBIRunner`:
```bash
prior:
  module: 'sbi.utils'
  class: 'BoxUniform'
  args:
    low: [0,0,0]
    high: [1,1,1]

model:
  module: 'sbi.inference'
  class: 'SNPE_C'
  name: 'run_SNPE'
  neural_posteriors:
    - model: 'maf'
      hidden_features: 50 
      num_transforms: 5
      signature: 'maf1'
    - model: 'maf'
      hidden_features: 50 
      num_transforms: 5
      signature: 'maf2'
    - model: 'mdn'
      hidden_features: 50 
      num_components: 2
    
train_args:
  training_batch_size: 32
  learning_rate: 0.001
  validation_fraction: 0.1
  stop_after_epochs: 20
  clip_max_norm: 5.0

device: 'cpu'
output_path: './toy'
```

The **prior** distribution specifies our prior belief on the true distribution of the inference parameters. The default here, `sbi.utils.BoxUniform`, is an extension of `torch.distribution.Uniform` modified to have a `log_prob` function which outputs a scalar value. In practice, our prior can be any [`torch.distribution`](https://pytorch.org/docs/stable/distributions.html) class which has a scalar `log_prob` function.

The **model** configuration specifies how we train our neural networks to produce posterior inference. There are three available classes of methods: posterior estimation ([`SNPE_A`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/inference/snpe/snpe_a.py#L24), [`SNPE_C`](https://github.com/mackelab/sbi/blob/main/sbi/inference/snpe/snpe_c.py)), likelihood estimation ([`SNLE_A`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/inference/snle/snle_a.py#L14)), and likelihood ratio estimation ([`SNRE_A`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/inference/snre/snre_a.py#L12), [`SNRE_B`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/inference/snre/snre_b.py#L12), [`SNRE_C`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/inference/snre/snre_b.py#L12)). We have provided links (and references therein) to the implementation of each of these methods, but for a general review of the differences between these models, see Section 3B in [Cranmer et al. 2020](https://arxiv.org/abs/1911.01429).

All implemented methods allow you to specify an ensemble of independently-trained neural networks. This ensembling is key to reducing our sensitivity to stochasticity in the training process. However, the choice of model affects what neural network architectures are implemented on the backend. See the references to each individual model to see what architecture configurations are available.
- `SNPE` models build architectures using [`sbi.utils.posterior_nn`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/utils/get_nn_models.py#L177) and support:
  - Mixture Density Networks ([`mdn`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/neural_nets/mdn.py#L14))
  - Masked Autoregressive Flows ([`maf`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/neural_nets/flow.py#L97))
  - Neural Spline Flows ([`nsf`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/neural_nets/flow.py#L293))
  - Masked Autoencoders for Density Estimation ([`made`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/neural_nets/flow.py#L24)). 
- `SNLE` models use [`sbi.utils.likelihood_nn`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/utils/get_nn_models.py#L93) and support:
  - Mixture Density Networks ([`mdn`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/neural_nets/mdn.py#L14))
  - Masked Autoregressive Flows ([`maf`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/neural_nets/flow.py#L97))
  - Neural Spline Flows ([`nsf`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/neural_nets/flow.py#L293))
  - Masked Autoencoders for Density Estimation ([`made`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/neural_nets/flow.py#L24)). 
- `SNRE` models use [`sbi.utils.classifier_nn`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/utils/get_nn_models.py#L19) and support:
  - Linear classifiers ([`linear`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/neural_nets/classifier.py#L85))
  - Multilayer Perceptrons ([`mlp`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/neural_nets/classifier.py#L136))
  - ResNets ([`resnet`](https://github.com/mackelab/sbi/blob/6c4fa7a6fd254d48d0c18640c832f2d80ab2257a/sbi/neural_nets/classifier.py#L194)).

Within the model configuration, you can specify a name for your ensemble posterior and a name for each individual neural network, with the `name` and `signature` parameters, respectively. These names are used to save the trained models to file, and can be referenced later in the validation stage using its `posterior_path` argument.

The **embedding_net** configuration allows one to specify additional neural layers which will prepend the input layer of the above neural density estimators. The default `sbi` architectures listed above are generally quite shallow, so its a good idea to make use of embedding architectures, especially for complex data. We include a fully-connected network (`ili.embedding.FCN`), but we also have an example of a CNN-like embedding network in [tutorial.ipynb](notebooks/tutorial.ipynb).

The **train_args** are used to configure the training optimizer and early stopping criterion. All `sbi` models use the Adam optimizer. Lastly, **device** specifies whether to use Pytorch's `cpu` or `cuda` backend, and **output_path** specifies where to save your models after they are done training.

Notes: Newly added `sbi` training engines, such as `MNLE`, don't yet work in our framework. Also, the `nsf` and `made` architectures unfortunately don't work if you're using `cuda` for GPU acceleration (yet!).


### SBIRunnerSequential
`SBIRunnerSequential` differs from `SBIRunner` in that it allows for multiround sampling for active learning inference. The configuration for `SBIRunnerSequential` is almost exactly the same as that of `SBIRunner`, except there is an added customizable parameter in `train_args` called `num_round` which specifies the number of rounds of simulation-training that the runner should do during training.

### PydelfiRunner
Here's an example configuration for `PydelfiRunner`
```bash
n_params: 3
n_data: 10

prior:
  module: 'pydelfi.priors'
  class: 'Uniform'
  args:
    lower: [0,0,0]
    upper: [1,1,1]

model:
  module: 'ili.inference.pydelfi_wrappers'
  class: 'DelfiWrapper'
  name: 'run_delfi'
  kwargs:
    nwalkers: 20
  nets:
    - module: 'pydelfi.ndes'
      class: 'MixtureDensityNetwork'
      args: 
        n_components: 12
        n_hidden: [64,64]
        activations: ['tanh','tanh']
    - module: 'pydelfi.ndes'
      class: 'ConditionalMaskedAutoregressiveFlow'
      args: 
        n_hiddens: [50,50]
        n_mades: 2
        act_fun: 'tanh'
    
train_args:
  batch_size: 32
  epochs: 300

output_path: 'toy'
```
The `PydelfiRunner` configuration is very similar  to that of `SBIRunner`. The biggest differences are that there are no explicit embedding networks and there is only one inference engine, `DelfiWrapper`. The `DelfiWrapper` engine follows a likelihood estimation methodology, based on the same paper as that for the `SNLE_A` implementatioin in `sbi`. 

The `prior` configuration can be any of the implemented distributions in [`pydelfi.priors`](https://github.com/justinalsing/pydelfi/blob/master/pydelfi/priors.py). The training engine supports the following models in `pydelfi.ndes`:
- [`MixtureDensityNetwork`](https://github.com/justinalsing/pydelfi/blob/9b1bf65c7171d4bd8b10a903b01a6fad5311e5ba/pydelfi/ndes.py#L268C7-L268C28)
- [`ConditionalMaskedAutoregressiveFlow`](https://github.com/justinalsing/pydelfi/blob/9b1bf65c7171d4bd8b10a903b01a6fad5311e5ba/pydelfi/ndes.py#L191)
- [`ConditionalGaussianMade`](https://github.com/justinalsing/pydelfi/blob/9b1bf65c7171d4bd8b10a903b01a6fad5311e5ba/pydelfi/ndes.py#L6)

Unlike in `sbi`, one can customize the number of hidden layers and types of activation functions in each of these architectures.

The training procedure is controlled through the `train_args` parameters. `PydelfiRunner` uses an Adam optimizer and enforces a strict maximum number of training epochs, without using an early stopping criterion. Lastly, you must manually specify the dimensionality of your parameter and data vectors (in `n_params` and `n_data`), whereas these are handled automatically in `sbi`.

## Validation
Here's an example configuration for a `ValidationRunner` object.
```bash
backend: 'sbi'

posterior_path: './toy/run_SNPE_posterior.pkl'
output_path: './toy'
labels: ['t1', 't2', 't3']

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
- `sbi`'s `SNLE` or `SNRE` models can use either the `emcee` or `pyro` samplers. The `pyro` samplers include several MCMC methods like slice sampling (`'slice_np'`, `'slice_np_vectorized'`), Hamiltonian Monte Carlo (`'hmc'`), and the NUTS sampler (`'nuts'`). From my experience, `slice_np_vectorized` works the fastest on CPU architectures for simple posteriors.
- `sbi`'s `SNPE` models can use any of the `emcee` or `pyro` samplers. However, as they are amortized posterior estimators, they can also do fast direct estimation of the `log_prob` of samples, thus allowing for super fast Rejection Sampling. It is recommended to use this with the `'direct'` sample method for `SNPE` models.

The `ensemble_mode` parameter allows you to specify whether you want to sample jointly from the ensemble of neural networks trained in your inference stage (`True`) or from each one individually (`False`). This can be useful for analyzing multiple trained architectures individually or for debugging for issues in training.

The `sampler_params` interface for specifying the number, length, and thinning of MCMC chains has been made identical for all implemented samplers.

## Notes
All of the above configuration details are subject to change with the constant evolution of `ltu-ili`. If you notice anything to be not as documented, please write us an issue!

Happy Inference!