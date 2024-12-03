LtU-ILI
=======
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[allc]: https://img.shields.io/badge/all_contributors-12-orange.svg?style=flat-square 'Number of contributors on All-Contributors'
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![All Contributors][allc]](https://github.com/maho3/ltu-ili/tree/main#contributors-)
[![unittest](https://github.com/maho3/ltu-ili/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/maho3/ltu-ili/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/maho3/ltu-ili/graph/badge.svg?token=8QNMK453GE)](https://codecov.io/gh/maho3/ltu-ili)
[![docs](https://readthedocs.org/projects/ltu-ili/badge/?version=latest)](https://ltu-ili.readthedocs.io/en/latest/?badge=latest)

The **Learning the Universe Implicit Likelihood Inference (LtU-ILI)** pipeline is an all-in-one framework for performing machine learning parameter inference in astrophysics and cosmology. Given labeled training data ${(x_i,\theta_i)}_{i=1}^N$ or a stochastic simulator $x(\theta)$, LtU-ILI is designed to automatically train state-of-the-art neural networks to learn the data-parameter relationship and produce robust, well-calibrated posterior inference.

The pipeline is quick and easy to set up; here's an example of training a [Masked Autoregressive Flow (MAF)](https://arxiv.org/abs/1705.07057) network to predict a posterior over parameters $y$, given input data $x$:

```python
...  # Imports

X, Y = load_data()                              # Load training data and parameters
loader = ili.data.NumpyLoader(X, Y)             # Create a data loader

trainer = ili.inference.InferenceRunner.load(
  backend = 'sbi', engine='NPE',                # Choose a backend and inference engine (here, Neural Posterior Estimation)
  prior = ili.utils.Uniform(low=-1, high=1),    # Define a prior 
  # Define a neural network architecture (here, MAF)
  nets = [ili.utils.load_nde_sbi(engine='NPE', model='maf')]  
)

posterior, _ = trainer(loader)                  # Run training to map data -> parameters

samples = posterior.sample(                     # Generate 1000 samples from the posterior for input x[0]
  x=X[0], sample_shape=(1000,)
)
```
Beyond this simple example, LtU-ILI comes with a wide range of customizable complexity, including:
  * Posterior-, Likelihood-, and Ratio-Estimation methods for ILI, including Sequential learning analogs
  * Various neural density estimators (Mixture Density Networks, Conditional Normalizing Flows, ResNet-like ratio classifiers)
  * Fully-customizable, exotic embedding networks (including CNNs and Graph Neural Networks)
  * A unified interface for multiple ILI backends ([sbi](https://github.com/sbi-dev/sbi), [pydelfi](https://github.com/justinalsing/pydelfi), and [lampe](https://lampe.readthedocs.io/en/stable/))
  * Multiple marginal and multivariate posterior coverage metrics
  * Jupyter and command-line interfaces
  * A parallelizable configuration framework for efficient hyperparameter tuning and production runs


For more details on the motivation, design, and theoretical background of this project, see the [software release paper](https://doi.org/10.33232/001c.120559).



## Getting Started 
To install LtU-ILI, follow the instructions in [INSTALL.md](INSTALL.md).

To get started, try out the tutorial for the Jupyter notebook interface in [notebooks/tutorial.ipynb](https://github.com/maho3/ltu-ili/blob/main/notebooks/tutorial.ipynb) or the command line interface in [examples/](https://github.com/maho3/ltu-ili/tree/main/examples).

## API Documentation
The documentation for this project can be found [at this link](https://ltu-ili.readthedocs.io/en/latest/).

## References
We keep an updated repository of relevant interesting papers and resources [at this link](https://hackmd.io/8inFGHxxTmye4wtPaFXRWA).

## Contributing
Before contributing, please familiarize yourself with the contribution workflow described in [CONTRIBUTING.md](CONTRIBUTING.md).

## Recent Changes
We keep a record of recent changes in the project in the [CHANGELOG.md](CHANGELOG.md) file. If you are interested in the latest features, bug fixes, and improvements, you can find them there.

## Contact
If you have comments, questions, or feedback, please [write us an issue](https://github.com/maho3/ltu-ili/issues). The current leads of the Learning the Universe ILI working group are Benjamin Wandelt (benwandelt@gmail.com) and Matthew Ho (matthew.annam.ho@gmail.com).

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://maho3.github.io/"><img src="https://avatars.githubusercontent.com/u/11132524?v=4?s=100" width="100px;" alt="Matt Ho"/><br /><sub><b>Matt Ho</b></sub></a><br /><a href="https://github.com/maho3/ltu-ili/commits?author=maho3" title="Code">💻</a> <a href="#design-maho3" title="Design">🎨</a> <a href="#example-maho3" title="Examples">💡</a> <a href="https://github.com/maho3/ltu-ili/commits?author=maho3" title="Documentation">📖</a> <a href="https://github.com/maho3/ltu-ili/pulls?q=is%3Apr+reviewed-by%3Amaho3" title="Reviewed Pull Requests">👀</a> <a href="#infra-maho3" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#content-maho3" title="Content">🖋</a> <a href="#research-maho3" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://deaglanbartlett.github.io/"><img src="https://avatars.githubusercontent.com/u/47668431?v=4?s=100" width="100px;" alt="Deaglan Bartlett"/><br /><sub><b>Deaglan Bartlett</b></sub></a><br /><a href="https://github.com/maho3/ltu-ili/commits?author=DeaglanBartlett" title="Code">💻</a> <a href="#design-DeaglanBartlett" title="Design">🎨</a> <a href="#example-DeaglanBartlett" title="Examples">💡</a> <a href="https://github.com/maho3/ltu-ili/commits?author=DeaglanBartlett" title="Documentation">📖</a> <a href="https://github.com/maho3/ltu-ili/pulls?q=is%3Apr+reviewed-by%3ADeaglanBartlett" title="Reviewed Pull Requests">👀</a> <a href="#infra-DeaglanBartlett" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#content-DeaglanBartlett" title="Content">🖋</a> <a href="#research-DeaglanBartlett" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/CompiledAtBirth"><img src="https://avatars.githubusercontent.com/u/47000650?v=4?s=100" width="100px;" alt="Nicolas Chartier"/><br /><sub><b>Nicolas Chartier</b></sub></a><br /><a href="#example-CompiledAtBirth" title="Examples">💡</a> <a href="https://github.com/maho3/ltu-ili/commits?author=CompiledAtBirth" title="Documentation">📖</a> <a href="#research-CompiledAtBirth" title="Research">🔬</a> <a href="https://github.com/maho3/ltu-ili/commits?author=CompiledAtBirth" title="Code">💻</a> <a href="#design-CompiledAtBirth" title="Design">🎨</a> <a href="https://github.com/maho3/ltu-ili/pulls?q=is%3Apr+reviewed-by%3ACompiledAtBirth" title="Reviewed Pull Requests">👀</a> <a href="#content-CompiledAtBirth" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/florpi"><img src="https://avatars.githubusercontent.com/u/15879020?v=4?s=100" width="100px;" alt="Carolina Cuesta"/><br /><sub><b>Carolina Cuesta</b></sub></a><br /><a href="https://github.com/maho3/ltu-ili/commits?author=florpi" title="Code">💻</a> <a href="#design-florpi" title="Design">🎨</a> <a href="#example-florpi" title="Examples">💡</a> <a href="https://github.com/maho3/ltu-ili/commits?author=florpi" title="Documentation">📖</a> <a href="https://github.com/maho3/ltu-ili/pulls?q=is%3Apr+reviewed-by%3Aflorpi" title="Reviewed Pull Requests">👀</a> <a href="#research-florpi" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AsianTaco"><img src="https://avatars.githubusercontent.com/u/42298902?v=4?s=100" width="100px;" alt="Simon"/><br /><sub><b>Simon</b></sub></a><br /><a href="https://github.com/maho3/ltu-ili/commits?author=AsianTaco" title="Code">💻</a> <a href="#example-AsianTaco" title="Examples">💡</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/axellapel"><img src="https://avatars.githubusercontent.com/u/69917993?v=4?s=100" width="100px;" alt="Axel Lapel"/><br /><sub><b>Axel Lapel</b></sub></a><br /><a href="https://github.com/maho3/ltu-ili/commits?author=axellapel" title="Code">💻</a> <a href="#research-axellapel" title="Research">🔬</a> <a href="#example-axellapel" title="Examples">💡</a></td>
      </tr>
      <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://pablo-lemos.github.io"><img src="https://avatars.githubusercontent.com/u/38078898?v=4?s=100" width="100px;" alt="Pablo Lemos"/><br /><sub><b>Pablo Lemos</b></sub></a><br /><a href="#design-Pablo-Lemos" title="Design">🎨</a> <a href="https://github.com/maho3/ltu-ili/commits?author=Pablo-Lemos" title="Code">💻</a></td>
        <td align="center" valign="top" width="14.28%"><a href="http://christopherlovell.co.uk"><img src="https://avatars.githubusercontent.com/u/4648092?v=4?s=100" width="100px;" alt="Chris Lovell"/><br /><sub><b>Chris Lovell</b></sub></a><br /><a href="#research-christopherlovell" title="Research">🔬</a> <a href="#example-christopherlovell" title="Examples">💡</a> <a href="#data-christopherlovell" title="Data">🔣</a> <a href="#content-christopherlovell" title="Content">🖋</a></td>
        <td align="center" valign="top" width="14.28%"><a href="https://tlmakinen.github.io/"><img src="https://avatars.githubusercontent.com/u/29409312?v=4?s=100" width="100px;" alt="T. Lucas Makinen"/><br /><sub><b>T. Lucas Makinen</b></sub></a><br /><a href="https://github.com/maho3/ltu-ili/commits?author=tlmakinen" title="Code">💻</a> <a href="#research-tlmakinen" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://modichirag.github.io/"><img src="https://avatars.githubusercontent.com/u/13356766?v=4?s=100" width="100px;" alt="Chirag Modi"/><br /><sub><b>Chirag Modi</b></sub></a><br /><a href="#design-modichirag" title="Design">🎨</a> <a href="https://github.com/maho3/ltu-ili/commits?author=modichirag" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/shivampcosmo"><img src="https://avatars.githubusercontent.com/u/32287865?v=4?s=100" width="100px;" alt="Shivam Pandey"/><br /><sub><b>Shivam Pandey</b></sub></a><br /><a href="#research-shivampcosmo" title="Research">🔬</a> <a href="#example-shivampcosmo" title="Examples">💡</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://luciaperez.owlstown.net"><img src="https://avatars.githubusercontent.com/u/26099741?v=4?s=100" width="100px;" alt="L.A. Perez"/><br /><sub><b>L.A. Perez</b></sub></a><br /><a href="#research-laperezNYC" title="Research">🔬</a> <a href="#content-laperezNYC" title="Content">🖋</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## Acknowledgements

This work is supported by the Simons Foundation through the [Simons Collaboration on Learning the Universe](https://www.learning-the-universe.org/).

