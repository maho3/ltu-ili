This directory contains several example scripts for running the ltu-ili inference pipeline, including support for both the sbi and pydelfi backends.

In these examples, we demonstrate parameter inference for two experiments:
* `toy`: A toy experiment with 3 parameters and a 10-dimensional data vector. The data is generated from a non-linear function of the parameters, with Gaussian noise added. The data generation occurs natively in the example scripts.
* `quijote`: An example of cosmological parameter inference from 1D power spectrum analysis, using the Quiijote latin hypercube simulations. The power spectra data can be generated from raw Quijote halo catalogs using [ili-summarizer](https://github.com/florpi/ili-summarizer), or downloaded as a pre-generated catalog from the LtU repository on OSN.

Each script queries the data, inference, and validation configuration files in the [data](./data), [infer](./infer), and [val](./val) subfolders, respectively. These files contain details specifying where the data is loaded from, how the inference is performed, and what validation metrics are computed.

To get started, one can run the [toy_sbi.py](./toy_sbi.py) script as follows:
```bash
cd examples
python toy_sbi.py
```
This will run a full inference pipeline, from data loading to model training to validation. Trained models and validation results will then be saved in the `toy` subfolder.

To try other inference models (i.e. likelihood or ratio estimation) with the toy data, you can also use:
```bash
python toy_sbi.py --model SNLE
python toy_sbi.py --model SNRE
```