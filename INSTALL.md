Installation
============

There are two backends to ltu-ili which provide access to different inference engines, namely [sbi](https://github.com/mackelab/sbi) and [pydelfi](https://github.com/justinalsing/pydelfi). This codebase provides simultaneous support for both of these backends, but they cannot (yet!) be installed at the same time due to dependency issues. As a result, we recommend you install them separately in two distinct virtual environments.

**Note**: The pydelfi backend requires tensorflow==1.15, which fails on Mac OS with M1/M2 chips.

## Basic installation

Here is a quick guide for getting started with the ltu-ili framework.

1. First, clone the main branch of the ltu-ili repository onto your local machine.
    ```bash
    git clone git@github.com:maho3/ltu-ili.git
    ```
2. Next, create a new virtual environment with an appropriate python version for your choice of backend (i.e. Python>=3.7 for sbi or Python==3.6 for pydelfi). Then, install the appropriate configuration of ltu-ili into your environment with `pip install -e`. Below is code for installing ltu-ili with the sbi backend with [conda](https://docs.anaconda.com/):
    ```bash
    # install with sbi backend
    conda create -n ili-sbi python=3.10
    conda activate ili-sbi
    pip install -e "ltu-ili[sbi]"
    ```
    and below is the equivalent for the pydelfi backend:
    ```bash
    # install with pydelfi backend
    conda create -n ili-pydelfi python=3.6
    conda activate ili-pydelfi
    pip install -e "ltu-ili[pydelfi]"
    ```

3. Next, manually install the [ili-summarizer](https://github.com/florpi/ili-summarizer) dependency into the same environment with:
    ```bash
    git clone git@github.com:florpi/ili-summarizer.git
    pip install -e ili-summarizer
    ```
Note, the above command installs `ili-summarizer` without summary calculation backends (e.g. without `nbodykit`, `pycorr`, `jax`, `kymatio`). If you wish to use these, you can replace the above command with `pip install -e 'ili-summarizer[backends]'`, though this is incompatible with our `pydelfi` version.

After this, ltu-ili and all its required dependencies should be correctly set up in your virtual environment.
## Verify installation

### Toy example
You can verify that the installation is working by running the toy example
```bash
   cd ltu-ili/examples
   
   # sbi backend
   python toy_sbi.py
   
   # pydelfi backend 
   python toy_pydelfi.py
```
After the script completes, you should be able to find some metric summaries and plots in the **examples/toy** folder.

### Quijote example
You can also download pre-processed Quijote two point correlation function (TPCF) summaries and run example cosmological inference. Quijote TPCF summaries can be found in the LtU Open Storage Network data repository ([https://sdsc.osn.xsede.org/learningtheuniverse](https://sdsc.osn.xsede.org/learningtheuniverse)). See the data access instructions in [DATA.md](DATA.md) for more details.

Store this data in a subfolder called `ltu-ili-data/` and run the inference using
```bash
   cd examples
   
   # sbi backend
   python quijote_sbi.py 

   # pydelfi backend
   python3 quijote_pydelfi.py
```
In case you want to use a mixture density network as density estimator instead of a normalizing flow, then execute
```bash
   python quijote_sbi.py --cfginfer configs/infer/quijote_sbi_MDN.yaml
```

## Jupyter Interface
You can also interact with ltu-ili through an iPython environment such as jupyter. See the [notebooks/tutorial.ipynb](notebooks/tutorial.ipynb) notebook for a guide to the jupyter interface.