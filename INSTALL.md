# Installation Instructions


## Basic installation

Here is a quick guide for getting started with the ltu-ili framework.

First, clone the main branch of the ltu-ili repository onto your local machine.
```bash
    git clone git@github.com:maho3/ltu-ili.git
    cd ltu-ili
```
Next, setup your environment to support the required dependencies for ltu-ili. ltu-ili requires a Python version >=3.7. The list of required modules is given in [requirements.txt](requirements.txt). There are two ways to install these, either using an environment manager such as [conda](https://docs.anaconda.com/) or the default Python installer pip. While we recommend the former, we give instructions for both methods.
### pip
1. Ensure that your Python version is >=3.7. If not, [install the appropriate version](https://www.python.org/downloads/).
```bash
    python --version
```
2. Ensure that your pip is at the latest version.
```bash
    pip install --upgrade pip
```
3. Install additional dependencies from forked repository [ili-summarizer](https://github.com/florpi/ili-summarizer).
```bash
   git clone git@github.com:florpi/ili-summarizer.git
   cd ili-summarizer
   pip install -e .
```
4. Finally setup the project via
```bash
    pip install -e .
```

### conda
1. Ensure that you have anaconda3 installed by following its [installation instructions](https://docs.anaconda.com/anaconda/install/index.html).
2. Create a new virtual environment with a Python version >=3.7 and required dependencies. For example,
```bash
    conda env create -f environment.yml
```
3. Activate the newly created environment.
```bash
    conda activate ili_env 
```
5. Install additional dependencies from forked repository [ili-summarizer](https://github.com/florpi/ili-summarizer).
```bash
   git clone git@github.com:florpi/ili-summarizer.git
   cd ili-summarizer
   pip install -e .
```
6. Finally setup the project via
```bash
    pip install -e .
```

