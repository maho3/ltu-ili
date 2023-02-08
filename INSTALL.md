# Installation Instructions


## Basic installation

Here is a quick guide for getting started with the ltu-ili framework.

First, clone the main branch of the ltu-ili repository onto your local machine.
```bash
    git clone git@github.com:maho3/ltu-ili.git
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
   pip install -e ili-summarizer
```
4. Finally install the project via
```bash
  pip install -e ltu-ili
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
   pip install -e ili-summarizer
```
6. Finally setup the project via
```bash
    pip install -e ltu-ili
```

### Verify installation

You can verify that the installation is working by running the toy example
```bash
   cd ltu-ili/examples
   python example_inference.py 
```
After the script completed, you should be able to find some metric summaries and plots in the **examples/toy** folder.

It's also possible to use pre-processed Quijote two point correlation function (TPCF) summaries and run example cosmological inference. Quijote TPCF summaries can be found in the LtU Open Storage Network data repository ([https://sdsc.osn.xsede.org/learningtheuniverse](https://sdsc.osn.xsede.org/learningtheuniverse)). See the download instructions in the #ili-wg Slack channel or ping Matt Ho if you need help getting this data. 

Store this data in a subfolder called `ltu-ili-data/` and run the inference using
```bash
   cd examples
   python quijote_inference.py 
```
In case you want to use a mixture density network as density estimator instead of a normalizing flow, then execute
```bash
   python quijote_inference.py --cfginfer configs/infer/quijote_MDN.yaml
```
