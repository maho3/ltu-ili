Installation
============

There are two distinct installations of ltu-ili which provide access to different inference engines, namely the pytorch ([sbi](https://github.com/mackelab/sbi) and [lampe](https://lampe.readthedocs.io/en/stable/index.html)) installations.

[!Caution]
The previously supported `pydelfi` backend is no longer maintained and requires tensorflow==1.15, which fails on Mac OS with M1/M2 chips and requires Python3.6 which is well past end-of-life.
We have retained a `pydelfi` branch, see the INSTALLTION.md on that branch.

## Basic installation

### Install from PyPI
```bash
# Install base package
pip install ltu-ili

# Then install your preferred backend
pip install "ltu-ili[pytorch]"  # for sbi/lampe

### Install from source

#### [`uv`](https://docs.astral.sh/uv/) (recommended for development)

```bash
git clone https://github.com/maho3/ltu-ili.git
cd ltu-ili
uv sync --all-extras --dev --python 3.10
source .venv/bin/activate
```

#### `conda`
```bash
# Clone and install with pytorch backend (sbi/lampe)
conda create -n ili-torch python=3.10 
conda activate ili-torch
pip install --upgrade pip
git clone https://github.com/maho3/ltu-ili.git
cd ltu-ili
pip install ".[pytorch]" --group dev
```

## Verify installation

### Toy example
You can verify that the installation is working by running the toy example
```bash
   cd ltu-ili/examples
   
   # sbi backend
   python toy_sbi.py
```
After the script completes, you should be able to find some metric summaries and plots in the **examples/toy** folder.

### Quijote example
You can also download pre-processed Quijote two point correlation function (TPCF) summaries and run example cosmological inference. Quijote TPCF summaries can be found in the LtU Open Storage Network data repository ([https://sdsc.osn.xsede.org/learningtheuniverse](https://sdsc.osn.xsede.org/learningtheuniverse)). Contact Matt Ho (matthew.annam.ho@gmail) for access.

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
You can also interact with ltu-ili through an iPython environment such as jupyter. See the [notebooks folder](https://github.com/maho3/ltu-ili/blob/main/notebooks/) for a comprehensive guide to the jupyter interface.
