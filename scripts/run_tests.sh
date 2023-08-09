#!/bin/bash
set -o verbose
set -e

# setup conda
source ~/anaconda3/etc/profile.d/conda.sh
conda env remove -n ili-sbi
conda env remove -n ili-pydelfi
conda env remove -n ili-docs

cd ../../

# install with sbi backend
conda create -n ili-sbi python=3.10 -y
conda activate ili-sbi
python3 -V
pip install -e "ltu-ili[sbi]"

# download and install ili-summarizer
rm -rf test_dir
mkdir test_dir
cd test_dir
git clone https://github.com/florpi/ili-summarizer.git
pip install -e ili-summarizer

# run sbi backend examples 
cd ../ltu-ili/examples
python3 toy_sbi.py

conda deactivate

# install with pydelfi backend
conda create -n ili-pydelfi python=3.6 -y
conda activate ili-pydelfi
cd ../../
pip install -e "ltu-ili[pydelfi]"
cd test_dir
pip install -e ili-summarizer

# run pydelfi backend examples
cd ../ltu-ili/examples
python3 toy_pydelfi.py

conda deactivate

# make documentation
conda create -n ili-docs python=3.10 -y
conda activate ili-docs
cd ../docs
pip install -r requirements.txt
make clean
make html
conda deactivate
