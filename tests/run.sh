#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

cd ..

conda activate ili-sbi
pip3 install coverage pytest
echo "Running sbi tests"
COVERAGE_FILE=.coverage_sbi python3 -m coverage run --source=ili -m pytest tests/test_sbi.py
conda deactivate

conda activate ili-pydelfi
pip3 install coverage pytest
echo "Running pydelfi tests"
COVERAGE_FILE=.coverage_pydelfi python3 -m coverage run --source=ili -m pytest tests/test_pydelfi.py
conda deactivate

coverage combine .coverage_sbi .coverage_pydelfi
coverage xml
coverage report -m
