#!/bin/bash
cd ..

conda activate ili_dev_sbi
echo "Running sbi tests locally"
COVERAGE_FILE=.coverage_sbi python3 -m coverage run --source=ili -m pytest tests/test_sbi.py
conda deactivate

coverage combine .coverage_sbi
coverage xml
coverage report -m
