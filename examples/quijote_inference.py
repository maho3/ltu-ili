import os
import numpy as np
from ili.inference.runner_sbi import SBIRunner
from ili.validation.runner import ValidationRunner


# train a model to infer x -> theta. save it as toy/posterior.pkl
runner = SBIRunner.from_config('configs/sample_quijoteTPCF_config.yaml')
runner()

# use the trained posterior model to predict on a single example from the test set
valrunner = ValidationRunner.from_config('configs/sample_quijotevalidation_config.yaml')
valrunner()
