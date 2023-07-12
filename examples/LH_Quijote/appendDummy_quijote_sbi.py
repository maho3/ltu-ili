#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:29:55 2023

@author: chartier
"""

#%% Import necessary packages

import os
import numpy as np
from ili.dataloaders import StaticNumpyLoader
from ili.inference.runner_sbi import SBIRunner
from ili.validation.runner import ValidationRunner # ie TARP metric
from ili.validation.metrics import PlotSinglePosterior, TARP # Both inherit the class BaseMetric in validation/metrics

#%% For logs and results

dataDir="todo"
resultDir="todo"

# reload all simulator examples as a dataloader
all_loader = StaticNumpyLoader.from_config("configs/data/sample.yaml")

# train a model to infer x -> theta. save it as toy/posterior.pkl
runner = SBIRunner.from_config("configs/infer/sample_sbi.yaml")
runner(loader=all_loader)
