import torch
import numpy as np

from ili.dataloaders import SBISimulator
from ili.inference.runner_sbi import SBIRunnerSequential
from ili.validation.runner import ValidationRunner

_ = torch.manual_seed(0)

# create toy 'simulations'
def simulator(params):
    x = np.arange(10)
    y = params @ np.array([np.sin(x), x ** 2, x])
    y += np.random.randn(len(params),len(x))
    return y

theta0 = np.zeros((1,3))+0.5
x0 = simulator(theta0)
np.save('toy/thetaobs.npy', theta0[0])
np.save('toy/xobs.npy', x0[0])

# setup a dataloader which can simulate
all_loader = SBISimulator.from_config("configs/data/multiround.yaml")
all_loader.set_simulator(simulator)

# train a model to infer x -> theta. save it as toy/posterior.pkl
runner = SBIRunnerSequential.from_config("configs/infer/multiround_ensemble.yaml")
runner(loader=all_loader)

# use the trained posterior model to predict on a single example from the test set
val_runner = ValidationRunner.from_config("configs/val/sample_sbi.yaml")
val_runner(loader=all_loader, prior=runner.prior)
