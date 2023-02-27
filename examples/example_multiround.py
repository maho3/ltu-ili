import torch

from ili.dataloaders import SBISimulator
from ili.inference.runner_sbi import SBIRunner
from ili.validation.runner import ValidationRunner

_ = torch.manual_seed(0)

def my_fun(theta):
    return theta + 1.0 + torch.randn_like(theta) * 0.1

num_rounds = 2
theta0 = torch.zeros(3,)
x0 = my_fun(theta0)

# setup a dataloader which can simulate
all_loader = SBISimulator.from_config("configs/data/multiround.yaml")
all_loader.set_simulator(my_fun)

# train a model to infer x -> theta. save it as toy/posterior.pkl
runner = SBIRunner.from_config("configs/infer/multiround_ensemble.yaml")
runner(loader=all_loader, num_rounds=num_rounds, x_obs=x0)

# use the trained posterior model to predict on a single example from the test set
val_runner = ValidationRunner.from_config("configs/val/multiround.yaml")
val_runner(loader=all_loader, prior=runner.prior, x_obs=x0, theta_obs=theta0)

