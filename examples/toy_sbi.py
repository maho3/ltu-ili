import os
import numpy as np
from ili.dataloaders import StaticNumpyLoader
from ili.inference.runner_sbi import SBIRunner
from ili.validation.runner import ValidationRunner


# create toy 'simulations'
def simulator(params):
    # params += np.random.normal(size=len(x))
    x = np.arange(10)
    y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
    y += np.random.randn(len(x))
    return y


theta = np.random.rand(2000, 3)  # 2000 simulations, 3 parameters
x = np.array([simulator(t) for t in theta])

# save them as numpy files
if not os.path.isdir("toy"):
    os.mkdir("toy")
np.save("toy/theta.npy", theta)
np.save("toy/x.npy", x)

# reload all simulator examples as a dataloader
all_loader = StaticNumpyLoader.from_config("configs/data/toy.yaml")

# train a model to infer x -> theta. save it as toy/posterior.pkl
runner = SBIRunner.from_config("configs/infer/toy_sbi.yaml")
runner(loader=all_loader)

# use the trained posterior model to predict on a single example from the test set
val_runner = ValidationRunner.from_config("configs/val/toy_sbi.yaml")
val_runner(loader=all_loader)
