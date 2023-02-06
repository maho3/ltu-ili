import os
import numpy as np
from ili.dataloaders import StaticNumpyLoader
from ili.inference.runner_pydelfi import DelfiRunner
from ili.validation.runner import ValidationRunner

# create toy 'simulations'
def simulator(params):
    x = np.arange(10)
    y = params[0] * np.sin(x) + params[1] * x ** 2 - 3 * params[2] * x
    y += np.random.normal(len(x))
    return y


theta = np.random.rand(2000, 3)  # 2000 simulations, 3 parameters
x = np.array([simulator(t) for t in theta])

# save them as numpy files
if not os.path.isdir("toy"):
    os.mkdir("toy")
np.save("toy/theta.npy", theta)
np.save("toy/x.npy", x)


# reload all simulator examples as a dataloader
all_loader = StaticNumpyLoader.from_config("configs/data/sample.yaml")

# train a model to infer x -> theta. save it as toy/posterior.pkl
runner = DelfiRunner.from_config("configs/infer/sample_delfi.yaml")
runner(loader=all_loader)

# use the trained posterior model to predict on a single example from the test set
val_runner = ValidationRunner.from_config("configs/val/sample_pydelfi.yaml")
val_runner(loader=all_loader)
