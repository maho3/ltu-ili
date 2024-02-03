import os
import numpy as np
from ili.dataloaders import SBISimulator
from ili.inference import InferenceRunner
from ili.validation import ValidationRunner


def simulator(params):
    # create toy simulations
    # NOTE: For use with SBISimulator, this function takes
    # a tuple of parameters and outputs data of shape (1, *data_shape)
    x = np.arange(10)
    y = params @ np.array([3*np.sin(x), x ** 2, -2*x])
    y += np.random.randn(len(params), len(x))
    return y


if __name__ == '__main__':
    # construct a working directory
    if not os.path.isdir("toy"):
        os.mkdir("toy")

    # simulate a single test observation and save as numpy files
    theta0 = np.zeros((1, 3))+0.5
    x0 = simulator(theta0)
    np.save('toy/thetafid.npy', theta0[0])
    np.save('toy/xobs.npy', x0[0])

    # setup a dataloader which can simulate
    all_loader = SBISimulator.from_config("configs/data/toy_multiround.yaml")
    all_loader.set_simulator(simulator)

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    runner = InferenceRunner.from_config(
        "configs/infer/toy_sbi_multiround.yaml")
    runner(loader=all_loader)

    # use the trained posterior model to predict on a single example from
    # the test set
    val_runner = ValidationRunner.from_config(
        "configs/val/toy_sbi_multiround.yaml")
    val_runner(loader=all_loader)
