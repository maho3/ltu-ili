import os
import argparse
import numpy as np
from ili.dataloaders import StaticNumpyLoader
from ili.inference.runner_sbi import SBIRunner
from ili.validation.runner import ValidationRunner


def simulator(params):
    # create toy simulations
    x = np.arange(10)
    y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
    y += np.random.randn(len(x))
    return y


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run SBI inference for toy data.")
    parser.add_argument(
        "--model", type=str,
        default="SNPE",
        help="Configuration file to use for model training.")
    args = parser.parse_args()

    # construct a working directory
    if not os.path.isdir("toy"):
        os.mkdir("toy")

    # simulate data and save as numpy files
    theta = np.random.rand(200, 3)  # 200 simulations, 3 parameters
    x = np.array([simulator(t) for t in theta])
    np.save("toy/theta.npy", theta)
    np.save("toy/x.npy", x)

    # reload all simulator examples as a dataloader
    all_loader = StaticNumpyLoader.from_config("configs/data/toy.yaml")

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    runner = SBIRunner.from_config(f"configs/infer/toy_sbi_{args.model}.yaml")
    runner(loader=all_loader)

    # use the trained posterior model to predict on a single example from
    # the test set
    val_runner = ValidationRunner.from_config(
        f"configs/val/toy_sbi_{args.model}.yaml")
    val_runner(loader=all_loader)
