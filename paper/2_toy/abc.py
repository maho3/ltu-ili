from os.path import join
import argparse
from ili.dataloaders import SBISimulator, StaticNumpyLoader
from ili.validation.runner import ValidationRunner
from gen import simulator


if __name__ == '__main__':
    # parse arguments
    model = 'abc'
    cfgdir = '.'

    # reload all simulator examples as a dataloader
    train_loader = SBISimulator.from_config(join(cfgdir, "data_train.yaml"))
    train_loader.set_simulator(simulator)

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    from ili.inference.runner_sbi import ABCRunner as Runner
    runner = Runner.from_config(join(cfgdir, f"inf_{model}.yaml"))
    runner(loader=train_loader)
