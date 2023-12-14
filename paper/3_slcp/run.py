from os.path import join
import argparse
from ili.dataloaders import SBISimulator, StaticNumpyLoader
from ili.validation.runner import ValidationRunner
from gen import simulator


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run SBI inference for toy data.")
    parser.add_argument(
        "--model", type=str,
        default="npe",
        help="Configuration file to use for model training.")
    parser.add_argument(
        "--cfgdir", type=str,
        default='.',)
    args = parser.parse_args()
    model = args.model
    cfgdir = args.cfgdir

    # reload all simulator examples as a dataloader
    train_loader = SBISimulator.from_config(join(cfgdir, "data_train.yaml"))
    train_loader.set_simulator(simulator)

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    if model == 'pydelfi':
        from ili.inference.runner_pydelfi import DelfiRunner as Runner
    else:
        from ili.inference.runner_sbi import SBIRunnerSequential as Runner
    runner = Runner.from_config(join(cfgdir, f"inf_{model}.yaml"))
    runner(loader=train_loader)

    # use the trained posterior model to predict on a single example from
    # the test set
    test_loader = StaticNumpyLoader.from_config(join(cfgdir, "data_test.yaml"))
    val_runner = ValidationRunner.from_config(
        join(cfgdir, f"val_{model}.yaml"))
    val_runner(loader=test_loader)
