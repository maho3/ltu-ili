from os.path import join
from ili.dataloaders import StaticNumpyLoader
from ili.inference.runner_sbi import SBIRunner as Runner
from ili.validation.runner import ValidationRunner


if __name__ == '__main__':
    model = 'npe'
    cfgdir = '.'

    # reload all simulator examples as a dataloader
    all_loader = StaticNumpyLoader.from_config(join(cfgdir, "data.yaml"))

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    runner = Runner.from_config(join(cfgdir, f"inf_{model}.yaml"))
    runner(loader=all_loader)

    # use the trained posterior model to predict on a single example from
    # the test set
    val_runner = ValidationRunner.from_config(
        join(cfgdir, f"val_{model}.yaml"))
    val_runner(loader=all_loader)
