from os.path import join
import argparse
from sbibm.metrics.c2st import c2st


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run SBI inference for toy data.")
    parser.add_argument(
        '--data', type=str)
    parser.add_argument(
        "--inf", type=str)
    parser.add_argument(
        "--val", type=str)
    args = parser.parse_args()
    data = args.data
    model = args.inf
    val = args.val

    print(f"Configuration:\n\t{data}\n\t{model}\n\t{val}")

    # reload all simulator examples as a dataloader
    if 'seq' in data:
        loader = SBISimulator.from_config(data)
        loader.set_simulator(simulator)
    else:
        loader = StaticNumpyLoader.from_config(data)

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    if model == 'pydelfi':
        from ili.inference.runner_pydelfi import DelfiRunner as Runner
    if 'seq' in data:
        from ili.inference.runner_sbi import SBIRunnerSequential as Runner
    else:
        from ili.inference.runner_sbi import SBIRunner as Runner
    runner = Runner.from_config(model)
    runner(loader=loader)

    # use the trained posterior model to predict on a single example from
    # the test set
    val_runner = ValidationRunner.from_config(val)
    val_runner(loader=loader)
