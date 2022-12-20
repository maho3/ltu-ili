import argparse

from ili.inference.runner_sbi import SBIRunner
from ili.validation.runner import ValidationRunner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run SBI inference for quijote test data.")
    parser.add_argument("--cfgtrain", type=str,
                        default="configs/sample_quijoteTPCF_config.yaml",
                        help="Configuration file to use for inference training")
    parser.add_argument("--cfgval", type=str,
                        default="configs/sample_quijotevalidation_config.yaml",
                        help="Configuration file to use for inference validation")

    args = parser.parse_args()

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    runner = SBIRunner.from_config(args.cfgtrain)
    runner()

    # use the trained posterior model to predict on a single example from the test set
    valrunner = ValidationRunner.from_config(args.cfgval)
    valrunner()
