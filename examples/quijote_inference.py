import argparse

from ili.inference.runner_sbi import SBIRunner
from ili.validation.runner import ValidationRunner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run SBI inference for quijote test data.')
    parser.add_argument('--use_mdn', action='store_true',
                        help='Use mixture density model as density estimator instead of normalizing flow')

    script_args = parser.parse_args()

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    if script_args.use_mdn:
        runner = SBIRunner.from_config("configs/sample_quijoteTPCF_mdn_config.yaml")
        runner()
    else:
        runner = SBIRunner.from_config("configs/sample_quijoteTPCF_config.yaml")
        runner()

    # use the trained posterior model to predict on a single example from the test set
    valrunner = ValidationRunner.from_config("configs/sample_quijotevalidation_config.yaml")
    valrunner()