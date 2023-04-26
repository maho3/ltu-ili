import argparse
from ili.dataloaders import SummarizerDatasetLoader
from ili.inference.runner_sbi import SBIRunner
from ili.validation.runner import ValidationRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run SBI inference for quijote test data.")
    parser.add_argument("--cfgdata", type=str,
                        default="configs/data/quijote_TPCF.yaml",
                        help="Configuration file to use for dataloaders")
    parser.add_argument("--cfginfer", type=str,
                        default="configs/infer/quijote_sbi_MAF.yaml",
                        help="Configuration file to use for inference training")
    parser.add_argument("--cfgval", type=str,
                        default="configs/val/quijote_sbi.yaml",
                        help="Configuration file to use for inference validation")

    args = parser.parse_args()

    train_loader = SummarizerDatasetLoader.from_config(args.cfgdata, stage='train')
    val_loader = SummarizerDatasetLoader.from_config(args.cfgdata, stage='val')

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    runner = SBIRunner.from_config(args.cfginfer)
    runner(loader=train_loader)

    # use the trained posterior model to predict on a single example from the test set
    val_runner = ValidationRunner.from_config(args.cfgval)
    val_runner(loader=val_loader)
