import argparse
from ili.dataloaders import SummarizerDatasetLoader
from ili.inference.runner_pydelfi import DelfiRunner
from ili.validation.runner import ValidationRunner

# pydelfi produces a lot of DivideByZero errors on TPCF data, but still works
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run pyDELFI inference for quijote test data.")
    parser.add_argument("--cfgdata", type=str,
                        default="configs/data/quijote_TPCF.yaml",
                        help="Configuration file to use for dataloaders")
    parser.add_argument("--cfginfer", type=str,
                        default="configs/infer/quijote_pydelfi_CMAF.yaml",
                        help="Configuration file to use for inference training")
    parser.add_argument("--cfgval", type=str,
                        default="configs/val/quijote_pydelfi.yaml",
                        help="Configuration file to use for inference validation")

    args = parser.parse_args()

    train_loader = SummarizerDatasetLoader.from_config(args.cfgdata, stage='train')
    val_loader = SummarizerDatasetLoader.from_config(args.cfgdata, stage='val')
    print(train_loader.get_all_data()[0].shape)

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    runner = DelfiRunner.from_config(args.cfginfer)
    runner(loader=train_loader)

    # use the trained posterior model to predict on a single example from the test set
    val_runner = ValidationRunner.from_config(args.cfgval)
    val_runner(loader=val_loader)
