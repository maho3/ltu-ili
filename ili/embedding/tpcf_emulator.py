import os
from typing import List, Tuple, Dict

import torch
import pytorch_lightning as pl
import xarray as xr
from fcn import FCN

from ili.inference.loaders import SummarizerDatasetLoader


class QuijoteData(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "../../data/tpcf_ili-summarizer/z_0.50/",
        n_summary: int = 74,
        param_names: List[str] = ["Omega_m", "Omega_b", "h", "sigma_8", "n_s"],
    ):
        """
        Class to load the Quijote redshift-space multipoles of the 2PCF, pre-process it,
        and prepare the train, validation and test datasets for the emulator

            Args:
                data_dir (str): path to the data directory
                n_summary (int): size of the summary data vector
                param_names (List[str]): parameters to fit

            Raises:
                FileNotFoundError: does not initialize properly if the data path is incorrect
        """
        super().__init__()
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")
        self.n_summary = n_summary
        self.param_names = param_names

    def load_separation(self) -> torch.Tensor:
        """
        Loads the 2PCF separation bins from a .nc file of the Quijote data

        Returns:
            separation (torch.Tensor): a Tensor of the separation bins

        Raises:
            FileNotFoundError: raises an exception if the data path is incorrect
        """
        # TODO: get the r array directly from the SummarizerDatasetLoader
        try:
            separation = torch.from_numpy(
                xr.open_dataarray(self.data_dir + "quijote_node0.nc").sel(ells=0).r.to_numpy()
            ).float()
            return separation
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Error reading data file: {err}")

    def load_parameters_and_summaries(self, stage: str, multipole: str = "monopole") -> Tuple[torch.Tensor]:
        """
        Loads the cosmological parameters and either monopole, quadrupole or hexadecapole
        of the 2pt correlation function for a given stage

        Args:
            stage (str): the stage for which to load the parameters and summaries (can be "train", "val" or "test")
            multipole (str): the multipole of the 2PCF to train on (can be "monopole", "quadrupole" or "hexadecapole")

        Returns:
            Tuple:
                parameters (torch.Tensor): the cosmological parameters
                summaries (torch.Tensor): the summaries [s²ξ_i(s)]
        """
        loader = SummarizerDatasetLoader(
            stage=stage,
            data_dir=self.data_dir,
            summary_root_file="quijote",
            param_file="latin_hypercube_params.txt",
            train_test_split_file="quijote_train_test_val.json",
            param_names=self.param_names,
        )
        if multipole == "monopole":
            tpcfs = torch.from_numpy(loader.get_all_data()[:, : self.n_summary]).float()
        elif multipole == "quadrupole":
            tpcfs = torch.from_numpy(loader.get_all_data()[:, self.n_summary : int(2 * self.n_summary)]).float()
        elif multipole == "hexadecapole":
            tpcfs = torch.from_numpy(
                loader.get_all_data()[:, int(2 * self.n_summary) : int(3 * self.n_summary)]
            ).float()
        else:
            raise Exception("Invalid name for the required summary")
        parameters = torch.from_numpy(loader.get_all_parameters()).float()
        # Rescaled multipole s^2*Xi(s)
        separation = self.load_separation()
        summaries = separation * separation * tpcfs
        return parameters, summaries

    def get_mean_and_std(self, tensor: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Calculates the mean and standard deviation of a given tensor

        Args:
            tensor (torch.Tensor): the tensor for which to calculate the mean and standard deviation

        Returns:
            Tuple:
                mean (torch.Tensor): the mean of the tensor
                std (torch.Tensor): the standard deviation of the tensor
        """
        mean = tensor.mean(axis=0)
        std = tensor.std(axis=0)
        return mean, std

    def normalize_parameters_and_summaries(
        self, parameters: torch.Tensor, summaries: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Normalizes the parameters and summaries and returns the associated means and standard deviations
        for both input and output vetors

        Args:
            parameters (torch.Tensor): the parameters to be normalized
            summaries (torch.Tensor): the summaries to be normalized

        Returns:
            Tuple: (normalized_parameters, normalized_summaries)
        """
        parameters_mean, parameters_std = self.get_mean_and_std(parameters)
        summaries_mean, summaries_std = self.get_mean_and_std(summaries)
        normalized_parameters = (parameters - parameters_mean) / parameters_std
        normalized_summaries = (summaries - summaries_mean) / summaries_std
        return normalized_parameters, normalized_summaries

    def prepare_data(self):
        """
        Loads and normalizes the data for train, val, and test datasets.
        """
        # Training set
        train_parameters, train_summaries = self.load_parameters_and_summaries(stage="train")
        normalized_train_parameters, normalized_train_summaries = self.normalize_parameters_and_summaries(
            train_parameters, train_summaries
        )
        self.train_dataset = torch.utils.data.TensorDataset(normalized_train_parameters, normalized_train_summaries)

        # Validation set
        val_parameters, val_summaries = self.load_parameters_and_summaries(stage="val")
        normalized_val_parameters, normalized_val_summaries = self.normalize_parameters_and_summaries(
            val_parameters, val_summaries
        )
        self.val_dataset = torch.utils.data.TensorDataset(normalized_val_parameters, normalized_val_summaries)

        # Test set
        test_parameters, test_summaries = self.load_parameters_and_summaries(stage="test")
        normalized_test_parameters, normalized_test_summaries = self.normalize_parameters_and_summaries(
            test_parameters, test_summaries
        )
        self.test_dataset = torch.utils.data.TensorDataset(normalized_test_parameters, normalized_test_summaries)

    def setup(self, stage: str):
        """
        Sets up the data for the specified stage.

        Args:
            stage (str): The stage for which to prepare the data. One of "train", "val", or "test"
        """
        self.prepare_data()

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Creates and returns a data loader for the training dataset.

        Returns:
            torch.utils.data.DataLoader: A DataLoader instance for the training dataset.

        """
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=250, num_workers=4, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Creates and returns a data loader for the validation dataset.

        Returns:
            torch.utils.data.DataLoader: A DataLoader instance for the validation dataset.

        """
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=50, num_workers=4, shuffle=True)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Creates and returns a data loader for the test dataset.

        Returns:
            torch.utils.data.DataLoader: A DataLoader instance for the test dataset.

        """
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=50, num_workers=4, shuffle=True)


class TpcfEmulator(pl.LightningModule):
    def __init__(
        self,
        n_input: int,
        n_summary: int,
        n_hidden: List[int],
        data_module: pl.LightningDataModule = QuijoteData(),
        act_fn: str = "SiLU",
    ):
        """
        Emulator class to train and perform fast summary statistics generation from a standard
        set of cosmological parameters

        Args:
            n_input (int): number of inputs to the model
            n_summary (int): number of summaries
            n_hidden (List[int]): list of hidden layer sizes
            data_module (pl.LightningDataModule): data module to prepare and setting up the train, test and validation data
            act_fn (str, optional): activation function to use in the model. Defaults to "SiLU"

        """
        super().__init__()
        self.model = FCN(n_summary, n_hidden, act_fn)
        self.data_module = data_module
        self.model.initialize_model(n_input)
        self.n_summary = n_summary

    def forward(self, cosmo_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model

        Args:
            cosmo_params (torch.Tensor): cosmological parameters, input to the model

        Returns:
            torch.Tensor: summary statistics, output of the model
        """
        return self.model.mlp(cosmo_params)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a training step for one batch of data

        Args:
            batch (Tuple): a tuple of (parameters, summaries)
            batch_idx (int): the current batch index

        Returns:
            torch.Tensor: the computed loss for this batch
        """
        parameters, summaries = batch
        summaries_predictions = self(parameters)
        train_loss = torch.nn.functional.mse_loss(summaries_predictions, summaries)
        self.log("train_loss", train_loss, on_epoch=True)
        return train_loss

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a validation step for one batch of data

        Args:
            batch (Tuple): a tuple of (parameters, summaries)
            batch_idx (int): the current batch index

        Returns:
            torch.Tensor: the computed loss for this batch
        """
        parameters, summaries = batch
        summaries_predictions = self(parameters)
        val_loss = torch.nn.functional.mse_loss(summaries_predictions, summaries)
        self.log("val_loss", val_loss, on_epoch=True)
        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer used for training

        Returns:
            torch.optim.Optimizer: the configured optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.2, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_avg_loss"}

    def training_epoch_end(self, outputs: List[Dict]):
        """
        Perform any necessary processing at the end of a training epoch

        Args:
            outputs (List[Dict]): a list of dictionaries containing the outputs from each training step in the epoch
        """
        train_losses = [output["loss"] for output in outputs]
        avg_loss = torch.stack(train_losses).mean()
        self.log("train_avg_loss", avg_loss, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs):
        """
        Perform any necessary processing at the end of a validation epoch

        Args:
            outputs (List[Dict]): a list of dictionaries containing the outputs from each validation step in the epoch
        """
        print("Outputs:", outputs)  # added line to print outputs
        val_losses = [output for output in outputs]
        avg_val_loss = torch.stack(val_losses).mean()
        self.log("val_avg_loss", avg_val_loss, on_epoch=True, prog_bar=True)


if __name__ == "__main__":

    import argparse

    # Argument parsing
    parser = argparse.ArgumentParser(
        prog="QuickSummary",
        description="Training script for the 2PCF monopole emulator based on the standard cosmology Quijote dataset.",
    )
    parser.add_argument("--n_epochs", type=int, required=True, help="Number of epochs required for the training (int)")
    parser.add_argument("--path_logs", default="logs", type=str, required=False, help="Path for the output logs (str)")
    parser.add_argument("--with-gpu", action="store_true", required=False, help="Accelerate the training with a GPU")
    parser.add_argument(
        "--cosmo_params",
        nargs="+",
        default=["Omega_m", "Omega_b", "h", "sigma_8", "n_s"],
        help="List of names for the required input cosmological parameters (List[str]). Available parameters are Omega_m, Omega_b, h, sigma_8 and n_s",
    )
    parser.add_argument(
        "--n_bins",
        default=74,
        type=int,
        required=False,
        help="Binning for the summary statistic: size of the training vector (int)",
    )
    parser.add_argument(
        "--number_hidden",
        type=int,
        default=4,
        required=False,
        help="Number of hidden layers for the MLP",
    )
    parser.add_argument(
        "--size_hidden",
        type=int,
        default=256,
        required=False,
        help="Lengths of the hidden layers of the MLP (constant number of nodes across hidden layers)",
    )
    args = parser.parse_args()

    # Tensorboard logger
    logger = pl.loggers.TensorBoardLogger(args.path_logs)

    # Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_avg_loss", patience=15)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="best_model_epoch_{epoch:02d}-val_loss_{val_loss:.2f}.ckpt",
        save_top_k=1,
        monitor="val_avg_loss",
        mode="min",
    )

    # Data module
    data = QuijoteData("../../data/tpcf_ili-summarizer/z_0.50/", n_summary=args.n_bins, param_names=args.cosmo_params)

    # 2PCF emulator
    architecture = [args.size_hidden for i in range(args.number_hidden)]
    emulator = TpcfEmulator(n_input=len(args.cosmo_params), n_summary=args.n_bins, n_hidden=architecture)

    # Training class instance
    if args.with_gpu and torch.cuda.is_available():
        trainer = pl.Trainer(
            max_epochs=args.n_epochs,
            default_root_dir=args.path_logs,
            logger=logger,
            accelerator="gpu",
            devices=torch.cuda.device_count(),
        )
    elif args.with_gpu and not torch.cuda.is_available():
        raise Exception("No GPU available for training")
    else:
        trainer = pl.Trainer(
            max_epochs=args.n_epochs, default_root_dir=args.path_logs, logger=logger, accelerator="cpu"
        )

    # Training of the emulator
    trainer.fit(emulator, data)
