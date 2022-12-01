import yaml
import time
import logging
import importlib
import pickle
import torch
import sbi
from pathlib import Path
from ili.inference.loaders import BaseLoader
from torch.distributions import Independent
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from typing import Dict, Any

logging.basicConfig(level = logging.INFO)

default_config = Path(__file__).parent.parent / 'examples/configs/sample_inference_config.yaml'

class SBIRunner:
    def __init__(
        self,
        loader: BaseLoader,
        prior: Independent,
        model: PosteriorEstimator,
        train_args: Dict,
        output_path: Path,
    ):
        """Class to train posterior inference models using the sbi package

        Args:
            loader (BaseLoader): data loader with stored summary-parameter pairs
            prior (Independent): prior on the parameters
            model (PosteriorEstimator): sbi posterior estimator for doing parameter inference
            train_args (Dict): dictionary of hyperparameters for training
            output_path (Path): path where to store outputs
        """
        self.loader = loader
        self.prior = prior
        self.model = model
        self.train_args = train_args
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        config_path
    )->"SBIRunner":
        """Create an sbi runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file. Defaults to default_config.
        Returns:
            SBIRunner: the sbi runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

        loader = cls.load_object(config['loader'])
        prior = cls.load_object(config['prior'])

        # prior object and device needed for model instantiation
        config['model']['args']['prior'] = prior
        config['model']['args']['device'] = config['device']
        model = cls.load_object(config['model'])

        train_args = config['train_args']
        output_path = Path(config['output_path'])

        return cls(
            loader=loader,
            prior=prior,
            model=model,
            train_args = train_args,
            output_path=output_path
        )

    @classmethod
    def load_object(cls, config) -> Any:
        """Load the right object, according to config file
        Args:
            config (Dict): dictionary with the configuration for the object
        Returns:
            object (Any): the object of choice
        """
        module = importlib.import_module(config["module"])
        return getattr(
            module,
            config["class"],
        )(**config["args"])

    def __call__(
        self
    ):
        """Train your posterior and save it to file
        """
        t0 = time.time()

        x = torch.Tensor(self.loader.get_all_data())
        theta = torch.Tensor(self.loader.get_all_parameters())

        # TODO: this is maybe not the best place to train-test split
        split_ind = int(0.9*len(self.loader))
        x_train, theta_train = x[:split_ind], theta[:split_ind]

        # train
        _ = self.model.append_simulations(theta_train, x_train).train(**self.train_args)
        posterior = self.model.build_posterior()

        # save posterior
        with open(self.output_path / "posterior.pkl", "wb") as handle:
            pickle.dump(posterior, handle)

        logging.info(f'It took {time.time() - t0} seconds to train all models.')

# Todo: Add cross-validation