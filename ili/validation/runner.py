import yaml
import time
import logging
import importlib
import pickle
import torch
from pathlib import Path
from typing import List, Optional
from torch.distributions import Independent
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from ili.dataloaders import BaseLoader
from ili.validation.metrics import BaseMetric
from ili.utils import load_from_config

logging.basicConfig(level=logging.INFO)

default_config = (
    Path(__file__).parent.parent / "examples/configs/sample.yaml"
)


class ValidationRunner:
    def __init__(
        self,
        posterior: NeuralPosterior,
        metrics: List[BaseMetric],
        output_path: Path,
    ):
        """Class to measure validation metrics of posterior inference models

        Args:
            loader (BaseLoader): data loader with stored summary-parameter pairs
            posterior (NeuralPosterior): trained sbi posterior inference engine
            metrics (List[BaseMetric]): list of metric objects to measure on the test set
            output_path (Path): path where to store outputs
        """
        self.posterior = posterior
        self.metrics = metrics
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path) -> "ValidationRunner":
        """Create an validation runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file. Defaults to default_config.
        Returns:
            SBIRunner: the validation runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

        posterior = cls.load_posterior(config["posterior_path"])
        output_path = Path(config["output_path"])

        metrics = {}
        for key, value in config["metrics"].items():
            value["args"]["output_path"] = output_path
            metrics[key] = load_from_config(value)

        return cls(posterior=posterior, metrics=metrics, output_path=output_path)

    @classmethod
    def load_posterior(cls, path):
        """Load a pretrained sbi posterior from file
        Args:
            path (Path): path to stored .pkl of trained sbi posterior
        Returns:
            posterior (NeuralPosterior): the posterior of interest
        """
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def __call__(
            self, 
            loader, 
            prior: Optional[Independent] = None, 
    ):
        """Run your validation metrics and save them to file

        Args:
            loader (BaseLoader): data loader with stored summary-parameter pairs
            or has ability to simulate summary-parameter pairs 
            prior (Independent): prior on the parameters
        """
        t0 = time.time()

        # NOTE: sbi posteriors only accept torch.Tensor inputs
        if hasattr(loader, 'simulate'):
            if prior is None:
                raise Exception('prior is None')
            theta_test, x_test = loader.simulate(prior)
            x_obs = loader.get_obs_data()
            theta_obs = loader.get_obs_parameters()
        else:
            x_test = torch.Tensor(loader.get_all_data())
            theta_test = torch.Tensor(loader.get_all_parameters())
            x_obs = None
            theta_obs = None
        # evaluate metrics
        for metric in self.metrics.values():
            metric(self.posterior, x_test, theta_test, x_obs=x_obs, theta_obs=theta_obs)

        logging.info(f"It took {time.time() - t0} seconds to run all metrics.")
