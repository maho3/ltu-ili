import importlib
import logging
import pickle
import time
import yaml
from pathlib import Path
from typing import List

from ili.dataloaders import BaseLoader
from ili.utils import load_from_config
from ili.validation.metrics import BaseMetric

try:
    import torch
    from sbi.inference.posteriors.base_posterior import NeuralPosterior
    ModelClass = NeuralPosterior
except ModuleNotFoundError:
    from ili.inference.pydelfi_wrappers import DelfiWrapper
    ModelClass = DelfiWrapper

logging.basicConfig(level=logging.INFO)


class ValidationRunner:
    def __init__(
        self,
        posterior: ModelClass,
        metrics: List[BaseMetric],
        backend: str,
        output_path: Path,
    ):
        """Class to measure validation metrics of posterior inference models

        Args:
            posterior (ModelClass): trained sbi posterior inference engine
            metrics (List[BaseMetric]): list of metric objects to measure on the test set
            backend (str): the backend for the posterior models (either 'sbi' or 'pydelfi')
            output_path (Path): path where to store outputs
        """
        self.posterior = posterior
        self.metrics = metrics
        self.backend = backend
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path) -> "ValidationRunner":
        """Create a validation runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file. Defaults to default_config.
        Returns:
            ValidationRunner: the validation runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

        backend = config['backend']
        if backend == 'sbi':
            posterior = cls.load_posterior_sbi(config["posterior_path"])
        elif backend == 'pydelfi':
            posterior = DelfiWrapper.load_engine(config["meta_path"])
        else:
            raise NotImplementedError
        output_path = Path(config["output_path"])

        metrics = {}
        for key, value in config["metrics"].items():
            value["args"]["backend"] = backend
            value["args"]["output_path"] = output_path
            metrics[key] = load_from_config(value)

        return cls(backend=backend, posterior=posterior, metrics=metrics, output_path=output_path)

    @classmethod
    def load_posterior_sbi(cls, path):
        """Load a pretrained sbi posterior from file
        Args:
            path (Path): path to stored .pkl of trained sbi posterior
        Returns:
            posterior (NeuralPosterior): the posterior of interest
        """
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def __call__(self, loader):
        """Run your validation metrics and save them to file

        Args:
            loader (BaseLoader): data loader with stored summary-parameter pairs
        """
        t0 = time.time()

        x_test = loader.get_all_data()
        theta_test = loader.get_all_parameters()
        # evaluate metrics
        for metric in self.metrics.values():
            metric(self.posterior, x_test, theta_test)

        logging.info(f"It took {time.time() - t0} seconds to run all metrics.")
