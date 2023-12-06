"""
Module to run validation metrics on posterior inference models
"""

import logging
import pickle
import time
import yaml
import matplotlib as mpl
from pathlib import Path
from typing import List, Optional
from ili.validation.metrics import _BaseMetric
from ili.utils import load_from_config

try:
    from sbi.inference.posteriors.base_posterior import NeuralPosterior
    ModelClass = NeuralPosterior
    from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
except ModuleNotFoundError:
    from ili.inference.pydelfi_wrappers import DelfiWrapper
    ModelClass = DelfiWrapper

logging.basicConfig(level=logging.INFO)


class ValidationRunner:
    """Class to measure validation metrics of posterior inference models

    Args:
        posterior (ModelClass): trained sbi posterior inference engine
        metrics (List[_BaseMetric]): list of metric objects to measure on
            the test set
        backend (str): the backend for the posterior models
            ('sbi' or 'pydelfi')
        output_path (Path): path where to store outputs
    """

    def __init__(
        self,
        posterior: ModelClass,  # see imports
        metrics: List[_BaseMetric],
        backend: str,
        output_path: Path,
        ensemble_mode: Optional[bool] = True,
        name: Optional[str] = "",
        signatures: Optional[List[str]] = [],
    ):
        self.posterior = posterior
        self.metrics = metrics
        self.backend = backend
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)
        self.ensemble_mode = ensemble_mode
        self.name = name
        self.signatures = signatures

    @classmethod
    def from_config(cls, config_path) -> "ValidationRunner":
        """Create a validation runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file.
        Returns:
            ValidationRunner: the validation runner specified by the config
                file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

        backend = config['backend']
        if backend == 'sbi':
            posterior_ensemble = cls.load_posterior_sbi(
                config["posterior_path"])
            signatures = posterior_ensemble.signatures
        elif backend == 'pydelfi':
            posterior_ensemble = DelfiWrapper.load_engine(config["meta_path"])
            signatures = [""]*posterior_ensemble.num_components
        else:
            raise NotImplementedError
        name = posterior_ensemble.name
        output_path = Path(config["output_path"])
        if "style_path" in config:
            mpl.style.use(config["style_path"])
        if "ensemble_mode" in config:
            ensemble_mode = config["ensemble_mode"]
        else:
            ensemble_mode = True

        logging.info("Number of posteriors in the ensemble is "
                     f"{posterior_ensemble.num_components}")
        if ensemble_mode:
            logging.info(
                "Metrics are computed for the ensemble posterior estimate.")
        else:
            logging.info(
                "Metrics are computed for each posterior in the ensemble.")

        metrics = {}
        for key, value in config["metrics"].items():
            value["args"]["backend"] = backend
            value["args"]["output_path"] = output_path
            value["args"]["labels"] = config["labels"]
            metrics[key] = load_from_config(value)

        return cls(
            backend=backend,
            posterior=posterior_ensemble,
            metrics=metrics,
            output_path=output_path,
            ensemble_mode=ensemble_mode,
            name=name,
            signatures=signatures,
        )

    @classmethod
    def load_posterior_sbi(cls, path):
        """Load a pretrained sbi posterior from file

        Args:
            path (Path): path to stored .pkl of trained sbi posterior
        Returns:
            posterior (ModelClass): the posterior of interest
        """
        with open(path, "rb") as handle:
            posterior = pickle.load(handle)
        if not hasattr(posterior, 'name'):
            posterior.name = ''
        return posterior

    def __call__(
            self,
            loader,
            x_obs=None,
            theta_obs=None
    ):
        """Run your validation metrics and save them to file

        Args:
            loader (BaseLoader): data loader with stored data-parameter
                pairs or has ability to simulate data-parameter pairs
        """
        t0 = time.time()

        # load data
        x_test = loader.get_all_data()
        theta_test = loader.get_all_parameters()
        if hasattr(loader, 'simulate'):
            x_obs = loader.get_obs_data()
            theta_obs = loader.get_obs_parameters()

        if ((not self.ensemble_mode) and (self.backend == 'sbi') and
                isinstance(self.posterior, NeuralPosteriorEnsemble)):
            n = 0
            for posterior_model in self.posterior.posteriors:
                signature = self.signatures[n]
                n += 1
                for metric in self.metrics.values():
                    logging.info(
                        f"Running metric {metric.__class__.__name__}.")
                    metric(posterior_model, x_test, theta_test, x_obs=x_obs,
                           theta_obs=theta_obs, signature=signature)
        else:
            # evaluate metrics
            signature = self.name+"".join(self.signatures)
            for metric in self.metrics.values():
                logging.info(f"Running metric {metric.__class__.__name__}.")
                metric(self.posterior, x_test, theta_test,
                       x_obs=x_obs, theta_obs=theta_obs, signature=signature)

        logging.info(f"It took {time.time() - t0} seconds to run all metrics.")
