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
from ili.dataloaders import _BaseLoader
from ili.validation.metrics import _BaseMetric
from ili.utils import load_from_config

try:
    from sbi.inference.posteriors.base_posterior import NeuralPosterior
    ModelClass = NeuralPosterior
    from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
    interface = 'torch'
except ModuleNotFoundError:
    from ili.inference.pydelfi_wrappers import DelfiWrapper
    ModelClass = DelfiWrapper
    interface = 'tensorflow'

logging.basicConfig(level=logging.INFO)


class ValidationRunner:
    """Class to measure validation metrics of posterior inference models

    Args:
        posterior (ModelClass): trained sbi posterior inference engine
        metrics (List[_BaseMetric]): list of metric objects to measure on
            the test set
        out_dir (Path): directory where to load posterior and store outputs
        ensemble_mode (Optional[bool], optional): whether to evaluate metrics
            on each posterior in the ensemble separately or on the ensemble
            posterior. Defaults to True.
        name (Optional[str], optional): name of the posterior. Defaults to "".
        signatures (Optional[List[str]], optional): list of signatures for
            each posterior in the ensemble. Defaults to [].
    """

    def __init__(
        self,
        posterior: ModelClass,  # see imports
        metrics: List[_BaseMetric],
        out_dir: Path,
        ensemble_mode: Optional[bool] = True,
        name: Optional[str] = "",
        signatures: Optional[List[str]] = [],
    ):
        self.posterior = posterior
        self.metrics = metrics
        self.out_dir = out_dir
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ensemble_mode = ensemble_mode
        self.name = name
        self.signatures = signatures

    @classmethod
    def from_config(cls, config_path, **kwargs) -> "ValidationRunner":
        """Create a validation runner from a yaml config file

        Args:
            config_path (Path): path to config file.
            **kwargs: optional keyword arguments to overload config file
        Returns:
            ValidationRunner: the validation runner specified by the config
                file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

        # optionally overload config file with kwargs
        config.update(kwargs)

        out_dir = Path(config["out_dir"])

        global interface
        if interface == 'torch':
            posterior_ensemble = cls.load_posterior_sbi(
                out_dir / config["posterior_file"])
            signatures = posterior_ensemble.signatures
        elif interface == 'tensorflow':
            posterior_ensemble = DelfiWrapper.load_engine(
                out_dir / config["posterior_file"])
            signatures = [""]*posterior_ensemble.num_components  # TODO: fix
        else:
            raise NotImplementedError
        name = posterior_ensemble.name
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
            value["args"]["out_dir"] = out_dir
            value["args"]["labels"] = config["labels"]
            metrics[key] = load_from_config(value)

        return cls(
            posterior=posterior_ensemble,
            metrics=metrics,
            out_dir=out_dir,
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

    def __call__(self, loader: _BaseLoader):
        """Run your validation metrics and save them to file

        Args:
            loader (_BaseLoader): data loader with stored data-parameter
                pairs or has ability to simulate data-parameter pairs
        """
        t0 = time.time()

        # load data
        x_test = loader.get_all_data()
        theta_test = loader.get_all_parameters()

        # load observations for inference (defaults to None)
        x_obs = loader.get_obs_data()
        theta_fid = loader.get_fid_parameters()

        # evaluate metrics on each posterior in the ensemble separately
        global interface
        if ((not self.ensemble_mode) and (interface == 'torch') and
                isinstance(self.posterior, NeuralPosteriorEnsemble)):
            n = 0
            for posterior_model in self.posterior.posteriors:
                signature = self.signatures[n]
                n += 1
                for metric in self.metrics.values():
                    logging.info(
                        f"Running metric {metric.__class__.__name__}.")
                    metric(posterior_model, x_test, theta_test, x_obs=x_obs,
                           theta_fid=theta_fid, signature=signature)
        # evaluate metrics on the ensemble posterior
        else:
            # evaluate metrics
            signature = self.name+"".join(self.signatures)
            for metric in self.metrics.values():
                logging.info(f"Running metric {metric.__class__.__name__}.")
                metric(self.posterior, x_test, theta_test,
                       x_obs=x_obs, theta_fid=theta_fid, signature=signature)

        logging.info(f"It took {time.time() - t0} seconds to run all metrics.")
