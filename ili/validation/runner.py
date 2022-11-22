import yaml
import time
import logging
import importlib
import pickle
import torch
from pathlib import Path

logging.basicConfig(level = logging.INFO)

default_config = Path(__file__).parent.parent / 'examples/configs/sample_inference_config.yaml'


class ValidationRunner:
    def __init__(
        self,
        loader,
        posterior,
        metrics,
        output_path
    ):
        """Class to run training and validation of posterior inference models using the sbi package

        """
        self.loader = loader
        self.posterior = posterior
        self.metrics = metrics
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        config_path
    ):
        """Create an sbi runner from a yaml config file

        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

        loader = cls.load_object(config['loader'])
        posterior = cls.load_posterior(config['posterior_path'])
        output_path = Path(config['output_path'])

        metrics = {}
        for key, value in config['metrics'].items():
            value['args']['output_path'] = output_path
            metrics[key] = cls.load_object(value)

        return cls(
            loader=loader,
            posterior=posterior,
            metrics=metrics,
            output_path=output_path
        )

    @classmethod
    def load_object(cls, config):
        """Load the right object, according to config file
        Args:
            config (Dict): dictionary with the configuration for the object
        Returns:
            object (): the object of choice
        """
        module = importlib.import_module(config["module"])
        return getattr(
            module,
            config["class"],
        )(**config["args"])

    @classmethod
    def load_posterior(cls, path):
        with open(path, "rb") as handle:
            return pickle.load(handle)

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
        x_test, theta_test = x[split_ind:], theta[split_ind:]

        # evaluate metrics
        for metric in self.metrics.values():
            metric(self.posterior, x_test, theta_test)

        logging.info(f'It took {time.time() - t0} seconds to run all metrics.')