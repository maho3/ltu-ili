import yaml
import time
import logging
import importlib
import pickle
import torch
import torch.nn as nn
import sbi
from pathlib import Path
from ili.inference.loaders import BaseLoader
from torch.distributions import Independent
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)

default_config = (
    Path(__file__).parent.parent / "examples/configs/sample_inference_config.yaml"
)


class SBIRunner:
    def __init__(
        self,
        loader: BaseLoader,
        prior: Independent,
        model: PosteriorEstimator,
        embedding_net: nn.Module,
        train_args: Dict,
        output_path: Path,
    ):
        """Class to train posterior inference models using the sbi package

        Args:
            loader (BaseLoader): data loader with stored summary-parameter pairs
            prior (Independent): prior on the parameters
            model (PosteriorEstimator): sbi posterior estimator for doing parameter inference
            embedding_net (nn.MOdule): neural network to compress high dimensional data into lower dimensionality
            train_args (Dict): dictionary of hyperparameters for training
            output_path (Path): path where to store outputs
        """
        self.loader = loader
        self.prior = prior
        self.model = model
        self.embedding_net = embedding_net
        self.train_args = train_args
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path) -> "SBIRunner":
        """Create an sbi runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file. Defaults to default_config.
        Returns:
            SBIRunner: the sbi runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)
        loader = cls.load_object(config["loader"])
        prior = cls.load_object(config["prior"])
        if "embedding_net" in config:
            embedding_net = cls.load_object(
                config=config["embedding_net"],
            )
        else:
            embedding_net = nn.Identity() 
        model = cls.load_inference_model(
            prior=prior,
            device=config["device"],
            embedding_net=embedding_net,
            inference_config=config["model"],
        )
        train_args = config["train_args"]
        output_path = Path(config["output_path"])
        return cls(
            loader=loader,
            prior=prior,
            model=model,
            embedding_net=embedding_net,
            train_args=train_args,
            output_path=output_path,
        )

    @classmethod
    def load_object(cls, config: Dict) -> Any:
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

    @classmethod
    def load_inference_model(
        cls,
        prior: Independent,
        device: str,
        embedding_net: nn.Module,
        inference_config: Dict,
    ) -> "InferenceModel":
        """Load the inference model

        Args:
            prior (Independent): prior for parameters to infer
            device (str): cpu or gpu
            embedding_net (nn.Module): neural network to compress high dimensional data
            inference_config (Dict): configuration for the inference module

        Returns:
            InferenceModel: model to fit posterior
        """
        posterior_config = inference_config["posterior_nn"]
        neural_posterior = sbi.utils.posterior_nn(
            embedding_net=embedding_net,
            **posterior_config,
        )
        module = importlib.import_module(inference_config["module"])
        inference_class = getattr(module, inference_config["class"])
        return inference_class(
            prior=prior, density_estimator=neural_posterior, device=device
        )

    def __call__(self):
        """Train your posterior and save it to file"""
        t0 = time.time()

        x = torch.Tensor(self.loader.get_all_data())
        if not isinstance(self.embedding_net, nn.Identity()):
            self.embedding_net.initalize_model(n_input=x.shape[-1])
        theta = torch.Tensor(self.loader.get_all_parameters())
        # train
        _ = self.model.append_simulations(theta, x).train(**self.train_args)
        posterior = self.model.build_posterior()
        # save posterior
        with open(self.output_path / "posterior.pkl", "wb") as handle:
            pickle.dump(posterior, handle)

        logging.info(f"It took {time.time() - t0} seconds to train all models.")
