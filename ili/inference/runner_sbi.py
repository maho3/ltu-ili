import yaml
import time
import logging
import importlib
import pickle
import torch
import torch.nn as nn
import sbi
from pathlib import Path
from typing import Dict, Any, List, Callable
from torch.distributions import Independent
from sbi.inference import NeuralInference
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from ili.inference.loaders import BaseLoader

logging.basicConfig(level=logging.INFO)

default_config = (
    Path(__file__).parent.parent / "examples/configs/sample_inference_config.yaml"
)


class SBIRunner:
    def __init__(
        self,
        loader: BaseLoader,
        prior: Independent,
        inference_class: NeuralInference,
        neural_posteriors: List[Callable],
        device: str,
        embedding_net: nn.Module,
        train_args: Dict,
        output_path: Path,
    ):
        """Class to train posterior inference models using the sbi package

        Args:
            loader (BaseLoader): data loader with stored summary-parameter pairs
            prior (Independent): prior on the parameters
            inference_class (NeuralInference): sbi inference class used to that train neural posteriors
            neural_posteriors (List[Callable]): list of neural posteriors to train
            embedding_net (nn.Module): neural network to compress high dimensional data into lower dimensionality
            train_args (Dict): dictionary of hyperparameters for training
            output_path (Path): path where to store outputs
        """
        self.loader = loader
        self.prior = prior
        self.inference_class = inference_class
        self.neural_posteriors = neural_posteriors
        self.device = device
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
        inference_class = cls.load_inference_class(
            inference_module=config["model"]["module"],
            inference_class=config["model"]["class"],
        )
        neural_posteriors = cls.load_neural_posteriors(
            embedding_net=embedding_net,
            posteriors_config=config["model"]["neural_posteriors"],
        )
        train_args = config["train_args"]
        output_path = Path(config["output_path"])
        return cls(
            loader=loader,
            prior=prior,
            inference_class=inference_class,
            neural_posteriors=neural_posteriors,
            device=config["device"],
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
    def load_inference_class(
        cls,
        inference_module: str,
        inference_class: str,
    ) -> NeuralInference:
        """Neural inference class used to train posterior

        Args:
            inference_module (str): module from which to import class
            inference_class (str): class name

        Returns:
            NeuralInference: neural inference class
        """
        module = importlib.import_module(inference_module)
        return getattr(module, inference_class)

    @classmethod
    def load_neural_posteriors(
        cls,
        embedding_net: nn.Module,
        posteriors_config: List[Dict],
    ) -> List[Callable]:
        """Load the inference model

        Args:
            embedding_net (nn.Module): neural network to compress high dimensional data
            posterior_config(List[Dict]): list with configurations for each neural posterior
            model in the ensemble

        Returns:
            List[Callable]: list of neural posterior models with forward methods
        """
        neural_posteriors = []
        for model_args in posteriors_config:
            neural_posteriors.append(
                sbi.utils.posterior_nn(
                    embedding_net=embedding_net,
                    **model_args,
                )
            )
        return neural_posteriors

    def __call__(self):
        """Train your posterior and save it to file"""
        t0 = time.time()
        x = torch.Tensor(self.loader.get_all_data())
        theta = torch.Tensor(self.loader.get_all_parameters())
        posteriors, val_loss = [], []
        for n, posterior in enumerate(self.neural_posteriors):
            logging.info(
                f"Training model {n+1} out of {len(self.neural_posteriors)} ensemble models"
            )
            model = self.inference_class(
                prior=self.prior,
                density_estimator=posterior,
                device=self.device,
            )
            model = model.append_simulations(theta, x)
            if not isinstance(self.embedding_net, nn.Identity):
                self.embedding_net.initalize_model(n_input=x.shape[-1])
            density_estimator = model.train(
                **self.train_args,
            )
            posteriors.append(model.build_posterior(density_estimator))
            val_loss += model.summary["best_validation_log_prob"]
        posterior = NeuralPosteriorEnsemble(
            posteriors=posteriors,
            weights=torch.tensor([float(vl) for vl in val_loss]),
        )
        with open(self.output_path / "posterior.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
        logging.info(f"It took {time.time() - t0} seconds to train all models.")
