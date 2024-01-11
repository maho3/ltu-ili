"""
Module to train posterior inference models using the sbi package
"""

import json
import yaml
import time
import logging
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Callable, Optional
from torch.distributions import Independent
from sbi.inference import NeuralInference
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from ili.dataloaders import _BaseLoader
from ili.utils import load_class, load_from_config, load_nde_sbi

logging.basicConfig(level=logging.INFO)

default_config = (
    Path(__file__).parent.parent / "examples/configs/sample_sbi.yaml"
)


class _BaseRunner():
    def __init__(
        self,
        prior: Independent,
        inference_class: NeuralInference,
        train_args: Dict = {},
        out_dir: Path = None,
        device: str = 'cpu',
        name: Optional[str] = "",
    ):
        self.prior = prior
        self.inference_class = inference_class
        self.class_name = inference_class.__name__
        self.train_args = train_args
        self.device = device
        self.name = name
        self.out_dir = out_dir
        if self.out_dir is not None:
            self.out_dir = Path(self.out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)


class SBIRunner(_BaseRunner):
    """Class to train posterior inference models using the sbi package

    Args:
        prior (Independent): prior on the parameters
        inference_class (NeuralInference): sbi inference class used to
            train neural posteriors
        nets (List[Callable]): list of neural nets for amortized posteriors,
            likelihood models, or ratio classifiers
        embedding_net (nn.Module): neural network to compress high
            dimensional data into lower dimensionality
        train_args (Dict): dictionary of hyperparameters for training
        out_dir (Path): directory where to store outputs
        proposal (Independent): proposal distribution from which existing
            simulations were run, for single round inference only. By default,
            sbi will set proposal = prior unless a proposal is specified.
            While it is possible to choose a prior on parameters different
            than the proposal for SNPE, we advise to leave proposal to None
            unless for test purposes.
        name (str): name of the model (for saving purposes)
        signatures (List[str]): list of signatures for each neural net
    """

    def __init__(
        self,
        prior: Independent,
        inference_class: NeuralInference,
        nets: List[Callable],
        train_args: Dict = {},
        out_dir: Path = None,
        device: str = 'cpu',
        embedding_net: nn.Module = None,
        proposal: Independent = None,
        name: Optional[str] = "",
        signatures: Optional[List[str]] = None,
    ):
        super().__init__(
            prior=prior,
            inference_class=inference_class,
            train_args=train_args,
            out_dir=out_dir,
            device=device,
            name=name,
        )
        if proposal is None:
            self.proposal = prior
        else:
            self.proposal = proposal
        self.nets = nets
        self.embedding_net = embedding_net
        self.train_args = train_args
        if "num_round" in train_args:
            self.num_rounds = train_args["num_round"]
            self.train_args.pop("num_round")
        else:
            self.num_rounds = 1
        self.signatures = signatures
        if self.signatures is None:
            self.signatures = [""]*len(self.nets)

    @classmethod
    def from_config(cls, config_path: Path, **kwargs) -> "SBIRunner":
        """Create an sbi runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file
            **kwargs: optional keyword arguments to overload config file
        Returns:
            SBIRunner: the sbi runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

        # optionally overload config with kwargs
        config.update(kwargs)

        # load prior distribution
        config['prior']['args']['device'] = config['device']
        prior = load_from_config(config["prior"])

        # load proposal distributions
        proposal = None
        if "proposal" in config:
            config['proposal']['args']['device'] = config['device']
            proposal = load_from_config(config["proposal"])

        # load embedding net
        if "embedding_net" in config:
            embedding_net = load_from_config(
                config=config["embedding_net"],
            )
        else:
            embedding_net = nn.Identity()

        # load logistics
        train_args = config["train_args"]
        out_dir = Path(config["out_dir"])
        if "name" in config["model"]:
            name = config["model"]["name"]+"_"
        else:
            name = ""
        signatures = []
        for type_nn in config["model"]["nets"]:
            signatures.append(type_nn.pop("signature", ""))

        # load inference class and neural nets
        inference_class = load_class(
            module_name=config["model"]["module"],
            class_name=config["model"]["class"],
        )
        nets = [load_nde_sbi(config['model']['class'],
                             embedding_net=embedding_net,
                             **model_args)
                for model_args in config['model']['nets']]

        # initialize
        return cls(
            prior=prior,
            proposal=proposal,
            inference_class=inference_class,
            nets=nets,
            device=config["device"],
            embedding_net=embedding_net,
            train_args=train_args,
            out_dir=out_dir,
            signatures=signatures,
            name=name,
        )

    def _setup_engine(self, net: nn.Module):
        """Instantiate an sbi inference engine (SNPE/SNLE/SNRE)."""
        if ("NPE" in self.class_name) or ("NLE" in self.class_name):
            return self.inference_class(
                prior=self.prior,
                density_estimator=net,
                device=self.device,
            )
        elif ("NRE" in self.class_name):
            return self.inference_class(
                prior=self.prior,
                classifier=net,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Model class {self.class_name} not supported. "
                "Please choose one of SNPE, SNLE, or SNRE."
            )

    def _train_round(self, models: List[NeuralInference],
                     x: torch.Tensor, theta: torch.Tensor,
                     proposal: Optional[Independent]):
        """Train a single round of inference for an ensemble of models."""
        posteriors, summaries = [], []
        for i, model in enumerate(models):
            logging.info(f"Training model {i+1} / {len(models)}.")

            # append simulations
            if ("NPE" in self.class_name):
                model = model.append_simulations(theta, x, proposal=proposal)
            else:
                model = model.append_simulations(theta, x)

            # train
            _ = model.train(**self.train_args)

            # save model
            posteriors.append(model.build_posterior())
            summaries.append(model.summary)

        # ensemble all trained models, weighted by validation loss
        val_logprob = torch.tensor(
            [float(x["best_validation_log_prob"][0]) for x in summaries]
        ).to(self.device)
        # Exponentiate with numerical stability
        weights = torch.exp(val_logprob - val_logprob.max())
        weights /= weights.sum()

        posterior_ensemble = NeuralPosteriorEnsemble(
            posteriors=posteriors,
            weights=weights,
            theta_transform=posteriors[0].theta_transform
        )  # raises warning due to bug in sbi

        # record the name of the ensemble
        posterior_ensemble.name = self.name
        posterior_ensemble.signatures = self.signatures

        return posterior_ensemble, summaries

    def _save_models(self, posterior_ensemble: NeuralPosteriorEnsemble,
                     summaries: List[Dict]):
        """Save models to file."""

        logging.info(f"Saving model to {self.out_dir}")
        str_p = self.name + "posterior.pkl"
        str_s = self.name + "summary.json"
        with open(self.out_dir / str_p, "wb") as handle:
            pickle.dump(posterior_ensemble, handle)
        with open(self.out_dir / str_s, "w") as handle:
            json.dump(summaries, handle)

    def __call__(self, loader: _BaseLoader, seed: int = None):
        """Train your posterior and save it to file

        Args:
            loader (_BaseLoader): dataloader with stored data-parameter pairs
            seed (int): torch seed for reproducibility
        """

        # set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # setup training engines for each model in the ensemble
        logging.info(f"MODEL INFERENCE CLASS: {self.class_name}")
        models = [self._setup_engine(net) for net in self.nets]

        # load single-round data
        x = torch.Tensor(loader.get_all_data()).to(self.device)
        theta = torch.Tensor(loader.get_all_parameters()).to(self.device)

        # instantiate embedding_net architecture, if necessary
        if self.embedding_net and hasattr(self.embedding_net, 'initalize_model'):
            self.embedding_net.initalize_model(n_input=x.shape[-1])

        # train a single round of inference
        t0 = time.time()
        posterior_ensemble, summaries = self._train_round(
            models=models,
            x=x,
            theta=theta,
            proposal=self.proposal,
        )
        logging.info(f"It took {time.time() - t0} seconds to train models.")

        # save if output path is specified
        if self.out_dir is not None:
            self._save_models(posterior_ensemble, summaries)

        return posterior_ensemble, summaries


class SBIRunnerSequential(SBIRunner):
    """
    Class to train posterior inference models using the sbi package with
    multiple rounds
    """

    def __call__(self, loader: _BaseLoader, seed: int = None):
        """Train your posterior and save it to file

        Args:
            loader (_BaseLoader): data loader with ability to simulate
                data-parameter pairs
        """
        # Check arguments
        if not hasattr(loader, "get_obs_data"):
            raise ValueError(
                "For sequential inference, the loader must have a method "
                "get_obs_data() that returns the observed data."
            )
        if not hasattr(loader, "simulate"):
            raise ValueError(
                "For sequential inference, the loader must have a method "
                "simulate() that returns simulated data-parameter pairs."
            )

        # set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # setup training engines for each model in the ensemble
        logging.info(f"MODEL INFERENCE CLASS: {self.class_name}")
        models = [self._setup_engine(net) for net in self.nets]

        # load observed and pre-run data
        x_obs = loader.get_obs_data()

        # pre-run data
        if len(loader) > 0:
            logging.info(
                "The first round of inference will use existing sims from the "
                "loader. Make sure that the simulations were run from the "
                "given proposal distribution for consistency.")
            x = torch.Tensor(loader.get_all_data()).to(self.device)
            theta = torch.Tensor(loader.get_all_parameters()).to(self.device)
        # no pre-run data
        else:
            logging.info(
                "The first round of inference will simulate from the given "
                "proposal or prior.")
            theta, x = loader.simulate(self.proposal)
            x = torch.Tensor(x).to(self.device)
            theta = torch.Tensor(theta).to(self.device)

        # instantiate embedding_net architecture, if necessary
        if self.embedding_net and hasattr(self.embedding_net, 'initalize_model'):
            self.embedding_net.initalize_model(n_input=x.shape[-1])

        # train multiple rounds of inference
        t0 = time.time()
        for rnd in range(self.num_rounds):
            logging.info(f"Running round {rnd+1} / {self.num_rounds}")

            # train a round of inference
            posterior_ensemble, summaries = self._train_round(
                models=models,
                x=x,
                theta=theta,
                proposal=self.proposal,
            )

            # update proposal for next round
            self.proposal = posterior_ensemble.set_default_x(x_obs)
        logging.info(f"It took {time.time() - t0} seconds to train models.")

        if self.out_dir is not None:
            self._save_models(posterior_ensemble, summaries)

        return posterior_ensemble, summaries


class ABCRunner(_BaseRunner):
    """Class to run ABC inference models using the sbi package"""

    @classmethod
    def from_config(cls, config_path: Path, **kwargs) -> "ABCRunner":
        """Create an sbi runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file
            **kwargs: optional keyword arguments to overload config file

        Returns:
            SBIRunner: the sbi runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

        # optionally overload config with kwargs
        config.update(kwargs)

        # load prior distribution
        prior = load_from_config(config["prior"])

        # load inference class
        inference_class = load_class(
            module_name=config["model"]["module"],
            class_name=config["model"]["class"],
        )

        # load logistics
        train_args = config["train_args"]
        out_dir = Path(config["out_dir"])
        name = ""
        if "name" in config["model"]:
            name = config["model"]["name"]+"_"

        return cls(
            prior=prior,
            inference_class=inference_class,
            device=config["device"],
            train_args=train_args,
            out_dir=out_dir,
            name=name,
        )

    def __call__(self, loader: _BaseLoader, seed: int = None):
        """Train your posterior and save it to file

        Args:
            loader (_BaseLoader): dataloader with stored data-parameter pairs
            seed (int): torch seed for reproducibility
        """
        t0 = time.time()

        logging.info(f"MODEL INFERENCE CLASS: {self.inference_class.__name__}")

        x_obs = loader.get_obs_data()

        # setup and train each architecture
        model = self.inference_class(
            prior=self.prior,
            simulator=loader.simulator
        )
        samples = model(x_obs, return_summary=False, **self.train_args)

        # save if output path is specified
        if self.out_dir is not None:
            str_p = self.name + "samples.pkl"
            with open(self.out_dir / str_p, "wb") as handle:
                pickle.dump(samples, handle)

        logging.info(
            f"It took {time.time() - t0} seconds to run the model.")
        return samples
