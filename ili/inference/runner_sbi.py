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
from typing import Dict, List, Callable, Optional, Union
from torch.distributions import Distribution
from sbi.inference import NeuralInference
from .base import _BaseRunner
try:  # sbi > 0.22.0
    from sbi.utils.posteriors import EnsemblePosterior
except ImportError:  # sbi < 0.22.0
    from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble as EnsemblePosterior

from ili.dataloaders import _BaseLoader
from ili.utils import load_class, load_from_config, load_nde_sbi, update

logging.basicConfig(level=logging.INFO)


class SBIRunner(_BaseRunner):
    """Class to train posterior inference models using the sbi package.
    Follows methodology of:
        * engine='NPE': https://arxiv.org/abs/1905.07488
        * engine='NLE': https://arxiv.org/abs/1805.07226
        * engine='NRE': https://arxiv.org/pdf/2002.03712

    Args:
        prior (Distribution): prior on the parameters
        engine (str): type of inference engine to use (NPE, NLE, NRE, or
            any sbi inference engine; see _setup_engine)
        nets (List[Callable]): list of neural nets for amortized posteriors,
            likelihood models, or ratio classifiers
        train_args (Dict): dictionary of hyperparameters for training
        out_dir (str, Path): directory where to store outputs
        proposal (Distribution): proposal distribution from which existing
            simulations were run, for single round inference only. By default,
            sbi will set proposal = prior unless a proposal is specified.
        name (str): name of the model (for saving purposes)
        signatures (List[str]): list of signatures for each neural net
    """

    def __init__(
        self,
        prior: Distribution,
        engine: str,
        nets: List[Callable],
        train_args: Dict = {},
        out_dir: Union[str, Path] = None,
        device: str = 'cpu',
        proposal: Distribution = None,
        name: Optional[str] = "",
        signatures: Optional[List[str]] = None,
    ):
        super().__init__(
            prior=prior,
            train_args=train_args,
            out_dir=out_dir,
            device=device,
            name=name,
        )
        if proposal is None:
            self.proposal = prior
        else:
            self.proposal = proposal
        self.engine = engine
        # Below, to handle the repeats
        nets_list = []
        for net_el in nets:
            if isinstance(net_el, List):
                for net in net_el:
                    nets_list.append(net)
            else:
                nets_list.append(net_el)
        self.nets = nets_list

        self.num_rounds = self.train_args.pop("num_round", 1)

        train_default = dict(
            training_batch_size=50,
            learning_rate=5e-4,
            validation_fraction=0.1,
            stop_after_epochs=20,
            clip_max_norm=5,
        )
        train_default.update(self.train_args)
        self.train_args = train_default

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
        update(config, **kwargs)

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
        engine = config["model"]["engine"]
        nets = []

        # For every different nets architecture
        for model_args in config['model']['nets']:
            if "repeats" in model_args:
                n_size = model_args["repeats"]
                model_args.pop("repeats")
            else:
                n_size = 1

            # Repeat to have an ensemble of n_size >=1 of the same nets architecture
            for n in range(n_size):
                nets.append(load_nde_sbi(config['model']['engine'],
                                         embedding_net=embedding_net,
                                         **model_args))

        # initialize
        return cls(
            prior=prior,
            proposal=proposal,
            engine=engine,
            nets=nets,
            device=config["device"],
            train_args=train_args,
            out_dir=out_dir,
            signatures=signatures,
            name=name,
        )

    def _setup_engine(self, net: nn.Module):
        """Instantiate an sbi inference engine (SNPE/SNLE/SNRE)."""
        if self.engine[0] == 'S':
            engine_name = self.engine
        else:
            engine_name = 'S'+self.engine
        try:
            inference_class = load_class('sbi.inference', engine_name)
        except ImportError:
            raise ValueError(
                f"Model class {self.engine} not supported. "
                "Please choose one of NPE/NLE/NRE or SNPE/SNLE/SNRE or "
                "an inference class in sbi.inference."
            )

        if ("NPE" in self.engine) or ("NLE" in self.engine):
            return inference_class(
                prior=self.prior,
                density_estimator=net,
                device=self.device,
            )
        elif ("NRE" in self.engine):
            return inference_class(
                prior=self.prior,
                classifier=net,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Model class {self.engine} not supported with SBIRunner.")

    def _train_round(self, models: List[NeuralInference],
                     x: torch.Tensor, theta: torch.Tensor,
                     proposal: Optional[Distribution]):
        """Train a single round of inference for an ensemble of models."""

        # append data to models
        for model in models:
            if ("NPE" in self.engine):
                model = model.append_simulations(theta, x, proposal=proposal)
            else:
                model = model.append_simulations(theta, x)

        # get all previous simulations
        starting_round = 0  # NOTE: won't work for SNPE_A, but we don't use it
        x, _, _ = model.get_simulations(starting_round)

        # split into training and validation randomly
        num_examples = x.shape[0]
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int(
            (1 - self.train_args['validation_fraction']) * num_examples)
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        posteriors, summaries = [], []
        for i, model in enumerate(models):
            logging.info(f"Training model {i+1} / {len(models)}.")

            # hack to initialize sbi model without training (ref. issue #127)
            first_round = False
            if model._neural_net is None:
                model.train(learning_rate=self.train_args['learning_rate'],
                            resume_training=False,
                            max_num_epochs=-1)
                model._epochs_since_last_improvement = 0
                first_round = True

            # set train/validation splits
            model.train_indices = train_indices
            model.val_indices = val_indices

            # train
            if ("NPE" in self.engine) & first_round:
                model.train(**self.train_args, resume_training=True,
                            force_first_round_loss=True)
            else:
                model.epoch, model._val_log_prob = 0, float("-Inf")
                model.train(**self.train_args,  resume_training=True)

            # save model
            posteriors.append(model.build_posterior())
            summaries.append(model.summary)

        # ensemble all trained models, weighted by validation loss
        val_logprob = torch.tensor(
            [-1.*float(x["best_validation_loss"][-1]) for x in summaries]).to(self.device)

        # Exponentiate with numerical stability
        weights = torch.exp(val_logprob - val_logprob.max())
        weights /= weights.sum()
        posterior_ensemble = EnsemblePosterior(
            posteriors=posteriors,
            weights=weights,
            theta_transform=posteriors[0].theta_transform
        )  # raises warning due to bug in sbi

        # record the name of the ensemble
        posterior_ensemble.name = self.name
        posterior_ensemble.signatures = self.signatures

        return posterior_ensemble, summaries

    def _save_models(self, posterior_ensemble: EnsemblePosterior,
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
        logging.info(f"MODEL INFERENCE CLASS: {self.engine}")
        models = [self._setup_engine(net) for net in self.nets]

        # load single-round data
        x = torch.Tensor(loader.get_all_data()).to(self.device)
        theta = torch.Tensor(loader.get_all_parameters()).to(self.device)

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
    multiple rounds.

    Follows methodology of:
        * engine='SNPE': https://arxiv.org/abs/1905.07488
        * engine='SNLE': https://arxiv.org/abs/1805.07226
        * engine='SNRE': https://arxiv.org/pdf/2002.03712
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
        logging.info(f"MODEL INFERENCE CLASS: {self.engine}")
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

            if rnd < self.num_rounds - 1:
                # simulate new data for next round
                theta, x = loader.simulate(self.proposal)
                x = torch.Tensor(x).to(self.device)
                theta = torch.Tensor(theta).to(self.device)

        logging.info(f"It took {time.time() - t0} seconds to train models.")

        if self.out_dir is not None:
            self._save_models(posterior_ensemble, summaries)

        return posterior_ensemble, summaries


class ABCRunner(_BaseRunner):
    """Class to run ABC inference models using the sbi package"""

    def __init__(
            self,
            prior: Distribution,
            engine: str,
            train_args: Dict = {},
            out_dir: Union[str, Path] = None,
            device: str = 'cpu',
            name: Optional[str] = "",
    ):
        super().__init__(
            prior=prior,
            train_args=train_args,
            out_dir=out_dir,
            device=device,
            name=name,
        )
        self.engine = engine

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
        update(config, **kwargs)

        # load prior distribution
        prior = load_from_config(config["prior"])

        # parse inference engine
        engine = config["model"]["engine"]

        # load logistics
        train_args = config["train_args"]
        out_dir = Path(config["out_dir"])
        name = ""
        if "name" in config["model"]:
            name = config["model"]["name"]+"_"

        return cls(
            prior=prior,
            engine=engine,
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

        logging.info(f"MODEL INFERENCE CLASS: {self.engine}")

        x_obs = loader.get_obs_data()

        # setup and train each architecture
        inference_class = load_class('sbi.inference', self.engine)
        model = inference_class(
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
