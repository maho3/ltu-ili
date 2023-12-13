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
import sbi
from pathlib import Path
from typing import Dict, List, Callable, Optional
from torch.distributions import Independent
from sbi.inference import NeuralInference
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from ili.dataloaders import _BaseLoader
from ili.utils import load_class, load_from_config

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
        output_path: Path = None,
        device: str = 'cpu',
        name: Optional[str] = "",
    ):
        self.prior = prior
        self.inference_class = inference_class
        self.class_name = inference_class.__name__
        self.train_args = train_args
        self.device = device
        self.name = name
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path = Path(self.output_path)
            self.output_path.mkdir(parents=True, exist_ok=True)


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
        output_path (Path): path where to store outputs
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
        output_path: Path = None,
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
            output_path=output_path,
            device=device,
            name=name,
        )
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
    def from_config(cls, config_path: Path) -> "SBIRunner":
        """Create an sbi runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file
        Returns:
            SBIRunner: the sbi runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

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

        # load inference class and neural nets
        inference_class = load_class(
            module_name=config["model"]["module"],
            class_name=config["model"]["class"],
        )
        nets = cls.load_nets(
            embedding_net=embedding_net,
            class_name=config["model"]["class"],
            posteriors_config=config["model"]["nets"],
        )

        # load logistics
        train_args = config["train_args"]
        output_path = Path(config["output_path"])
        if "name" in config["model"]:
            name = config["model"]["name"]+"_"
        else:
            name = ""
        signatures = []
        for type_nn in config["model"]["nets"]:
            if "signature" in type_nn:
                signatures.append(type_nn["signature"] + "_")
            else:
                signatures.append("")
        return cls(
            prior=prior,
            proposal=proposal,
            inference_class=inference_class,
            nets=nets,
            device=config["device"],
            embedding_net=embedding_net,
            train_args=train_args,
            output_path=output_path,
            signatures=signatures,
            name=name,
        )

    @classmethod
    def load_nets(
        cls,
        class_name: str,
        posteriors_config: List[Dict],
        embedding_net: nn.Module = nn.Identity(),
    ) -> List[Callable]:
        """Load the inference model

        Args:
            embedding_net (nn.Module): neural network to compress data
            class_name (str): name of the inference class
            posterior_config(List[Dict]): list with configurations for each
                neural posterior model in the ensemble

        Returns:
            List[Callable]: list of pytorch neural network models with forward
                methods
        """
        # determine the correct model type
        def _build_model(embedding_net, model_args):
            if "SNPE" in class_name:
                return sbi.utils.posterior_nn(
                    embedding_net=embedding_net, **model_args)
            elif "SNLE" in class_name or "MNLE" in class_name:
                return sbi.utils.likelihood_nn(
                    embedding_net=embedding_net, **model_args)
            elif "SNRE" in class_name or "BNRE" in class_name:
                return sbi.utils.classifier_nn(
                    embedding_net_x=embedding_net, **model_args)
            else:
                raise ValueError(
                    f"Model class {class_name} not supported. "
                    "Please choose one of SNPE, SNLE, or SNRE."
                )
        return [_build_model(embedding_net, model_args)
                for model_args in posteriors_config]

    def _setup_SNPE(self, net: nn.Module, theta: torch.Tensor, x: torch.Tensor):
        """Instantiate and train an amoritized posterior SNPE model."""
        model = self.inference_class(
            prior=self.prior,
            density_estimator=net,
            device=self.device,
        )
        model = model.append_simulations(theta, x, proposal=self.proposal)
        return model

    def _setup_SNLE(self, net: nn.Module, theta: torch.Tensor, x: torch.Tensor):
        """Instantiate and train a likelihood estimation SNLE model."""
        model = self.inference_class(
            prior=self.prior,
            density_estimator=net,
            device=self.device,
        )
        model = model.append_simulations(theta, x)
        return model

    def _setup_SNRE(self, net: nn.Module, theta: torch.Tensor, x: torch.Tensor):
        """Instantiate and train a ratio estimation SNRE model."""
        model = self.inference_class(
            prior=self.prior,
            classifier=net,
            device=self.device,
        )
        model = model.append_simulations(theta, x)
        return model

    def __call__(self, loader: _BaseLoader, seed: int = None):
        """Train your posterior and save it to file

        Args:
            loader (_BaseLoader): dataloader with stored data-parameter pairs
            seed (int): torch seed for reproducibility
        """
        t0 = time.time()
        x = torch.Tensor(loader.get_all_data()).to(self.device)
        theta = torch.Tensor(loader.get_all_parameters()).to(self.device)

        logging.info(f"MODEL INFERENCE CLASS: {self.class_name}")

        # instantiate embedding_net architecture, if necessary
        if self.embedding_net and hasattr(self.embedding_net, 'initalize_model'):
            self.embedding_net.initalize_model(n_input=x.shape[-1])

        # setup and train each architecture
        posteriors, summaries = [], []
        for n, net in enumerate(self.nets):
            logging.info(
                f"Training model {n+1} out of {len(self.nets)}"
                " ensemble models"
            )
            # set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
            # setup training class
            if "SNPE" in self.class_name:
                model = self._setup_SNPE(net, theta, x)
            elif "SNLE" in self.class_name or "MNLE" in self.class_name:
                model = self._setup_SNLE(net, theta, x)
            elif "SNRE" in self.class_name or "BNRE" in self.class_name:
                model = self._setup_SNRE(net, theta, x)

            # train
            _ = model.train(**self.train_args)

            # save model
            posteriors.append(model.build_posterior())
            summaries.append(model.summary)

        # ensemble all trained models, weighted by validation loss
        val_logprob = torch.tensor(
            [float(x["best_validation_log_prob"][0]) for x in summaries]
        ).to(self.device)
        # Subtract maximum loss to improve numerical stability of exp
        # (cancels in next line)
        weights = torch.exp(val_logprob - val_logprob.max())
        weights /= weights.sum()

        posterior_ensemble = NeuralPosteriorEnsemble(
            posteriors=posteriors,
            weights=weights,
            theta_transform=posteriors[0].theta_transform
        )  # raises warning due to bug in sbi
        posterior_ensemble.name = self.name
        posterior_ensemble.signatures = self.signatures

        # save if output path is specified
        if self.output_path is not None:
            str_p = self.name + "posterior.pkl"
            str_s = self.name + "summary.json"
            with open(self.output_path / str_p, "wb") as handle:
                pickle.dump(posterior_ensemble, handle)
            with open(self.output_path / str_s, "w") as handle:
                json.dump(summaries, handle)

        logging.info(
            f"It took {time.time() - t0} seconds to train all models.")
        return posterior_ensemble, summaries


class SBIRunnerSequential(SBIRunner):
    """
    Class to train posterior inference models using the sbi package with
    multiple rounds
    """

    def __call__(self, loader: _BaseLoader):
        """Train your posterior and save it to file

        Args:
            loader (_BaseLoader): data loader with ability to simulate
                data-parameter pairs
        """
        t0 = time.time()
        x_obs = loader.get_obs_data()

        all_model = []
        for n, posterior in enumerate(self.nets):
            all_model.append(self.inference_class(
                prior=self.prior,
                density_estimator=posterior,
                device=self.device,
            ))
        proposal = self.prior

        # loader has x and theta attributes, both default values are None
        # Even in multiround inference, we can take advantage of prerun
        # simulation-parameter pairs
        x = loader.get_all_data()
        theta = loader.get_all_parameters()
        if x is not None and theta is not None:
            theta = torch.Tensor(theta).to(self.device)
            x = torch.Tensor(x).to(self.device)
            prerun_sims = True
            logging.info(
                "The first round of inference will use existing sims from the "
                "loader. Make sure that the simulations were run from the "
                "given prior for consistency.")
        else:
            prerun_sims = False
            logging.info(
                "The first round of inference will simulate from the given prior."
            )

        # Start multiround inference
        for rnd in range(self.num_rounds):
            t1 = time.time()
            logging.info(
                f"Running round {rnd+1} of {self.num_rounds}"
            )

            if rnd == 0 and prerun_sims:
                pass  # in that case theta and x were set before the loop on rnd
            else:
                theta, x = loader.simulate(proposal)
                theta, x = torch.Tensor(theta).to(
                    self.device), torch.Tensor(x).to(self.device)

            posteriors, summaries = [], []
            for i in range(len(self.nets)):
                logging.info(
                    f"Training model {n+1} out of "
                    f"{len(self.nets)} ensemble models"
                )
                if not isinstance(self.embedding_net, nn.Identity):
                    self.embedding_net.initalize_model(n_input=x.shape[-1])
                density_estimator = \
                    all_model[i].append_simulations(theta, x, proposal).train(
                        **self.train_args,
                    )
                posteriors.append(
                    all_model[i].build_posterior(density_estimator))
                summaries.append(all_model[i].summary)

            val_logprob = torch.tensor(
                [float(x["best_validation_log_prob"][0]) for x in summaries]
            ).to(self.device)
            # Subtract maximum loss to improve numerical stability of exp
            # (cancels in next line)
            weights = torch.exp(val_logprob - val_logprob.max())
            weights /= weights.sum()

            posterior_ensemble = NeuralPosteriorEnsemble(
                posteriors=posteriors,
                weights=weights)  # raises warning due to bug in sbi
            posterior_ensemble.name = self.name
            posterior_ensemble.signatures = self.signatures

            logging.info(f"Network signatures: {self.signatures}")

            str_p = self.name + f"posterior_{rnd}.pkl"
            str_s = self.name + f"summary_{rnd}.json"
            with open(self.output_path / str_p, "wb") as handle:
                pickle.dump(posterior_ensemble, handle)
            with open(self.output_path / str_s, "w") as handle:
                json.dump(summaries, handle)

            proposal = posterior_ensemble.set_default_x(x_obs)
            logging.info(
                f"It took {time.time()-t1} seconds to complete round {rnd+1}.")

        if self.output_path is not None:
            str_p = self.name + "posterior.pkl"
            str_s = self.name + "summary.json"
            with open(self.output_path / str_p, "wb") as handle:
                pickle.dump(posterior_ensemble, handle)
            with open(self.output_path / str_s, "w") as handle:
                json.dump(summaries, handle)
        logging.info(
            f"It took {time.time() - t0} seconds to train all models.")


class ABCRunner(_BaseRunner):
    """Class to run ABC inference models using the sbi package"""

    @classmethod
    def from_config(cls, config_path: Path) -> "ABCRunner":
        """Create an sbi runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file
        Returns:
            SBIRunner: the sbi runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

        # load prior distribution
        prior = load_from_config(config["prior"])

        # load inference class
        inference_class = load_class(
            module_name=config["model"]["module"],
            class_name=config["model"]["class"],
        )

        # load logistics
        train_args = config["train_args"]
        output_path = Path(config["output_path"])
        name = ""
        if "name" in config["model"]:
            name = config["model"]["name"]+"_"

        return cls(
            prior=prior,
            inference_class=inference_class,
            device=config["device"],
            train_args=train_args,
            output_path=output_path,
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

        # save model

        # save if output path is specified
        if self.output_path is not None:
            str_p = self.name + "samples.pkl"
            with open(self.output_path / str_p, "wb") as handle:
                pickle.dump(samples, handle)

        logging.info(
            f"It took {time.time() - t0} seconds to run the model.")
        return samples
