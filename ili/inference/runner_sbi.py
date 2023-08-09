"""
Module to train posterior inference models using the sbi package
"""


import yaml
import time
import logging
import pickle
import torch
import torch.nn as nn
import sbi
from pathlib import Path
from typing import Dict, List, Callable
from torch.distributions import Independent
from sbi.inference import NeuralInference
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from ili.utils import load_class, load_from_config

logging.basicConfig(level=logging.INFO)

default_config = (
    Path(__file__).parent.parent / "examples/configs/sample_sbi.yaml"
)


class SBIRunner:
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
    """

    def __init__(
        self,
        prior: Independent,
        inference_class: NeuralInference,
        nets: List[Callable],
        device: str,
        embedding_net: nn.Module,
        train_args: Dict,
        output_path: Path,
        proposal: Independent = None,
    ):
        self.prior = prior
        self.proposal = proposal
        self.inference_class = inference_class
        self.class_name = inference_class.__name__
        self.nets = nets
        self.device = device
        self.embedding_net = embedding_net
        self.train_args = train_args
        if "num_round" in train_args:
            self.num_rounds = train_args["num_round"]
            self.train_args.pop("num_round")
        else:
            self.num_rounds = 1
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

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
        prior = load_from_config(config["prior"])
        if "proposal" in config:
            proposal = load_from_config(config["proposal"])
        else:
            proposal = None
        if "proposal" in config:
            proposal = load_from_config(config["proposal"])
        else:
            proposal = None
        if "embedding_net" in config:
            embedding_net = load_from_config(
                config=config["embedding_net"],
            )
        else:
            embedding_net = nn.Identity()
        inference_class = load_class(
            module_name=config["model"]["module"],
            class_name=config["model"]["class"],
        )
        nets = cls.load_nets(
            embedding_net=embedding_net,
            class_name=config["model"]["class"],
            posteriors_config=config["model"]["nets"],
        )
        train_args = config["train_args"]
        output_path = Path(config["output_path"])
        return cls(
            prior=prior,
            proposal=proposal,
            inference_class=inference_class,
            nets=nets,
            device=config["device"],
            embedding_net=embedding_net,
            train_args=train_args,
            output_path=output_path,
        )

    @classmethod
    def load_nets(
        cls,
        embedding_net: nn.Module,
        class_name: str,
        posteriors_config: List[Dict],
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
            elif "SNLE" in class_name:
                return sbi.utils.likelihood_nn(
                    embedding_net=embedding_net, **model_args)
            elif "SNRE" in class_name:
                return sbi.utils.classifier_nn(
                    embedding_net_x=embedding_net, **model_args)
            else:
                raise ValueError(
                    f"Model class {class_name} not supported. "
                    "Please choose one of SNPE, SNLE, or SNRE."
                )

        return [_build_model(embedding_net, model_args)
                for model_args in posteriors_config]

    def _train_model_SNPE(self, net, theta, x):
        """Instantiate and train an amoritized posterior SNPE model."""
        model = self.inference_class(
            prior=self.prior,
            density_estimator=net,
            device=self.device,
        )
        model = model.append_simulations(theta, x, proposal=self.proposal)
        if not isinstance(self.embedding_net, nn.Identity):
            self.embedding_net.initalize_model(n_input=x.shape[-1])
        _ = model.train(**self.train_args)
        return model

    def _train_model_SNLE(self, net, theta, x):
        """Instantiate and train a likelihood estimation SNLE model."""
        model = self.inference_class(
            prior=self.prior,
            density_estimator=net,
            device=self.device,
        )
        model = model.append_simulations(theta, x)
        if not isinstance(self.embedding_net, nn.Identity):
            self.embedding_net.initalize_model(n_input=x.shape[-1])
        _ = model.train(**self.train_args)
        return model

    def _train_model_SNRE(self, net, theta, x):
        """Instantiate and train a ratio estimation SNRE model."""
        model = self.inference_class(
            prior=self.prior,
            classifier=net,
            device=self.device,
        )
        model = model.append_simulations(theta, x)
        if not isinstance(self.embedding_net, nn.Identity):
            self.embedding_net.initalize_model(n_input=x.shape[-1])
        _ = model.train(**self.train_args)
        return model

    def __call__(self, loader, seed=None):
        """Train your posterior and save it to file

        Args:
            loader (BaseLoader): dataloader with stored summary-parameter pairs
            seed (int): torch seed for reproducibility
        """

        t0 = time.time()
        x = torch.Tensor(loader.get_all_data())
        theta = torch.Tensor(loader.get_all_parameters())
        posteriors, val_logprob = [], []
        for n, posterior in enumerate(self.nets):
            logging.info(
                f"Training model {n+1} out of {len(self.nets)}"
                " ensemble models"
            )
            # set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)

            if "SNPE" in self.class_name:
                model = self._train_model_SNPE(posterior, theta, x)
            elif "SNLE" in self.class_name:
                model = self._train_model_SNLE(posterior, theta, x)
            elif "SNRE" in self.class_name:
                model = self._train_model_SNRE(posterior, theta, x)

            # save model
            posteriors.append(model.build_posterior())
            val_logprob += model.summary["best_validation_log_prob"]

        # ensemble all trained models, weighted by validation loss
        posterior = NeuralPosteriorEnsemble(
            posteriors=posteriors,
            weights=torch.tensor([float(vl) for vl in val_logprob]),
        )
        with open(self.output_path / "posterior.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
        logging.info(
            f"It took {time.time() - t0} seconds to train all models.")


class SBIRunnerSequential(SBIRunner):
    """
    Class to train posterior inference models using the sbi package with
    multiple rounds
    """

    def __call__(self, loader):
        """Train your posterior and save it to file

        Args:
            loader (BaseLoader): data loader with ability to simulate
                summary-parameter pairs
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

        for rnd in range(self.num_rounds):
            t1 = time.time()
            logging.info(
                f"Running round {rnd+1} of {self.num_rounds}"
            )
            theta, x = loader.simulate(proposal)
            theta, x = torch.Tensor(theta), torch.Tensor(x)
            posteriors, val_logprob = [], []
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
                val_logprob.append(
                    all_model[i].summary["best_validation_log_prob"][-1])

            val_logprob = torch.tensor([float(vl) for vl in val_logprob])
            # Subtract maximum loss to improve numerical stability of exp
            # (cancels in next line)
            val_logprob = torch.exp(val_logprob - val_logprob.max())
            val_logprob /= val_logprob.sum()

            posterior = NeuralPosteriorEnsemble(
                posteriors=posteriors,
                weights=val_logprob
            )

            with open(self.output_path / f"posterior_{rnd}.pkl", "wb") as f:
                pickle.dump(posterior, f)
            proposal = posterior.set_default_x(x_obs)
            logging.info(
                f"It took {time.time()-t1} seconds to complete round {rnd+1}.")

        with open(self.output_path / "posterior.pkl", "wb") as f:
            pickle.dump(posterior, f)
        logging.info(
            f"It took {time.time() - t0} seconds to train all models.")
