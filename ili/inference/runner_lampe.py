"""
Module to train posterior inference models using the lampe package
"""

import json
import yaml
import time
import logging
import pickle
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import lampe
from pathlib import Path
from typing import Dict, List, Callable, Optional
from torch.distributions import Distribution
from ili.dataloaders import _BaseLoader
from ili.utils import load_from_config, LampeEnsemble, load_nde_lampe

logging.basicConfig(level=logging.INFO)


class LampeRunner():
    """Class to train NPE posterior inference models using the lampe package.
    Follows methodology of: https://arxiv.org/abs/1711.01861

    Args:
        prior (Distribution): prior on the parameters
        nets (List[Callable]): list of neural nets for amortized posteriors,
            likelihood models, or ratio classifiers
        engine (str): name of the engine class (NPE only)
        train_args (Dict): dictionary of hyperparameters for training
        out_dir (Path): directory where to store outputs
        device (str): device to run on
        proposal (Distribution): proposal distribution from which existing
            simulations were run, for single round inference only. By default,
            we will set proposal = prior unless a proposal is specified.
        name (str): name of the model (for saving purposes)
        signatures (List[str]): list of signatures for each neural net
    """

    def __init__(
        self,
        prior: Distribution,
        nets: List[Callable],
        engine: str = 'NPE',
        train_args: Dict = {},
        out_dir: Path = None,
        device: str = 'cpu',
        proposal: Distribution = None,
        name: Optional[str] = "",
        signatures: Optional[List[str]] = None,
    ):
        self.prior = prior
        self.nets = nets
        if engine != 'NPE':
            logging.warning(
                'lampe only supports NPE engine. Engine set to NPE.')
        self.engine = 'NPE'
        self.train_args = dict(
            training_batch_size=50, learning_rate=5e-4,
            stop_after_epochs=30, clip_max_norm=5,
            max_epochs=int(1e10),
            validation_fraction=0.1)
        self.train_args.update(train_args)
        self.out_dir = out_dir
        if self.out_dir is not None:
            self.out_dir = Path(self.out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        if proposal is None:
            self.proposal = prior
        else:
            self.proposal = proposal
        self.name = name
        self.signatures = signatures
        if self.signatures is None:
            self.signatures = [""]*len(self.nets)

    @classmethod
    def from_config(cls, config_path: Path, **kwargs) -> "LampeRunner":
        """Create a lampe runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file
            **kwargs: optional keyword arguments to overload config file
        Returns:
            LampeRunner: the lampe runner specified by the config file
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
        nets = [load_nde_lampe(embedding_net=embedding_net,
                               device=config["device"],
                               **model_args)
                for model_args in config['model']['nets']]

        # initialize
        return cls(
            prior=prior,
            nets=nets,
            train_args=train_args,
            out_dir=out_dir,
            device=config["device"],
            proposal=proposal,
            name=name,
            signatures=signatures,
        )

    def _prepare_loader(self, loader: _BaseLoader):
        """Prepare a loader for training."""
        if (hasattr(loader, "train_loader") and
                hasattr(loader, "val_loader")):
            train_loader, val_loader = loader.train_loader, loader.val_loader
        elif (hasattr(loader, "get_all_data") and
                hasattr(loader, "get_all_parameters")):
            x, theta = loader.get_all_data(), loader.get_all_parameters()

            # move to device
            x = torch.Tensor(x).to(self.device)
            theta = torch.Tensor(theta).to(self.device)

            # split data into train and validation
            mask = torch.randperm(len(x)) < int(
                self.train_args['validation_fraction']*len(x))
            x_train, x_val = x[~mask], x[mask]
            theta_train, theta_val = theta[~mask], theta[mask]

            data_train = TensorDataset(x_train, theta_train)
            data_val = TensorDataset(x_val, theta_val)
            train_loader = DataLoader(
                data_train, shuffle=True,
                batch_size=self.train_args["training_batch_size"])
            val_loader = DataLoader(
                data_val, shuffle=False,
                batch_size=self.train_args["training_batch_size"])
        else:
            raise ValueError("Loader must be a subclass of _BaseLoader.")
        return train_loader, val_loader

    def _loss(self, model, theta, x):
        """Return neg importance-weighted probability as loss."""
        log_posterior = model(theta, x)
        if self.prior == self.proposal:
            return -log_posterior.mean()

        log_prior = self.prior.log_prob(theta)
        log_proposal = self.proposal.log_prob(theta)

        negloss = torch.exp(log_prior - log_proposal) * log_posterior
        return -negloss.mean()

    def _train_epoch(self, model, train_loader, val_loader, stepper):
        """Train a single epoch of a neural network model."""
        model.train()

        loss_train, count = [], 0
        for x, theta in train_loader:
            x, theta = x.to(self.device), theta.to(self.device)
            loss_train.append(
                stepper(self._loss(model, theta, x)) * len(theta))
            count += len(theta)
        loss_train = torch.stack(loss_train).sum().item()/count

        model.eval()
        with torch.no_grad():
            loss_val, count = [], 0
            for x, theta in val_loader:
                x, theta = x.to(self.device), theta.to(self.device)
                loss_val.append(self._loss(model, theta, x) * len(theta))
                count += len(theta)
            loss_val = torch.stack(loss_val).sum().item()/count
        return loss_train, loss_val

    def _train_round(self, models: List[Callable],
                     train_loader: DataLoader, val_loader: DataLoader):
        """Train a single round of inference for an ensemble of models."""

        # initialize models
        x_, y_ = next(iter(train_loader))
        models_rnd = [
            model(x_, y_, self.prior).to(self.device)
            for model in models
        ]

        posteriors, summaries = [], []
        for i, model in enumerate(models_rnd):
            logging.info(f"Training model {i+1} / {len(models_rnd)}.")

            # define optimizer
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.train_args["learning_rate"]
            )
            stepper = lampe.utils.GDStep(
                optimizer, clip=self.train_args["clip_max_norm"])

            # train model
            best_val = float('inf')
            wait = 0
            summary = {'training_log_probs': [], 'validation_log_probs': []}
            with tqdm(iter(range(self.train_args["max_epochs"])),
                      unit=' epochs') as tq:
                for epoch in tq:
                    loss_train, loss_val = self._train_epoch(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        stepper=stepper,
                    )
                    tq.set_postfix(
                        loss=loss_train,
                        loss_val=loss_val,
                    )
                    summary['training_log_probs'].append(-loss_train)
                    summary['validation_log_probs'].append(-loss_val)

                    # check for convergence
                    if loss_val < best_val:
                        best_val = loss_val
                        best_model = deepcopy(model.state_dict())
                        wait = 0
                    elif wait > self.train_args["stop_after_epochs"]:
                        break
                    else:
                        wait += 1
                else:
                    logging.warning(
                        "Training did not converge in "
                        f"{self.train_args['max_epochs']} epochs.")
                summary['best_validation_log_prob'] = -best_val
                summary['epochs_trained'] = epoch

            # save model
            model.load_state_dict(best_model)
            posteriors.append(model)
            summaries.append(summary)

        # ensemble all trained models, weighted by validation loss
        val_logprob = torch.tensor(
            [float(x["best_validation_log_prob"]) for x in summaries]
        ).to(self.device)
        # Exponentiate with numerical stability
        weights = torch.exp(val_logprob - val_logprob.max())
        weights /= weights.sum()

        posterior_ensemble = LampeEnsemble(posteriors, weights)

        # record the name of the ensemble
        posterior_ensemble.name = self.name
        posterior_ensemble.signatures = self.signatures

        return posterior_ensemble, summaries

    def _save_models(self, posterior_ensemble: LampeEnsemble,
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
        logging.info("MODEL INFERENCE CLASS: NPE")

        # load single-round data
        train_loader, val_loader = self._prepare_loader(loader)

        # train a single round of inference
        t0 = time.time()
        posterior_ensemble, summaries = self._train_round(
            models=self.nets,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        logging.info(f"It took {time.time() - t0} seconds to train models.")

        # save if output path is specified
        if self.out_dir is not None:
            self._save_models(posterior_ensemble, summaries)

        return posterior_ensemble, summaries
