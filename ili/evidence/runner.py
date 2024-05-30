"""
Module to contain evidence estimators.

TODO:
  * Add embedding networks
  * Add ensembling
"""
import logging
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import harmonic as hm
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from .utils import POPExpLoss, ExpLoss, EvidenceNetworkSimple

logging.basicConfig(level=logging.INFO)


class HarmonicEvidence():
    def __init__(self):
        self.evidence = None

    def from_nde(self, npe, nle, x, shape=(10000,), **sample_kwargs):
        """Estimate evidence from a posterior object.

        Args:
            npe (_type_): A Neural Posterior Estimator object with a
                .sample() method.
            nle (_type_): A Neural Likelihood Estimator object with a
                .potential() method.
            x (_type_): The input data to the posterior.
            shape (tuple, optional): The shape of the samples to draw from the
                posterior. Defaults to (10000,).
            **sample_kwargs: Additional keyword arguments to pass to the
                posterior's .sample() method.
        """
        samples = npe.sample(shape, x, **sample_kwargs)
        lnprob = nle.potential(samples, x)
        self.from_samples(samples, lnprob)

    def from_samples(self, samples, lnprob):
        """Estimate evidence from samples and log-probabilities.
        # TODO: Add support for multiple chains

        Args:
            samples (array-like): An (N, D) array of samples,
                where N is the number of samples, D is the dimensionality of
                the samples.
            lnprob (array-like): An (N,) array of log-likelihoods for each
                sample
        """
        if len(samples.shape) != 2:
            raise ValueError('Samples shape must be (N, D).')
        if len(samples) != len(lnprob):
            raise ValueError('Input samples and lnprob must be same length.')
        self._calculate_evidence(samples, lnprob)

    def _calculate_evidence(self, samples, lnprob):
        """Internal function to calculate evidence with the targeted harmonic
        mean estimator.

        Args:
            samples (array-like): An (N, D) array of samples, where C is the
                number of chains, N is the number of samples, and D is the
                dimensionality of the samples.
            lnprob (array-like): An (N,) array of proxy log-probabilities for
                each sample.
        """
        samples = np.asarray(samples)
        lnprob = np.asarray(lnprob)
        ndim = samples.shape[-1]

        # Split samples into train/test
        chains = hm.Chains(ndim)
        chains.add_chain(samples, lnprob)
        chains.split_into_blocks(100)
        chains_train, chains_infer = hm.utils.split_data(
            chains, training_proportion=0.5)

        # Select RealNVP Model
        n_scaled_layers = 2
        n_unscaled_layers = 4
        temperature = 0.8

        model = hm.model.RealNVPModel(
            ndim,
            n_scaled_layers=n_scaled_layers,
            n_unscaled_layers=n_unscaled_layers,
            standardize=True,
            temperature=temperature
        )
        epochs_num = 20
        # Train model
        model.fit(chains_train.samples, epochs=epochs_num, verbose=True)

        # Save harmonic's evidence class
        self.evidence = hm.Evidence(chains_infer.nchains, model)
        self.evidence.add_chains(chains_infer)

    def get_evidence(self):
        if self.evidence is None:
            raise ValueError('Evidence has not been computed.')
        return self.evidence.compute_evidence()

    def get_ln_evidence(self):
        if self.evidence is None:
            raise ValueError('Evidence has not been computed.')
        return self.evidence.compute_ln_evidence()

    def get_bayes_factor(self, ev2):
        if self.evidence is None:
            raise ValueError('Evidence has not been computed.')
        if ev2.evidence is None:
            raise ValueError('Evidence for model 2 has not been computed.')
        return hm.evidence.compute_bayes_factor(self.evidence, ev2.evidence)


class K_EvidenceNetwork():
    """Trains an evidence network to estimate the Bayes Factor K for two models."""

    def __init__(
        self,
        layer_width=16,
        added_layers=3,
        batch_norm_flag=1,
        alpha=2,
        train_args={},
        device='cpu'
    ):
        self.layer_width = layer_width
        self.added_layers = added_layers
        self.batch_norm_flag = batch_norm_flag
        self.alpha = alpha
        self.train_args = dict(
            training_batch_size=32, learning_rate=5e-4,
            stop_after_epochs=30, clip_max_norm=5,
            max_epochs=int(1e10),
            validation_fraction=0.1)
        self.train_args.update(train_args)
        self.device = device

        self.loss_fn = ExpLoss()

        self.best_val = float('inf')
        self.model = None

    def _loss(self, model, theta, x):
        """Compute the loss function for a given model."""
        logK = model(x)
        return self.loss_fn(logK, theta)

    def _train_epoch(self, model, train_loader, val_loader, optimizer):
        """Train a single epoch of a neural network model."""
        model.train()

        loss_train, count = 0, 0
        for x, theta in train_loader:
            x, theta = x.to(self.device), theta.to(self.device)
            optimizer.zero_grad()
            loss = self._loss(model, theta, x)
            loss.backward()

            # Clip gradients
            norm = nn.utils.clip_grad_norm_(
                model.parameters(), self.train_args['clip_max_norm'])
            # Step
            if norm.isfinite():
                optimizer.step()
            # Record
            loss_train += loss.item() * len(theta)
            count += len(theta)
        loss_train = loss_train/count

        model.eval()
        with torch.no_grad():
            loss_val, count = 0, 0
            for x, theta in val_loader:
                x, theta = x.to(self.device), theta.to(self.device)
                loss_val += self._loss(model, theta, x).item() * len(theta)
                count += len(theta)
            loss_val = loss_val/count
        return loss_train, loss_val

    def train(self, loader1, loader2, show_progress_bars=True):
        # Aggregate data from different models, and label them
        x1 = loader1.get_all_data().astype(np.float32)
        x2 = loader2.get_all_data().astype(np.float32)
        x = np.concatenate([x1, x2], axis=0)
        labels = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))], axis=0)

        # Train/Validation split
        train_mask = (
            np.random.rand(len(x)) > self.train_args['validation_fraction'])
        x_train = x[train_mask]
        labels_train = labels[train_mask]
        x_val = x[~train_mask]
        labels_val = labels[~train_mask]

        # Create data loaders
        train_data = TensorDataset(torch.tensor(
            x_train), torch.tensor(labels_train))
        val_data = TensorDataset(torch.tensor(x_val), torch.tensor(labels_val))
        train_loader = DataLoader(
            train_data,
            batch_size=self.train_args['training_batch_size'],
            shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.train_args['training_batch_size'],
            shuffle=False, drop_last=True
        )

        # Create model
        ndim = x1.shape[-1]
        model = EvidenceNetworkSimple(
            ndim,
            self.layer_width,
            self.added_layers,
            self.batch_norm_flag,
            self.alpha
        )
        model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.train_args['learning_rate'])

        # Train model
        wait = 0
        summary = {'training_loss': [], 'validation_loss': []}
        best_model = model.state_dict()
        with tqdm(iter(range(self.train_args["max_epochs"])),
                  unit=' epochs') as tq:
            for epoch in tq:
                loss_train, loss_val = self._train_epoch(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer
                )
                tq.set_postfix(
                    loss1=loss_train-1,
                    loss_val1=loss_val-1,
                )
                summary['training_loss'].append(loss_train)
                summary['validation_loss'].append(loss_val)

                # check for convergence
                if loss_val < self.best_val:
                    self.best_val = loss_val
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
            summary['best_validation_loss'] = self.best_val
            summary['epochs_trained'] = epoch
        self.model = model
        # self.model.load_state_dict(best_model)
        return summary

    def predict(self, x):
        """Predict the log-Bayes Ratio for a given input."""
        if self.model is None:
            raise ValueError("Model has not been trained.")
        x = np.atleast_2d(x)
        x = torch.tensor(x.astype(np.float32)).to(self.device)
        return self.model(x)
