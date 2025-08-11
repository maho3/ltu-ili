"""
Module to contain evidence estimators.

TODO:
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
from .utils import ExpLoss, EvidenceNetwork

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

    def get_ln_bayes_factor(self, ev2):
        if self.evidence is None:
            raise ValueError('Evidence has not been computed.')
        if ev2.evidence is None:
            raise ValueError('Evidence for model 2 has not been computed.')
        return hm.evidence.compute_ln_bayes_factor(self.evidence, ev2.evidence)


class K_EvidenceNetwork():
    """
    A runner class to train an EvidenceNetwork for approximating the Bayes Factor (K)
    between two competing models.

    This class handles data preparation, training loops, validation, early stopping,
    and model saving. It can optionally use a custom embedding network to process
    complex data modalities before the main fully-connected layers.

    Args:
        embedding_net (nn.Module, optional): A PyTorch module to preprocess input
            data. If None, an identity mapping is used. Defaults to None.
        layer_width (int, optional): The number of neurons in the hidden layers of
            the EvidenceNetwork. Defaults to 16.
        added_layers (int, optional): The number of residual blocks in the
            EvidenceNetwork. Defaults to 3.
        batch_norm_flag (int, optional): A flag to enable (1) or disable (0)
            batch normalization. Defaults to 1.
        alpha (int, optional): The exponent for the leaky_parity_odd_power
            activation function. Defaults to 2.
        train_args (dict, optional): A dictionary of training hyperparameters to
            override the defaults. Defaults to {}.
        device (str, optional): The device to run training on (e.g., 'cpu' or
            'cuda'). Defaults to 'cpu'.

    Attributes:
        model (EvidenceNetwork): The trained PyTorch model. This is None until
            the `train` method is successfully called.
        best_val (float): The best validation loss achieved during training.
        loss_fn (nn.Module): The loss function instance used for training.
    """

    def __init__(
        self,
        embedding_net=None,
        layer_width=16,
        added_layers=3,
        batch_norm_flag=1,
        alpha=2,
        train_args={},
        device='cpu'
    ):
        self.embedding_net = embedding_net or nn.Identity()
        self.layer_width = layer_width
        self.added_layers = added_layers
        self.batch_norm_flag = batch_norm_flag
        self.alpha = alpha
        self.train_args = dict(
            training_batch_size=32, learning_rate=1e-5,
            stop_after_epochs=30, clip_max_norm=5,
            max_epochs=int(1e4),
            validation_fraction=0.1)
        self.train_args.update(train_args)
        self.device = device

        self.embedding_net.to(self.device)

        self.loss_fn = ExpLoss()
        self.best_val = float('inf')
        self.model = None

    def _loss(self, model, theta, x):
        """Compute the loss function for a given model."""
        logK = model(x)
        return self.loss_fn(logK, theta.view(-1, 1))

    def _train_epoch(self, model, train_loader, val_loader, optimizer):
        """Train a single epoch of a neural network model."""
        model.train()
        loss_train, count_train = [], 0
        for x, theta in train_loader:
            x, theta = x.to(self.device), theta.to(self.device)
            optimizer.zero_grad()
            loss = self._loss(model, theta, x)
            loss.backward()

            norm = nn.utils.clip_grad_norm_(
                model.parameters(), self.train_args['clip_max_norm'])

            if torch.isfinite(norm):
                optimizer.step()
            # Record
            loss_train.append(loss.item() * len(theta))
            count_train += len(theta)
        loss_train = sum(loss_train) / count_train

        model.eval()
        with torch.no_grad():
            loss_val, count_val = [], 0
            for x, theta in val_loader:
                x, theta = x.to(self.device), theta.to(self.device)
                loss_val.append(self._loss(
                    model, theta, x).item() * len(theta))
                count_val += len(theta)
        loss_val = sum(loss_val) / count_val
        return loss_train, loss_val

    def train(self, loader1, loader2, show_progress_bars=True):
        """
        Trains the evidence network.

        Args:
            loader1: A data loader object for the first model. Must have a
                     `get_all_data()` method.
            loader2: A data loader object for the second model.
            show_progress_bars (bool): Whether to display tqdm progress bars.
        """
        # Aggregate data from different models, and label them
        # Assumes loaders have a 'get_all_data' method returning numpy arrays
        x1 = loader1.get_all_data().astype(np.float32)
        x2 = loader2.get_all_data().astype(np.float32)
        x = np.concatenate([x1, x2], axis=0)
        labels = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))], axis=0)

        # Train/Validation split
        indices = np.random.permutation(len(x))
        val_size = int(len(x) * self.train_args['validation_fraction'])
        val_indices, train_indices = indices[:val_size], indices[val_size:]
        x_train, labels_train = x[train_indices], labels[train_indices]
        x_val, labels_val = x[val_indices], labels[val_indices]

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

        # Determine the input dimension for the dense network
        if not isinstance(self.embedding_net, nn.Identity):
            with torch.no_grad():
                # Use a single-item batch to find the output dimension
                sample_input = next(iter(train_loader))[0].to(self.device)
                embedded_output = self.embedding_net(sample_input)
                ndim = embedded_output.shape[-1]
        else:
            # If no embedding net, the input dim is from the data itself
            ndim = x_train.shape[-1]

        # Create model, passing the embedding net
        model = EvidenceNetwork(
            input_size=ndim,
            embedding_net=self.embedding_net,
            layer_width=self.layer_width,
            added_layers=self.added_layers,
            batch_norm_flag=self.batch_norm_flag,
            alpha=self.alpha
        )
        model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.train_args['learning_rate'])

        # Train model
        wait = 0
        summary = {'training_loss': [], 'validation_loss': []}
        best_model_state = deepcopy(model.state_dict())

        with tqdm(iter(range(self.train_args["max_epochs"])), unit=' epochs',
                  disable=not show_progress_bars) as tq:
            for epoch in tq:
                loss_train, loss_val = self._train_epoch(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer
                )
                if show_progress_bars:
                    tq.set_postfix(
                        loss_train=f"{loss_train:.4f}",
                        loss_val=f"{loss_val:.4f}")

                summary['training_loss'].append(loss_train)
                summary['validation_loss'].append(loss_val)

                if loss_val < self.best_val:
                    self.best_val = loss_val
                    best_model_state = deepcopy(model.state_dict())
                    wait = 0
                elif wait >= self.train_args["stop_after_epochs"]:
                    logging.info(f"Stopping early after {epoch} epochs.")
                    break
                else:
                    wait += 1
            else:
                logging.warning(
                    f"Training did not converge in {self.train_args['max_epochs']} epochs.")

        summary['best_validation_loss'] = self.best_val
        summary['epochs_trained'] = epoch + 1
        self.model = model
        self.model.load_state_dict(best_model_state)
        return summary

    def predict(self, x):
        """Predict the log-Bayes Ratio for a given input."""
        if self.model is None:
            raise ValueError(
                "Model has not been trained yet. Please call .train() first.")
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(np.atleast_2d(
                x).astype(np.float32)).to(self.device)
            return self.model(x_tensor).cpu().numpy()
