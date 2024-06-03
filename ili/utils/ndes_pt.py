"""
Module to provide loading functions for ndes in various backends.

All Mixture Density Networks (mdn) have the configuration:
    hidden_features (int): width of hidden layers (each MDN has 3 hidden layers)
    num_components (int): number of Gaussian components in the mixture model

All flow-based models (maf, nsf, made) have the configuration:
    hidden_features (int): width of hidden layers in the coupling layers
    num_transforms (int): number of coupling layers

Linear classifiers (linear) have no arguments.

MLP and ResNet classifiers (mlp, resnet) have the configuration:
    hidden_features (int): width of hidden layers (each has 2 hidden layers)
"""

import logging

import numpy as np
import sbi
import torch
from torch import nn
import lampe
import zuko
from tqdm import tqdm
from typing import List, Any, Optional
from copy import deepcopy
from torch.distributions import Distribution
from torch.distributions.transforms import (
    identity_transform, AffineTransform, Transform)


def load_nde_sbi(
        engine: str,
        model: str,
        embedding_net: nn.Module = nn.Identity(),
        **model_args):
    """Load an nde from sbi.

    Args:
        engine (str): engine to use. 
            One of: NPE, NLE, NRE, SNPE, SNLE, or SNRE.
        model (str): model to use. 
            One of: mdn, maf, nsf, made, linear, mlp, resnet.
        embedding_net (nn.Module, optional): embedding network to use.
            Defaults to nn.Identity().
        **model_args: additional arguments to pass to the model.
    """
    # load NRE models (linear, mlp, resnet)
    if 'NRE' in engine:
        if model not in ['linear', 'mlp', 'resnet']:
            raise ValueError(f"Model {model} not implemented for {engine}.")
        return sbi.utils.classifier_nn(
            model=model, embedding_net_x=embedding_net, **model_args)

    if model not in ['mdn', 'maf', 'nsf', 'made']:
        raise ValueError(f"Model {model} not implemented for {engine}.")

    if (model == 'mdn'):
        # check for arguments
        if not (set(model_args.keys()) <= {'hidden_features', 'num_components'}):
            raise ValueError(f"Model {model} arguments mispecified.")
    else:
        # check for arguments
        if not (set(model_args.keys()) <= {'hidden_features', 'num_transforms'}):
            raise ValueError(f"Model {model} arguments mispecified.")

    # Load NPE models (mdn, maf, nsf, made)
    if 'NPE' in engine:
        return sbi.utils.posterior_nn(
            model=model, embedding_net=embedding_net, **model_args)

    # Load NLE models (mdn, maf, nsf, made)
    if 'NLE' in engine:
        if not isinstance(embedding_net, nn.Identity):
            logging.warning(
                "Using an embedding_net with NLE models compresses theta, not "
                "x as might be expected.")
        return sbi.utils.likelihood_nn(
            model=model, embedding_net=embedding_net, **model_args)

    raise ValueError(f"Engine {engine} not implemented.")


class LampeNPE(nn.Module):
    """Simple wrapper to add an embedding network to an NPE model."""

    def __init__(
        self,
        nde: nn.Module,
        prior: Distribution,
        embedding_net: nn.Module = nn.Identity(),
        x_transform: Transform = identity_transform,
        theta_transform: Transform = identity_transform
    ):
        super().__init__()
        self.nde = nde
        self.prior = prior
        self.embedding_net = embedding_net
        self.x_transform = x_transform
        self.theta_transform = theta_transform
        self._device = 'cpu'
        self.max_sample_size = 1000

    def forward(
        self,
        theta: torch.Tensor,
        x: Any
    ) -> torch.Tensor:
        # check inputs
        if isinstance(x, (list, np.ndarray)):
            x = torch.Tensor(x)
        if isinstance(theta, (list, np.ndarray)):
            theta = torch.Tensor(theta)
        x = x.to(self._device)
        theta = theta.to(self._device)

        logprob = self.nde(
            self.theta_transform.inv(theta),
            self.embedding_net(self.x_transform.inv(x)))
        log_abs_det_jacobian = self.theta_transform.log_abs_det_jacobian(
            theta, theta  # just for shape
        )  # for Affine/IdentityTransform, this outputs a constant
        return logprob - log_abs_det_jacobian

    potential = forward

    def flow(self, x: torch.Tensor):  # -> Distribution
        if hasattr(x, 'float'):
            x = x.float()
        return self.nde.flow(
            self.embedding_net(self.x_transform.inv(x)).float())

    def sample(
        self,
        shape: tuple,
        x: torch.Tensor,
        show_progress_bars: bool = True
    ) -> torch.Tensor:
        """Accept-reject sampling"""

        # check inputs
        if isinstance(x, (list, np.ndarray)):
            x = torch.Tensor(x)
        x = x.to(self._device)

        # sample
        num_samples = np.prod(shape)
        if num_samples == 0:
            return torch.empty(shape)
        pbar = tqdm(
            disable=not show_progress_bars,
            total=num_samples,
            desc=f"Drawing {num_samples} posterior samples",
        )

        batch_size = min(self.max_sample_size, num_samples)
        num_remaining = num_samples
        accepted = []
        while num_remaining > 0:
            candidates = self.theta_transform(
                self.flow(x).sample((batch_size,)))
            are_accepted = self.prior.support.check(candidates)
            samples = candidates[are_accepted]
            accepted.append(samples)

            num_remaining -= len(samples)
            pbar.update(len(samples))
        pbar.close()

        samples = torch.cat(accepted, dim=0)[:num_samples]
        return samples.reshape(*shape, -1)

    def to(self, device):
        self._device = device
        return super().to(device)


class LampeEnsemble(nn.Module):
    """Simple module to wrap an ensemble of NPE models."""

    def __init__(
        self,
        posteriors: List[LampeNPE],
        weights: torch.Tensor
    ):
        super().__init__()
        self.posteriors = nn.ModuleList(posteriors)
        self.weights = weights
        assert len(self.posteriors) == len(self.weights)
        self.prior = self.posteriors[0].prior
        self._device = posteriors[0]._device
        self.num_components = len(self.posteriors)

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            weight * npe(theta, x)
            for weight, npe in zip(self.weights, self.posteriors)
        ], dim=-1)

    potential = forward

    def sample(
        self,
        shape: tuple,
        x: Any,
        show_progress_bars: bool = True
    ):
        # determine number of samples per model
        num_samples = np.prod(shape)
        per_model = torch.round(
            num_samples * self.weights/self.weights.sum())  # .numpy().astype(int)
        if show_progress_bars:
            logging.info(f"Sampling models with {per_model} samples each.")

        # sample
        samples = torch.cat([
            nde.sample((int(N),), x, show_progress_bars=show_progress_bars)
            for nde, N in zip(self.posteriors, per_model)
        ], dim=0)
        samples = samples[:num_samples]
        return samples.reshape(*shape, -1)

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.forward(theta, x).sum(dim=-1).detach()

    def to(self, device):
        self._device = device
        return super().to(device)


def load_nde_lampe(
    model: str,
    embedding_net: nn.Module = nn.Identity(),
    device: Optional[str] = 'cpu',
    x_normalize: bool = True,
    theta_normalize: bool = True,
    **model_args
):
    """Load an nde from lampe.
    Models include:
        - mdn: Mixture Density Network (https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf)
        - maf: Masked Autoregressive Flow (https://arxiv.org/abs/1705.07057)
        - nsf: Neural Spline Flow (https://arxiv.org/abs/1906.04032)
        - cnf: Continuous Normalizing Flow (https://arxiv.org/abs/1810.01367)
        - nice: Non-linear Independent Components Estimation (https://arxiv.org/abs/1410.8516)
        - gf: Gaussianization Flow (https://arxiv.org/abs/2003.01941)
        - sospf: Sum-of-Squares Polynomial Flow (https://arxiv.org/abs/1905.02325)
        - naf: Neural Autoregressive Flow (https://arxiv.org/abs/1804.00779)
        - unaf: Unconstrained Neural Autoregressive Flow (https://arxiv.org/abs/1908.05164)

    For more info, see zuko at https://zuko.readthedocs.io/en/stable/index.html

    Args:
        model (str): model to use.
            One of: mdn, maf, nsf, ncsf, cnf, nice, sospf, gf, naf.
        embedding_net (nn.Module, optional): embedding network to use.
            Defaults to nn.Identity().
        device (str, optional): device to use. Defaults to 'cpu'.
        x_normalize (bool, optional): whether to z-normalize x.
            Defaults to True.
        theta_normalize (bool, optional): whether to z-normalize theta.
            Defaults to True.
        **model_args: additional arguments to pass to the model.
    """
    if model == 'mdn':  # for mixture density networks
        if not (set(model_args.keys()) <= {'hidden_features', 'num_components'}):
            raise ValueError(f"Model {model} arguments mispecified.")
        model_args['hidden_features'] = [model_args['hidden_features']] * 3
        model_args['components'] = model_args.pop('num_components', 2)
        flow_class = zuko.flows.mixture.GMM
    elif model == 'cnf':  # for continuous flow models
        # number of time embeddings
        model_args['hidden_features'] = [
            model_args['hidden_features']] * 2
        model_args['freqs'] = model_args.pop('num_transforms', 2)
        flow_class = zuko.flows.continuous.CNF
    else:  # for all discrete flow models
        if not (set(model_args.keys()) <= {'hidden_features', 'num_transforms'}):
            raise ValueError(f"Model {model} arguments mispecified.")
        model_args['hidden_features'] = [
            model_args['hidden_features']] * 2
        model_args['transforms'] = model_args.pop('num_transforms', 2)

        if model == 'maf':
            flow_class = zuko.flows.autoregressive.MAF
        elif model == 'nsf':
            flow_class = zuko.flows.spline.NSF
        elif model == 'nice':
            flow_class = zuko.flows.coupling.NICE
        elif model == 'gf':
            flow_class = zuko.flows.gaussianization.GF
        elif model == 'sospf':
            flow_class = zuko.flows.polynomial.SOSPF
        elif model == 'naf':
            flow_class = zuko.flows.neural.NAF
        elif model == 'unaf':
            flow_class = zuko.flows.neural.UNAF

    embedding_net = deepcopy(embedding_net)

    def net_constructor(x_batch, theta_batch, prior):
        if hasattr(embedding_net, 'initalize_model'):
            embedding_net.initalize_model(x_batch.shape[-1])

        # pass data through embedding network
        z_batch = embedding_net(x_batch)
        z_shape = z_batch.shape[1:]
        theta_shape = theta_batch.shape[1:]

        if (len(z_shape) > 1):
            raise ValueError("Embedding network must return a vector.")
        if (len(theta_shape) > 1):
            raise ValueError("Parameters theta must be a vector.")

        # instantiate a neural density estimator
        nde = lampe.inference.NPE(
            theta_dim=theta_shape[0],
            x_dim=z_shape[0],
            build=flow_class,
            **model_args
        ).to(device)

        # determine transformations
        x_transform = identity_transform
        theta_transform = identity_transform

        if x_normalize:
            x_mean = x_batch.mean(dim=0).to(device)
            x_std = x_batch.std(dim=0).to(device)

            # avoid division by zero
            x_std[x_std == 0] = 1
            x_std = torch.clamp(x_std, min=1e-16)

            # z-normalize x
            x_transform = AffineTransform(
                loc=x_mean, scale=x_std, event_dim=1)

        if theta_normalize:
            theta_mean = theta_batch.mean(dim=0).to(device)
            theta_std = theta_batch.std(dim=0).to(device)

            # avoid division by zero
            theta_std[theta_std == 0] = 1
            theta_std = torch.clamp(theta_std, min=1e-16)

            # z-normalize theta
            theta_transform = AffineTransform(
                loc=theta_mean, scale=theta_std, event_dim=1)

        npe = LampeNPE(
            nde=nde,
            embedding_net=embedding_net,
            prior=prior,
            x_transform=x_transform,
            theta_transform=theta_transform
        ).to(device)
        return npe

    return net_constructor
