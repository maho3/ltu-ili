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
        if embedding_net != nn.Identity():
            logging.warning(
                "Using an embedding_net with NLE models compresses theta, not "
                "x as might be expected.")
        return sbi.utils.likelihood_nn(
            model=model, embedding_net=embedding_net, **model_args)

    raise ValueError(f"Engine {engine} not implemented.")


class NPEWithEmbedding(nn.Module):
    """Simple wrapper to add an embedding network to an NPE model."""

    def __init__(self, nde, embedding_net=nn.Identity()):
        super().__init__()
        self.nde = nde
        self.embedding_net = embedding_net

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.nde(theta, self.embedding_net(x))

    def flow(self, x: torch.Tensor):  # -> Distribution
        return self.nde.flow(self.embedding_net(x))

    def sample(self, x: torch.Tensor, num_samples: int = 1):
        return self.flow(x).sample((num_samples,)).cpu()


class LampeEnsemble(nn.Module):
    """Simple module to wrap an ensemble of NPE models."""

    def __init__(self, npes: NPEWithEmbedding, weights: torch.Tensor, theta_transform=None):
        super().__init__()
        self.npes = nn.ModuleList(npes)
        self.weights = weights
        assert len(self.npes) == len(self.weights)
        self.theta_transform = theta_transform

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            weight * nde(theta, x)
            for weight, nde in zip(self.weights, self.npes)
        ], dim=-1)

    def sample(self, shape: tuple, x0: torch.Tensor):
        num_samples = np.prod(shape)
        per_model = torch.round(
            num_samples * self.weights/self.weights.sum()).numpy().astype(int)
        samples = torch.cat([
            nde.sample(x0, num_samples=N)
            for nde, N in zip(self.npes, per_model)
        ], dim=0)
        if self.theta_transform is not None:
            samples = self.theta_transform(samples)
        return samples.reshape(*shape, -1)

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.theta_transform is not None:
            theta = self.theta_transform.inv(theta)
        return self.forward(theta, x).sum(dim=-1).detach()


def load_nde_lampe(
        model: str,
        embedding_net: nn.Module = nn.Identity(),
        **model_args):
    """Load an nde from lampe.

    Args:
        model (str): model to use.
            One of: mdn, maf, nsf, made, linear, mlp, resnet.
        embedding_net (nn.Module, optional): embedding network to use.
            Defaults to nn.Identity().
        **model_args: additional arguments to pass to the model.
    """
    if model == 'mdn':
        if not (set(model_args.keys()) <= {'hidden_features', 'num_components'}):
            raise ValueError(f"Model {model} arguments mispecified.")
        model_args['hidden_features'] = [model_args['hidden_features']] * 3
        model_args['components'] = model_args.pop('num_components', 2)
        flow_class = zuko.flows.mixture.GMM
    else:
        if not (set(model_args.keys()) <= {'hidden_features', 'num_transforms'}):
            raise ValueError(f"Model {model} arguments mispecified.")
        model_args['hidden_features'] = [
            model_args['hidden_features']] * 2
        model_args['transforms'] = model_args.pop('num_transforms', 2)

    if model == 'maf':
        flow_class = zuko.flows.autoregressive.MAF

    def net_constructor(x_batch, theta_batch):
        z_batch = embedding_net(x_batch)
        z_shape = z_batch.shape[1:]
        theta_shape = theta_batch.shape[1:]

        if (len(z_shape) > 1):
            raise ValueError("Embedding network must return a vector.")
        if (len(theta_shape) > 1):
            raise ValueError("Parameters theta must be a vector.")

        nde = lampe.inference.NPE(
            theta_dim=theta_shape[0],
            x_dim=z_shape[0],
            build=flow_class,
            **model_args
        )
        return NPEWithEmbedding(
            nde=nde,
            embedding_net=embedding_net
        )

    return net_constructor
