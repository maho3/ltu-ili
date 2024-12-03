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
import warnings
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
        if isinstance(self.nde.flow, zuko.flows.spline.NCSF):
            if (theta < -np.pi).any() or (theta > np.pi).any():
                raise ValueError(
                    "Encountered parameters outside of [-pi,pi]. "
                    "This is not supported by the chosen NDE, Neural Circular "
                    "Spline Flow (ncsf)."
                )

        # move them to device
        x = x.to(self._device)
        theta = theta.to(self._device)

        logprob = self.nde(
            self.theta_transform.inv(theta),
            self.embedding_net(self.x_transform.inv(x)))
        log_abs_det_jacobian = self.theta_transform.log_abs_det_jacobian(
            theta, theta  # just for shape
        )  # for Affine/IdentityTransform, this outputs a constant
        if len(log_abs_det_jacobian.shape) > 1:
            # this happens with the identity_transform, but it should be
            # equivalent to a scalar. See: https://github.com/pytorch/pytorch/blob/5c2584a14c2283514703a17cba0a57c8bfb0d977/torch/distributions/transforms.py#L363
            log_abs_det_jacobian = log_abs_det_jacobian.sum(dim=1)
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
        if isinstance(shape, int):
            shape = (shape,)

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
        tries = 0
        while num_remaining > 0:
            candidates = self.theta_transform(
                self.flow(x).sample((batch_size,)))
            are_accepted = self.prior.support.check(candidates)
            samples = candidates[are_accepted]
            accepted.append(samples)

            num_remaining -= len(samples)
            pbar.update(len(samples))
            tries += 1
            if tries > 10*len(samples)/batch_size:  # 10x the expected number of tries
                warnings.warn(
                    "Direct sampling took too long. The posterior is poorly "
                    "constrained within the prior support. Consider using "
                    "emcee sampling or using a larger prior support. Returning"
                    " prior samples.")
                return self.prior.sample(shape)
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
        if isinstance(shape, int):
            shape = (shape,)

        # determine number of samples per model
        num_samples = np.prod(shape)
        per_model = torch.round(
            num_samples * self.weights/self.weights.sum())
        if show_progress_bars:
            logging.info(
                f"Sampling models with {per_model.int().tolist()} "
                "samples each.")

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
    engine: str = 'NPE',
    **model_args
):
    """Load an nde from lampe.
    Models include:
        - mdn: Mixture Density Network (https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf)
        - maf: Masked Autoregressive Flow (https://arxiv.org/abs/1705.07057)
        - nsf: Neural Spline Flow (https://arxiv.org/abs/1906.04032)
        - ncsf: Neural Circular Spline Flow (https://arxiv.org/abs/2002.02428)
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
        engine (str, optional): dummy argument to match sbi interface.
            Must be set to 'NPE' or will be overwritten.
        **model_args: additional arguments to pass to the model.
    """
    if 'NPE' not in engine:
        raise ValueError(
            f'Engine {engine} not supported in lampe backend. '
            'You probably meant to specify engine="NPE" or to use the NLE or NRE'
            ' engines in the sbi or pydelfi backends.')

    model = model.lower()

    # check the model parameterizations
    if model == 'mdn':
        model_defaults = dict(hidden_features=16, num_components=3)
    else:
        model_defaults = dict(hidden_features=16, num_transforms=2)
    if not (set(model_args.keys()) <= set(model_defaults.keys())):
        raise ValueError(
            f"Model {model} arguments mispecified. Extra arguments found: "
            f"{set(model_args.keys()) - set(model_defaults.keys())}.")

    # set defaults
    model_args = {**model_defaults, **model_args}

    # setup models
    if model == 'mdn':  # for mixture density networks
        model_args['hidden_features'] = [model_args['hidden_features']] * 3
        model_args['components'] = model_args.pop('num_components', 2)
        flow_class = zuko.flows.mixture.GMM
    else:
        if model == 'cnf':  # for continuous flow models
            # number of time embeddings
            model_args['hidden_features'] = [
                model_args['hidden_features']] * 2
            model_args['freqs'] = model_args.pop('num_transforms', 2)
            flow_class = zuko.flows.continuous.CNF
        else:  # for all discrete flow models
            model_args['hidden_features'] = [
                model_args['hidden_features']] * 2
            model_args['transforms'] = model_args.pop('num_transforms', 2)

            if model == 'maf':
                flow_class = zuko.flows.autoregressive.MAF
            elif model == 'nsf':
                flow_class = zuko.flows.spline.NSF
            elif model == 'ncsf':
                logging.warning(
                    "You've selected a Neural Circular Spline Flow, for "
                    "which parameters are expected to be restricted to [-pi,pi]."
                )
                flow_class = zuko.flows.spline.NCSF
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
            else:
                raise ValueError(f"Model {model} not implemented.")

    embedding_net = deepcopy(embedding_net)

    net_constructor = _Lampe_Net_Constructor(
        flow_class, embedding_net, model_args,
        device, x_normalize, theta_normalize)

    return net_constructor


class _Lampe_Net_Constructor():
    """
    Simple, functional wrapper to add an embedding network
    to a Lampe NPE model.
    Attributes:
        flow_class (class): The class of the flow model to be used.
        embedding_net (torch.nn.Module): The embedding network to process input data.
        model_args (dict): Arguments to be passed to the flow model.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
        x_normalize (bool): Whether to normalize the input data.
        theta_normalize (bool): Whether to normalize the parameter data.
    Methods:
        __call__(x_batch, theta_batch, prior):
            Constructs and returns a LampeNPE model with the given data and prior.
            Args:
                x_batch (torch.Tensor): Batch of input data.
                theta_batch (torch.Tensor): Batch of parameter data.
                prior (torch.distributions.Distribution): Prior distribution for the parameters.
            Returns:
                LampeNPE: An instance of the LampeNPE model.
    """

    def __init__(self, flow_class, embedding_net, model_args,
                 device, x_normalize, theta_normalize):
        self.flow_class = flow_class
        self.embedding_net = embedding_net
        self.model_args = model_args
        self.device = device
        self.x_normalize = x_normalize
        self.theta_normalize = theta_normalize

    def to(self, device):
        self.device = device
        return self

    def __print__(self):
        return (
            f"This is a constructor for a Lampe NPE model with the following attributes:\n"
            f"Flow Class: {self.flow_class}\n"
            f"Embedding Network: {self.embedding_net}\n"
            f"Model Arguments: {self.model_args}\n"
            f"Device: {self.device}\n"
        )

    def __call__(self, x_batch, theta_batch, prior):

        # pass data through embedding network
        z_batch = self.embedding_net(x_batch.cpu())
        self.embedding_net = self.embedding_net.to(self.device)
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
            build=self.flow_class,
            **self.model_args
        ).to(self.device)

        # determine transformations
        x_transform = identity_transform
        theta_transform = identity_transform

        if self.x_normalize:
            x_mean = x_batch.mean(dim=0).to(self.device)
            x_std = x_batch.std(dim=0).to(self.device)

            # avoid division by zero
            x_std = torch.clamp(x_std, min=1e-16)

            # z-normalize x
            x_transform = AffineTransform(
                loc=x_mean, scale=x_std, event_dim=1)

        if self.theta_normalize:
            theta_mean = theta_batch.mean(dim=0).to(self.device)
            theta_std = theta_batch.std(dim=0).to(self.device)

            # avoid division by zero
            theta_std = torch.clamp(theta_std, min=1e-16)

            # z-normalize theta
            theta_transform = AffineTransform(
                loc=theta_mean, scale=theta_std, event_dim=1)
        npe = LampeNPE(
            nde=nde,
            embedding_net=self.embedding_net,
            prior=prior,
            x_transform=x_transform,
            theta_transform=theta_transform
        ).to(self.device)
        return npe
