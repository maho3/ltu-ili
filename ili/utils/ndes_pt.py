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
from sklearn.preprocessing import StandardScaler
from torch.distributions import Distribution
from torch.distributions.transforms import (
    identity_transform, AffineTransform, Transform)

try:  # sbi > 0.22.0
    from sbi.neural_nets import posterior_nn
    from sbi import neural_nets
except ImportError:  # sbi <= 0.22.0
    from sbi import utils as neural_nets


def load_nde_sbi(
        engine: str,
        model: str,
        embedding_net: nn.Module = nn.Identity(),
        repeats=1,
        **model_args):
    """Load an nde from sbi.

    Args:
        engine (str): engine to use.
            One of: NPE, NLE, NRE, SNPE, SNLE, or SNRE.
        model (str): model to use.
            One of: mdn, maf, nsf, made, linear, mlp, resnet.
        embedding_net (nn.Module, optional): embedding network to use.
            Defaults to nn.Identity().
        repeats (int, optional): number of models to load. Defaults to 1.
        **model_args: additional arguments to pass to the model.
    """
    # load NRE models (linear, mlp, resnet)
    if 'NRE' in engine:
        if model not in ['linear', 'mlp', 'resnet']:
            raise ValueError(f"Model {model} not implemented for {engine}.")
        return [
            neural_nets.classifier_nn(
                model=model, embedding_net_x=embedding_net,
                **model_args) for _ in range(repeats)
        ]

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
    # Please use `from sbi.neural_nets import posterior_nn` in the future (not sbi.utils.posterior_nn)
    # Load NPE models (mdn, maf, nsf, made)
    if 'NPE' in engine:
        return [
            neural_nets.posterior_nn(
                model=model, embedding_net=embedding_net,
                **model_args) for _ in range(repeats)
        ]

    # Load NLE models (mdn, maf, nsf, made)
    if 'NLE' in engine:
        if not isinstance(embedding_net, nn.Identity):
            logging.warning(
                "Using an embedding_net with NLE models compresses theta, not "
                "x as might be expected.")
        return [
            neural_nets.likelihood_nn(
                model=model, embedding_net=embedding_net,
                **model_args) for _ in range(repeats)
        ]

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

    def _embed(self, x: torch.Tensor):
        """Precompute the per-observation context of the conditional flow.

        Split out of ``flow`` so that batched sampling runs the embedding
        network once for the whole batch, rather than once per accept/reject
        iteration. Subclasses whose conditional distribution is not built from
        a single embedding vector (e.g. moment networks) override this
        together with ``_flow_from_embedding``.
        """
        if hasattr(x, 'float'):
            x = x.float()
        return self.embedding_net(self.x_transform.inv(x)).float()

    def _flow_from_embedding(self, embedding, index: torch.Tensor):
        """Conditional flow for the subset ``index`` of a precomputed context."""
        return self.nde.flow(embedding[index])

    def sample(
        self,
        shape: tuple,
        x: torch.Tensor,
        show_progress_bars: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Accept-reject sampling of the posterior for a single observation.

        Args:
            shape (tuple): shape of the sample to draw
            x (torch.Tensor): single observation to condition on
            show_progress_bars (bool, optional): whether to show a progress
                bar. Defaults to True.
            **kwargs: forwarded to ``sample_batched``.

        Returns:
            torch.Tensor: samples of shape (*shape, npars)
        """
        if isinstance(shape, int):
            shape = (shape,)
        if np.prod(shape) == 0:
            return torch.empty(shape)

        if isinstance(x, (list, np.ndarray)):
            x = torch.Tensor(x)
        x = x.to(self._device)

        # a single observation is promoted to a batch of one
        if x.dim() == 1:
            x = x.unsqueeze(0)
        samples = self.sample_batched(
            shape, x, show_progress_bars=show_progress_bars, **kwargs)
        if samples.shape[-2] == 1:
            samples = samples.squeeze(-2)
        return samples

    def sample_batched(
        self,
        shape: tuple,
        x: torch.Tensor,
        show_progress_bars: bool = True,
        max_candidate_batch: int = 250_000,
        max_oversample: int = 1000,
    ) -> torch.Tensor:
        """Accept-reject sampling of the posterior for a batch of observations.

        All observations are drawn from simultaneously: each iteration
        conditions the flow on every observation that still needs samples and
        rejects candidates outside the prior support in one vectorized pass.
        Observations drop out of the batch as they fill up, and the number of
        candidates drawn per iteration adapts to the measured acceptance rate,
        so a poorly-constrained posterior costs extra candidates rather than
        extra sequential iterations.

        Args:
            shape (tuple): shape of the sample to draw per observation
            x (torch.Tensor): batch of observations, of shape (nobs, *x.shape)
            show_progress_bars (bool, optional): whether to show a progress
                bar. Defaults to True.
            max_candidate_batch (int, optional): cap on the total number of
                candidate parameter vectors drawn per iteration, summed over
                the observations still being sampled. Bounds peak memory.
                Defaults to 250,000.
            max_oversample (int, optional): give up on an observation once it
                has drawn this many times more candidates than the number of
                samples requested. Defaults to 1000.

        Returns:
            torch.Tensor: samples of shape (*shape, nobs, npars)
        """
        if isinstance(shape, int):
            shape = (shape,)

        if isinstance(x, (list, np.ndarray)):
            x = torch.Tensor(x)
        x = x.to(self._device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        nobs = x.shape[0]

        num_samples = int(np.prod(shape))
        if num_samples == 0 or nobs == 0:
            return torch.empty((*shape, nobs, 0))

        device = self._device
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                embedding = self._embed(x)

                # per-observation bookkeeping
                filled = torch.zeros(nobs, dtype=torch.long, device=device)
                drawn = torch.zeros(nobs, dtype=torch.long, device=device)
                naccept = torch.zeros(nobs, dtype=torch.long, device=device)
                failed = torch.zeros(nobs, dtype=torch.bool, device=device)
                active = torch.arange(nobs, device=device)
                samples = None  # allocated once the parameter dim is known

                pbar = tqdm(
                    disable=not show_progress_bars,
                    total=num_samples * nobs,
                    desc=f"Drawing {num_samples} posterior samples for "
                         f"{nobs} observation(s)",
                )

                flow = self._flow_from_embedding(embedding, active)
                # first iteration has no acceptance estimate yet, so ask for
                # exactly what is needed and let the estimate refine it
                batch_size = max(
                    1, min(num_samples, max_candidate_batch // nobs))

                while len(active) > 0:
                    candidates = self.theta_transform(
                        flow.sample((batch_size,)))  # (batch, nactive, npars)

                    # check if the dimensions have been reduced by the prior
                    raw_check = self.prior.support.check(candidates)
                    if raw_check.dim() == candidates.dim():
                        are_accepted = raw_check.all(dim=-1)
                    else:
                        are_accepted = raw_check

                    if samples is None:
                        samples = torch.empty(
                            (num_samples, nobs, candidates.shape[-1]),
                            dtype=candidates.dtype, device=candidates.device)

                    # scatter each observation's accepted draws into its slot,
                    # continuing from however many it had already accepted
                    offset = filled[active]
                    slot = are_accepted.long().cumsum(dim=0) - 1 + offset
                    keep = are_accepted & (slot < num_samples)
                    isamp, iact = keep.nonzero(as_tuple=True)
                    samples[slot[isamp, iact], active[iact]] = \
                        candidates[isamp, iact]

                    nnew = are_accepted.sum(dim=0)
                    filled[active] = torch.clamp(
                        offset + nnew, max=num_samples)
                    naccept[active] += nnew
                    drawn[active] += batch_size
                    pbar.update(int((filled[active] - offset).sum()))

                    # give up on observations whose posterior barely overlaps
                    # the prior support: 10x the draws the measured acceptance
                    # rate says are needed, and a hard cap for rates too low
                    # to estimate at all
                    rate = naccept[active].double() / drawn[active].double()
                    budget = torch.where(
                        rate > 0, 10 * num_samples / rate.clamp(min=1e-12),
                        torch.full_like(rate, float('inf')))
                    budget = budget.clamp(
                        max=float(max_oversample * num_samples))
                    give_up = ((filled[active] < num_samples) &
                               (drawn[active].double() > budget))

                    if give_up.any():
                        idx = active[give_up]
                        # match the single-observation behaviour: discard the
                        # partial draws and fall back to the prior
                        prior_samples = self.prior.sample(
                            (num_samples * len(idx),))
                        samples[:, idx] = prior_samples.reshape(
                            num_samples, len(idx), -1).to(
                                device=samples.device, dtype=samples.dtype)
                        failed[idx] = True
                        pbar.update(
                            int((num_samples - filled[idx]).sum()))

                    done = (filled[active] >= num_samples) | give_up
                    if done.any():
                        active = active[~done]
                        if len(active) == 0:
                            break
                        flow = self._flow_from_embedding(embedding, active)

                    # size the next draw from the measured acceptance rate.
                    # the median keeps well-behaved observations from
                    # over-drawing on behalf of the worst one; they finish and
                    # leave the batch, and the estimate rises for those left
                    rate = (naccept[active].double() /
                            drawn[active].double()).clamp(min=1e-12)
                    needed = (num_samples - filled[active]).double() / rate
                    target = int(1.2 * torch.median(needed).item()) + 1
                    cap = max(1, max_candidate_batch // len(active))
                    batch_size = int(min(max(target, min(cap, 32)), cap))

                pbar.close()

                if failed.any():
                    nfail = int(failed.sum())
                    rates = (naccept[failed].double() /
                             drawn[failed].double().clamp(min=1))
                    warnings.warn(
                        f"Direct sampling took too long for {nfail} of {nobs} "
                        "observation(s). The posterior is poorly constrained "
                        "within the prior support there (median acceptance "
                        f"rate: {rates.median():.4%}). Consider using emcee "
                        "sampling or using a larger prior support. Returning "
                        "prior samples for those observations."
                    )

                return samples.reshape(*shape, nobs, samples.shape[-1])
        finally:
            self.train(was_training)

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

    def _per_model_samples(self, num_samples: int, show_progress_bars: bool):
        """Number of samples to draw from each ensemble member."""
        per_model = torch.ceil(
            num_samples * self.weights / self.weights.sum())
        if show_progress_bars:
            logging.info(
                f"Sampling models with {per_model.int().tolist()} "
                "samples each.")
        return per_model

    def sample(
        self,
        shape: tuple,
        x: Any,
        show_progress_bars: bool = True,
        **kwargs
    ):
        if isinstance(shape, int):
            shape = (shape,)

        num_samples = np.prod(shape)
        per_model = self._per_model_samples(num_samples, show_progress_bars)

        # sample
        samples = torch.cat([
            nde.sample((int(N),), x, show_progress_bars=show_progress_bars,
                       **kwargs)
            for nde, N in zip(self.posteriors, per_model)
        ], dim=0)
        samples = samples[:num_samples]
        return samples.reshape(*shape, -1)

    def sample_batched(
        self,
        shape: tuple,
        x: Any,
        show_progress_bars: bool = True,
        **kwargs
    ):
        """Sample the ensemble for a batch of observations simultaneously.

        Args:
            shape (tuple): shape of the sample to draw per observation
            x (Any): batch of observations, of shape (nobs, *x.shape)

        Returns:
            torch.Tensor: samples of shape (*shape, nobs, npars)
        """
        if isinstance(shape, int):
            shape = (shape,)

        num_samples = np.prod(shape)
        per_model = self._per_model_samples(num_samples, show_progress_bars)

        # concatenate members along the sample axis, as in sample()
        samples = torch.cat([
            nde.sample_batched(
                (int(N),), x, show_progress_bars=show_progress_bars, **kwargs)
            for nde, N in zip(self.posteriors, per_model)
        ], dim=0)
        samples = samples[:num_samples]
        return samples.reshape(*shape, samples.shape[-2], samples.shape[-1])

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
    repeats=1,
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
        - moment: Moment Network (https://arxiv.org/abs/2011.05991). Not a flow;
            approximates the posterior as a single full-covariance Gaussian via
            an internal joint (mean + covariance) training procedure.
            Its own trunk is a constant-width MLP, configured like mdn:
            hidden_features, hidden_depth, activation. Any funnel-shaped
            compression belongs in embedding_net (e.g. embedding_net='fun'),
            not in this model's own architecture args.

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

    # Moment Networks (Jeffrey & Wandelt 2020) have a bespoke joint training
    # path and are built by a separate constructor. They are lampe-only
    # NPE models that approximate the posterior as a full-covariance Gaussian.
    if model == 'moment':
        from ili.inference.lampe_moment import _Lampe_Moment_Constructor
        moment_defaults = dict(
            hidden_features=64, hidden_depth=3, activation='relu')
        extra = set(model_args.keys()) - set(moment_defaults.keys())
        # warn (don't error) on flow-specific kwargs that don't apply here
        for k in ('num_transforms', 'num_components'):
            if k in extra:
                logging.warning(
                    f"Argument '{k}' does not apply to model 'moment' and "
                    "will be ignored.")
                extra.discard(k)
                model_args.pop(k)
        if extra:
            raise ValueError(
                f"Model moment arguments mispecified. Extra arguments found: "
                f"{extra}.")
        if 'activation' in model_args and \
                model_args['activation'] not in (
                    'relu', 'tanh', 'elu', 'gelu', 'silu', 'leaky_relu'):
            raise ValueError(
                f"Unknown activation '{model_args['activation']}' for model "
                "'moment'.")
        moment_args = {**moment_defaults, **model_args}
        embedding_net = deepcopy(embedding_net)
        return [
            _Lampe_Moment_Constructor(
                embedding_net, moment_args, device,
                x_normalize, theta_normalize) for _ in range(repeats)
        ]

    # check the model parameterizations
    if model == 'mdn':
        model_defaults = dict(hidden_features=16, hidden_depth=3,
                              num_components=3)
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
        model_args['hidden_features'] = [model_args['hidden_features']] * \
            model_args.pop('hidden_depth', 3)
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

    net_constructor = [
        _Lampe_Net_Constructor(
            flow_class, embedding_net, model_args,
            device, x_normalize, theta_normalize) for _ in range(repeats)
    ]

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

    def __call__(self, train_loader, prior):

        # pass data through embedding network
        x_batch, theta_batch = next(iter(train_loader))
        dtype = x_batch.dtype
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
            scaler = StandardScaler()
            for x_batch, _ in train_loader:
                x_batch = x_batch.cpu().numpy()
                scaler.partial_fit(x_batch)

            x_mean = torch.tensor(scaler.mean_, dtype=dtype).to(self.device)
            x_std = torch.tensor(scaler.scale_, dtype=dtype).to(self.device)

            # avoid division by zero
            x_std = torch.clamp(x_std, min=1e-16)

            # z-normalize x
            x_transform = AffineTransform(
                loc=x_mean, scale=x_std, event_dim=1)

        if self.theta_normalize:
            scaler = StandardScaler()
            for _, theta_batch in train_loader:
                theta_batch = theta_batch.cpu().numpy()
                scaler.partial_fit(theta_batch)

            theta_mean = torch.tensor(
                scaler.mean_, dtype=dtype).to(self.device)
            theta_std = torch.tensor(
                scaler.scale_, dtype=dtype).to(self.device)

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
