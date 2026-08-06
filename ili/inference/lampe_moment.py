"""
Standalone trainer and estimator for Moment Networks
(Jeffrey & Wandelt 2020, arXiv:2011.05991) in the lampe backend.

A Moment Network does not learn a full joint density like the zuko flows that
the lampe backend usually wraps. Instead it directly regresses the posterior
mean and covariance, with the mean network (f_mu) and covariance network
(f_cov) optimized jointly in a single pass:

    f_mu(x)  -> mu
    f_cov(x) -> raw Cholesky entries -> L -> Sigma = L L^T

    resid  = theta - mu.detach()        # stop-gradient: cov term doesn't
                                         # backprop into f_mu
    target = vech(resid resid^T)        # residual products, i<=j
    loss   = logsum_sq(theta - mu) + logsum_sq(vech(Sigma) - target)

where ``logsum_sq(err) = sum_j log(sum_batch(err_j^2) + floor)``: a per-output
loss that sums squared error over the batch *before* taking a log, rather than
averaging error directly (plain MSE). This keeps gradients comparably scaled
across output dimensions (mean components and covariance entries alike) even
though their natural magnitudes differ substantially, which is what makes the
joint fit converge reliably in practice.

Because E[resid resid^T | x] = Sigma(x), driving vech(Sigma) toward the
residual-product targets pushes L L^T towards the true conditional covariance,
while the Cholesky parameterization (lower triangular, softplus on the
diagonal) guarantees Sigma is positive-definite by construction.

The resulting posterior is a single (full-covariance) Gaussian approximation.
It is *not* a flexible density -- it is unimodal and Gaussian -- so on its own
it can only capture Gaussian posteriors. It is most useful as one member of an
ensemble alongside real flows (maf/nsf/...), where it is sampled and log_prob'd
identically via a torch MultivariateNormal.

This module is self-contained: the joint training lives entirely inside
``train_moment_network``, and the trained ``MomentNetworkEstimator`` is a
drop-in ensemble member (it subclasses ``LampeNPE`` to reuse its accept-reject
sampling and device handling).
"""

import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, MultivariateNormal
from torch.distributions.transforms import (
    identity_transform, AffineTransform)
from sklearn.preprocessing import StandardScaler

from ili.utils.ndes_pt import LampeNPE

_ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'elu': nn.ELU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'leaky_relu': nn.LeakyReLU,
}


def _make_mlp(in_dim, out_dim, hidden_features, hidden_depth, activation):
    """Build a simple constant-width MLP trunk with ``hidden_depth`` hidden
    layers of width ``hidden_features`` (mirrors the mdn model's own
    architecture args). Any funnel-shaped compression belongs upstream, in
    the embedding_net (e.g. embedding_net='fun'), not in this trunk."""
    act = _ACTIVATIONS[activation]
    layers, d = [], in_dim
    for _ in range(hidden_depth):
        layers += [nn.Linear(d, hidden_features), act()]
        d = hidden_features
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class MomentNetworkEstimator(LampeNPE):
    """A trained Moment Network posterior, interchangeable with lampe flows.

    Predicts a full-covariance Gaussian posterior N(mu(x), Sigma(x)) with
    Sigma = L L^T for a lower-triangular Cholesky factor L. Subclasses
    ``LampeNPE`` so that ``sample`` (accept-reject within prior support),
    ``to``, and the ensemble interface are inherited unchanged; only the
    conditional distribution (``flow``) and ``forward`` (log_prob) are
    overridden.

    All internal networks operate in normalized theta/x space; the
    ``theta_transform`` maps predictions back to physical parameter space
    (matching how ``LampeNPE`` handles normalization).
    """

    is_moment = True

    def __init__(
        self,
        emb_mu: nn.Module,
        mu_net: nn.Module,
        emb_cov: nn.Module,
        cov_net: nn.Module,
        n_params: int,
        prior: Distribution,
        x_transform=identity_transform,
        theta_transform=identity_transform,
    ):
        nn.Module.__init__(self)
        self.emb_mu = emb_mu
        self.mu_net = mu_net
        self.emb_cov = emb_cov
        self.cov_net = cov_net
        self.n_params = n_params
        self.prior = prior
        self.x_transform = x_transform
        self.theta_transform = theta_transform
        self._device = 'cpu'
        self.max_sample_size = 1000

        # lower-triangular (row-major, i>=j) index helpers, incl. diagonal
        rows, cols = torch.tril_indices(n_params, n_params)
        self.register_buffer('_tril_rows', rows)
        self.register_buffer('_tril_cols', cols)
        self.register_buffer('_is_diag', (rows == cols))

    @property
    def n_cov_outputs(self) -> int:
        return self.n_params * (self.n_params + 1) // 2

    def _build_L(self, raw: torch.Tensor) -> torch.Tensor:
        """Map a flat (B, n(n+1)/2) network output to a valid Cholesky factor.

        The diagonal entries are passed through a softplus (plus a small floor)
        to guarantee strict positivity, so ``L @ L.T`` is always
        positive-definite. Off-diagonal entries are used as-is.
        """
        B = raw.shape[0]
        n = self.n_params
        # softplus + floor on the diagonal entries only; leave off-diag as-is
        vals = torch.where(
            self._is_diag.unsqueeze(0),
            F.softplus(raw) + 1e-6,
            raw,
        )
        L = raw.new_zeros(B, n, n)
        L[:, self._tril_rows, self._tril_cols] = vals
        return L

    def predict_moments(self, x):
        """Return (mu, L) in normalized theta space for input data x."""
        if isinstance(x, (list, np.ndarray)):
            x = torch.Tensor(x)
        x = x.to(self._device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        xin = self.x_transform.inv(x).float()
        mu = self.mu_net(self.emb_mu(xin))
        L = self._build_L(self.cov_net(self.emb_cov(xin)))
        return mu, L

    def flow(self, x):  # -> Distribution (normalized theta space)
        """Conditional posterior as a MultivariateNormal in normalized space."""
        mu, L = self.predict_moments(x)
        return MultivariateNormal(mu, scale_tril=L)

    def forward(self, theta: torch.Tensor, x) -> torch.Tensor:
        """Log-probability of theta under the Gaussian posterior given x.

        Mirrors ``LampeNPE.forward``: evaluates in normalized theta space and
        corrects by the theta_transform Jacobian, so it is directly comparable
        to the other ensemble members' log-probabilities.
        """
        if isinstance(x, (list, np.ndarray)):
            x = torch.Tensor(x)
        if isinstance(theta, (list, np.ndarray)):
            theta = torch.Tensor(theta)
        x = x.to(self._device)
        theta = theta.to(self._device)

        logprob = self.flow(x).log_prob(self.theta_transform.inv(theta))
        log_abs_det_jacobian = self.theta_transform.log_abs_det_jacobian(
            theta, theta)  # constant for Affine/Identity; just for shape
        if len(log_abs_det_jacobian.shape) > 1:
            log_abs_det_jacobian = log_abs_det_jacobian.sum(dim=1)
        return logprob - log_abs_det_jacobian

    potential = forward

    def log_prob(self, theta: torch.Tensor, x) -> torch.Tensor:
        return self.forward(theta, x)


class _Lampe_Moment_Constructor():
    """Deferred builder for a Moment Network, mirroring ``_Lampe_Net_Constructor``.

    Constructed by ``load_nde_lampe(model='moment', ...)`` and called later as
    ``constructor(train_loader, prior)`` to build an (untrained)
    ``MomentNetworkEstimator``. The actual joint fit is performed
    separately by ``train_moment_network``; the runner detects this net via the
    ``is_moment`` marker and routes it there.
    """

    is_moment = True

    def __init__(self, embedding_net, model_args, device,
                 x_normalize, theta_normalize):
        self.embedding_net = embedding_net
        self.model_args = model_args
        self.device = device
        self.x_normalize = x_normalize
        self.theta_normalize = theta_normalize

    def to(self, device):
        self.device = device
        return self

    def _fit_transforms(self, train_loader, dtype):
        """Compute z-normalization transforms over the training set."""
        x_transform = identity_transform
        theta_transform = identity_transform

        if self.x_normalize:
            scaler = StandardScaler()
            for x_batch, _ in train_loader:
                scaler.partial_fit(x_batch.cpu().numpy())
            x_mean = torch.tensor(scaler.mean_, dtype=dtype).to(self.device)
            x_std = torch.clamp(
                torch.tensor(scaler.scale_, dtype=dtype).to(self.device),
                min=1e-16)
            x_transform = AffineTransform(loc=x_mean, scale=x_std, event_dim=1)

        if self.theta_normalize:
            scaler = StandardScaler()
            for _, theta_batch in train_loader:
                scaler.partial_fit(theta_batch.cpu().numpy())
            theta_mean = torch.tensor(
                scaler.mean_, dtype=dtype).to(self.device)
            theta_std = torch.clamp(
                torch.tensor(scaler.scale_, dtype=dtype).to(self.device),
                min=1e-16)
            theta_transform = AffineTransform(
                loc=theta_mean, scale=theta_std, event_dim=1)

        return x_transform, theta_transform

    def __call__(self, train_loader, prior) -> MomentNetworkEstimator:
        # infer input/output dimensions from a batch
        x_batch, theta_batch = next(iter(train_loader))
        dtype = x_batch.dtype

        # two independent embedding_net instances, one per stage (matches the
        # reference's independently-initialized networks)
        emb_mu = deepcopy(self.embedding_net).to(self.device)
        emb_cov = deepcopy(self.embedding_net).to(self.device)

        z_batch = emb_mu(x_batch.cpu())
        z_shape = z_batch.shape[1:]
        theta_shape = theta_batch.shape[1:]
        if len(z_shape) > 1:
            raise ValueError("Embedding network must return a vector.")
        if len(theta_shape) > 1:
            raise ValueError("Parameters theta must be a vector.")
        z_dim = z_shape[0]
        n_params = theta_shape[0]
        n_cov = n_params * (n_params + 1) // 2

        hidden_features = self.model_args['hidden_features']
        hidden_depth = self.model_args['hidden_depth']
        activation = self.model_args['activation']

        mu_net = _make_mlp(
            z_dim, n_params, hidden_features, hidden_depth, activation
        ).to(self.device)
        cov_net = _make_mlp(
            z_dim, n_cov, hidden_features, hidden_depth, activation
        ).to(self.device)

        x_transform, theta_transform = self._fit_transforms(
            train_loader, dtype)

        estimator = MomentNetworkEstimator(
            emb_mu=emb_mu, mu_net=mu_net,
            emb_cov=emb_cov, cov_net=cov_net,
            n_params=n_params, prior=prior,
            x_transform=x_transform, theta_transform=theta_transform,
        ).to(self.device)
        return estimator


def _fit_stage(modules, loss_fn, train_loader, val_loader,
               train_args, device, desc='', verbose=True):
    """Train a set of modules to convergence via MSE with early stopping.

    Generic single-stage optimizer loop, following the same early-stopping /
    patience / lr / scheduler conventions as the shared lampe trainer.
    Returns (train_trace, val_trace, best_val_loss) and restores the best
    weights.
    """
    from tqdm import tqdm

    params = [p for m in modules for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=train_args["learning_rate"],
        weight_decay=train_args["weight_decay"])
    clip = train_args["clip_max_norm"]
    early_stopping = train_args.get("early_stopping", True)
    max_epochs = int(train_args["max_epochs"])

    scheduler_name = train_args.get('lr_scheduler', 'ReduceLROnPlateau')
    if scheduler_name == 'ReduceLROnPlateau':
        if train_args["lr_decay_factor"] < 1:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=train_args["lr_decay_factor"],
                patience=train_args["lr_patience"])
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 1.0)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=0)
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")

    best_val = float('inf')
    best_state = [deepcopy(m.state_dict()) for m in modules]
    wait = 0
    train_trace, val_trace = [], []

    with tqdm(range(max_epochs), unit=' epochs', disable=not verbose,
              desc=desc) as tq:
        for epoch in tq:
            for m in modules:
                m.train()
            tot, cnt = 0.0, 0
            for x, theta in train_loader:
                x, theta = x.to(device), theta.to(device)
                optimizer.zero_grad()
                loss = loss_fn(x, theta)
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(params, clip)
                optimizer.step()
                tot += loss.item() * len(theta)
                cnt += len(theta)
            train_loss = tot / cnt

            for m in modules:
                m.eval()
            with torch.no_grad():
                tot, cnt = 0.0, 0
                for x, theta in val_loader:
                    x, theta = x.to(device), theta.to(device)
                    tot += loss_fn(x, theta).item() * len(theta)
                    cnt += len(theta)
                val_loss = tot / cnt

            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

            train_trace.append(train_loss)
            val_trace.append(val_loss)
            tq.set_postfix(mse=train_loss, mse_val=val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = [deepcopy(m.state_dict()) for m in modules]
                wait = 0
            elif early_stopping:
                if wait > train_args["stop_after_epochs"]:
                    break
                wait += 1
        else:
            if early_stopping:
                logging.warning(
                    f"Moment network stage '{desc}' did not converge in "
                    f"{max_epochs} epochs.")

    for m, s in zip(modules, best_state):
        m.load_state_dict(s)
    return train_trace, val_trace, best_val


_VAR_FLOOR = 1e-6  # matches the softplus floor used in MomentNetworkEstimator._build_L


def _logsum_sq(err):
    """sum_j log(sum_batch(err_j^2) + floor): keeps gradients comparably
    scaled across output dimensions of differing natural magnitude, unlike
    plain per-sample MSE."""
    return torch.log((err ** 2).sum(0) + _VAR_FLOOR).sum()


def train_moment_network(estimator: MomentNetworkEstimator,
                         train_loader, val_loader,
                         train_args, device, verbose=True):
    """Jointly fit the mean and covariance networks of a Moment Network.

    Both networks are optimized together in a single pass: the mean loss
    backpropagates into ``emb_mu``/``mu_net``, while the covariance loss
    backpropagates into ``emb_cov``/``cov_net`` only (the residuals it
    targets use a stop-gradient copy of mu, so the mean net isn't pulled by
    the covariance term). Returns a summary dict shaped like the shared lampe
    trainer's summaries (so ensembling/plotting code is unchanged).
    """
    tt = estimator.theta_transform
    xt = estimator.x_transform
    rows = estimator._tril_rows
    cols = estimator._tril_cols

    def loss_fn(x, theta):
        xin = xt.inv(x).float()
        theta_n = tt.inv(theta)

        mu = estimator.mu_net(estimator.emb_mu(xin))
        term_mean = _logsum_sq(theta_n - mu)

        resid = theta_n - mu.detach()                     # stop-gradient
        L = estimator._build_L(estimator.cov_net(estimator.emb_cov(xin)))
        Sigma = L @ L.transpose(-1, -2)
        pred = Sigma[:, rows, cols]                        # vech(Sigma)
        target = (resid.unsqueeze(-1) * resid.unsqueeze(-2))[:, rows, cols]
        term_cov = _logsum_sq(pred - target)

        return term_mean + term_cov

    if verbose:
        logging.info(
            "Moment network: joint training of mean + covariance nets.")
    train_trace, val_trace, _ = _fit_stage(
        [estimator.emb_mu, estimator.mu_net,
         estimator.emb_cov, estimator.cov_net],
        loss_fn, train_loader, val_loader, train_args, device,
        desc='moment', verbose=verbose)

    # ----- final validation log-prob (comparable to flow members) -----
    estimator.eval()
    with torch.no_grad():
        tot, cnt = 0.0, 0
        for x, theta in val_loader:
            x, theta = x.to(device), theta.to(device)
            tot += estimator.log_prob(theta, x).sum().item()
            cnt += len(theta)
        best_val_logprob = tot / cnt

    summary = {
        'training_log_probs': [-v for v in train_trace],
        'validation_log_probs': [-v for v in val_trace],
        'best_validation_log_prob': best_val_logprob,
        'epochs_trained': len(val_trace),
    }
    return summary
