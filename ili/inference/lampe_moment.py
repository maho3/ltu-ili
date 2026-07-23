"""
Standalone two-stage trainer and estimator for Moment Networks
(Jeffrey & Wandelt 2020, arXiv:2011.05991) in the lampe backend.

A Moment Network does not learn a full joint density like the zuko flows that
the lampe backend usually wraps. Instead it directly regresses the posterior
mean and covariance:

    Stage 1 (mean network):
        f_mu(x) -> mu, trained with plain MSE against the true parameters theta.

    Stage 2 (covariance network), trained *after* Stage 1 is frozen:
        resid    = theta - f_mu(x)              # frozen, no gradient
        target   = vech(resid resid^T)          # residual products, i<=j
        f_cov(x) -> raw Cholesky entries -> L -> Sigma = L L^T
        loss     = MSE(vech(Sigma), target)     # plain MSE

    Because E[resid resid^T | x] = Sigma(x), minimizing the MSE between
    vech(Sigma) and the residual-product targets drives L L^T towards the true
    conditional covariance, while the Cholesky parameterization (lower
    triangular, softplus on the diagonal) guarantees Sigma is positive-definite
    by construction. This reconciles the reference implementation's
    "regress toward residual products" targets with the requirement to output a
    valid covariance.

The resulting posterior is a single (full-covariance) Gaussian approximation.
It is *not* a flexible density -- it is unimodal and Gaussian -- so on its own
it can only capture Gaussian posteriors. It is most useful as one member of an
ensemble alongside real flows (maf/nsf/...), where it is sampled and log_prob'd
identically via a torch MultivariateNormal.

This module is self-contained: the two-stage training lives entirely inside
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
    """Build a simple MLP trunk with ``hidden_depth`` hidden layers."""
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
        """Return (mu, L) in normalized theta space for input data x.

        mu has shape (B, n_params); L is a (B, n_params, n_params) lower-
        triangular Cholesky factor with Sigma = L @ L.T.
        """
        if isinstance(x, (list, np.ndarray)):
            x = torch.Tensor(x)
        x = x.to(self._device)
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
    ``MomentNetworkEstimator``. The actual two-stage fit is performed
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
            theta_mean = torch.tensor(scaler.mean_, dtype=dtype).to(self.device)
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

        x_transform, theta_transform = self._fit_transforms(train_loader, dtype)

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
    patience / lr conventions as the shared lampe trainer. Returns
    (train_trace, val_trace, best_val_loss) and restores the best weights.
    """
    from tqdm import tqdm

    params = [p for m in modules for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=train_args["learning_rate"],
        weight_decay=train_args["weight_decay"])
    clip = train_args["clip_max_norm"]
    early_stopping = train_args.get("early_stopping", True)
    max_epochs = int(train_args["max_epochs"])

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


def train_moment_network(estimator: MomentNetworkEstimator,
                         train_loader, val_loader,
                         train_args, device, verbose=True):
    """Run the two-stage Moment Network fit in place on ``estimator``.

    Stage 1 fits the mean network with MSE against theta. Stage 2 freezes it,
    derives residual-product targets, and fits the covariance network so that
    vech(L L^T) matches those targets. Returns a summary dict shaped like the
    shared lampe trainer's summaries (so ensembling/plotting code is unchanged).
    """
    tt = estimator.theta_transform
    xt = estimator.x_transform
    rows = estimator._tril_rows
    cols = estimator._tril_cols

    # ----- Stage 1: mean network (plain MSE against true theta) -----
    def loss_mu(x, theta):
        z = estimator.emb_mu(xt.inv(x).float())
        mu = estimator.mu_net(z)
        return F.mse_loss(mu, tt.inv(theta))

    if verbose:
        logging.info("Moment network: training Stage 1 (mean network).")
    s1_train, s1_val, _ = _fit_stage(
        [estimator.emb_mu, estimator.mu_net], loss_mu,
        train_loader, val_loader, train_args, device,
        desc='moment-mean', verbose=verbose)

    # freeze Stage 1
    for p in estimator.emb_mu.parameters():
        p.requires_grad_(False)
    for p in estimator.mu_net.parameters():
        p.requires_grad_(False)
    estimator.emb_mu.eval()
    estimator.mu_net.eval()

    # ----- Stage 2: covariance network (MSE toward residual products) -----
    def loss_cov(x, theta):
        xin = xt.inv(x).float()
        with torch.no_grad():
            mu = estimator.mu_net(estimator.emb_mu(xin))
        resid = tt.inv(theta) - mu                       # frozen residuals
        L = estimator._build_L(estimator.cov_net(estimator.emb_cov(xin)))
        Sigma = L @ L.transpose(-1, -2)
        pred = Sigma[:, rows, cols]                      # vech(Sigma)
        outer = resid.unsqueeze(-1) * resid.unsqueeze(-2)
        target = outer[:, rows, cols]                    # vech(resid resid^T)
        return F.mse_loss(pred, target)

    if verbose:
        logging.info("Moment network: training Stage 2 (covariance network).")
    s2_train, s2_val, _ = _fit_stage(
        [estimator.emb_cov, estimator.cov_net], loss_cov,
        train_loader, val_loader, train_args, device,
        desc='moment-cov', verbose=verbose)

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
        # generic keys expected by ensembling / plotting code
        'training_log_probs': [-v for v in (s1_train + s2_train)],
        'validation_log_probs': [-v for v in (s1_val + s2_val)],
        'best_validation_log_prob': best_val_logprob,
        'epochs_trained': len(s1_val) + len(s2_val),
        # moment-network specific diagnostics
        'stage1_train_mse': s1_train,
        'stage1_val_mse': s1_val,
        'stage2_train_mse': s2_train,
        'stage2_val_mse': s2_val,
    }
    return summary
