"""
Neural Information Theory and Mutual Information Estimation for Oxideon.
Includes Mutual Information Neural Estimator (MINE, Belghazi et al., ICML 2018),
Jensen-Shannon MI estimator, Kraskov-Stogbauer-Grassberger (KSG) k-NN MI,
and closed-form Gaussian differential entropy & KL divergence.
"""
import math
from typing import Tuple, Optional
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy
from .nn import Module, Linear


def _logsumexp_1d(x: Tensor) -> Tensor:
    m = float(np.max(x.to_numpy()))
    return ((x - m).exp().sum() + 1e-12).log() + m


def _digamma(x: float) -> float:
    """Accurate digamma psi(x) via recurrence + asymptotic expansion."""
    val = float(x)
    if val <= 0.0:
        return -1e9
    res = 0.0
    while val < 6.0:
        res -= 1.0 / val
        val += 1.0
    inv = 1.0 / val
    inv2 = inv * inv
    res += math.log(val) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
    return res


class MINE(Module):
    """
    Mutual Information Neural Estimator (Belghazi et al., ICML 2018).
    Estimates I(X; Y) via Donsker-Varadhan ('dv'), f-divergence ('nwj'), or
    Jensen-Shannon ('jsd') representations while preserving the Oxideon autograd tape.
    """
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.fc1 = Linear(x_dim + y_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3 = Linear(hidden_dim, 1)

    def _score(self, x: Tensor, y: Tensor) -> Tensor:
        xy = Tensor.cat([x, y], dim=-1)
        h = self.fc1(xy).relu()
        h2 = self.fc2(h).relu()
        return self.fc3(h2)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        estimator: str = "dv",
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            mi_estimate: Differentiable scalar tensor estimate of mutual information I(X; Y)
            loss: Differentiable negative lower bound for gradient-descent minimization
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Length mismatch: x has {x.shape[0]}, y has {y.shape[0]}")
        n = x.shape[0]
        if n == 0:
            zero = _from_numpy(np.array(0.0, dtype=np.float32))
            return zero, zero

        t_joint = self._score(x, y).reshape([n])

        y_np = y.to_numpy()
        shuffle_idx = np.random.permutation(n)
        y_shuffled = _from_numpy(y_np[shuffle_idx].astype(np.float32))
        t_marginal = self._score(x, y_shuffled).reshape([n])

        est = estimator.strip().lower()
        if est == "dv":
            e_joint = t_joint.mean()
            log_e_marginal = _logsumexp_1d(t_marginal) - math.log(float(n))
            mi = e_joint - log_e_marginal
        elif est in ("nwj", "fdiv"):
            mi = t_joint.mean() - (t_marginal - 1.0).exp().mean()
        elif est == "jsd":
            # -softplus(-t_joint) - softplus(t_marginal)
            sp_joint = (-t_joint).softplus()
            sp_marg = t_marginal.softplus()
            mi = (sp_joint * -1.0).mean() - sp_marg.mean()
        else:
            raise ValueError(f"Unknown MINE estimator {estimator!r}; expected 'dv', 'nwj', or 'jsd'")

        loss = -mi
        mi_nonneg = mi.relu()
        return mi_nonneg, loss


def ksg_mutual_information(x: Tensor, y: Tensor, k: int = 3) -> float:
    """
    Kraskov-Stogbauer-Grassberger (KSG-1, 2004) k-NN non-parametric mutual information estimator:
    I(X; Y) = psi(k) + psi(N) - <psi(n_x + 1) + psi(n_y + 1)>
    """
    x_np = np.asarray(x.to_numpy(), dtype=np.float64)
    y_np = np.asarray(y.to_numpy(), dtype=np.float64)
    if x_np.ndim == 1:
        x_np = x_np[:, None]
    if y_np.ndim == 1:
        y_np = y_np[:, None]
    if x_np.shape[0] != y_np.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    n = x_np.shape[0]
    if n <= k:
        raise ValueError(f"Number of samples ({n}) must be > k ({k})")

    dx = np.max(np.abs(x_np[:, None, :] - x_np[None, :, :]), axis=-1)
    dy = np.max(np.abs(y_np[:, None, :] - y_np[None, :, :]), axis=-1)
    dz = np.maximum(dx, dy)
    np.fill_diagonal(dz, np.inf)

    # Distance to k-th nearest neighbor in jointChebyshev metric
    eps = np.partition(dz, k - 1, axis=1)[:, k - 1]
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)

    nx = np.sum(dx < eps[:, None] - 1e-12, axis=1)
    ny = np.sum(dy < eps[:, None] - 1e-12, axis=1)

    avg_psi = np.mean([_digamma(nx_i + 1) + _digamma(ny_i + 1) for nx_i, ny_i in zip(nx, ny)])
    mi = _digamma(k) + _digamma(n) - float(avg_psi)
    return float(max(0.0, mi))


def differential_entropy_gaussian(cov: Tensor) -> float:
    """Computes exact differential entropy of a multivariate Gaussian: 0.5 * log((2*pi*e)^d * det(Sigma))."""
    sigma = cov.to_numpy()
    if sigma.ndim == 1:
        d = len(sigma)
        logdet = float(np.sum(np.log(np.maximum(sigma, 1e-12))))
        entropy = 0.5 * d * (1.0 + np.log(2.0 * np.pi)) + 0.5 * logdet
        return float(entropy)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"cov must be 1D or square 2D matrix, got shape {sigma.shape}")
    d = sigma.shape[0]
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        return 0.0
    entropy = 0.5 * d * (1.0 + np.log(2.0 * np.pi)) + 0.5 * logdet
    return float(entropy)


def kl_divergence_gaussians(
    mu_p: Tensor,
    cov_p: Tensor,
    mu_q: Tensor,
    cov_q: Tensor,
) -> float:
    """Closed-form KL(N(mu_p, Sigma_p) || N(mu_q, Sigma_q))."""
    mp = np.asarray(mu_p.to_numpy(), dtype=np.float64).reshape(-1)
    mq = np.asarray(mu_q.to_numpy(), dtype=np.float64).reshape(-1)
    sp = np.asarray(cov_p.to_numpy(), dtype=np.float64)
    sq = np.asarray(cov_q.to_numpy(), dtype=np.float64)
    if sp.ndim == 1:
        sp = np.diag(sp)
    if sq.ndim == 1:
        sq = np.diag(sq)
    d = len(mp)
    sq_inv = np.linalg.pinv(sq)
    diff = mq - mp
    tr_term = float(np.trace(sq_inv @ sp))
    quad_term = float(diff.T @ sq_inv @ diff)
    _, logdet_p = np.linalg.slogdet(sp)
    _, logdet_q = np.linalg.slogdet(sq)
    kl = 0.5 * (tr_term + quad_term - d + float(logdet_q - logdet_p))
    return float(max(0.0, kl))
