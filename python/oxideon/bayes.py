"""
Bayesian Neural Networks (BNNs) and Variational Inference for Oxideon.
Implements differentiable Bayes by Backprop (Blundell et al.) with reparameterization trick,
ELBO loss, and Monte Carlo epistemic uncertainty estimation.
"""
import math
from typing import Optional, Tuple
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter


class BayesLinear(Module):
    """
    Bayesian Linear layer with Gaussian variational posterior q(w|theta) ~ N(mu, sigma^2)
    and Gaussian prior p(w) ~ N(0, prior_std^2), preserving the autograd tape through
    `weight_mu`, `weight_rho`, `bias_mu`, and `bias_rho`.
    """
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = float(prior_std)

        w_mu = (np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)).astype(np.float32)
        w_rho = np.full((out_features, in_features), -3.0, dtype=np.float32)
        b_mu = np.zeros(out_features, dtype=np.float32)
        b_rho = np.full(out_features, -3.0, dtype=np.float32)

        self.weight_mu = Parameter(_from_numpy(w_mu))
        self.weight_rho = Parameter(_from_numpy(w_rho))
        self.bias_mu = Parameter(_from_numpy(b_mu))
        self.bias_rho = Parameter(_from_numpy(b_rho))

    def _sample_weights(self) -> Tuple[Tensor, Tensor]:
        w_sigma = self.weight_rho.softplus()
        b_sigma = self.bias_rho.softplus()

        eps_w = _from_numpy(np.random.randn(self.out_features, self.in_features).astype(np.float32))
        eps_b = _from_numpy(np.random.randn(self.out_features).astype(np.float32))

        w = self.weight_mu + w_sigma * eps_w
        b = self.bias_mu + b_sigma * eps_b
        return w, b

    def kl_divergence(self) -> Tensor:
        """Computes differentiable KL divergence KL(q(theta) || p(theta)) to Gaussian prior."""
        w_sigma = self.weight_rho.softplus() + 1e-7
        b_sigma = self.bias_rho.softplus() + 1e-7
        prior_var = self.prior_std ** 2
        log_prior_std = math.log(self.prior_std)

        kl_w = (
            (w_sigma.log() * -1.0 + log_prior_std)
            + (w_sigma * w_sigma + self.weight_mu * self.weight_mu) * (0.5 / prior_var)
            - 0.5
        ).sum()
        kl_b = (
            (b_sigma.log() * -1.0 + log_prior_std)
            + (b_sigma * b_sigma + self.bias_mu * self.bias_mu) * (0.5 / prior_var)
            - 0.5
        ).sum()
        return kl_w + kl_b

    def forward(self, x: Tensor, sample: bool = True) -> Tensor:
        if sample:
            w, b = self._sample_weights()
        else:
            w = self.weight_mu
            b = self.bias_mu

        return x.matmul(w.transpose(0, 1)) + b.reshape([1, self.out_features])


class ELBOLoss:
    """Differentiable Evidence Lower Bound (ELBO) loss combining empirical loss and beta-weighted KL divergence."""
    def __init__(self, beta: float = 1.0, criterion: str = "mse"):
        self.beta = float(beta)
        self.criterion = criterion

    def __call__(self, preds: Tensor, targets: Tensor, kl: Tensor) -> Tensor:
        if list(preds.shape) != list(targets.shape):
            raise ValueError(f"Shape mismatch: preds shape {preds.shape} != targets shape {targets.shape}")

        if self.criterion == "mse":
            diff = preds - targets
            data_loss = (diff * diff).mean()
        elif self.criterion == "bce":
            p_clip = preds.clamp(1e-7, 1.0 - 1e-7)
            one_minus_t = (targets * -1.0) + 1.0
            one_minus_p = (p_clip * -1.0) + 1.0
            data_loss = ((targets * p_clip.log() + one_minus_t * one_minus_p.log()) * -1.0).mean()
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

        return data_loss + kl * self.beta


class BayesPredictor:
    """Predictive inference for Bayesian models using Monte Carlo sampling."""
    def __init__(self, model: Module, num_samples: int = 20):
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        self.model = model
        self.num_samples = num_samples

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Runs M stochastic forward passes; returns mean prediction and epistemic variance."""
        samples = []
        for _ in range(self.num_samples):
            pred = self.model(x).to_numpy()
            samples.append(pred)

        samples_arr = np.stack(samples, axis=0)
        mean_pred = np.mean(samples_arr, axis=0)
        var_pred = np.var(samples_arr, axis=0)
        return _from_numpy(mean_pred.astype(np.float32)), _from_numpy(var_pred.astype(np.float32))
