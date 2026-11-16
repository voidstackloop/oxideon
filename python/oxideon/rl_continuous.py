"""
Continuous Action Space Reinforcement Learning for Oxideon.
Includes Gaussian squashed policies (SAC actor), Twin Q-Critics (TD3 / SAC), and entropy-regularized objectives.
"""
from typing import Tuple, Optional
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy
from .nn import Module, Linear


class GaussianActor(Module):
    """
    Stochastic policy for continuous action spaces.
    Outputs mean and log_std, sampling actions a = tanh(u) with exact log-probability computation.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, log_std_min: float = -20.0, log_std_max: float = 2.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = Linear(state_dim, hidden_dim)
        self.mean_layer = Linear(hidden_dim, action_dim)
        self.log_std_layer = Linear(hidden_dim, action_dim)

    def forward(self, state: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            action: squashed action in range (-1, 1)
            log_prob: log probability log pi(a|s) with tanh Jacobian correction
        """
        single = state.ndim == 1
        s_tensor = state.unsqueeze(0) if single else state

        h = self.net(s_tensor).relu()
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()

        if deterministic:
            action = mean.tanh()
            log_prob = (mean * 0.0 + log_std * 0.0).sum(dim=-1)
        else:
            eps = _from_numpy(np.random.randn(*mean.shape).astype(np.float32))
            u = mean + std * eps
            action = u.tanh()

            # Log prob under Gaussian using standard normal sample eps directly
            log_2pi = 1.8378770664093456
            log_prob_u = ((eps * eps + log_std * 2.0 + log_2pi) * (-0.5)).sum(dim=-1)

            # Correction for tanh squashing: - sum log(1 - tanh(u)^2 + 1e-6)
            squash_correction = (1.0 - action * action + 1e-6).log().sum(dim=-1)
            log_prob = log_prob_u - squash_correction

        if single:
            return action.squeeze(0), log_prob.squeeze(0)
        return action, log_prob


class TwinCritic(Module):
    """
    Twin Q-Networks Q1(s, a) and Q2(s, a) to mitigate positive overestimation bias in continuous RL.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = state_dim + action_dim
        self.q1_fc1 = Linear(in_dim, hidden_dim)
        self.q1_fc2 = Linear(hidden_dim, 1)

        self.q2_fc1 = Linear(in_dim, hidden_dim)
        self.q2_fc2 = Linear(hidden_dim, 1)

    def forward(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        is_1d = state.ndim == 1 and action.ndim == 1
        s = state.unsqueeze(0) if is_1d else state
        a = action.unsqueeze(0) if is_1d else action
        if s.shape[0] != a.shape[0]:
            raise ValueError(f"Batch dimension mismatch: state has batch {s.shape[0]}, action has batch {a.shape[0]}")
        sa_t = Tensor.cat([s, a], dim=-1)

        q1 = self.q1_fc2(self.q1_fc1(sa_t).relu())
        q2 = self.q2_fc2(self.q2_fc1(sa_t).relu())
        if is_1d:
            q1 = q1.reshape([-1])
            q2 = q2.reshape([-1])
        return q1, q2

    def min_q(self, state: Tensor, action: Tensor) -> Tensor:
        q1, q2 = self.forward(state, action)
        return (q1 + q2 - (q1 - q2).abs()) * 0.5


class SACLoss:
    """Soft Actor-Critic loss calculation for critic and entropy-regularized actor."""
    def __init__(self, alpha: float = 0.2, gamma: float = 0.99):
        self.alpha = alpha
        self.gamma = gamma

    def critic_loss(self, q1: Tensor, q2: Tensor, target_q: Tensor) -> Tensor:
        t = target_q.reshape(list(q1.shape)) if list(target_q.shape) != list(q1.shape) else target_q
        diff1 = q1 - t
        diff2 = q2 - t
        return ((diff1 * diff1).mean() + (diff2 * diff2).mean()) * 0.5

    def actor_loss(self, min_q: Tensor, log_prob: Tensor) -> Tensor:
        q_flat = min_q.reshape([-1])
        lp_flat = log_prob.reshape([-1])
        return (lp_flat * float(self.alpha) - q_flat).mean()


__all__ = [
    "GaussianActor",
    "TwinCritic",
    "SACLoss",
]
