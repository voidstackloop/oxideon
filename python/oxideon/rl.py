"""Deep Reinforcement Learning (RL) module for Oxideon.

Provides experience replay memories, policy/value networks, and actor-critic utilities:
  - ReplayBuffer: Standard circular experience replay buffer
  - PrioritizedReplayBuffer: Proportional Prioritized Experience Replay (Schaul et al., 2016)
  - DQNPolicy: Deep Q-Network policy with epsilon-greedy exploration
  - compute_gae: Generalized Advantage Estimation (Schulman et al., 2015)
  - PPOAgent: Discrete Actor-Critic agent for Proximal Policy Optimization
"""

from typing import Tuple, Optional, Any, List, Dict, Union
import random
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, Sequential, ReLU, Parameter


class ReplayBuffer:
    """Fixed-capacity circular replay buffer for off-policy RL."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int = 1):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.size = 0

    def push(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
    ):
        """Stores a transition tuple in the buffer."""
        s = state.to_numpy() if isinstance(state, _Tensor) else np.asarray(state, dtype=np.float32)
        a = action.to_numpy() if isinstance(action, _Tensor) else np.asarray(action, dtype=np.float32)
        ns = next_state.to_numpy() if isinstance(next_state, _Tensor) else np.asarray(next_state, dtype=np.float32)

        self.states[self.idx] = s.reshape(-1)
        self.actions[self.idx] = a.reshape(-1)
        self.rewards[self.idx, 0] = float(reward)
        self.next_states[self.idx] = ns.reshape(-1)
        self.dones[self.idx, 0] = 1.0 if done else 0.0

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, seed: Optional[int] = None
    ) -> Tuple[_Tensor, _Tensor, _Tensor, _Tensor, _Tensor]:
        """Samples a uniform mini-batch of transitions as Oxideon Tensors."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer with size {self.size}")

        if seed is not None:
            rng = np.random.default_rng(seed)
            indices = rng.choice(self.size, size=batch_size, replace=False)
        else:
            indices = np.random.choice(self.size, size=batch_size, replace=False)

        return (
            _from_numpy(np.ascontiguousarray(self.states[indices])),
            _from_numpy(np.ascontiguousarray(self.actions[indices])),
            _from_numpy(np.ascontiguousarray(self.rewards[indices])),
            _from_numpy(np.ascontiguousarray(self.next_states[indices])),
            _from_numpy(np.ascontiguousarray(self.dones[indices])),
        )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay (PER) using rank/proportional probability."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
    ):
        super().__init__(capacity, state_dim, action_dim)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
    ):
        idx = self.idx
        super().push(state, action, reward, next_state, done)
        self.priorities[idx] = self.max_priority

    def sample(
        self, batch_size: int, seed: Optional[int] = None
    ) -> Tuple[_Tensor, _Tensor, _Tensor, _Tensor, _Tensor, np.ndarray, _Tensor]:
        """Samples transitions weighted by TD error priorities, returning IS weights."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer with size {self.size}")

        prios = self.priorities[: self.size]
        probs = np.power(prios, self.alpha)
        probs_sum = np.sum(probs)
        if probs_sum <= 0:
            probs = np.ones(self.size, dtype=np.float32) / self.size
        else:
            probs = probs / probs_sum

        if seed is not None:
            rng = np.random.default_rng(seed)
            indices = rng.choice(self.size, size=batch_size, p=probs, replace=False)
        else:
            indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)

        total = self.size
        weights = np.power(total * probs[indices], -self.beta)
        weights = weights / np.maximum(np.max(weights), 1e-12)

        return (
            _from_numpy(np.ascontiguousarray(self.states[indices])),
            _from_numpy(np.ascontiguousarray(self.actions[indices])),
            _from_numpy(np.ascontiguousarray(self.rewards[indices])),
            _from_numpy(np.ascontiguousarray(self.next_states[indices])),
            _from_numpy(np.ascontiguousarray(self.dones[indices])),
            indices,
            _from_numpy(np.ascontiguousarray(weights.astype(np.float32).reshape(-1, 1))),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: Any):
        """Updates sample priorities based on absolute TD errors."""
        errs = td_errors.to_numpy() if isinstance(td_errors, _Tensor) else np.asarray(td_errors)
        prios = np.abs(errs).reshape(-1) + self.eps
        for idx, prio in zip(indices, prios):
            self.priorities[idx] = float(prio)
            self.max_priority = max(self.max_priority, float(prio))


class DQNPolicy(Module):
    """Deep Q-Network (Mnih et al., 2015) mapping states to action values."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = Sequential(
            Linear(state_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, action_dim),
        )

    def forward(self, state: _Tensor) -> _Tensor:
        return self.net(state)

    def select_action(self, state: Any, epsilon: float = 0.0, seed: Optional[int] = None) -> int:
        """Selects an action using epsilon-greedy exploration."""
        rng = random.Random(seed) if seed is not None else random
        if rng.random() < epsilon:
            return rng.randint(0, self.action_dim - 1)

        if not isinstance(state, _Tensor):
            s_np = np.ascontiguousarray(np.asarray(state, dtype=np.float32).reshape(1, -1))
            state = _from_numpy(s_np)
        elif len(state.shape) == 1:
            state = _from_numpy(np.ascontiguousarray(state.to_numpy().reshape(1, -1)))

        q_vals = self.forward(state).to_numpy()
        return int(np.argmax(q_vals[0]))


def compute_gae(
    rewards: Any,
    values: Any,
    dones: Any,
    next_value: Any,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[Any, Any]:
    """Computes Generalized Advantage Estimation (GAE) and discounted returns."""
    if any(isinstance(x, _Tensor) for x in (rewards, values, dones, next_value)):
        t_values = values if isinstance(values, _Tensor) else _from_numpy(np.asarray(values, dtype=np.float32))
        t_rewards = rewards if isinstance(rewards, _Tensor) else _from_numpy(np.asarray(rewards, dtype=np.float32))
        t_dones = dones if isinstance(dones, _Tensor) else _from_numpy(np.asarray(dones, dtype=np.float32))
        t_next = next_value if isinstance(next_value, _Tensor) else _from_numpy(np.asarray([next_value], dtype=np.float32))

        t_val_flat = t_values.reshape([-1])
        t_rew_flat = t_rewards.reshape([-1])
        t_dn_flat = t_dones.reshape([-1])
        t_next_flat = t_next.reshape([-1])

        T = t_val_flat.shape[0]
        vals = [t_val_flat.narrow(0, i, 1) for i in range(T)] + [t_next_flat.narrow(0, 0, 1)]
        rews = [t_rew_flat.narrow(0, i, 1) for i in range(T)]
        dns = [t_dn_flat.narrow(0, i, 1) for i in range(T)]

        adv_list = []
        last_adv = _from_numpy(np.zeros(1, dtype=np.float32))

        for t in reversed(range(T)):
            non_terminal = (dns[t] * -1.0) + 1.0
            delta = rews[t] + (vals[t + 1] * (float(gamma) * non_terminal)) - vals[t]
            adv_t = delta + (last_adv * (float(gamma * gae_lambda) * non_terminal))
            last_adv = adv_t
            adv_list.append(adv_t)

        advantages = _Tensor.cat(list(reversed(adv_list)), dim=0)
        returns = advantages + t_val_flat
        return advantages, returns

    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError(f"Length mismatch: rewards has {len(rewards)}, values has {len(values)}, dones has {len(dones)}")
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0

    next_val_f = float(next_value) if not hasattr(next_value, "item") else float(next_value.item())
    vals = np.array([float(v) for v in values] + [next_val_f], dtype=np.float32)
    rews = np.array([float(r) for r in rewards], dtype=np.float32)
    dns = np.array([float(d) for d in dones], dtype=np.float32)

    for t in reversed(range(T)):
        non_terminal = 1.0 - dns[t]
        delta = rews[t] + gamma * vals[t + 1] * non_terminal - vals[t]
        advantages[t] = last_adv = delta + gamma * gae_lambda * non_terminal * last_adv

    returns = advantages + vals[:T]
    return advantages, returns


class PPOAgent(Module):
    """Actor-Critic agent architecture for Proximal Policy Optimization."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Sequential(
            Linear(state_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, action_dim),
        )
        self.critic = Sequential(
            Linear(state_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1),
        )

    def forward(self, state: _Tensor) -> Tuple[_Tensor, _Tensor]:
        """Returns (action_logits, state_value)."""
        logits = self.actor(state)
        value = self.critic(state)
        return logits, value


__all__ = [
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "DQNPolicy",
    "compute_gae",
    "PPOAgent",
]
