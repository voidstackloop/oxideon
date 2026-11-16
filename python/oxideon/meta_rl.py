"""
Meta-Reinforcement Learning and Few-Shot Policy Adaptation for Oxideon.
Includes RL^2 recurrent context policies (Duan et al., 2016), episodic trajectory rollouts,
and MAML inner-loop policy adaptation.
"""
from typing import Tuple, Dict, Optional, Sequence, Union
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, Parameter


class RL2ContextPolicy(Module):
    """
    RL^2 Recurrent Meta-RL policy (Duan et al., 2016).
    Input vector at timestep t: [state_t, prev_action_{t-1}, prev_reward_{t-1}, prev_done_{t-1}].
    Internal recurrent state acts as fast dynamic adaptation across episodic horizons while
    preserving the Oxideon autograd computation graph.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        in_dim = state_dim + action_dim + 2
        self.fc_in = Linear(in_dim, hidden_dim)
        self.fc_rec = Linear(hidden_dim, hidden_dim)
        self.policy_head = Linear(hidden_dim, action_dim)
        self.value_head = Linear(hidden_dim, 1)

    def step(
        self,
        state: Tensor,
        prev_action: Tensor,
        prev_reward: float,
        prev_done: float,
        hidden_state: Optional[Union[Tensor, np.ndarray]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Single meta-RL environment step preserving autograd through `state`, `prev_action`, and `hidden_state`.
        Returns:
            action_logits: (1, action_dim)
            value: (1, 1)
            next_hidden: (1, hidden_dim) Tensor
        """
        if state.numel() != self.state_dim:
            raise ValueError(f"state dimension mismatch: expected {self.state_dim}, got {state.numel()}")
        if prev_action.numel() != self.action_dim:
            raise ValueError(f"action dimension mismatch: expected {self.action_dim}, got {prev_action.numel()}")

        s_1d = state.reshape([1, self.state_dim])
        a_1d = prev_action.reshape([1, self.action_dim])
        r_d = _from_numpy(np.array([[float(prev_reward), float(prev_done)]], dtype=np.float32))
        feat = Tensor.cat([s_1d, a_1d, r_d], dim=-1)

        if hidden_state is None:
            h_prev = Tensor.zeros([1, self.hidden_dim])
        elif isinstance(hidden_state, Tensor):
            h_prev = hidden_state.reshape([1, self.hidden_dim])
        else:
            h_prev = _from_numpy(np.asarray(hidden_state, dtype=np.float32).reshape(1, self.hidden_dim))

        h_next = (self.fc_in(feat) + self.fc_rec(h_prev)).tanh()
        logits = self.policy_head(h_next)
        val = self.value_head(h_next)
        return logits, val, h_next

    def rollout_trajectory(
        self,
        states: Tensor,
        prev_actions: Tensor,
        prev_rewards: Sequence[float],
        prev_dones: Sequence[float],
        init_hidden: Optional[Union[Tensor, np.ndarray]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Rolls out an entire trajectory of T steps through the RL^2 recurrent context policy."""
        t_steps = states.shape[0]
        if prev_actions.shape[0] != t_steps or len(prev_rewards) != t_steps or len(prev_dones) != t_steps:
            raise ValueError("states, prev_actions, prev_rewards, and prev_dones must share length T")

        logits_list = []
        vals_list = []
        h = init_hidden
        for t in range(t_steps):
            s_t = states.select(0, t)
            a_t = prev_actions.select(0, t)
            lg, vl, h = self.step(s_t, a_t, float(prev_rewards[t]), float(prev_dones[t]), h)
            logits_list.append(lg)
            vals_list.append(vl)

        all_logits = Tensor.cat(logits_list, dim=0)
        all_vals = Tensor.cat(vals_list, dim=0)
        return all_logits, all_vals, h


class MAMLPolicyAdapter:
    """Computes one-step fast gradient adaptation for MAML policy parameters."""
    @staticmethod
    def adapt(
        weights: Dict[str, np.ndarray],
        advantages: np.ndarray,
        log_prob_grads: Dict[str, np.ndarray],
        inner_lr: float = 0.05,
    ) -> Dict[str, np.ndarray]:
        """One-step gradient ascent on policy objective: theta' = theta + alpha * grad."""
        adapted = {}
        adv_mean = float(np.mean(advantages)) if len(advantages) > 0 else 0.0
        for k in weights:
            if k in log_prob_grads:
                g = log_prob_grads[k] * adv_mean
                adapted[k] = weights[k] + inner_lr * g
            else:
                adapted[k] = weights[k].copy()
        return adapted
