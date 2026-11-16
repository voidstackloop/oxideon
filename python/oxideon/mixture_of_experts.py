"""
Mixture of Experts (MoE) and Sparse Routing Architectures for Oxideon.
Includes Top-k noisy gating, Switch Transformer load balancing loss, router z-loss, and sparse MoE layers.
"""
from typing import Tuple, List, Optional
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, ModuleList, _check_repeats


class TopKGating(Module):
    """
    Top-k gating router with optional Gaussian exploration noise and load balancing metrics.
    """
    def __init__(self, d_model: int, num_experts: int = 4, top_k: int = 2, noisy_gating: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noisy_gating = noisy_gating

        self.w_gate = Linear(d_model, num_experts)
        self.w_noise = Linear(d_model, num_experts) if noisy_gating else None
        self._last_expert_weights: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Returns:
            topk_indices: (B, top_k) expert assignments
            topk_weights: (B, top_k) normalized routing weights
            aux_loss: Switch Transformer load balancing penalty
            z_loss: Logit stabilization z-loss
        """
        logits_t = self.w_gate(x)

        if self.noisy_gating and self.w_noise is not None:
            noise_logits = self.w_noise(x)
            noise_std = (noise_logits.clamp(-20.0, 20.0).exp() + 1.0).log()
            noise_eps = _from_numpy(np.random.randn(*logits_t.shape).astype(np.float32))
            logits_t = logits_t + noise_std * noise_eps

        logits = logits_t.to_numpy()
        probs_t = logits_t.softmax(-1)
        probs = probs_t.to_numpy()

        # Router Z-loss: mean square of logsumexp of logits
        max_l = np.max(logits, axis=-1, keepdims=True)
        lse = max_l + np.log(np.sum(np.exp(logits - max_l), axis=-1, keepdims=True))
        z_loss = float(np.mean(lse ** 2))

        # Top-k selection
        topk_indices = np.argsort(-logits, axis=-1)[:, :self.top_k]
        batch_indices = np.arange(len(logits))[:, None]
        topk_probs = probs[batch_indices, topk_indices]
        # Renormalize top-k probabilities to sum to 1
        topk_weights = topk_probs / (np.sum(topk_probs, axis=-1, keepdims=True) + 1e-9)

        # Differentiable masked expert routing weights of shape (B, num_experts)
        topk_mask = np.zeros_like(probs, dtype=np.float32)
        topk_mask[batch_indices, topk_indices] = 1.0
        masked_probs_t = probs_t * _from_numpy(topk_mask)
        denom_t = masked_probs_t.sum(dim=-1, keepdim=True) + 1e-9
        self._last_expert_weights = masked_probs_t / denom_t

        # Load balancing auxiliary loss: N * sum_e (fraction_dispatched_e * mean_prob_e)
        fraction_dispatched = np.mean(topk_mask, axis=0)
        mean_prob = np.mean(probs, axis=0)
        aux_loss = float(self.num_experts * np.sum(fraction_dispatched * mean_prob))

        return topk_indices, topk_weights, aux_loss, z_loss


class MoELayer(Module):
    """
    Sparse Mixture of Experts feed-forward layer.
    Routes tokens dynamically to top-k expert sub-networks while preserving autograd.
    """
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        num_experts = _check_repeats("num_experts", num_experts)
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKGating(d_model=d_model, num_experts=num_experts, top_k=top_k)

        # Build expert networks: each expert is a 2-layer MLP
        self.experts_fc1 = ModuleList([Linear(d_model, hidden_dim) for _ in range(num_experts)])
        self.experts_fc2 = ModuleList([Linear(hidden_dim, d_model) for _ in range(num_experts)])

    def forward(self, x: Tensor) -> Tuple[Tensor, float, float]:
        """
        Returns:
            output: Tensor of matching leading shapes and d_model
            aux_loss: Load balancing penalty
            z_loss: Router z-loss
        """
        orig_shape = list(x.shape)
        if len(orig_shape) == 0 or int(np.prod(orig_shape)) == 0:
            return x, 0.0, 0.0

        single = len(orig_shape) == 1
        x_2d = x.unsqueeze(0) if single else x.reshape([-1, orig_shape[-1]])

        topk_indices, topk_weights, aux_loss, z_loss = self.router(x_2d)
        expert_weights_t = self.router._last_expert_weights

        total_out: Optional[Tensor] = None
        for e in range(self.num_experts):
            if not np.any(topk_indices == e):
                continue

            h = self.experts_fc1[e](x_2d).relu()
            exp_out = self.experts_fc2[e](h)

            if expert_weights_t is not None:
                e_mask = np.zeros((1, self.num_experts), dtype=np.float32)
                e_mask[0, e] = 1.0
                w_e = (expert_weights_t * _from_numpy(e_mask)).sum(dim=-1, keepdim=True)
            else:
                w_np = np.zeros((x_2d.shape[0], 1), dtype=np.float32)
                b_idx, k_idx = np.where(topk_indices == e)
                w_np[b_idx, 0] = topk_weights[b_idx, k_idx]
                w_e = _from_numpy(w_np)

            contrib = exp_out * w_e
            total_out = contrib if total_out is None else (total_out + contrib)

        if total_out is None:
            total_out = x_2d * 0.0

        if single:
            return total_out.squeeze(0), aux_loss, z_loss
        return total_out.reshape(orig_shape), aux_loss, z_loss


__all__ = [
    "TopKGating",
    "MoELayer",
]
