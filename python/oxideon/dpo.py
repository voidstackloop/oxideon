"""Preference Optimization for LLM alignment beyond standard pairwise DPO.

Implements:
  - ORPOLoss: Reference-free Odds Ratio Preference Optimization (Hong et al., 2024)
  - KTOLoss: Kahneman-Tversky Optimization for unpaired desirable/undesirable feedback (Ethayarajh et al., 2024)
  - SimPOLoss: Simple Preference Optimization with reference-free reward margin (Meng et al., 2024)
"""

from typing import Tuple, Dict, Any, Union, Optional
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module


class ORPOLoss(Module):
    """Reference-free Odds Ratio Preference Optimization (ORPO) loss (Hong et al., 2024).

    $$\\mathcal{L}_{ORPO} = \\mathcal{L}_{SFT} - \\lambda \\log \\sigma\\left(\\log \\frac{\\text{odds}(y_w)}{\\text{odds}(y_l)}\\right)$$
    """

    def __init__(self, lambda_coeff: float = 0.1):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def forward(
        self,
        policy_chosen_logps: _Tensor,
        policy_rejected_logps: _Tensor,
        sft_loss: Optional[_Tensor] = None,
    ) -> Tuple[_Tensor, Dict[str, float]]:
        """Computes differentiable ORPO loss combining odds ratio penalty with SFT loss."""
        pi_w_np = policy_chosen_logps.to_numpy().flatten()
        pi_l_np = policy_rejected_logps.to_numpy().flatten()
        if len(pi_w_np) != len(pi_l_np):
            raise ValueError(f"Length mismatch: chosen has {len(pi_w_np)}, rejected has {len(pi_l_np)}")
        if len(pi_w_np) == 0:
            total_loss_val = 0.0 if sft_loss is None else float(sft_loss.to_numpy().flat[0])
            metrics = {
                "orpo_loss": total_loss_val,
                "odds_ratio_loss": 0.0,
                "log_odds_ratio": 0.0,
            }
            out_t = sft_loss if sft_loss is not None else _from_numpy(np.array([0.0], dtype=np.float32))
            return out_t, metrics

        pi_w = policy_chosen_logps.reshape([-1]).clamp(-30.0, -1e-7)
        pi_l = policy_rejected_logps.reshape([-1]).clamp(-30.0, -1e-7)

        p_w = pi_w.exp()
        p_l = pi_l.exp()

        odds_w = p_w / (1.0 - p_w + 1e-8)
        odds_l = p_l / (1.0 - p_l + 1e-8)

        log_odds_ratio = (odds_w + 1e-8).log() - (odds_l + 1e-8).log()
        sig_lor = log_odds_ratio.clamp(-30.0, 30.0).sigmoid()
        or_loss = -(sig_lor + 1e-12).log()
        mean_or_loss_t = or_loss.mean()

        total_loss_t = mean_or_loss_t * float(self.lambda_coeff)
        if sft_loss is not None:
            total_loss_t = total_loss_t + sft_loss.mean()

        mean_or_val = float(mean_or_loss_t.to_numpy().flat[0])
        total_val = float(total_loss_t.to_numpy().flat[0])
        lor_val = float(log_odds_ratio.mean().to_numpy().flat[0])

        metrics = {
            "orpo_loss": total_val,
            "odds_ratio_loss": mean_or_val,
            "log_odds_ratio": lor_val,
        }
        return total_loss_t, metrics


class KTOLoss(Module):
    """Kahneman-Tversky Optimization (KTO) loss for unpaired binary human feedback (Ethayarajh et al., 2024).

    Optimizes for human utility based on prospect theory with loss aversion penalty.
    """

    def __init__(self, beta: float = 0.1, desirable_weight: float = 1.0, undesirable_weight: float = 1.33):
        super().__init__()
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_logps: _Tensor,
        reference_logps: _Tensor,
        labels: _Tensor,  # 1 for desirable, 0 for undesirable
        kl_ref: float = 0.0,
    ) -> Tuple[_Tensor, Dict[str, float]]:
        """Computes differentiable KTO loss over a batch of independently labeled responses."""
        pi_np = policy_logps.to_numpy().flatten()
        ref_np = reference_logps.to_numpy().flatten()
        lbls_np = labels.to_numpy().flatten()
        if not (len(pi_np) == len(ref_np) == len(lbls_np)):
            raise ValueError("Length mismatch among policy_logps, reference_logps, and labels")
        if len(pi_np) == 0:
            metrics = {
                "kto_loss": 0.0,
                "mean_reward": 0.0,
                "reward_margin": 0.0,
            }
            return _from_numpy(np.array([0.0], dtype=np.float32)), metrics

        pi_t = policy_logps.reshape([-1])
        ref_t = reference_logps.reshape([-1])
        rewards_t = (pi_t - ref_t) * float(self.beta)

        loss_desirable_t = 1.0 - (rewards_t - float(kl_ref)).clamp(-30.0, 30.0).sigmoid()
        loss_undesirable_t = 1.0 - (float(kl_ref) - rewards_t).clamp(-30.0, 30.0).sigmoid()

        des_mask = _from_numpy((lbls_np > 0.5).astype(np.float32))
        undes_mask = _from_numpy((lbls_np <= 0.5).astype(np.float32))

        weighted_loss_t = (
            des_mask * (loss_desirable_t * float(self.desirable_weight))
            + undes_mask * (loss_undesirable_t * float(self.undesirable_weight))
        )
        mean_loss_t = weighted_loss_t.mean()

        rewards_np = rewards_t.to_numpy().flatten()
        mean_loss_val = float(mean_loss_t.to_numpy().flat[0])
        metrics = {
            "kto_loss": mean_loss_val,
            "mean_reward": float(np.mean(rewards_np)),
            "reward_margin": (
                float(np.mean(rewards_np[lbls_np > 0.5]) - np.mean(rewards_np[lbls_np <= 0.5]))
                if np.any(lbls_np > 0.5) and np.any(lbls_np <= 0.5)
                else 0.0
            ),
        }
        return mean_loss_t, metrics


class SimPOLoss(Module):
    """Simple Preference Optimization (SimPO) reference-free loss with target reward margin (Meng et al., 2024)."""

    def __init__(self, beta: float = 2.0, gamma: float = 0.5):
        super().__init__()
        self.beta = float(beta)
        self.gamma = float(gamma)

    def forward(
        self,
        chosen_avg_logps: _Tensor,
        rejected_avg_logps: _Tensor,
    ) -> Tuple[_Tensor, Dict[str, float]]:
        w = chosen_avg_logps.reshape([-1])
        l = rejected_avg_logps.reshape([-1])
        if w.shape[0] != l.shape[0]:
            raise ValueError(f"Length mismatch: chosen has {w.shape[0]}, rejected has {l.shape[0]}")
        logits = (w - l - self.gamma) * self.beta
        loss_t = -(logits.clamp(-30.0, 30.0).sigmoid() + 1e-12).log().mean()
        metrics = {
            "simpo_loss": float(loss_t.to_numpy().flat[0]),
            "reward_margin": float(((w - l) * self.beta).mean().to_numpy().flat[0]),
        }
        return loss_t, metrics


__all__ = [
    "ORPOLoss",
    "KTOLoss",
    "SimPOLoss",
]
