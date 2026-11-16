"""Post-training alignment, preference optimization, and RLHF for Oxideon.

Implements state-of-the-art preference loss objectives and reward modeling:
  - DPOLoss: Direct Preference Optimization (Rafailov et al., 2023)
  - KTOLoss: Kahneman-Tversky Optimization for unpaired binary feedback (Ethayarajh et al., 2024)
  - SimPOLoss: Reference-free Simple Preference Optimization (Meng et al., 2024)
  - RewardModelHead: Scalar score projection module for reward modeling
  - PairwisePreferenceDataset: Container for (prompt, chosen, rejected) alignment pairs
  - RLOOLoss: REINFORCE leave-one-out baselines over K samples per prompt
  - ReMaxLoss: greedy-decoding baseline for low-variance RLHF
  - DrGRPOLoss: debiased GRPO with group-mean (no std) advantages
  - DAPOLoss: asymmetric clip-higher, dynamic sampling, overlong shaping
  - PPOClipLoss: clipped surrogate with value loss and clip-higher support
  - ProcessRewardHead / PRMLoss / aggregate_process_scores: step supervision
"""

from typing import Tuple, Optional, Any, List, Dict
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, Parameter


def _logsigmoid(z: _Tensor) -> _Tensor:
    return (z.sigmoid() + 1e-12).log()


class DPOLoss(Module):
    """Direct Preference Optimization (DPO) objective with full Tensor autograd and label smoothing."""

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        policy_chosen_logps: _Tensor,
        policy_rejected_logps: _Tensor,
        reference_chosen_logps: _Tensor,
        reference_rejected_logps: _Tensor,
    ) -> Tuple[_Tensor, _Tensor, _Tensor]:
        """Calculates DPO loss, chosen rewards, and rejected rewards.

        Returns: (loss, chosen_rewards, rejected_rewards)
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = (pi_logratios - ref_logratios) * self.beta
        pos_loss = _logsigmoid(logits) * (-1.0)
        neg_loss = _logsigmoid(logits * (-1.0)) * (-1.0)
        losses = pos_loss * (1.0 - self.label_smoothing) + neg_loss * self.label_smoothing

        chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta

        loss_tensor = losses.mean().reshape([1])
        return loss_tensor, chosen_rewards, rejected_rewards


class SimPOLoss(Module):
    """Simple Preference Optimization (SimPO) loss (reference-model-free) with Tensor autograd."""

    def __init__(self, beta: float = 2.0, gamma: float = 0.5):
        super().__init__()
        self.beta = float(beta)
        self.gamma = float(gamma)

    def forward(
        self,
        chosen_logps: _Tensor,
        rejected_logps: _Tensor,
        chosen_lens: Optional[Any] = None,
        rejected_lens: Optional[Any] = None,
    ) -> _Tensor:
        """Computes reference-free length-normalized margin loss."""
        w = chosen_logps
        l = rejected_logps

        if chosen_lens is not None:
            w_len = np.maximum(np.asarray(chosen_lens, dtype=np.float32), 1.0)
            w = w * _from_numpy(np.ascontiguousarray((1.0 / w_len).astype(np.float32)))
        if rejected_lens is not None:
            l_len = np.maximum(np.asarray(rejected_lens, dtype=np.float32), 1.0)
            l = l * _from_numpy(np.ascontiguousarray((1.0 / l_len).astype(np.float32)))

        logits = (w - l) * self.beta - self.gamma
        losses = _logsigmoid(logits) * (-1.0)
        return losses.mean().reshape([1])


class KTOLoss(Module):
    """Kahneman-Tversky Optimization (KTO) loss for unpaired feedback with Tensor autograd."""

    def __init__(
        self,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
    ):
        super().__init__()
        self.beta = float(beta)
        self.desirable_weight = float(desirable_weight)
        self.undesirable_weight = float(undesirable_weight)

    def forward(
        self,
        policy_logps: _Tensor,
        reference_logps: _Tensor,
        kl: float,
        is_desirable: Any,
    ) -> _Tensor:
        desirable = np.asarray(is_desirable, dtype=bool)
        des_mask = _from_numpy(np.ascontiguousarray(desirable.astype(np.float32)))
        undes_mask = _from_numpy(np.ascontiguousarray((~desirable).astype(np.float32)))

        u = (policy_logps - reference_logps) * self.beta
        sig_des = (u - float(kl)).sigmoid()
        sig_undes = (float(kl) - u).sigmoid()

        loss_des = (1.0 - sig_des) * self.desirable_weight
        loss_undes = (1.0 - sig_undes) * self.undesirable_weight
        losses = loss_des * des_mask + loss_undes * undes_mask
        return losses.mean().reshape([1])


class RewardModelHead(Module):
    """Linear projection from model hidden dimension to scalar reward."""

    def __init__(self, in_features: int, bias: bool = True):
        super().__init__()
        self.proj = Linear(in_features, 1, bias=bias)

    def forward(self, hidden_states: _Tensor) -> _Tensor:
        """Returns scalar reward tensor [B, 1] or [B]."""
        return self.proj(hidden_states)


class PairwisePreferenceDataset:
    """Dataset storing pairwise preference comparisons."""

    def __init__(
        self,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
    ):
        if not (len(prompts) == len(chosen) == len(rejected)):
            raise ValueError("prompts, chosen, and rejected must have identical lengths")
        self.prompts = prompts
        self.chosen = chosen
        self.rejected = rejected

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            "prompt": self.prompts[idx],
            "chosen": self.chosen[idx],
            "rejected": self.rejected[idx],
        }


__all__ = [
    "DPOLoss",
    "SimPOLoss",
    "KTOLoss",
    "RewardModelHead",
    "PairwisePreferenceDataset",
    "RLOOLoss",
    "ReMaxLoss",
    "DrGRPOLoss",
    "DAPOLoss",
    "PPOClipLoss",
    "ProcessRewardHead",
    "aggregate_process_scores",
    "PRMLoss",
]


def _as_matrix(tensor: _Tensor, name: str) -> np.ndarray:
    arr = np.asarray(tensor.to_numpy(), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty [batch, group] matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_vector(values: Any, name: str, n: int) -> np.ndarray:
    arr = np.asarray(values.to_numpy() if isinstance(values, _Tensor) else values, dtype=np.float64).flatten()
    if arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} entries, expected {n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_rewards(rewards: Any, shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(
        rewards.to_numpy() if isinstance(rewards, _Tensor) else rewards, dtype=np.float64
    )
    if arr.shape != shape:
        raise ValueError(f"rewards shape {arr.shape} != expected {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("rewards must be finite")
    return arr


def _scalar_loss(value: float) -> _Tensor:
    return _from_numpy(np.array([float(value)], dtype=np.float32))


class RLOOLoss(Module):
    """REINFORCE Leave-One-Out baseline (Ahmadian et al., 2024) with Tensor autograd."""

    def __init__(self, normalize_advantages: bool = False):
        super().__init__()
        self.normalize_advantages = normalize_advantages

    def forward(
        self, policy_logps: _Tensor, rewards: Any
    ) -> Tuple[_Tensor, Dict[str, float]]:
        """Computes -mean((r - leave-one-out-mean) * logp) over [B, K]."""
        logp = _as_matrix(policy_logps, "policy_logps")
        rew = _as_rewards(rewards, logp.shape)
        k = logp.shape[1]
        if k < 2:
            raise ValueError(f"RLOO needs K >= 2 samples per prompt, got K = {k}")
        baseline = (rew.sum(axis=1, keepdims=True) - rew) / (k - 1)
        adv = rew - baseline
        if self.normalize_advantages:
            std = adv.std()
            adv = (adv - adv.mean()) / (std if std > 1e-8 else 1.0)
        adv_t = _from_numpy(np.ascontiguousarray(adv.astype(np.float32)))
        loss_t = (policy_logps * adv_t).mean() * (-1.0)
        loss = float(loss_t.to_numpy().item())
        metrics = {
            "rloo_loss": loss,
            "mean_advantage": float(np.mean(adv)),
            "mean_reward": float(np.mean(rew)),
        }
        return loss_t.reshape([1]), metrics


class ReMaxLoss(Module):
    """ReMax baseline from greedy decoding (Li et al., 2023) with Tensor autograd."""

    def forward(
        self, policy_logps: _Tensor, rewards: Any, greedy_rewards: Any
    ) -> Tuple[_Tensor, Dict[str, float]]:
        """Computes -mean((r - r_greedy) * logp) over flat batches."""
        logp = np.asarray(policy_logps.to_numpy(), dtype=np.float64).flatten()
        if logp.shape[0] < 1 or not np.all(np.isfinite(logp)):
            raise ValueError("policy_logps must be a non-empty finite batch")
        rew = _as_vector(rewards, "rewards", logp.shape[0])
        greedy = _as_vector(greedy_rewards, "greedy_rewards", logp.shape[0])
        adv = rew - greedy
        adv_t = _from_numpy(np.ascontiguousarray(adv.reshape(policy_logps.shape).astype(np.float32)))
        loss_t = (policy_logps * adv_t).mean() * (-1.0)
        loss = float(loss_t.to_numpy().item())
        metrics = {
            "remax_loss": loss,
            "mean_advantage": float(np.mean(adv)),
            "mean_reward": float(np.mean(rew)),
        }
        return loss_t.reshape([1]), metrics


class DrGRPOLoss(Module):
    """Debiased GRPO (Liu et al., 2025): group-mean baseline, no std normalization."""

    def forward(
        self, policy_logps: _Tensor, rewards: Any
    ) -> Tuple[_Tensor, Dict[str, float]]:
        """Computes -mean((r - group_mean) * logp) over [B, K]."""
        logp = _as_matrix(policy_logps, "policy_logps")
        rew = _as_rewards(rewards, logp.shape)
        adv = rew - rew.mean(axis=1, keepdims=True)
        adv_t = _from_numpy(np.ascontiguousarray(adv.astype(np.float32)))
        loss_t = (policy_logps * adv_t).mean() * (-1.0)
        loss = float(loss_t.to_numpy().item())
        metrics = {
            "drgrpo_loss": loss,
            "mean_advantage": float(np.mean(adv)),
            "mean_reward": float(np.mean(rew)),
            "frac_zero_advantage": float(np.mean(np.abs(adv) < 1e-12)),
        }
        return loss_t.reshape([1]), metrics


class DAPOLoss(Module):
    """Decoupled Clip and Dynamic Sampling Policy Optimization (Yu et al., 2025)."""

    def __init__(
        self,
        clip_low: float = 0.2,
        clip_high: float = 0.28,
        overlong_cache: Optional[int] = None,
    ):
        super().__init__()
        if clip_low < 0 or clip_high < 0:
            raise ValueError("clip bounds must be non-negative")
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.overlong_cache = overlong_cache

    def _shape_lengths(self, lengths: Any, shape: tuple, max_len: int) -> np.ndarray:
        lens = np.asarray(
            lengths.to_numpy() if isinstance(lengths, _Tensor) else lengths,
            dtype=np.float64,
        )
        if lens.shape != shape:
            raise ValueError(f"lengths shape {lens.shape} != expected {shape}")
        if (lens < 0).any():
            raise ValueError("lengths must be non-negative")
        cache = self.overlong_cache if self.overlong_cache is not None else max_len // 4
        if cache < 1:
            raise ValueError("overlong cache must cover at least one token")
        floor = max_len - cache
        penalty = np.zeros_like(lens)
        over = lens > max_len
        soft = (lens > floor) & ~over
        penalty[soft] = (floor - lens[soft]) / cache
        penalty[over] = -1.0
        return penalty

    def forward(
        self,
        policy_logps: _Tensor,
        old_logps: _Tensor,
        rewards: Any,
        lengths: Any = None,
        max_len: Optional[int] = None,
    ) -> Tuple[_Tensor, Dict[str, float]]:
        """Clipped surrogate over [B, K] with dynamic sampling of flat groups."""
        logp = _as_matrix(policy_logps, "policy_logps")
        old = _as_matrix(old_logps, "old_logps")
        if old.shape != logp.shape:
            raise ValueError(f"old_logps shape {old.shape} != {logp.shape}")
        rew = _as_rewards(rewards, logp.shape)
        if lengths is not None:
            if max_len is None or max_len < 1:
                raise ValueError("overlong shaping needs max_len >= 1")
            rew = rew + self._shape_lengths(lengths, logp.shape, max_len)
        mean = rew.mean(axis=1, keepdims=True)
        std = rew.std(axis=1, keepdims=True)
        live = (std > 1e-8).flatten()
        adv = np.zeros_like(rew)
        adv[live] = (rew[live] - mean[live]) / std[live]
        ratio = np.exp(np.clip(logp - old, -30.0, 30.0))
        clipped = np.clip(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high)
        unclipped_wins = (ratio * adv <= clipped * adv).astype(np.float32)
        live_mask = np.repeat(live, logp.shape[1]).reshape(logp.shape).astype(np.float32)
        n_live = int(live.sum()) * logp.shape[1]

        ratio_t = (policy_logps - old_logps.detach()).clamp(-30.0, 30.0).exp()
        clipped_t = ratio_t.clamp(1.0 - self.clip_low, 1.0 + self.clip_high)
        adv_t = _from_numpy(np.ascontiguousarray(adv.astype(np.float32)))
        win_t = _from_numpy(np.ascontiguousarray(unclipped_wins))
        live_t = _from_numpy(np.ascontiguousarray(live_mask))
        terms_t = (ratio_t * adv_t * win_t + clipped_t * adv_t * (1.0 - win_t)) * live_t
        loss_t = terms_t.sum() * (-1.0 / float(n_live)) if n_live else policy_logps.sum() * 0.0
        loss = float(loss_t.to_numpy().item())
        metrics = {
            "dapo_loss": loss,
            "frac_filtered": float(1.0 - live.mean()),
            "clipfrac_low": float(np.mean(ratio < 1.0 - self.clip_low)),
            "clipfrac_high": float(np.mean(ratio > 1.0 + self.clip_high)),
            "mean_advantage": float(np.mean(adv)),
        }
        return loss_t.reshape([1]), metrics


class PPOClipLoss(Module):
    """Clipped PPO surrogate with value loss and optional clip-higher (VAPO/DAPO style)."""

    def __init__(self, clip: float = 0.2, clip_high: Optional[float] = None, vf_coef: float = 0.5):
        super().__init__()
        if clip < 0:
            raise ValueError("clip must be non-negative")
        if clip_high is not None and clip_high < 0:
            raise ValueError("clip_high must be non-negative")
        self.clip = float(clip)
        self.clip_high = float(clip if clip_high is None else clip_high)
        self.vf_coef = float(vf_coef)

    def forward(
        self,
        policy_logps: _Tensor,
        old_logps: _Tensor,
        advantages: Any,
        returns: Any,
        values: Any,
    ) -> Tuple[_Tensor, Dict[str, float]]:
        """Policy + value loss over flat per-sequence batches with Tensor autograd."""
        logp = np.asarray(policy_logps.to_numpy(), dtype=np.float64).flatten()
        old = np.asarray(old_logps.to_numpy(), dtype=np.float64).flatten()
        if logp.shape != old.shape or logp.shape[0] < 1:
            raise ValueError("policy/old logps must be non-empty flat batches of equal size")
        if not np.all(np.isfinite(logp)) or not np.all(np.isfinite(old)):
            raise ValueError("policy/old logps must be finite")
        n = logp.shape[0]
        adv = _as_vector(advantages, "advantages", n)
        ret = _as_vector(returns, "returns", n)
        val = _as_vector(values, "values", n)
        ratio = np.exp(np.clip(logp - old, -30.0, 30.0))
        clipped = np.clip(ratio, 1.0 - self.clip, 1.0 + self.clip_high)
        unclipped_wins = (ratio * adv <= clipped * adv).astype(np.float32)

        pi_flat = policy_logps.reshape([n])
        old_flat = old_logps.detach().reshape([n])
        ratio_t = (pi_flat - old_flat).clamp(-30.0, 30.0).exp()
        clipped_t = ratio_t.clamp(1.0 - self.clip, 1.0 + self.clip_high)
        adv_t = _from_numpy(np.ascontiguousarray(adv.astype(np.float32)))
        win_t = _from_numpy(np.ascontiguousarray(unclipped_wins))
        surr_t = ratio_t * adv_t * win_t + clipped_t * adv_t * (1.0 - win_t)
        policy_loss_t = surr_t.mean() * (-1.0)

        ret_t = _from_numpy(np.ascontiguousarray(ret.astype(np.float32)))
        val_t = values.reshape([n]) if isinstance(values, _Tensor) else _from_numpy(np.ascontiguousarray(val.astype(np.float32)))
        diff_v = ret_t - val_t
        value_loss_t = (diff_v * diff_v).mean()
        total_t = policy_loss_t + value_loss_t * self.vf_coef

        policy_loss = float(policy_loss_t.to_numpy().item())
        value_loss = float(value_loss_t.to_numpy().item())
        total = float(total_t.to_numpy().item())
        ret_var = float(np.var(ret))
        metrics = {
            "ppo_loss": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "clipfrac": float(np.mean((ratio < 1.0 - self.clip) | (ratio > 1.0 + self.clip_high))),
            "explained_variance": float(1.0 - np.var(ret - val) / ret_var) if ret_var > 1e-12 else 0.0,
        }
        return total_t.reshape([1]), metrics


class ProcessRewardHead(Module):
    """Per-token process reward scores for PRM-style step supervision."""

    def __init__(self, in_features: int):
        super().__init__()
        self.proj = Linear(in_features, 1)

    def forward(self, hidden_states: _Tensor) -> _Tensor:
        """Scores [B, S, H] hidden states into [B, S] step logits while preserving autograd."""
        out = self.proj(hidden_states)
        sh = list(out.shape)
        if len(sh) != 3 or sh[2] != 1:
            raise ValueError(f"expected [B, S, 1] projection, got {tuple(sh)}")
        return out.reshape([sh[0], sh[1]])


def aggregate_process_scores(
    scores: _Tensor, mask: Any = None, method: str = "mean"
) -> _Tensor:
    """Aggregates [B, S] step scores into per-response rewards."""
    arr = np.asarray(scores.to_numpy(), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("scores must be a non-empty [B, S] matrix")
    if mask is None:
        keep = np.ones_like(arr, dtype=bool)
    else:
        keep = np.asarray(
            mask.to_numpy() if isinstance(mask, _Tensor) else mask, dtype=bool
        )
        if keep.shape != arr.shape:
            raise ValueError(f"mask shape {keep.shape} != scores shape {arr.shape}")
        if not keep.any(axis=1).all():
            raise ValueError("every response needs at least one valid step")
    if method == "mean":
        out = np.stack([arr[i][keep[i]].mean() for i in range(arr.shape[0])])
    elif method == "min":
        out = np.stack([arr[i][keep[i]].min() for i in range(arr.shape[0])])
    elif method == "last":
        out = np.stack([arr[i][keep[i]][-1] for i in range(arr.shape[0])])
    elif method == "prod":
        prob = 1.0 / (1.0 + np.exp(-np.clip(arr, -30.0, 30.0)))
        out = np.stack([prob[i][keep[i]].prod() for i in range(arr.shape[0])])
    else:
        raise ValueError(f"unknown aggregation '{method}' (mean/min/prod/last)")
    return _from_numpy(np.ascontiguousarray(out.astype(np.float32)))


class PRMLoss(Module):
    """Binary cross-entropy over per-step process labels with padding mask and Tensor autograd."""

    def forward(
        self, scores: _Tensor, labels: Any, mask: Any = None
    ) -> Tuple[_Tensor, Dict[str, float]]:
        """BCE-with-logits between step scores and 0/1 step labels."""
        arr = np.asarray(scores.to_numpy(), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 1:
            raise ValueError("scores must be a non-empty [B, S] matrix")
        lab = np.asarray(
            labels.to_numpy() if isinstance(labels, _Tensor) else labels, dtype=np.float64
        )
        if lab.shape != arr.shape or not np.all((lab == 0) | (lab == 1)):
            raise ValueError("labels must be 0/1 with the scores shape")
        if mask is None:
            keep = np.ones_like(arr, dtype=bool)
        else:
            keep = np.asarray(
                mask.to_numpy() if isinstance(mask, _Tensor) else mask, dtype=bool
            )
            if keep.shape != arr.shape:
                raise ValueError(f"mask shape {keep.shape} != scores shape {arr.shape}")
        if not keep.any():
            raise ValueError("mask leaves no supervised steps")

        lab_t = _from_numpy(np.ascontiguousarray(lab.astype(np.float32)))
        keep_t = _from_numpy(np.ascontiguousarray(keep.astype(np.float32)))
        probs = scores.sigmoid()
        eps = 1e-7
        bce_t = (lab_t * (probs + eps).log() + (1.0 - lab_t) * (1.0 - probs + eps).log()) * (-1.0)
        n_valid = float(np.sum(keep))
        loss_t = (bce_t * keep_t).sum() * (1.0 / n_valid)

        bce = np.maximum(arr, 0.0) - arr * lab + np.log1p(np.exp(-np.abs(arr)))
        loss = float(bce[keep].mean())
        pred = (arr > 0) == (lab > 0.5)
        metrics = {"prm_loss": loss, "step_accuracy": float(pred[keep].mean())}
        return loss_t.reshape([1]), metrics
