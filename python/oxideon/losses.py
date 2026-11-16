"""Advanced specialized loss functions for Oxideon."""

from typing import Optional, Any, Sequence, Union, Tuple, List
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy, cross_entropy as _native_ce
from .nn import (
    Module,
    Parameter,
    HuberLoss,
    CosineEmbeddingLoss,
    GaussianNLLLoss,
    CTCLoss,
)


class DiceLoss(Module):
    """Dice Loss for image segmentation: 1 - (2 * intersection + smooth) / (total + smooth)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, pred: _Tensor, target: _Tensor) -> _Tensor:
        if isinstance(pred, _Tensor) and (pred.requires_grad or (isinstance(target, _Tensor) and target.requires_grad)):
            t_t = target if isinstance(target, _Tensor) else _from_numpy(np.asarray(target, dtype=np.float32))
            p_flat = pred.reshape([-1])
            t_flat = t_t.reshape([-1])
            intersection = (p_flat * t_flat).sum()
            total = p_flat.sum() + t_flat.sum()
            dice = (intersection * 2.0 + self.smooth) / (total + self.smooth)
            return (dice * -1.0) + 1.0

        p_np = pred.to_numpy().reshape(-1)
        t_np = target.to_numpy().reshape(-1)

        intersection = np.sum(p_np * t_np)
        total = np.sum(p_np) + np.sum(t_np)

        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        loss = 1.0 - dice
        return _from_numpy(np.array(loss, dtype=np.float32))


class ContrastiveLoss(Module):
    """Contrastive loss for Siamese embedding networks:
    L = 0.5 * (1 - y) * D^2 + 0.5 * y * max(0, margin - D)^2
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = float(margin)

    def forward(self, x1: _Tensor, x2: _Tensor, target: _Tensor) -> _Tensor:
        diff = x1 - x2
        dist_sq = (diff * diff).sum(-1)
        dist = (dist_sq + 1e-12).sqrt()

        if (isinstance(x1, _Tensor) and x1.requires_grad) or (isinstance(x2, _Tensor) and x2.requires_grad):
            t_np = target.to_numpy().reshape(-1).astype(np.float32) if isinstance(target, _Tensor) else np.asarray(target, dtype=np.float32).reshape(-1)
            y_t = _from_numpy(np.ascontiguousarray(t_np))
            one_minus_y = _from_numpy(np.ascontiguousarray(1.0 - t_np))
            d_flat = dist.reshape([-1])
            dsq_flat = dist_sq.reshape([-1])
            margin_hinge = ((d_flat * -1.0) + self.margin).relu()
            loss_vec = one_minus_y * dsq_flat + y_t * (margin_hinge * margin_hinge)
            return loss_vec.mean() * 0.5

        dist_np = dist.to_numpy().reshape(-1)
        target_np = target.to_numpy().reshape(-1)

        loss_sim = (1.0 - target_np) * (dist_np ** 2)
        loss_dissim = target_np * (np.maximum(0.0, self.margin - dist_np) ** 2)

        loss = 0.5 * np.mean(loss_sim + loss_dissim)
        return _from_numpy(np.array(loss, dtype=np.float32))


class LabelSmoothingCrossEntropy(Module):
    """Cross Entropy with uniform label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= float(smoothing) <= 1.0:
            raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")
        self.smoothing = float(smoothing)

    def forward(self, logits: _Tensor, target: _Tensor) -> _Tensor:
        if len(logits.shape) != 2:
            raise ValueError(f"LabelSmoothingCrossEntropy expects 2D [B, C] logits, got shape {logits.shape}")
        log_probs = logits.log_softmax(-1)
        num_classes = logits.shape[-1]
        tgt = target.to_numpy().reshape(-1) if isinstance(target, _Tensor) else np.asarray(target).reshape(-1)
        if tgt.shape != (logits.shape[0],):
            raise ValueError(f"target length {tgt.shape[0]} must match batch size {logits.shape[0]}")
        if tgt.dtype.kind not in "iu" and np.any(tgt != np.floor(tgt)):
            raise ValueError("target must contain integers")
        if np.any(tgt < 0) or np.any(tgt >= num_classes):
            raise ValueError(f"target out of range [0, {num_classes})")

        if isinstance(logits, _Tensor) and logits.requires_grad:
            tgt_np = target.to_numpy().reshape(-1).astype(int) if isinstance(target, _Tensor) else np.asarray(target, dtype=int).reshape(-1)
            batch_size = logits.shape[0]
            one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
            one_hot[np.arange(batch_size), tgt_np] = 1.0
            nll_loss = ((log_probs * _from_numpy(one_hot)).sum(-1) * -1.0).mean()
            smooth_loss = log_probs.mean() * -1.0
            return nll_loss * (1.0 - self.smoothing) + smooth_loss * self.smoothing

        lp_np = log_probs.to_numpy()
        tgt_np = target.to_numpy().reshape(-1).astype(int)

        batch_size = lp_np.shape[0]
        nll_loss = -np.mean(lp_np[np.arange(batch_size), tgt_np])
        smooth_loss = -np.mean(lp_np)

        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return _from_numpy(np.array(loss, dtype=np.float32))


class ArcFaceLoss(Module):
    """Additive Angular Margin Loss (ArcFace) for deep representation learning."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = Parameter(_Tensor.randn([out_features, in_features]))

    def forward(self, embeddings: _Tensor, labels: _Tensor) -> _Tensor:
        if len(embeddings.shape) != 2:
            raise ValueError(f"ArcFaceLoss expects 2D [B, D] embeddings, got shape {embeddings.shape}")
        lbl = labels.to_numpy().reshape(-1) if isinstance(labels, _Tensor) else np.asarray(labels).reshape(-1)
        if lbl.shape != (embeddings.shape[0],):
            raise ValueError(f"labels length {lbl.shape[0]} must match batch size {embeddings.shape[0]}")
        if lbl.dtype.kind not in "iu" and np.any(lbl != np.floor(lbl)):
            raise ValueError("labels must contain integers")
        if np.any(lbl < 0) or np.any(lbl >= self.out_features):
            raise ValueError(f"labels out of range [0, {self.out_features})")
        if (isinstance(embeddings, _Tensor) and embeddings.requires_grad) or (isinstance(self.weight, _Tensor) and self.weight.requires_grad):
            emb_norm = embeddings / ((embeddings * embeddings).sum(dim=-1, keepdim=True) + 1e-8).sqrt()
            w_norm = self.weight / ((self.weight * self.weight).sum(dim=-1, keepdim=True) + 1e-8).sqrt()
            cosine = emb_norm.matmul(w_norm.transpose(0, 1)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            sine = ((cosine * cosine * -1.0 + 1.0).relu() + 1e-8).sqrt()
            phi = cosine * self.cos_m - sine * self.sin_m

            lbl_np = labels.to_numpy().reshape(-1).astype(int) if isinstance(labels, _Tensor) else np.asarray(labels, dtype=int).reshape(-1)
            batch_size = embeddings.shape[0]
            mask = np.zeros((batch_size, self.out_features), dtype=np.float32)
            valid = (lbl_np >= 0) & (lbl_np < self.out_features)
            mask[np.arange(batch_size)[valid], lbl_np[valid]] = 1.0
            mask_t = _from_numpy(mask)
            inv_mask_t = _from_numpy(1.0 - mask)

            logits = (cosine * inv_mask_t + phi * mask_t) * self.s
            log_probs = logits.log_softmax(-1)
            return ((log_probs * mask_t).sum(-1) * -1.0).mean()

        emb_np = embeddings.to_numpy()
        w_np = self.weight.to_numpy()

        emb_norm_np = emb_np / (np.linalg.norm(emb_np, axis=-1, keepdims=True) + 1e-8)
        w_norm_np = w_np / (np.linalg.norm(w_np, axis=-1, keepdims=True) + 1e-8)

        cosine = emb_norm_np @ w_norm_np.T
        cosine = np.clip(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        sine = np.sqrt(np.maximum(0.0, 1.0 - cosine ** 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        lbl_np = labels.to_numpy().reshape(-1).astype(int)
        batch_size = emb_norm_np.shape[0]
        out_np = cosine.copy()
        valid = (lbl_np >= 0) & (lbl_np < self.out_features)
        rows = np.arange(batch_size)[valid]
        cols = lbl_np[valid]
        out_np[rows, cols] = phi[rows, cols]

        scaled_logits = _from_numpy(np.ascontiguousarray(out_np * self.s, dtype=np.float32))
        return _native_ce(scaled_logits, labels)


__all__ = [
    "DiceLoss",
    "ContrastiveLoss",
    "LabelSmoothingCrossEntropy",
    "ArcFaceLoss",
    "HuberLoss",
    "CosineEmbeddingLoss",
    "GaussianNLLLoss",
    "CTCLoss",
]
