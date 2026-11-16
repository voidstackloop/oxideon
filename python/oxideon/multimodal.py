"""Multimodal Vision-Language representations, Cross-Attention, and CLIP contrastive loss for Oxideon.

Enables cross-modal alignment between images, text, and arbitrary modalities:
  - CLIPLoss: Symmetrical contrastive cross-entropy loss (Radford et al., 2021)
  - CrossAttentionBlock: Multi-head cross-attention querying across modalities
  - MultimodalProjection: L2-normalized linear projection heads
"""

from typing import Tuple, Optional, Any, List
import math
import numpy as np

from .oxideon import (
    Tensor as _Tensor,
    from_numpy as _from_numpy,
    cross_entropy,
)
from .nn import Module, Linear, Parameter


class CLIPLoss(Module):
    """Contrastive Language-Image Pre-training (CLIP) loss."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, image_embeddings: _Tensor, text_embeddings: _Tensor
    ) -> _Tensor:
        """Computes bidirectional cross-entropy loss between image and text embeddings [N, D]."""
        if image_embeddings.shape[0] != text_embeddings.shape[0]:
            raise ValueError(
                f"Batch dimension mismatch: image has {image_embeddings.shape[0]}, text has {text_embeddings.shape[0]}"
            )

        if image_embeddings.requires_grad or text_embeddings.requires_grad:
            img_norm = image_embeddings / ((image_embeddings * image_embeddings).sum(dim=-1, keepdim=True) + 1e-12).sqrt()
            txt_norm = text_embeddings / ((text_embeddings * text_embeddings).sum(dim=-1, keepdim=True) + 1e-12).sqrt()
            sim_i2t = img_norm.matmul(txt_norm.transpose(0, 1)) * (1.0 / max(self.temperature, 1e-4))
            sim_t2i = sim_i2t.transpose(0, 1)
            n = image_embeddings.shape[0]
            eye_mask = _from_numpy(np.eye(n, dtype=np.float32))
            loss_i2t = ((sim_i2t.log_softmax(-1) * eye_mask).sum(-1) * -1.0).mean()
            loss_t2i = ((sim_t2i.log_softmax(-1) * eye_mask).sum(-1) * -1.0).mean()
            return (loss_i2t + loss_t2i) * 0.5

        img = image_embeddings.to_numpy()
        txt = text_embeddings.to_numpy()

        # Normalize to unit sphere
        img = img / np.maximum(np.linalg.norm(img, axis=-1, keepdims=True), 1e-12)
        txt = txt / np.maximum(np.linalg.norm(txt, axis=-1, keepdims=True), 1e-12)

        # Cosine similarity matrix scaled by temperature [N, N]
        sim = (img @ txt.T) / max(self.temperature, 1e-4)

        n = sim.shape[0]
        labels = _from_numpy(np.arange(n, dtype=np.int64))

        sim_tensor_i2t = _from_numpy(np.ascontiguousarray(sim, dtype=np.float32))
        sim_tensor_t2i = _from_numpy(np.ascontiguousarray(sim.T, dtype=np.float32))

        loss_i2t = cross_entropy(sim_tensor_i2t, labels)
        loss_t2i = cross_entropy(sim_tensor_t2i, labels)

        total_loss = (loss_i2t + loss_t2i) * 0.5
        return total_loss


class CrossAttentionBlock(Module):
    """Multi-Head Cross-Attention querying target representations conditioned on context."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = Linear(query_dim, hidden_dim)
        self.k_proj = Linear(context_dim, hidden_dim)
        self.v_proj = Linear(context_dim, hidden_dim)
        self.out_proj = Linear(hidden_dim, query_dim)

    def forward(self, query: _Tensor, context: _Tensor) -> _Tensor:
        """Query [B, T_q, D_q] attends to Context [B, T_c, D_c]."""
        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)

        scale = 1.0 / math.sqrt(self.head_dim)
        head_outs = []
        for h in range(self.num_heads):
            q_h = q.narrow(2, h * self.head_dim, self.head_dim)
            k_h = k.narrow(2, h * self.head_dim, self.head_dim)
            v_h = v.narrow(2, h * self.head_dim, self.head_dim)
            scores_h = q_h.matmul(k_h.transpose(1, 2)) * scale
            attn_h = scores_h.softmax(-1)
            head_outs.append(attn_h.matmul(v_h))

        out = _Tensor.cat(head_outs, dim=2) if len(head_outs) > 1 else head_outs[0]
        return self.out_proj(out)


class MultimodalProjection(Module):
    """Projection head mapping modality features into shared embedding space with L2 normalization."""

    def __init__(self, in_dim: int, proj_dim: int = 256, normalize: bool = True):
        super().__init__()
        self.proj = Linear(in_dim, proj_dim)
        self.normalize = normalize

    def forward(self, x: _Tensor) -> _Tensor:
        out = self.proj(x)
        if self.normalize:
            if out.requires_grad:
                norm = ((out * out).sum(dim=-1, keepdim=True) + 1e-12).sqrt()
                return out / norm
            arr = out.to_numpy()
            norm = np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-12)
            out = _from_numpy(np.ascontiguousarray(arr / norm, dtype=np.float32))
        return out


__all__ = [
    "CLIPLoss",
    "CrossAttentionBlock",
    "MultimodalProjection",
]
