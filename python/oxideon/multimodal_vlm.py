"""Vision-Language Model (VLM) Connectors and Perceiver Resamplers.

Implements cross-modal alignment between visual backbones (ViT) and autoregressive LLMs:
- VLMProjection: 2-layer MLP projection mapping visual patch vectors to text embeddings (LLaVA style)
- PerceiverResampler: Cross-attention resampler compressing variable image tokens to fixed latents (Flamingo style)
- VisionLanguageConnector: Unified connector assembling multi-modal visual prefix sequences
"""

from typing import Optional, List, Tuple
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, LayerNorm, GELU, Parameter, ModuleList


class VLMProjection(Module):
    """2-layer MLP projector mapping visual patch features to LLM text embedding dimension."""

    def __init__(self, visual_dim: int, text_dim: int):
        super().__init__()
        self.proj1 = Linear(visual_dim, text_dim)
        self.act = GELU()
        self.proj2 = Linear(text_dim, text_dim)

    def forward(self, visual_tokens: _Tensor) -> _Tensor:
        """Args: visual_tokens: [batch, num_patches, visual_dim] -> [batch, num_patches, text_dim]"""
        return self.proj2(self.act(self.proj1(visual_tokens)))


class PerceiverResampler(Module):
    """Flamingo-style Perceiver Resampler using learnable latent queries to compress visual sequences."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        num_latents: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads

        # Learnable latent queries
        latents_init = (np.random.randn(num_latents, text_dim) * 0.02).astype(np.float32)
        self.latents = Parameter(_from_numpy(latents_init))

        self.visual_proj = Linear(visual_dim, text_dim)
        self.q_proj = Linear(text_dim, text_dim)
        self.k_proj = Linear(text_dim, text_dim)
        self.v_proj = Linear(text_dim, text_dim)
        self.out_proj = Linear(text_dim, text_dim)
        self.norm = LayerNorm(text_dim)

    def forward(self, visual_features: _Tensor) -> _Tensor:
        """Args: visual_features: [batch, num_patches, visual_dim] -> [batch, num_latents, text_dim]"""
        b = visual_features.shape[0]
        if b == 0:
            return _from_numpy(np.zeros((0, self.num_latents, self.text_dim), dtype=np.float32))

        latents_batch = _Tensor.cat([self.latents.unsqueeze(0) for _ in range(b)], dim=0)

        num_patches = visual_features.shape[1] if visual_features.ndim > 1 else 0
        if num_patches == 0:
            return self.norm(latents_batch)

        v_proj = self.visual_proj(visual_features)  # [b, num_patches, text_dim]

        # Multi-head cross-attention: Q from latents, K/V from visual_features
        q = self.q_proj(latents_batch).reshape([b, self.num_latents, self.num_heads, self.head_dim]).transpose(1, 2)
        k = self.k_proj(v_proj).reshape([b, num_patches, self.num_heads, self.head_dim]).transpose(1, 2)
        v = self.v_proj(v_proj).reshape([b, num_patches, self.num_heads, self.head_dim]).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = q.matmul(k.transpose(2, 3)) * scale
        attn = scores.softmax(-1)

        out = attn.matmul(v).transpose(1, 2).reshape([b, self.num_latents, self.text_dim])
        out_tensor = self.out_proj(out)
        res = latents_batch + out_tensor
        return self.norm(res)


class VisionLanguageConnector(Module):
    """End-to-end multi-modal vision-language connector prefixing image tokens to text prompts."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        mode: str = "mlp",
        num_latents: int = 32,
    ):
        super().__init__()
        self.mode = mode
        if mode == "resampler":
            self.projector = PerceiverResampler(visual_dim=visual_dim, text_dim=text_dim, num_latents=num_latents)
        else:
            self.projector = VLMProjection(visual_dim=visual_dim, text_dim=text_dim)

    def forward(self, visual_features: _Tensor, text_embeddings: _Tensor) -> _Tensor:
        """Concatenates projected visual tokens and text embeddings along the sequence dimension."""
        v_tokens = self.projector(visual_features)
        return _Tensor.cat([v_tokens, text_embeddings], dim=1)


__all__ = [
    "VLMProjection",
    "PerceiverResampler",
    "VisionLanguageConnector",
]
