"""Vision Transformer (ViT) architecture for visual representation learning.

Implements:
  - PatchEmbedding: Converts 2D images to sequence of flattened patch embeddings with [CLS] token and positional encoding.
  - TransformerEncoderBlock: Pre-LN multi-head self-attention and MLP feed-forward block.
  - VisionTransformer: Complete ViT model for image classification (Dosovitskiy et al., ICLR 2021).
"""

from typing import Optional, List
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, LayerNorm, GELU, Parameter, ModuleList, _check_repeats


class PatchEmbedding(Module):
    """2D Image to Patch Embedding projection layer with full Tensor autograd."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 64,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        self.proj = Linear(self.patch_dim, embed_dim)
        self.cls_token = Parameter(_Tensor.randn([1, 1, embed_dim]) * 0.02)
        self.pos_embed = Parameter(_Tensor.randn([1, self.num_patches + 1, embed_dim]) * 0.02)

    def forward(self, x: _Tensor) -> _Tensor:
        """Args: x: [batch, in_channels, img_size, img_size] -> [batch, num_patches + 1, embed_dim]"""
        sh = list(x.shape)
        if len(sh) == 3:
            x = x.reshape([1] + sh)
            sh = list(x.shape)
        if len(sh) != 4:
            raise ValueError(f"PatchEmbedding expects 3D [C, H, W] or 4D [B, C, H, W] tensor, got shape {tuple(sh)}")
        b, c, h, w = sh
        if h != self.img_size or w != self.img_size:
            raise ValueError(f"Image spatial dimensions ({h}, {w}) must match img_size ({self.img_size}, {self.img_size})")
        p = self.patch_size
        g = self.grid_size

        # Differentiable patch extraction via Tensor reshape + transpose:
        # [b, c, g, p, g, p] -> transpose(2, 3) -> [b, c, p, g, g, p] -> reshape [b, c*p, g*g, p]
        #Wait: to match exact (0, 2, 4, 1, 3, 5) permutation [b, gh, gw, c, ph, pw]:
        # Let's permute using sequential 2-axis transposes on 6D tensor:
        t = x.reshape([b, c, g, p, g, p])
        # axes: (0:b, 1:c, 2:gh, 3:ph, 4:gw, 5:pw)
        t = t.transpose(1, 2)  # (0:b, 1:gh, 2:c, 3:ph, 4:gw, 5:pw)
        t = t.transpose(2, 4)  # (0:b, 1:gh, 2:gw, 3:ph, 4:c, 5:pw)
        t = t.transpose(3, 4)  # (0:b, 1:gh, 2:gw, 3:c, 4:ph, 5:pw)
        patches_t = t.reshape([b, self.num_patches, self.patch_dim])

        projected = self.proj(patches_t)  # [b, num_patches, embed_dim]

        # Expand cls_token [1, 1, embed_dim] to [b, 1, embed_dim] via broadcasting addition with zeros
        zeros_cls = _from_numpy(np.zeros((b, 1, self.embed_dim), dtype=np.float32))
        cls_expanded = zeros_cls + self.cls_token
        tokens = _Tensor.cat([cls_expanded, projected], dim=1)  # [b, num_patches + 1, embed_dim]
        return tokens + self.pos_embed


class TransformerEncoderBlock(Module):
    """Standard Pre-LN Transformer encoder block with full Tensor autograd."""

    def __init__(self, embed_dim: int, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = Linear(embed_dim, embed_dim * 3)
        self.out_proj = Linear(embed_dim, embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp_fc1 = Linear(embed_dim, hidden_dim)
        self.act = GELU()
        self.mlp_fc2 = Linear(hidden_dim, embed_dim)

    def forward(self, x: _Tensor) -> _Tensor:
        normed = self.norm1(x)
        b, s, d = normed.shape
        qkv = self.qkv(normed)  # [b, s, 3 * d]
        qkv_r = qkv.reshape([b, s, 3, self.num_heads, self.head_dim])

        # Split along the 3-way QKV dimension using differentiable one-hot masks
        e_q = _from_numpy(np.array([1.0, 0.0, 0.0], dtype=np.float32).reshape(1, 1, 3, 1, 1))
        e_k = _from_numpy(np.array([0.0, 1.0, 0.0], dtype=np.float32).reshape(1, 1, 3, 1, 1))
        e_v = _from_numpy(np.array([0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 1, 3, 1, 1))

        q = (qkv_r * e_q).sum(dim=2).transpose(1, 2)  # [b, heads, s, head_dim]
        k = (qkv_r * e_k).sum(dim=2).transpose(1, 2)
        v = (qkv_r * e_v).sum(dim=2).transpose(1, 2)

        attn_scores = (q.matmul(k.transpose(2, 3))) * self.scale
        attn_weights = attn_scores.softmax(-1)

        attn_out = attn_weights.matmul(v).transpose(1, 2).reshape([b, s, d])
        attn_proj = self.out_proj(attn_out)

        x = x + attn_proj

        normed2 = self.norm2(x)
        mlp_out = self.mlp_fc2(self.act(self.mlp_fc1(normed2)))
        return x + mlp_out


class VisionTransformer(Module):
    """Vision Transformer (ViT) model for image classification."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        depth = _check_repeats("depth", depth)
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.blocks = ModuleList([
            TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes)

    def forward(self, x: _Tensor) -> _Tensor:
        """Args: x: [batch, in_channels, img_size, img_size] -> [batch, num_classes]"""
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Differentiably select [CLS] token at index 0 along sequence dim (dim=1)
        seq_len = x.shape[1]
        cls_mask = np.zeros((1, seq_len, 1), dtype=np.float32)
        cls_mask[0, 0, 0] = 1.0
        cls_t = (x * _from_numpy(cls_mask)).sum(dim=1)
        return self.head(cls_t)


__all__ = [
    "PatchEmbedding",
    "TransformerEncoderBlock",
    "VisionTransformer",
]
