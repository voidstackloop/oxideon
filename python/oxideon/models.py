"""Canonical deep learning neural network architectures for Oxideon.

Includes ResNet, VisionTransformer (ViT), and CausalLM (Autoregressive GPT).
"""

from typing import List, Optional, Tuple
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import (
    Module,
    Parameter,
    Sequential,
    ModuleList,
    Linear,
    Conv2d,
    LayerNorm,
    ReLU,
    Dropout,
    MultiheadAttention,
    AdaptiveAvgPool2d,
    Flatten,
    _check_repeats,
)


# ---------------------------------------------------------------------------
# 1. ResNet Architecture
# ---------------------------------------------------------------------------

class ResidualBlock(Module):
    """Basic Residual Block with identity skip connection: y = F(x) + x."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = LayerNorm(channels)
        self.relu = ReLU()
        self.conv2 = Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = LayerNorm(channels)

    def forward(self, x: _Tensor) -> _Tensor:
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + residual)


class ResNet18(Module):
    """ResNet-18 vision classifier model."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.stem = Sequential(
            Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            LayerNorm(hidden_dim),
            ReLU(),
        )
        self.layer1 = ResidualBlock(hidden_dim)
        self.layer2 = ResidualBlock(hidden_dim)
        self.pool = AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = Linear(hidden_dim, num_classes)

    def forward(self, x: _Tensor) -> _Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.pool(out)
        out = self.flatten(out)
        return self.fc(out)


# ---------------------------------------------------------------------------
# 2. Vision Transformer (ViT)
# ---------------------------------------------------------------------------

class VisionTransformer(Module):
    """Vision Transformer (ViT) with patch projection and transformer encoder blocks."""

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
        dropout: float = 0.0,
    ):
        super().__init__()
        depth = _check_repeats("depth", depth)
        if img_size <= 0 or patch_size <= 0 or in_channels <= 0 or num_classes <= 0 or embed_dim <= 0:
            raise ValueError("img_size, patch_size, in_channels, num_classes, and embed_dim must be positive")
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch projection
        self.patch_proj = Linear(self.patch_dim, embed_dim)
        self.cls_token = Parameter(_Tensor.randn([1, 1, embed_dim]))
        self.pos_embed = Parameter(_Tensor.randn([1, self.num_patches + 1, embed_dim]))
        self.dropout = Dropout(dropout) if dropout > 0.0 else ReLU()

        # Encoder layers
        self.layers = ModuleList()
        mlp_dim = int(embed_dim * mlp_ratio)
        for _ in range(depth):
            layer = Module()
            layer.norm1 = LayerNorm(embed_dim)
            layer.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            layer.norm2 = LayerNorm(embed_dim)
            layer.mlp = Sequential(
                Linear(embed_dim, mlp_dim),
                ReLU(),
                Linear(mlp_dim, embed_dim),
            )
            self.layers.append(layer)

        self.norm = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes)

    def _extract_patches(self, x: _Tensor) -> _Tensor:
        if len(x.shape) != 4:
            raise ValueError(f"VisionTransformer requires 4D images [B, C, H, W], got shape {x.shape}")
        N, C, H, W = x.shape
        p = self.patch_size
        if H % p != 0 or W % p != 0:
            raise ValueError(f"Spatial dimensions (H={H}, W={W}) must be divisible by patch_size={p}")
        h_p, w_p = H // p, W // p

        if x.requires_grad:
            patch_list = []
            for i in range(h_p):
                row_slice = x.narrow(2, i * p, p)
                for j in range(w_p):
                    patch = row_slice.narrow(3, j * p, p).reshape([N, 1, C * p * p])
                    patch_list.append(patch)
            return _Tensor.cat(patch_list, dim=1) if len(patch_list) > 1 else patch_list[0]

        arr = x.to_numpy()
        stacked = (
            arr.reshape(N, C, h_p, p, w_p, p)
            .transpose(0, 2, 4, 1, 3, 5)
            .reshape(N, h_p * w_p, C * p * p)
        )
        return _from_numpy(np.ascontiguousarray(stacked, dtype=np.float32))

    def forward(self, x: _Tensor) -> _Tensor:
        patches = self._extract_patches(x)
        x_proj = self.patch_proj(patches)

        batch_size = x.shape[0]
        cls_tokens = (
            _Tensor.cat([self.cls_token] * batch_size, dim=0)
            if batch_size > 1
            else self.cls_token
        )
        tokens = _Tensor.cat([cls_tokens, x_proj], dim=1)

        x_seq = tokens + self.pos_embed

        for layer in self.layers:
            normed = layer.norm1(x_seq)
            attn_out, _ = layer.attn(normed)
            x_seq = x_seq + attn_out
            x_seq = x_seq + layer.mlp(layer.norm2(x_seq))

        x_seq = self.norm(x_seq)
        cls_out = x_seq.narrow(1, 0, 1).reshape([batch_size, self.embed_dim])
        return self.head(cls_out)


# ---------------------------------------------------------------------------
# 3. Causal LM (Mini-GPT / TransformerLM)
# ---------------------------------------------------------------------------

class MiniGPT(Module):
    """Autoregressive decoder-only Causal Language Model with text generation capability."""

    def __init__(
        self,
        vocab_size: int = 1000,
        max_seq_len: int = 128,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        num_layers = _check_repeats("num_layers", num_layers)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.tok_embed = Linear(vocab_size, embed_dim, bias=False)
        self.pos_embed = Parameter(_Tensor.randn([1, max_seq_len, embed_dim]))

        self.layers = ModuleList()
        for _ in range(num_layers):
            layer = Module()
            layer.norm1 = LayerNorm(embed_dim)
            layer.attn = MultiheadAttention(embed_dim, num_heads)
            layer.norm2 = LayerNorm(embed_dim)
            layer.ffn = Sequential(
                Linear(embed_dim, embed_dim * 4),
                ReLU(),
                Linear(embed_dim * 4, embed_dim),
            )
            self.layers.append(layer)

        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx: _Tensor) -> _Tensor:
        arr = idx.to_numpy().astype(int)
        single = False
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)
            single = True
        elif arr.ndim != 2:
            raise ValueError(f"MiniGPT requires 1D or 2D token tensors [B, T], got shape {arr.shape}")
        B, T = arr.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        # One-hot embedding
        one_hot = np.zeros((B, T, self.vocab_size), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                val = arr[b, t]
                if 0 <= val < self.vocab_size:
                    one_hot[b, t, val] = 1.0

        x = self.tok_embed(_from_numpy(one_hot))
        pos = self.pos_embed.narrow(1, 0, T)
        x = x + pos

        for layer in self.layers:
            attn_out, _ = layer.attn(layer.norm1(x))
            x = x + attn_out
            x = x + layer.ffn(layer.norm2(x))

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if single:
            return logits.reshape([T, self.vocab_size])
        return logits

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ) -> List[int]:
        """Autoregressively generates new token IDs."""
        if not prompt_ids:
            raise ValueError("prompt_ids cannot be empty, need at least 1 token")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive (> 0), got {temperature}")
        tokens = list(prompt_ids)
        for _ in range(max_new_tokens):
            cur_seq = tokens[-self.max_seq_len :]
            inp = _from_numpy(np.array([cur_seq], dtype=np.float32))
            logits = self(inp)
            last_logits = logits.to_numpy()[0, -1, :] / max(temperature, 1e-4)

            # Softmax
            exp_l = np.exp(last_logits - np.max(last_logits))
            probs = exp_l / np.sum(exp_l)
            next_token = int(np.argmax(probs))
            tokens.append(next_token)
        return tokens


TransformerLM = MiniGPT

__all__ = [
    "ResidualBlock",
    "ResNet18",
    "VisionTransformer",
    "MiniGPT",
    "TransformerLM",
]
