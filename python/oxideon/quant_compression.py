"""Model Quantization, Sparsity, and Representation Compression.

Implements:
42. SmoothQuant: Activation-aware weight and activation smoothing (Xiao et al., 2023)
43. GPTQQuantizer: 2nd-order Taylor expansion inverse Hessian weight quantizer (Frantar et al., 2022)
44. SparseMagnitudePruner: Unstructured and N:M structured sparsity pruner (e.g. 2:4)
45. VectorQuantizedVAE: Discrete codebook autoencoder with commitment loss and perplexity
46. HQQQuantizer: Half-Quadratic Quantization for ultra-low bit calibration-free weights
"""

from typing import Optional, Tuple, Dict, Any, Union, List
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, Parameter


class SmoothQuant:
    """SmoothQuant: Smooths activation outliers by migrating scale to weights (Xiao et al., 2023)."""

    @staticmethod
    def calculate_scale(act_abs_max: np.ndarray, weight_abs_max: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Computes per-channel migration scale: s = (max|X|)^alpha / (max|W|)^(1 - alpha)."""
        act_pow = np.power(np.maximum(act_abs_max, 1e-5), alpha)
        w_pow = np.power(np.maximum(weight_abs_max, 1e-5), 1.0 - alpha)
        s = act_pow / (w_pow + 1e-8)
        s = np.clip(s, 1e-4, 1e4)
        return s

    @staticmethod
    def smooth_layer(
        weight: Union[_Tensor, np.ndarray],
        act_sample: Union[_Tensor, np.ndarray],
        alpha: float = 0.5,
    ) -> Tuple[_Tensor, _Tensor, _Tensor]:
        w = weight.to_numpy() if isinstance(weight, _Tensor) else np.array(weight, dtype=np.float32)
        x = act_sample.to_numpy() if isinstance(act_sample, _Tensor) else np.array(act_sample, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError(f"smooth_layer expects a 2D [out, in] weight, got shape {w.shape}")
        if x.ndim < 1 or x.shape[-1] != w.shape[1]:
            last = x.shape[-1] if x.ndim >= 1 else None
            raise ValueError(f"activation last dim {last} must match weight in_dim {w.shape[1]}")

        act_max = np.max(np.abs(x), axis=tuple(range(x.ndim - 1)))
        w_max = np.max(np.abs(w), axis=0)

        s = SmoothQuant.calculate_scale(act_max, w_max, alpha=alpha)

        w_smooth = (w * s[np.newaxis, :]).astype(np.float32)
        x_smooth = (x / s).astype(np.float32)

        return _from_numpy(w_smooth), _from_numpy(x_smooth), _from_numpy(s.astype(np.float32))


class GPTQQuantizer:
    """GPTQ: Accurate Post-Training Quantization for Deep Neural Networks (Frantar et al., 2022)."""

    @staticmethod
    def quantize_weight_col(
        w: np.ndarray,
        h_inv: np.ndarray,
        num_bits: int = 4,
        block_size: int = 128,
    ) -> np.ndarray:
        """Quantizes weights column by column with inverse Hessian error compensation."""
        if num_bits < 2:
            raise ValueError(f"num_bits must be >= 2, got {num_bits}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if w.ndim != 2:
            raise ValueError(f"weight must be 2D [out, in], got shape {w.shape}")
        w_q = w.copy()
        out_dim, in_dim = w.shape
        q_max = 2 ** (num_bits - 1) - 1
        q_min = -(2 ** (num_bits - 1))

        for c in range(in_dim):
            col = w_q[:, c]
            diag = h_inv[c, c] + 1e-7

            scale = (np.max(np.abs(col)) + 1e-8) / q_max
            col_q = np.clip(np.round(col / scale), q_min, q_max) * scale

            err = col - col_q
            w_q[:, c] = col_q

            end = min(c + block_size, in_dim)
            if c + 1 < end:
                update_slice = h_inv[c, c + 1 : end] / diag
                w_q[:, c + 1 : end] -= np.outer(err, update_slice)

        return w_q

    @staticmethod
    def quantize(
        weight: Union[_Tensor, np.ndarray],
        hessian: Optional[Union[_Tensor, np.ndarray]] = None,
        num_bits: int = 4,
    ) -> _Tensor:
        w = weight.to_numpy() if isinstance(weight, _Tensor) else np.array(weight, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError(f"weight must be 2D [out, in], got shape {w.shape}")
        in_dim = w.shape[1]

        if hessian is not None:
            h = hessian.to_numpy() if isinstance(hessian, _Tensor) else np.array(hessian, dtype=np.float32)
            if h.shape != (in_dim, in_dim):
                raise ValueError(f"hessian shape {h.shape} must match (in_dim, in_dim) ({in_dim}, {in_dim})")
            h_damped = h + 0.01 * np.eye(in_dim, dtype=np.float32) * np.mean(np.diag(h))
            try:
                h_inv = np.linalg.inv(h_damped)
            except np.linalg.LinAlgError:
                h_inv = np.eye(in_dim, dtype=np.float32)
        else:
            h_inv = np.eye(in_dim, dtype=np.float32)

        w_quant = GPTQQuantizer.quantize_weight_col(w, h_inv, num_bits=num_bits)
        return _from_numpy(w_quant.astype(np.float32))


class SparseMagnitudePruner:
    """Unstructured and N:M Structured Sparsity Pruner."""

    @staticmethod
    def prune_unstructured(tensor: Union[_Tensor, np.ndarray], sparsity: float = 0.5) -> Tuple[_Tensor, _Tensor]:
        arr = tensor.to_numpy() if isinstance(tensor, _Tensor) else np.array(tensor, dtype=np.float32)
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
        k = int(round(sparsity * arr.size))
        if k == 0:
            mask = np.ones_like(arr, dtype=np.float32)
        elif k >= arr.size:
            mask = np.zeros_like(arr, dtype=np.float32)
        else:
            flat = np.abs(arr).ravel()
            drop_idx = np.argpartition(flat, k)[:k]
            mask = np.ones_like(arr, dtype=np.float32)
            mask.flat[drop_idx] = 0.0

        pruned = arr * mask
        return _from_numpy(pruned), _from_numpy(mask)

    @staticmethod
    def prune_n_m_structured(tensor: Union[_Tensor, np.ndarray], n: int = 2, m: int = 4) -> Tuple[_Tensor, _Tensor]:
        arr = tensor.to_numpy() if isinstance(tensor, _Tensor) else np.array(tensor, dtype=np.float32)
        if not 0 <= n <= m:
            raise ValueError(f"require 0 <= n <= m, got n={n}, m={m}")
        orig_shape = arr.shape
        flat = arr.ravel()

        remainder = flat.size % m
        pad_len = (m - remainder) if remainder != 0 else 0
        if pad_len > 0:
            flat = np.pad(flat, (0, pad_len), mode="constant")

        blocks = flat.reshape(-1, m)
        mask_blocks = np.zeros_like(blocks, dtype=np.float32)

        abs_blocks = np.abs(blocks)
        top_indices = np.argsort(-abs_blocks, axis=1)[:, :n]
        rows = np.arange(blocks.shape[0])[:, None]
        mask_blocks[rows, top_indices] = 1.0

        pruned_flat = (blocks * mask_blocks).ravel()
        if pad_len > 0:
            pruned_flat = pruned_flat[:-pad_len]
            mask_flat = mask_blocks.ravel()[:-pad_len]
        else:
            mask_flat = mask_blocks.ravel()

        pruned = pruned_flat.reshape(orig_shape)
        mask = mask_flat.reshape(orig_shape)
        return _from_numpy(pruned.astype(np.float32)), _from_numpy(mask.astype(np.float32))


class VectorQuantizedVAE(Module):
    """Vector Quantized Variational AutoEncoder (VQ-VAE) (van den Oord et al., 2017)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_embeddings: int = 64, beta: float = 0.25):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_embeddings = num_embeddings
        self.beta = float(beta)

        self.encoder = Linear(in_dim, hidden_dim)
        self.decoder = Linear(hidden_dim, in_dim)

        codebook_init = (np.random.randn(num_embeddings, hidden_dim) / math.sqrt(hidden_dim)).astype(np.float32)
        self.embedding = Parameter(_from_numpy(codebook_init))

    def forward(self, x: _Tensor) -> Tuple[_Tensor, _Tensor, Dict[str, float]]:
        """Args:
            x: Input tensor [B, in_dim]
        Returns:
            reconstructed: [B, in_dim]
            loss: differentiable scalar loss (reconstruction + codebook + commitment)
            metrics: dict with 'perplexity', 'codebook_loss', 'commitment_loss', 'reconstruction_loss'
        """
        z_e_t = self.encoder(x)           # [B, D]
        z_e = z_e_t.to_numpy()
        b, d = z_e.shape
        e = self.embedding.to_numpy()     # [K, D]

        distances = (
            np.sum(z_e ** 2, axis=1, keepdims=True)
            + np.sum(e ** 2, axis=1)
            - 2.0 * np.matmul(z_e, e.T)
        )

        min_indices = np.argmin(distances, axis=1)
        one_hot = np.zeros((b, self.num_embeddings), dtype=np.float32)
        one_hot[np.arange(b), min_indices] = 1.0
        z_q_t = _from_numpy(one_hot).matmul(self.embedding)  # [B, D], differentiable w.r.t. self.embedding

        # VQ Losses (with stop-gradient via .detach()):
        # codebook_loss = ||sg[z_e] - e||^2
        # commitment_loss = ||z_e - sg[e]||^2
        diff_cb = z_e_t.detach() - z_q_t
        codebook_loss_t = (diff_cb * diff_cb).mean()
        diff_commit = z_e_t - z_q_t.detach()
        commitment_loss_t = (diff_commit * diff_commit).mean()
        vq_loss_t = codebook_loss_t + commitment_loss_t * self.beta

        # Straight-through estimator: z_q_ste = z_e + sg(z_q - z_e)
        z_q_ste = z_e_t + (z_q_t - z_e_t).detach()
        reconstruction = self.decoder(z_q_ste)

        diff_recon = x - reconstruction
        recon_loss_t = (diff_recon * diff_recon).mean()
        total_loss = recon_loss_t + vq_loss_t

        counts = np.bincount(min_indices, minlength=self.num_embeddings)
        probs = counts / (np.sum(counts) + 1e-12)
        perplexity = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))

        metrics = {
            "reconstruction_loss": float(recon_loss_t.to_numpy().flat[0]),
            "codebook_loss": float(codebook_loss_t.to_numpy().flat[0]),
            "commitment_loss": float(commitment_loss_t.to_numpy().flat[0]),
            "perplexity": perplexity,
        }
        return reconstruction, total_loss, metrics


class HQQQuantizer:
    """Half-Quadratic Quantization (HQQ) (Badri & Shayer, 2023)."""

    @staticmethod
    def quantize(
        weight: Union[_Tensor, np.ndarray],
        num_bits: int = 4,
        axis: int = 0,
    ) -> Tuple[_Tensor, _Tensor, _Tensor]:
        if num_bits < 1:
            raise ValueError(f"num_bits must be >= 1, got {num_bits}")
        w = weight.to_numpy() if isinstance(weight, _Tensor) else np.array(weight, dtype=np.float32)
        q_max = (1 << num_bits) - 1

        w_min = np.min(w, axis=axis, keepdims=True)
        w_max = np.max(w, axis=axis, keepdims=True)

        scale = (w_max - w_min) / max(q_max, 1)
        scale = np.maximum(scale, 1e-8)
        # Constant groups have no range to derive a scale from; anchor the scale
        # to the group's magnitude so the value round-trips instead of collapsing to ~0.
        scale = np.where(w_max == w_min, np.maximum(np.abs(w_min), 1e-8) / max(q_max, 1), scale)
        zero_point = np.round(-w_min / scale)
        zero_point = np.clip(zero_point, 0, q_max)

        # Half-quadratic alternating refinement of zero_point & quantized codes
        for _ in range(3):
            q = np.clip(np.round(w / scale + zero_point), 0, q_max)
            zero_point = np.clip(np.mean(q - w / scale, axis=axis, keepdims=True), 0, q_max)

        q = np.clip(np.round(w / scale + zero_point), 0, q_max)
        return _from_numpy(q.astype(np.float32)), _from_numpy(scale.astype(np.float32)), _from_numpy(zero_point.astype(np.float32))

    @staticmethod
    def dequantize(
        quantized: Union[_Tensor, np.ndarray],
        scale: Union[_Tensor, np.ndarray],
        zero_point: Union[_Tensor, np.ndarray],
    ) -> _Tensor:
        q = quantized.to_numpy() if isinstance(quantized, _Tensor) else np.array(quantized)
        s = scale.to_numpy() if isinstance(scale, _Tensor) else np.array(scale)
        z = zero_point.to_numpy() if isinstance(zero_point, _Tensor) else np.array(zero_point)
        w_rec = (q - z) * s
        return _from_numpy(w_rec.astype(np.float32))
