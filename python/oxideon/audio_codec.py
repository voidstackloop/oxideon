"""Neural Audio Codec and Residual Vector Quantization (RVQ).

Implements SoundStream / EnCodec neural audio compression (Zeghidour et al., 2021; Defossez et al., 2022):
- VectorQuantizer: Single-codebook discrete vector quantization with commitment loss and STE
- AudioResidualVQ: Multi-stage cascaded residual quantization
- AudioEncoder1D & AudioDecoder1D: 1D strided convolutional encoder and transposed decoder
"""

from typing import List, Tuple, Optional, Dict
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter, Linear, ModuleList, _check_repeats


class VectorQuantizer(Module):
    """Single-stage Vector Quantizer with straight-through gradient estimator."""

    def __init__(self, num_embeddings: int = 1024, embedding_dim: int = 128, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        limit = 1.0 / np.sqrt(embedding_dim)
        cb_init = np.random.uniform(-limit, limit, (num_embeddings, embedding_dim)).astype(np.float32)
        self.embedding = Parameter(_from_numpy(cb_init))

    def forward(self, z: _Tensor) -> Tuple[_Tensor, np.ndarray, float]:
        """Args:
            z: Continuous latent tensor [batch, seq_len, embedding_dim]

        Returns:
            Tuple of (quantized_tensor, discrete_indices, commitment_loss)
        """
        z_np = z.to_numpy()
        orig_shape = list(z_np.shape)
        if z_np.ndim == 1:
            z_np = z_np.reshape(1, 1, -1)
        elif z_np.ndim == 2:
            z_np = z_np.reshape(1, -1, orig_shape[-1])
        b, s, d = z_np.shape
        if b * s == 0:
            out_tensor = _from_numpy(np.empty(orig_shape, dtype=np.float32))
            idx_shape = orig_shape[:-1] if len(orig_shape) > 1 else ()
            return out_tensor, np.empty(idx_shape, dtype=np.int64), 0.0

        flat_z = z_np.reshape(-1, d)  # [N, d]
        cb = self.embedding.to_numpy()  # [K, d]

        z_sq = np.sum(flat_z ** 2, axis=1, keepdims=True)
        e_sq = np.sum(cb ** 2, axis=1, keepdims=True).T
        dist = z_sq - 2.0 * (flat_z @ cb.T) + e_sq

        indices = np.argmin(dist, axis=1)
        n_tokens = len(indices)

        # Differentiable codebook lookup via one-hot matmul + Straight-Through Estimator (STE)
        one_hot = np.zeros((n_tokens, self.num_embeddings), dtype=np.float32)
        one_hot[np.arange(n_tokens), indices] = 1.0
        z_q_2d = _from_numpy(one_hot).matmul(self.embedding)  # [N, d], flows grad to self.embedding
        z_q = z_q_2d.reshape(orig_shape)

        # STE: value is z_q, gradient flows to both z and self.embedding
        out_tensor = z + (z_q - z.detach())

        loss = float(self.commitment_cost * np.mean((z_np - cb[indices].reshape(z_np.shape)) ** 2))
        idx_out = indices.reshape(orig_shape[:-1]) if len(orig_shape) > 1 else indices
        return out_tensor, idx_out, loss


class AudioResidualVQ(Module):
    """Residual Vector Quantization (RVQ) across multiple cascaded codebook stages."""

    def __init__(self, num_quantizers: int = 4, num_embeddings: int = 1024, embedding_dim: int = 128):
        super().__init__()
        num_quantizers = _check_repeats("num_quantizers", num_quantizers)
        self.num_quantizers = num_quantizers
        self.quantizers = ModuleList([
            VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
            for _ in range(num_quantizers)
        ])

    def forward(self, z: _Tensor) -> Tuple[_Tensor, List[np.ndarray], float]:
        """Quantizes continuous latents by hierarchically encoding residual errors."""
        z_curr = z
        quantized_sum: Optional[_Tensor] = None
        all_codes: List[np.ndarray] = []
        total_loss = 0.0

        for vq in self.quantizers:
            q_stage, indices, stage_loss = vq(z_curr)
            quantized_sum = q_stage if quantized_sum is None else (quantized_sum + q_stage)
            all_codes.append(indices)
            total_loss += stage_loss
            # Next residual (detached quantized vector subtracted from z_curr)
            z_curr = z_curr - q_stage.detach()

        if quantized_sum is None:
            quantized_sum = z * 0.0
        return quantized_sum, all_codes, total_loss


class AudioEncoder1D(Module):
    """1D Strided Convolutional Encoder downsampling 1D audio waveforms into latent features."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 128, downsample_rate: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.downsample_rate = downsample_rate
        self.proj_in = Linear(in_channels * downsample_rate, hidden_dim)
        self.proj_out = Linear(hidden_dim, hidden_dim)

    def forward(self, x: _Tensor) -> _Tensor:
        """Args: x: 1D audio waveform [batch, num_samples] -> [batch, num_samples // downsample, hidden_dim]"""
        b, l = x.shape[0], x.shape[-1]
        cropped_len = (l // self.downsample_rate) * self.downsample_rate
        if cropped_len < l:
            # Differentiable crop via projection matrix
            crop_mat = np.zeros((l, cropped_len), dtype=np.float32)
            crop_mat[:cropped_len, :cropped_len] = np.eye(cropped_len, dtype=np.float32)
            x_cropped = x.matmul(_from_numpy(crop_mat))
        else:
            x_cropped = x
        framed = x_cropped.reshape([b, cropped_len // self.downsample_rate, self.in_channels * self.downsample_rate])
        h = self.proj_in(framed).relu()
        return self.proj_out(h)


class AudioDecoder1D(Module):
    """1D Transposed Decoder reconstructing 1D audio waveforms from quantized latents."""

    def __init__(self, hidden_dim: int = 128, out_channels: int = 1, upsample_rate: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.upsample_rate = upsample_rate
        self.proj = Linear(hidden_dim, out_channels * upsample_rate)

    def forward(self, z: _Tensor) -> _Tensor:
        """Args: z: [batch, seq_len, hidden_dim] -> [batch, seq_len * upsample_rate]"""
        h = self.proj(z)
        b, s, _ = h.shape
        return h.reshape([b, s * self.upsample_rate])


__all__ = [
    "VectorQuantizer",
    "AudioResidualVQ",
    "AudioEncoder1D",
    "AudioDecoder1D",
]
