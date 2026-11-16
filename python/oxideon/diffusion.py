"""Denoising Diffusion Probabilistic Models (DDPM / DDIM) module for Oxideon.

Implements forward noise scheduling, reverse generation sampling, time embeddings,
and residual U-Net backbones:
  - DDPMScheduler: Discrete Gaussian noise scheduler (Ho et al., 2020)
  - DDIMScheduler: Fast deterministic sampling scheduler (Song et al., 2020)
  - SinusoidalPositionalEmbedding: Continuous frequency embedding for diffusion timesteps
  - UNetModel: Multi-scale residual U-Net with time conditioning and skip connections
"""

from typing import Tuple, Optional, Any, List, Union
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, Sequential, ReLU, SiLU, Parameter, _check_positive


class DDPMScheduler:
    """Standard Denoising Diffusion Probabilistic Models (DDPM) scheduler."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        num_train_timesteps = _check_positive("num_train_timesteps", num_train_timesteps)
        self.num_train_timesteps = num_train_timesteps

        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = np.linspace(math.sqrt(beta_start), math.sqrt(beta_end), num_train_timesteps, dtype=np.float32) ** 2
        elif beta_schedule in ("cosine", "squaredcos_cap_v2"):
            s = 0.008
            steps = np.arange(num_train_timesteps + 1, dtype=np.float64) / num_train_timesteps
            alphas_bar = np.cos((steps + s) / (1.0 + s) * math.pi * 0.5) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
            self.betas = np.clip(betas, 1e-4, 0.9999).astype(np.float32)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.pad(self.alphas_cumprod[:-1], (1, 0), constant_values=1.0)

        # Calculations for diffusion posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(
        self,
        original_samples: _Tensor,
        noise: _Tensor,
        timesteps: Union[int, List[int], np.ndarray, _Tensor],
    ) -> _Tensor:
        """Adds Gaussian noise to inputs according to forward process q(x_t | x_0)."""
        x_0 = original_samples.to_numpy()

        if isinstance(timesteps, int):
            t = np.array([timesteps] * x_0.shape[0], dtype=np.int64)
        elif isinstance(timesteps, _Tensor):
            t = timesteps.to_numpy().astype(np.int64).reshape(-1)
        else:
            t = np.asarray(timesteps, dtype=np.int64).reshape(-1)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(-1, *([1] * (x_0.ndim - 1)))
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, *([1] * (x_0.ndim - 1)))

        if (isinstance(original_samples, _Tensor) and original_samples.requires_grad) or (
            isinstance(noise, _Tensor) and noise.requires_grad
        ):
            sa_t = _from_numpy(np.ascontiguousarray(sqrt_alpha, dtype=np.float32))
            soma_t = _from_numpy(np.ascontiguousarray(sqrt_one_minus_alpha, dtype=np.float32))
            return original_samples * sa_t + noise * soma_t

        eps = noise.to_numpy()
        noisy_samples = sqrt_alpha * x_0 + sqrt_one_minus_alpha * eps
        return _from_numpy(np.ascontiguousarray(noisy_samples, dtype=np.float32))

    def step(
        self,
        model_output: _Tensor,
        timestep: int,
        sample: _Tensor,
        seed: Optional[int] = None,
    ) -> _Tensor:
        """Single reverse diffusion denoising step p_theta(x_{t-1} | x_t)."""
        pred_eps = model_output.to_numpy()
        x_t = sample.to_numpy()

        t = timestep
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]

        # Compute mean
        mean = (1.0 / math.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus_alpha_bar) * pred_eps)

        if t > 0:
            rng = np.random.default_rng(seed)
            variance = self.posterior_variance[t]
            std = math.sqrt(variance)
            noise = rng.standard_normal(x_t.shape).astype(np.float32)
            prev_sample = mean + std * noise
        else:
            prev_sample = mean

        return _from_numpy(np.ascontiguousarray(prev_sample, dtype=np.float32))


class DDIMScheduler(DDPMScheduler):
    """Denoising Diffusion Implicit Models (DDIM) deterministic / accelerated sampler."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        num_train_timesteps = _check_positive("num_train_timesteps", num_train_timesteps)
        super().__init__(num_train_timesteps, beta_start, beta_end)
        self.timesteps = np.arange(num_train_timesteps - 1, -1, -1)

    def set_timesteps(self, num_inference_steps: int):
        """Discretizes inference trajectory into uniform intervals."""
        num_inference_steps = _check_positive("num_inference_steps", num_inference_steps)
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps = {num_inference_steps} exceeds "
                f"num_train_timesteps = {self.num_train_timesteps}"
            )
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (np.arange(num_inference_steps) * step_ratio).round().astype(np.int64)[::-1]

    def step(
        self,
        model_output: _Tensor,
        timestep: int,
        sample: _Tensor,
        eta: float = 0.0,
        seed: Optional[int] = None,
    ) -> _Tensor:
        """Deterministic / semi-deterministic reverse step."""
        pred_eps = model_output.to_numpy()
        x_t = sample.to_numpy()

        t = timestep
        alpha_bar_t = self.alphas_cumprod[t]
        step_ratio = self.num_train_timesteps // len(self.timesteps) if len(self.timesteps) > 0 else 1
        prev_t = t - step_ratio
        alpha_bar_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1.0

        # Predict x_0
        pred_x0 = (x_t - math.sqrt(1.0 - alpha_bar_t) * pred_eps) / math.sqrt(alpha_bar_t)

        # Variance
        sigma_t = eta * math.sqrt(max(0.0, (1 - alpha_bar_prev) / max(1 - alpha_bar_t, 1e-12) * (1 - alpha_bar_t / max(alpha_bar_prev, 1e-12))))

        # Direction pointing to x_t
        dir_xt = math.sqrt(max(0.0, 1.0 - alpha_bar_prev - sigma_t**2)) * pred_eps

        prev_sample = math.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
        if eta > 0:
            rng = np.random.default_rng(seed)
            noise = rng.standard_normal(x_t.shape).astype(np.float32)
            prev_sample += sigma_t * noise

        return _from_numpy(np.ascontiguousarray(prev_sample, dtype=np.float32))


class SinusoidalPositionalEmbedding(Module):
    """Sinusoidal frequency embeddings for diffusion timesteps."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: Union[int, np.ndarray, _Tensor]) -> _Tensor:
        if isinstance(timesteps, int):
            t = np.array([timesteps], dtype=np.float32)
        elif isinstance(timesteps, _Tensor):
            t = timesteps.to_numpy().astype(np.float32).reshape(-1)
        else:
            t = np.asarray(timesteps, dtype=np.float32).reshape(-1)

        half_dim = self.dim // 2
        freqs = np.exp(-math.log(self.max_period) * np.arange(half_dim, dtype=np.float32) / max(half_dim, 1))
        args = t[:, None] * freqs[None, :]
        emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
        if self.dim % 2 == 1:
            emb = np.pad(emb, ((0, 0), (0, 1)))
        return _from_numpy(np.ascontiguousarray(emb, dtype=np.float32))


class UNetModel(Module):
    """Residual U-Net model with time-step conditioning."""

    def __init__(
        self,
        in_dim: int = 16,
        hidden_dim: int = 64,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.time_embed = Sequential(
            SinusoidalPositionalEmbedding(time_embed_dim),
            Linear(time_embed_dim, time_embed_dim),
            SiLU(),
            Linear(time_embed_dim, hidden_dim),
        )

        # Encoder
        self.in_proj = Linear(in_dim, hidden_dim)
        self.enc1 = Linear(hidden_dim, hidden_dim)
        self.enc2 = Linear(hidden_dim, hidden_dim * 2)

        # Bottleneck
        self.mid = Linear(hidden_dim * 2, hidden_dim * 2)

        # Decoder with skip connections
        self.dec2 = Linear(hidden_dim * 4, hidden_dim)
        self.dec1 = Linear(hidden_dim * 2, hidden_dim)
        self.out_proj = Linear(hidden_dim, in_dim)
        self.act = SiLU()

    def forward(self, x: _Tensor, timesteps: Any) -> _Tensor:
        t_emb = self.time_embed(timesteps)

        # Enc 1
        h1 = self.act(self.in_proj(x) + t_emb)
        h1 = self.act(self.enc1(h1))

        # Enc 2
        h2 = self.act(self.enc2(h1))

        # Mid
        h_mid = self.act(self.mid(h2))

        # Dec 2 (skip h2) - preserve Tensor autograd via Tensor.cat
        h_dec2_in = _Tensor.cat([h_mid, h2], dim=-1)
        h_dec2 = self.act(self.dec2(h_dec2_in))

        # Dec 1 (skip h1)
        h_dec1_in = _Tensor.cat([h_dec2, h1], dim=-1)
        h_dec1 = self.act(self.dec1(h_dec1_in))

        return self.out_proj(h_dec1)


__all__ = [
    "DDPMScheduler",
    "DDIMScheduler",
    "SinusoidalPositionalEmbedding",
    "UNetModel",
]
