"""High-Order Diffusion Schedulers and Fast ODE Samplers.

Implements fast sampling for continuous-time and discrete-time diffusion models:
- EulerDiscreteScheduler: 1st-order discrete Euler solver with noise schedule interpolation
- DPMSolverMultistepScheduler: 2nd-order multistep DPM-Solver++ ODE solver (Lu et al., 2022)
- DDIMScheduler: Denoising Diffusion Implicit Models sampler (Song et al., 2020)
"""

from typing import List, Optional, Tuple, Union
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import _check_positive


class EulerDiscreteScheduler:
    """Euler Discrete diffusion step solver."""

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
        else:
            self.betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=np.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sigmas = np.sqrt((1.0 - self.alphas_cumprod) / self.alphas_cumprod)

        self.timesteps: np.ndarray = np.arange(0, num_train_timesteps)[::-1].copy()

    def set_timesteps(self, num_inference_steps: int):
        """Sets custom sampling timesteps across the diffusion trajectory."""
        num_inference_steps = _check_positive("num_inference_steps", num_inference_steps)
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps = {num_inference_steps} exceeds "
                f"num_train_timesteps = {self.num_train_timesteps}"
            )
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (np.arange(0, num_inference_steps) * step_ratio)[::-1].copy()

    def step(self, model_output: _Tensor, timestep: int, sample: _Tensor) -> _Tensor:
        """Predicts the sample at the previous timestep via Euler step."""
        idx = np.where(self.timesteps == timestep)[0]
        step_idx = int(idx[0]) if len(idx) > 0 else 0

        sigma = float(self.sigmas[timestep])
        dt = (
            float(self.sigmas[self.timesteps[step_idx + 1]]) - sigma
            if step_idx + 1 < len(self.timesteps)
            else -sigma
        )

        pred_original_sample = sample - model_output * sigma
        prev_sample = sample + ((sample - pred_original_sample) / sigma) * dt
        return prev_sample


class DPMSolverMultistepScheduler:
    """DPM-Solver++ 2M: 2nd-Order Multistep Diffusion ODE Solver (Lu et al., 2022).

    Enables high-fidelity diffusion generation in 15-20 steps.
    """

    def __init__(self, num_train_timesteps: int = 1000):
        num_train_timesteps = _check_positive("num_train_timesteps", num_train_timesteps)
        self.num_train_timesteps = num_train_timesteps
        self.betas = np.linspace(0.0001, 0.02, num_train_timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.lambda_t = 0.5 * np.log(self.alphas_cumprod / (1.0 - self.alphas_cumprod))

        self.timesteps: np.ndarray = np.arange(0, num_train_timesteps)[::-1].copy()
        self.model_outputs: List[_Tensor] = []

    def set_timesteps(self, num_inference_steps: int):
        num_inference_steps = _check_positive("num_inference_steps", num_inference_steps)
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps = {num_inference_steps} exceeds "
                f"num_train_timesteps = {self.num_train_timesteps}"
            )
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (np.arange(0, num_inference_steps) * step_ratio)[::-1].copy()
        self.model_outputs = []

    def step(self, model_output: _Tensor, timestep: int, sample: _Tensor) -> _Tensor:
        """Executes a 2nd-order Adams-Bashforth DPM-Solver++ multistep."""
        idx = np.where(self.timesteps == timestep)[0]
        step_idx = int(idx[0]) if len(idx) > 0 else 0
        t_next = int(self.timesteps[step_idx + 1]) if step_idx + 1 < len(self.timesteps) else 0

        lambda_curr = float(self.lambda_t[timestep])
        lambda_next = float(self.lambda_t[t_next])
        h = lambda_next - lambda_curr

        alpha_curr = float(np.sqrt(self.alphas_cumprod[timestep]))
        sigma_curr = float(np.sqrt(1.0 - self.alphas_cumprod[timestep]))
        alpha_next = float(np.sqrt(self.alphas_cumprod[t_next]))
        sigma_next = float(np.sqrt(1.0 - self.alphas_cumprod[t_next]))

        # Convert noise prediction eps to data prediction x0_pred = (x_t - sigma_t * eps) / alpha_t
        x0_curr = (sample - model_output * sigma_curr) * (1.0 / alpha_curr)
        self.model_outputs.append(x0_curr)

        # 1st-order DPM-Solver++ update
        exp_neg_h = math.exp(-h)
        x_pred = sample * (sigma_next / sigma_curr) - x0_curr * (alpha_next * (exp_neg_h - 1.0))

        # 2nd-order multistep correction when previous data prediction is available
        if len(self.model_outputs) >= 2:
            x0_prev = self.model_outputs[-2]
            d_slope = x0_curr - x0_prev
            x_pred = x_pred - d_slope * (0.5 * alpha_next * (exp_neg_h - 1.0))

        return x_pred


class DDIMScheduler:
    """Denoising Diffusion Implicit Models (DDIM) Sampler."""

    def __init__(self, num_train_timesteps: int = 1000, eta: float = 0.0):
        num_train_timesteps = _check_positive("num_train_timesteps", num_train_timesteps)
        self.num_train_timesteps = num_train_timesteps
        self.eta = float(eta)
        self.betas = np.linspace(0.0001, 0.02, num_train_timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.timesteps: np.ndarray = np.arange(0, num_train_timesteps)[::-1].copy()

    def set_timesteps(self, num_inference_steps: int):
        num_inference_steps = _check_positive("num_inference_steps", num_inference_steps)
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps = {num_inference_steps} exceeds "
                f"num_train_timesteps = {self.num_train_timesteps}"
            )
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (np.arange(0, num_inference_steps) * step_ratio)[::-1].copy()

    def step(self, model_output: _Tensor, timestep: int, sample: _Tensor) -> _Tensor:
        """Executes a single DDIM reverse step."""
        idx = np.where(self.timesteps == timestep)[0]
        step_idx = int(idx[0]) if len(idx) > 0 else 0
        if step_idx + 1 < len(self.timesteps):
            t_prev = int(self.timesteps[step_idx + 1])
            alpha_bar_prev = float(self.alphas_cumprod[t_prev])
        else:
            t_prev = -1
            alpha_bar_prev = 1.0

        alpha_bar_t = float(self.alphas_cumprod[timestep])

        sigma_t = 0.0
        if self.eta > 0.0 and alpha_bar_prev < 1.0:
            variance = ((1.0 - alpha_bar_prev) / max(1.0 - alpha_bar_t, 1e-12)) * (1.0 - alpha_bar_t / alpha_bar_prev)
            sigma_t = self.eta * math.sqrt(max(0.0, variance))

        pred_x0 = (sample - model_output * math.sqrt(1.0 - alpha_bar_t)) * (1.0 / math.sqrt(alpha_bar_t))
        dir_xt = model_output * math.sqrt(max(0.0, 1.0 - alpha_bar_prev - sigma_t * sigma_t))
        prev_x = pred_x0 * math.sqrt(alpha_bar_prev) + dir_xt

        if sigma_t > 0.0:
            noise = _from_numpy(np.random.randn(*sample.shape).astype(np.float32))
            prev_x = prev_x + noise * sigma_t

        return prev_x


__all__ = ["EulerDiscreteScheduler", "DPMSolverMultistepScheduler", "DDIMScheduler"]
