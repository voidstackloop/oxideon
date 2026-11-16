#!/usr/bin/env python3
"""Normalizing Flows, Diffusion Processes, and Fast High-Order Schedulers.

Demonstrates modern generative modeling paradigms in Oxideon:
  1. RealNVP Normalizing Flow: Exact log-likelihood density evaluation & inverse generation
  2. DDPMScheduler: Forward Gaussian diffusion noise corruption trajectory q(x_t | x_0)
  3. Comparative Fast ODE Samplers:
     - 1st-order Euler Discrete Scheduler
     - 2nd-order DPM-Solver++ Multistep ODE Solver (Lu et al., 2022)
     - Deterministic DDIM (Denoising Diffusion Implicit Models)
  4. End-to-end score-matching training loop with autograd backpropagation

Usage:
  python examples/flow_matching_and_diffusion.py
"""

import time
import numpy as np

import oxideon as ox
from oxideon import Tensor, from_numpy
from oxideon.flow import RealNVP, AffineCouplingLayer
from oxideon.diffusion import DDPMScheduler, SinusoidalPositionalEmbedding
from oxideon.diffusion_sampler import (
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)
from oxideon.optim import AdamW


def banner(title: str) -> None:
    print("\n" + "=" * 76)
    print(f"🌊 {title}")
    print("=" * 76)


def main():
    ox.manual_seed(42)
    np.random.seed(42)

    banner("Oxideon Normalizing Flows & High-Order Diffusion Schedulers")

    dim = 8
    num_samples = 64

    # -----------------------------------------------------------------------
    # Step 1: RealNVP Normalizing Flow (Exact Likelihood & Sampling)
    # -----------------------------------------------------------------------
    print("\n[Step 1] Invertible Generative Modeling via RealNVP Flow...")
    flow = RealNVP(in_features=dim, num_layers=4, hidden_dim=32)

    x_data = ox.randn([num_samples, dim])
    t0 = time.perf_counter()
    z_latents, log_det = flow.forward(x_data)
    t1 = time.perf_counter()

    # Exact log-likelihood computation via change of variables: log p(x) = log p(z) + log|det J|
    z_np = z_latents.to_numpy()
    log_pz = -0.5 * (np.sum(z_np ** 2, axis=-1) + dim * np.log(2.0 * np.pi))
    log_px = log_pz + log_det.to_numpy()

    print(f"  • Data dimension: {dim}, Sample count: {num_samples}")
    print(f"  • Forward encoding latency: {(t1 - t0) * 1000.0:.2f} ms")
    print(f"  • Latent representation shape: {z_latents.shape}")
    print(f"  • Mean log-likelihood log p(x): {float(np.mean(log_px)):.4f} nats")

    # Invertible generative sampling: z ~ N(0, I) -> x
    z_prior = ox.randn([16, dim])
    x_gen = flow.inverse(z_prior)
    print(f"  • Generated samples from prior: {x_gen.shape}")
    assert x_gen.shape == (16, dim)

    # -----------------------------------------------------------------------
    # Step 2: Forward Diffusion Process (DDPM Schedule)
    # -----------------------------------------------------------------------
    print("\n[Step 2] Forward Diffusion Trajectory q(x_t | x_0)...")
    ddpm = DDPMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
    time_embed = SinusoidalPositionalEmbedding(dim=32)

    timesteps = [0, 100, 250, 500, 750, 999]
    x_0 = ox.ones([4, dim])
    noise = ox.randn([4, dim])

    print("  Timestep t | Noise Alpha_bar | Signal Weight | Noise Weight | Mean Magnitude")
    print("  -----------|-----------------|---------------|--------------|---------------")

    for t in timesteps:
        alpha_bar = float(ddpm.alphas_cumprod[t])
        sqrt_alpha = np.sqrt(alpha_bar)
        sqrt_one_minus = np.sqrt(1.0 - alpha_bar)
        # Corrupt data: x_t = sqrt(alpha_bar)*x_0 + sqrt(1 - alpha_bar)*noise
        x_t = x_0 * float(sqrt_alpha) + noise * float(sqrt_one_minus)
        mag = float(np.mean(np.abs(x_t.to_numpy())))
        print(f"    {t:7d}  | {alpha_bar:15.4f} | {sqrt_alpha:13.4f} | {sqrt_one_minus:12.4f} | {mag:14.4f}")

    # -----------------------------------------------------------------------
    # Step 3: Comparative Fast ODE Samplers
    # -----------------------------------------------------------------------
    print("\n[Step 3] Comparative High-Order Diffusion ODE Solvers...")

    inference_steps = 15
    schedulers = {
        "1st-Order Euler": EulerDiscreteScheduler(num_train_timesteps=1000),
        "2nd-Order DPM-Solver++": DPMSolverMultistepScheduler(num_train_timesteps=1000),
        "Deterministic DDIM": DDIMScheduler(num_train_timesteps=1000),
    }

    # Dummy score-matching network model: predicts noise residual
    class DummyEpsilonModel(ox.nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.net = ox.nn.Sequential(
                ox.nn.Linear(d, 32),
                ox.nn.SiLU(),
                ox.nn.Linear(32, d),
            )
        def forward(self, x_in, t_val):
            return self.net(x_in) * 0.1

    model = DummyEpsilonModel(dim)

    print("  Scheduler                | Steps | Latency (ms) | Trajectory Norm")
    print("  -------------------------|-------|--------------|----------------")

    for name, sched in schedulers.items():
        sched.set_timesteps(num_inference_steps=inference_steps)
        latents = ox.randn([4, dim])

        t0 = time.perf_counter()
        for t in sched.timesteps:
            t_int = int(t)
            noise_pred = model(latents, t_int)
            step_res = sched.step(noise_pred, t_int, latents)
            latents = step_res.prev_sample if hasattr(step_res, "prev_sample") else step_res
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        final_norm = float(np.mean(np.linalg.norm(latents.to_numpy(), axis=-1)))
        print(f"  {name:24s} | {len(sched.timesteps):5d} | {elapsed_ms:12.2f} | {final_norm:15.4f}")

    # -----------------------------------------------------------------------
    # Step 4: Differentiable Score-Matching Training Loop
    # -----------------------------------------------------------------------
    print("\n[Step 4] Differentiable Score-Matching Autograd Optimization...")
    score_net = ox.nn.Sequential(
        ox.nn.Linear(dim, 32),
        ox.nn.GELU(),
        ox.nn.Linear(32, dim),
    )
    optimizer = AdamW(score_net.parameters(), lr=5e-3)

    target_noise = ox.randn([num_samples, dim])
    perturbed_x = x_data + target_noise * 0.5

    init_loss = 0.0
    final_loss = 0.0

    print("  Iter | Denoising Score MSE | Loss Reduction")
    print("  -----|---------------------|---------------")

    for it in range(1, 9):
        optimizer.zero_grad()
        pred_noise = score_net(perturbed_x)
        diff = pred_noise - target_noise
        loss = (diff * diff).mean()

        val = float(loss.to_numpy().flat[0])
        if it == 1:
            init_loss = val
        final_loss = val

        loss.backward()
        optimizer.step()

        reduction = (1.0 - val / init_loss) * 100.0 if init_loss > 0 else 0.0
        print(f"  {it:4d} | {val:19.4f} | {reduction:12.1f}%")

    print(f"\n  ✓ Denoising score network converged from {init_loss:.4f} down to {final_loss:.4f}.")
    assert final_loss < init_loss, "Expected denoising loss to decrease"

    banner("Normalizing Flows & Diffusion Showcase Completed Successfully! ✅")


if __name__ == "__main__":
    main()
