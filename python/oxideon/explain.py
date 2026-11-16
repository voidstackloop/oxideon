"""Interpretability, attribution, and explainable AI for Oxideon."""

from typing import Any, Callable, List, Optional
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy, grad as _native_grad


def _select_target_score(output: _Tensor, target_class: Optional[int]) -> _Tensor:
    if target_class is None:
        return output.sum()
    if len(output.shape) > 1:
        mask = np.zeros(output.shape, dtype=np.float32)
        mask[..., int(target_class)] = 1.0
        return (output * _from_numpy(mask)).sum()
    return output[int(target_class)]


class IntegratedGradients:
    """Integrated Gradients attribution algorithm for explaining neural network predictions."""

    def __init__(self, model: Any):
        self.model = model

    def attribute(
        self,
        inputs: _Tensor,
        target_class: Optional[int] = None,
        baselines: Optional[_Tensor] = None,
        steps: int = 20,
    ) -> _Tensor:
        """Computes path-integral attribution scores."""
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if baselines is not None and baselines.shape != inputs.shape:
            raise ValueError(f"baselines shape mismatch: expected {inputs.shape}, got {baselines.shape}")
        if baselines is None:
            baselines = _Tensor.zeros_like(inputs)

        inp_arr = inputs.to_numpy()
        base_arr = baselines.to_numpy()
        diff = inp_arr - base_arr

        total_grads = np.zeros_like(inp_arr)

        for step in range(1, steps + 1):
            alpha = float(step) / float(steps)
            interp = base_arr + alpha * diff
            interp_t = _from_numpy(np.ascontiguousarray(interp, dtype=np.float32)).detach().requires_grad_(True)

            output = self.model(interp_t)
            score = _select_target_score(output, target_class)

            score.backward()
            g = interp_t.grad()
            if g is not None:
                total_grads += g.to_numpy()

        avg_grads = total_grads / float(steps)
        attributions = diff * avg_grads
        return _from_numpy(np.ascontiguousarray(attributions, dtype=np.float32))


class SaliencyMap:
    """Vanilla gradient saliency map: |grad(f(x))|."""

    def __init__(self, model: Any):
        self.model = model

    def attribute(self, inputs: _Tensor, target_class: Optional[int] = None) -> _Tensor:
        inp = inputs.detach().requires_grad_(True)

        out = self.model(inp)
        score = _select_target_score(out, target_class)

        score.backward()
        g = inp.grad()
        if g is not None:
            saliency = np.abs(g.to_numpy())
            return _from_numpy(np.ascontiguousarray(saliency, dtype=np.float32))
        return _Tensor.zeros_like(inputs)


class SmoothGrad:
    """SmoothGrad attribution (Smilkov et al., 2017): averages gradients over Gaussian-perturbed inputs."""

    def __init__(
        self,
        model: Any,
        stdev: float = 0.1,
        num_samples: int = 20,
        seed: Optional[int] = None,
    ):
        if float(stdev) < 0.0:
            raise ValueError(f"stdev must be >= 0, got {stdev}")
        if int(num_samples) < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        self.model = model
        self.stdev = float(stdev)
        self.num_samples = int(num_samples)
        self.seed = seed

    def attribute(self, inputs: _Tensor, target_class: Optional[int] = None) -> _Tensor:
        inp_arr = inputs.to_numpy()
        rng = np.random.default_rng(self.seed)
        total_grads = np.zeros_like(inp_arr, dtype=np.float32)

        for _ in range(self.num_samples):
            noise = rng.normal(0.0, self.stdev, size=inp_arr.shape).astype(np.float32)
            noisy_t = _from_numpy(np.ascontiguousarray(inp_arr + noise, dtype=np.float32)).detach().requires_grad_(True)
            out = self.model(noisy_t)
            score = _select_target_score(out, target_class)
            score.backward()
            g = noisy_t.grad()
            if g is not None:
                total_grads += np.abs(g.to_numpy())

        avg_grads = total_grads / float(self.num_samples)
        return _from_numpy(np.ascontiguousarray(avg_grads, dtype=np.float32))


class FeatureAblation:
    """Perturbation-based feature ablation attribution."""

    def __init__(self, model: Any):
        self.model = model

    def attribute(self, inputs: _Tensor, target_class: Optional[int] = None) -> _Tensor:
        base_out = self.model(inputs).to_numpy()
        if target_class is not None:
            base_score = float(np.sum(base_out[..., target_class])) if base_out.ndim > 1 else float(base_out[target_class])
        else:
            base_score = float(np.sum(base_out))

        inp_arr = inputs.to_numpy()
        flat_inp = inp_arr.reshape(-1)
        importances = np.zeros_like(flat_inp)

        for i in range(len(flat_inp)):
            orig_val = flat_inp[i]
            flat_inp[i] = 0.0  # Ablate feature

            ablated_t = _from_numpy(np.ascontiguousarray(inp_arr, dtype=np.float32))
            abl_out = self.model(ablated_t).to_numpy()
            if target_class is not None:
                abl_score = float(np.sum(abl_out[..., target_class])) if abl_out.ndim > 1 else float(abl_out[target_class])
            else:
                abl_score = float(np.sum(abl_out))

            importances[i] = base_score - abl_score
            flat_inp[i] = orig_val

        return _from_numpy(np.ascontiguousarray(importances.reshape(inp_arr.shape), dtype=np.float32))


__all__ = [
    "IntegratedGradients",
    "SaliencyMap",
    "SmoothGrad",
    "FeatureAblation",
]
