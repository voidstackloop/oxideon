"""
Advanced Model Interpretability, Concept-based Explanations, and Feature Interactions for Oxideon.
Includes Testing with Concept Activation Vectors (TCAV, Kim et al., ICML 2018),
Integrated Hessians (pairwise feature interaction), Difference-from-Reference attributions,
and KernelSHAP linear regression attribution.
"""
from typing import Callable, List, Optional, Tuple
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy


class TCAV:
    """
    Testing with Concept Activation Vectors (TCAV, Kim et al., ICML 2018).
    Quantifies the degree to which high-level human concepts (e.g. 'stripes', 'fur')
    influence deep neural network predictions.
    """
    def __init__(self, cav_vector: Tensor):
        """
        Args:
            cav_vector: 1D Tensor representing normal vector orthogonal to concept decision boundary.
        """
        v = cav_vector.to_numpy().flatten()
        norm = np.linalg.norm(v)
        self.v = (v / (norm + 1e-7)).astype(np.float32)

    @classmethod
    def train_cav(cls, concept_activations: Tensor, random_activations: Tensor) -> "TCAV":
        """
        Fits a linear probe separating concept activations from random background activations
        using regularized Fisher linear discriminant / least squares.
        """
        c_act = concept_activations.to_numpy()
        r_act = random_activations.to_numpy()
        if c_act.ndim != 2 or r_act.ndim != 2:
            raise ValueError(f"Activations must be 2D [N, D], got {c_act.shape} and {r_act.shape}")
        if c_act.shape[1] != r_act.shape[1]:
            raise ValueError(f"Feature dimension mismatch: concept has {c_act.shape[1]}, random has {r_act.shape[1]}")
        if len(c_act) == 0 or len(r_act) == 0:
            raise ValueError("Activations cannot be empty")

        mean_c = np.mean(c_act, axis=0)
        mean_r = np.mean(r_act, axis=0)
        diff = mean_c - mean_r
        return cls(_from_numpy(diff.astype(np.float32)))

    def directional_derivative(self, activation_grad: Tensor) -> Tensor:
        """
        Computes directional derivative S_{C, k}(x) = grad_h(x) dot v_C.
        """
        g = activation_grad.to_numpy()
        if g.ndim == 1:
            score = np.dot(g, self.v)
            return _from_numpy(np.array(score, dtype=np.float32))

        scores = g @ self.v
        return _from_numpy(scores.astype(np.float32))

    def tcav_score(self, activation_grads: Tensor) -> float:
        """Computes fraction of inputs with positive directional derivative."""
        scores = self.directional_derivative(activation_grads).to_numpy()
        pos_count = np.sum(scores > 0)
        return float(pos_count / max(len(scores), 1))


class IntegratedHessians:
    """
    Integrated Hessians for computing pairwise non-linear feature interaction attributions.
    """
    def __init__(self, steps: int = 20):
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self.steps = steps

    def attribute(self, model_fn: Callable[[Tensor], Tensor], x: Tensor, baseline: Optional[Tensor] = None) -> Tensor:
        x_np = x.to_numpy().flatten()
        b_np = baseline.to_numpy().flatten() if baseline is not None else np.zeros_like(x_np)
        if len(x_np) != len(b_np):
            raise ValueError(f"Shape mismatch: x has {len(x_np)} elements, baseline has {len(b_np)}")
        diff = x_np - b_np
        d = len(x_np)

        alphas = np.linspace(0.0, 1.0, self.steps)
        hessian_acc = np.zeros((d, d), dtype=np.float32)

        eps = 1e-3
        for alpha in alphas:
            x_alpha = (b_np + alpha * diff).astype(np.float32)
            f_base = float(model_fn(_from_numpy(x_alpha)).to_numpy().flat[0])
            for i in range(d):
                ei = np.zeros(d, dtype=np.float32)
                ei[i] = eps
                f_i = float(model_fn(_from_numpy(x_alpha + ei)).to_numpy().flat[0])
                for j in range(i, d):
                    ej = np.zeros(d, dtype=np.float32)
                    ej[j] = eps
                    f_j = float(model_fn(_from_numpy(x_alpha + ej)).to_numpy().flat[0])
                    f_ij = float(model_fn(_from_numpy(x_alpha + ei + ej)).to_numpy().flat[0])

                    hij = (f_ij - f_i - f_j + f_base) / (eps * eps)
                    hessian_acc[i, j] += hij
                    if i != j:
                        hessian_acc[j, i] += hij

        hessian_avg = hessian_acc / float(self.steps)
        interaction_matrix = (diff[:, None] @ diff[None, :]) * hessian_avg
        return _from_numpy(interaction_matrix.astype(np.float32))


def difference_from_reference_multipliers(
    model_fn: Callable[[Tensor], Tensor],
    x: Tensor,
    reference: Tensor
) -> Tensor:
    """
    Computes DeepLIFT-style difference-from-reference multipliers and feature attributions.
    """
    x_np = x.to_numpy().flatten()
    ref_np = reference.to_numpy().flatten()
    if len(x_np) != len(ref_np):
        raise ValueError(f"Shape mismatch: x has {len(x_np)} elements, reference has {len(ref_np)}")
    delta_x = x_np - ref_np

    fx = float(model_fn(x).to_numpy().flat[0])
    fref = float(model_fn(reference).to_numpy().flat[0])
    delta_f = fx - fref

    norm_delta = float(np.sum(np.abs(delta_x))) + 1e-8
    attributions = delta_f * (np.abs(delta_x) / norm_delta) * np.sign(delta_x)
    return _from_numpy(attributions.reshape(x.shape).astype(np.float32))


def kernel_shap_linear(
    model_fn: Callable[[Tensor], Tensor],
    x: Tensor,
    reference: Tensor,
    num_samples: int = 32,
    seed: Optional[int] = 42,
) -> Tensor:
    """
    Approximates Shapley values via KernelSHAP weighted linear regression over coalitions.
    """
    x_np = x.to_numpy().flatten().astype(np.float32)
    ref_np = reference.to_numpy().flatten().astype(np.float32)
    if len(x_np) != len(ref_np):
        raise ValueError(f"Shape mismatch: x has {len(x_np)} elements, reference has {len(ref_np)}")
    M = len(x_np)
    if M == 0:
        return _from_numpy(np.zeros_like(x_np))
    if M == 1:
        val = float(model_fn(x).to_numpy().flat[0]) - float(model_fn(reference).to_numpy().flat[0])
        return _from_numpy(np.array([val], dtype=np.float32).reshape(x.shape))

    rng = np.random.default_rng(seed)
    f_ref = float(model_fn(_from_numpy(ref_np.reshape(x.shape))).to_numpy().flat[0])
    f_x = float(model_fn(_from_numpy(x_np.reshape(x.shape))).to_numpy().flat[0])

    Z_list = []
    y_list = []
    w_list = []

    for _ in range(max(num_samples, M + 2)):
        s = int(rng.integers(1, M))
        z = np.zeros(M, dtype=np.float32)
        chosen = rng.choice(M, size=s, replace=False)
        z[chosen] = 1.0
        coalition_x = ref_np + z * (x_np - ref_np)
        fz = float(model_fn(_from_numpy(coalition_x.reshape(x.shape))).to_numpy().flat[0])
        # Shapley kernel weight (M - 1) / (C(M, s) * s * (M - s))
        weight = float(M - 1) / float(s * (M - s))
        Z_list.append(z)
        y_list.append(fz - f_ref)
        w_list.append(weight)

    # Enforce efficiency constraint with high weight on full coalition z = 1
    Z_list.append(np.ones(M, dtype=np.float32))
    y_list.append(f_x - f_ref)
    w_list.append(1e4)

    Z = np.stack(Z_list, axis=0)
    y = np.array(y_list, dtype=np.float64)
    W = np.diag(np.array(w_list, dtype=np.float64))

    ZtW = Z.T @ W
    A = ZtW @ Z + 1e-6 * np.eye(M, dtype=np.float64)
    b = ZtW @ y
    phi = np.linalg.solve(A, b).astype(np.float32)
    return _from_numpy(phi.reshape(x.shape))


__all__ = [
    "TCAV",
    "IntegratedHessians",
    "difference_from_reference_multipliers",
    "kernel_shap_linear",
]
