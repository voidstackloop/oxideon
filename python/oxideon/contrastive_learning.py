"""
Multi-View Self-Supervised and Contrastive Representation Learning for Oxideon.
Includes differentiable Barlow Twins (Zbontar et al., ICML 2021), BYOL (Grill et al., NeurIPS 2020),
and SwAV online clustering with Sinkhorn-Knopp transport (Caron et al., NeurIPS 2020).
"""
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy


class BarlowTwinsLoss:
    """
    Differentiable Barlow Twins objective (Zbontar et al., ICML 2021).
    Minimizes redundancy by encouraging the empirical cross-correlation matrix
    between two augmented views to approach the identity matrix:
      L = sum_i (1 - C_ii)^2 + lambda * sum_i sum_{j != i} C_ij^2
    """
    def __init__(self, lambda_param: float = 0.005):
        self.lambda_param = float(lambda_param)

    def __call__(self, z1: Tensor, z2: Tensor) -> Tensor:
        if list(z1.shape) != list(z2.shape):
            raise ValueError(f"Shape mismatch: z1 {z1.shape} vs z2 {z2.shape}")
        if z1.ndim == 1:
            z1 = z1.unsqueeze(0)
            z2 = z2.unsqueeze(0)
        n, d = z1.shape[0], z1.shape[1]
        if n == 0:
            return _from_numpy(np.array(0.0, dtype=np.float32))

        z1_centered = z1 - z1.mean(dim=0, keepdim=True)
        z2_centered = z2 - z2.mean(dim=0, keepdim=True)
        std1 = ((z1_centered * z1_centered).mean(dim=0, keepdim=True) + 1e-7).sqrt()
        std2 = ((z2_centered * z2_centered).mean(dim=0, keepdim=True) + 1e-7).sqrt()

        z1_norm = z1_centered / std1
        z2_norm = z2_centered / std2

        c = z1_norm.transpose(0, 1).matmul(z2_norm) * (1.0 / float(n))
        eye = _from_numpy(np.eye(d, dtype=np.float32))
        off_mask = _from_numpy((1.0 - np.eye(d, dtype=np.float32)))

        diff_diag = (c - eye) * eye
        on_diag = (diff_diag * diff_diag).sum()
        off_diag = (c * c * off_mask).sum()
        return on_diag + off_diag * self.lambda_param


class BYOLLoss:
    """
    Differentiable Bootstrap Your Own Latent (BYOL) objective (Grill et al., NeurIPS 2020).
    Mean squared error between L2-normalized online predictions and target representations:
      L = 2 - 2 * (p dot z) / (||p|| * ||z||)
    """
    def __call__(self, p1: Tensor, z2: Tensor) -> Tensor:
        if list(p1.shape) != list(z2.shape):
            raise ValueError(f"Shape mismatch: p1 {p1.shape} vs z2 {z2.shape}")
        p_norm = p1 / ((p1 * p1).sum(dim=-1, keepdim=True) + 1e-12).sqrt()
        z_norm = z2 / ((z2 * z2).sum(dim=-1, keepdim=True) + 1e-12).sqrt()
        cosine_sim = (p_norm * z_norm).sum(dim=-1)
        return ((cosine_sim * -2.0) + 2.0).mean()


class SwAVLoss:
    """
    Differentiable Swapped Assignments between Views (SwAV, Caron et al., NeurIPS 2020).
    Computes stop-gradient online cluster code assignments via Sinkhorn-Knopp algorithm
    and differentiable cross-entropy between codes and predicted cluster log-probabilities.
    """
    def __init__(self, temperature: float = 0.1, sinkhorn_iters: int = 3, epsilon: float = 0.05):
        self.temperature = float(temperature)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.epsilon = float(epsilon)

    def _sinkhorn(self, scores: np.ndarray) -> np.ndarray:
        shifted = (scores - np.max(scores, axis=-1, keepdims=True)) / max(self.epsilon, 1e-6)
        q = np.exp(np.clip(shifted, -60.0, 0.0)).T  # (K, B)
        k_clusters, b_batch = q.shape
        q /= np.sum(q) + 1e-12

        for _ in range(self.sinkhorn_iters):
            q /= np.sum(q, axis=1, keepdims=True) + 1e-9
            q /= float(k_clusters)
            q /= np.sum(q, axis=0, keepdims=True) + 1e-9
            q /= float(b_batch)

        return (q * float(b_batch)).T.astype(np.float32)  # (B, K)

    def __call__(self, scores1: Tensor, scores2: Tensor) -> Tensor:
        s1_np = scores1.to_numpy()
        s2_np = scores2.to_numpy()

        # Stop-gradient optimal transport codes Q1 and Q2
        q1 = _from_numpy(self._sinkhorn(s1_np))
        q2 = _from_numpy(self._sinkhorn(s2_np))

        inv_temp = 1.0 / max(self.temperature, 1e-6)
        log_p1 = (scores1 * inv_temp).log_softmax(-1)
        log_p2 = (scores2 * inv_temp).log_softmax(-1)

        loss_12 = (q1 * log_p2).sum(dim=-1)
        loss_21 = (q2 * log_p1).sum(dim=-1)
        return (loss_12 + loss_21).mean() * -0.5
