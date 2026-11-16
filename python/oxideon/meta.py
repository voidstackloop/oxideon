"""Meta-Learning, Few-Shot Learning, and Prototypical Networks for Oxideon.

Enables fast adaptation and few-shot episodic evaluation on novel tasks:
  - compute_prototypes: Computes mean class prototype embeddings
  - PrototypicalNetworksLoss: Metric-based few-shot classification (Snell et al., 2017)
  - MAMLLinear: Functional weight linear layer for Model-Agnostic Meta-Learning (Finn et al., 2017)
"""

from typing import Tuple, Optional, Any, List, Dict
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter


def compute_prototypes(
    support_embeddings: Any, support_labels: Any, num_classes: int
) -> _Tensor:
    """Computes mean centroid prototype vectors per class [num_classes, D]."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    emb = support_embeddings.to_numpy() if isinstance(support_embeddings, _Tensor) else np.asarray(support_embeddings, dtype=np.float32)
    lbl = support_labels.to_numpy() if isinstance(support_labels, _Tensor) else np.asarray(support_labels, dtype=np.int64)
    lbl = lbl.reshape(-1)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    if len(emb) != len(lbl):
        raise ValueError(f"Length mismatch: support_embeddings has {len(emb)} samples, support_labels has {len(lbl)}")
    if len(lbl) > 0 and (np.any(lbl < 0) or np.any(lbl >= num_classes)):
        raise ValueError(f"support_labels out of bounds for num_classes={num_classes}")

    if isinstance(support_embeddings, _Tensor) and support_embeddings.requires_grad:
        s_t = support_embeddings.reshape([1, -1]) if len(support_embeddings.shape) == 1 else support_embeddings
        n_sup = len(lbl)
        w_mat = np.zeros((num_classes, n_sup), dtype=np.float32)
        for c in range(num_classes):
            mask = (lbl == c)
            cnt = int(np.sum(mask))
            if cnt > 0:
                w_mat[c, mask] = 1.0 / float(cnt)
        return _from_numpy(np.ascontiguousarray(w_mat)).matmul(s_t)

    d = emb.shape[1]
    prototypes = np.zeros((num_classes, d), dtype=np.float32)

    for c in range(num_classes):
        mask = lbl == c
        if np.any(mask):
            prototypes[c] = np.mean(emb[mask], axis=0)

    return _from_numpy(np.ascontiguousarray(prototypes, dtype=np.float32))


class PrototypicalNetworksLoss(Module):
    """Snell et al. (2017) Prototypical Networks Euclidean loss for few-shot episodes."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        query_embeddings: Any,
        prototypes: Any,
        query_labels: Any,
    ) -> Tuple[_Tensor, float]:
        """Calculates cross-entropy loss based on negative Euclidean distance to prototypes.

        Returns (loss_tensor, accuracy).
        """
        q = query_embeddings.to_numpy() if isinstance(query_embeddings, _Tensor) else np.asarray(query_embeddings, dtype=np.float32)
        proto = prototypes.to_numpy() if isinstance(prototypes, _Tensor) else np.asarray(prototypes, dtype=np.float32)
        lbl = query_labels.to_numpy() if isinstance(query_labels, _Tensor) else np.asarray(query_labels, dtype=np.int64)
        lbl = lbl.reshape(-1)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if proto.ndim == 1:
            proto = proto.reshape(1, -1)
        if len(q) != len(lbl):
            raise ValueError(f"Length mismatch: query_embeddings has {len(q)}, query_labels has {len(lbl)}")
        if len(lbl) == 0:
            raise ValueError("query_labels must not be empty")
        if np.any(lbl < 0) or np.any(lbl >= proto.shape[0]):
            raise ValueError(f"query_labels out of bounds for {proto.shape[0]} prototypes")

        # Distance matrix [N_query, num_classes]
        q_sq = np.sum(q**2, axis=1, keepdims=True)
        p_sq = np.sum(proto**2, axis=1, keepdims=True)
        dists = np.maximum(0.0, q_sq + p_sq.T - 2.0 * (q @ proto.T))

        logits = -dists
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        n = len(lbl)
        acc = float(np.mean(np.argmax(probs, axis=1) == lbl))

        if (isinstance(query_embeddings, _Tensor) and query_embeddings.requires_grad) or (
            isinstance(prototypes, _Tensor) and prototypes.requires_grad
        ):
            q_t = query_embeddings if isinstance(query_embeddings, _Tensor) else _from_numpy(q)
            p_t = prototypes if isinstance(prototypes, _Tensor) else _from_numpy(proto)
            if len(q_t.shape) == 1:
                q_t = q_t.reshape([1, -1])
            if len(p_t.shape) == 1:
                p_t = p_t.reshape([1, -1])
            q_sq_t = (q_t * q_t).sum(dim=1, keepdim=True)
            p_sq_t = (p_t * p_t).sum(dim=1, keepdim=True).transpose(0, 1)
            dists_t = (q_sq_t + p_sq_t - q_t.matmul(p_t.transpose(0, 1)) * 2.0).relu()
            logits_t = dists_t * -1.0
            log_probs = logits_t.log_softmax(-1)
            one_hot = np.zeros((n, proto.shape[0]), dtype=np.float32)
            one_hot[np.arange(n), lbl] = 1.0
            loss_t = ((log_probs * _from_numpy(one_hot)).sum(-1) * -1.0).mean()
            return loss_t, acc

        loss = -np.mean(np.log(np.maximum(probs[np.arange(n), lbl], 1e-12)))
        return _from_numpy(np.array([loss], dtype=np.float32)), acc


class MAMLLinear(Module):
    """Linear layer supporting external weights for MAML inner-loop adaptations."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be >= 1")
        self.in_features = in_features
        self.out_features = out_features

        bound = 1.0 / math.sqrt(in_features)
        w = np.random.uniform(-bound, bound, (in_features, out_features)).astype(np.float32)
        self.weight = Parameter(_from_numpy(w))
        if bias:
            b = np.zeros(out_features, dtype=np.float32)
            self.bias = Parameter(_from_numpy(b))
        else:
            self.bias = None

    def forward(
        self,
        x: _Tensor,
        weight: Optional[_Tensor] = None,
        bias: Optional[_Tensor] = None,
    ) -> _Tensor:
        w = weight if weight is not None else self.weight
        b = bias if bias is not None else self.bias

        out = x @ w
        if b is not None:
            out = out + b
        return out


__all__ = [
    "compute_prototypes",
    "PrototypicalNetworksLoss",
    "MAMLLinear",
]
