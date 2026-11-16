"""Extended Graph Neural Network Architectures, Embeddings, and Graph Metrics.

Implements:
30. GatedGraphConv: Gated Graph Sequence Neural Networks with GRU recurrence (Li et al., 2015)
31. GraphSAGE: Inductive representation learning via neighborhood aggregation (Hamilton et al., 2017)
32. GINConv: Graph Isomorphism Network with epsilon parameter and MLP (Xu et al., 2019)
33. LaplacianEigenmaps: Spectral graph embedding using normalized graph Laplacian eigenvectors
34. AdamicAdarIndex: Link prediction metric inversely weighting common neighbor degrees
35. JaccardSimilarityCoeff: Graph neighborhood overlap similarity coefficient matrix
"""

from typing import Optional, List, Tuple, Union, Callable
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .primitives import cat as _cat
from .nn import Module, Linear, Parameter


class GatedGraphConv(Module):
    """Gated Graph Neural Network layer (Li et al., 2015)."""

    def __init__(self, out_channels: int, num_layers: int = 3):
        super().__init__()
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.message_linear = Linear(out_channels, out_channels, bias=False)
        self.w_z = Linear(out_channels * 2, out_channels)
        self.w_r = Linear(out_channels * 2, out_channels)
        self.w_h = Linear(out_channels * 2, out_channels)

    def forward(self, x: _Tensor, adj: _Tensor) -> _Tensor:
        h = x
        for _ in range(self.num_layers):
            m = adj @ self.message_linear(h)
            concat = _cat([m, h], dim=-1)
            z = self.w_z(concat).sigmoid()
            r = self.w_r(concat).sigmoid()
            r_h = r * h
            cand = self.w_h(_cat([m, r_h], dim=-1)).tanh()
            h = (1.0 - z) * h + z * cand
        return h


class GraphSAGE(Module):
    """GraphSAGE layer (Hamilton et al., 2017)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
        normalize: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr.lower()
        self.normalize = normalize

        self.lin_self = Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = Linear(in_channels, out_channels, bias=bias)

        if self.aggr not in ("mean", "sum", "max"):
            raise ValueError(f"Unsupported aggregation: {aggr}. Choose from 'mean', 'sum', 'max'.")

    def forward(self, x: _Tensor, adj: _Tensor) -> _Tensor:
        if self.aggr == "mean":
            degree = adj.sum(1, True).clamp(1.0, None)
            neigh_agg = (adj @ x) / degree
        elif self.aggr == "sum":
            neigh_agg = adj @ x
        elif self.aggr == "max":
            n = x.shape[0]
            if n == 0:
                neigh_agg = x
            else:
                x_np = x.to_numpy()
                adj_np = adj.to_numpy()
                rows: List[_Tensor] = []
                for i in range(n):
                    nbr_idx = np.where(adj_np[i] > 0)[0]
                    if len(nbr_idx) > 0:
                        sub_np = x_np[nbr_idx]  # [K, D]
                        max_k = np.argmax(sub_np, axis=0)  # [D]
                        # Build selector matrix [1, N] per feature or weighted mask [N, D]
                        mask_nd = np.zeros_like(x_np, dtype=np.float32)
                        for d_idx in range(x_np.shape[1]):
                            mask_nd[nbr_idx[max_k[d_idx]], d_idx] = 1.0
                        row_t = (x * _from_numpy(mask_nd)).sum(dim=0, keepdim=True)
                        rows.append(row_t)
                    else:
                        sel_i = np.zeros((1, n), dtype=np.float32)
                        sel_i[0, i] = 1.0
                        rows.append(_from_numpy(sel_i).matmul(x))
                neigh_agg = _cat(rows, dim=0)

        out = self.lin_self(x) + self.lin_neigh(neigh_agg)

        if self.normalize:
            norm = (out * out).sum(1, True).sqrt() + 1e-12
            out = out / norm

        return out


class GINConv(Module):
    """Graph Isomorphism Network (GIN) layer (Xu et al., 2019)."""

    def __init__(self, in_channels: int, out_channels: int, eps: float = 0.0, train_eps: bool = False):
        super().__init__()
        self.train_eps = train_eps
        if train_eps:
            self.eps = Parameter(_from_numpy(np.array([eps], dtype=np.float32)))
        else:
            self.eps = eps

        self.mlp_1 = Linear(in_channels, out_channels * 2)
        self.mlp_2 = Linear(out_channels * 2, out_channels)

    def forward(self, x: _Tensor, adj: _Tensor) -> _Tensor:
        neigh_sum = adj @ x
        combined = (1.0 + self.eps) * x + neigh_sum
        h = self.mlp_1(combined).relu()
        return self.mlp_2(h)


class LaplacianEigenmaps:
    """Laplacian Eigenmaps spectral graph embedding (Belkin & Niyogi, 2003)."""

    @staticmethod
    def fit_transform(adj: Union[_Tensor, np.ndarray], n_components: int = 2, normalized: bool = True) -> _Tensor:
        a = adj.to_numpy() if isinstance(adj, _Tensor) else np.array(adj, dtype=np.float64)
        n = a.shape[0]

        w = (a + a.T) / 2.0
        degrees = np.sum(w, axis=1)

        if normalized:
            inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
            l_sym = np.eye(n, dtype=np.float32) - (w * inv_sqrt[:, None] * inv_sqrt[None, :])
            eigenvalues, eigenvectors = np.linalg.eigh(l_sym)
            y = eigenvectors * inv_sqrt[:, None]
        else:
            l = -w.astype(np.float32)
            np.fill_diagonal(l, degrees)
            eigenvalues, eigenvectors = np.linalg.eigh(l)
            y = eigenvectors

        idx = np.argsort(eigenvalues)
        selected_idx = idx[1 : n_components + 1]
        embedding = y[:, selected_idx].astype(np.float32)
        return _from_numpy(embedding)


class AdamicAdarIndex:
    """Adamic-Adar link prediction metric for graph node pairs."""

    @staticmethod
    def score_matrix(adj: Union[_Tensor, np.ndarray]) -> _Tensor:
        a = (adj.to_numpy() if isinstance(adj, _Tensor) else np.array(adj, dtype=np.float32)).copy()
        np.fill_diagonal(a, 0.0)
        a_bin = (a > 0).astype(np.float32)
        n = a_bin.shape[0]
        degrees = np.sum(a_bin, axis=1)

        weights = np.zeros(n, dtype=np.float32)
        valid = degrees > 1
        weights[valid] = 1.0 / np.log(degrees[valid].astype(np.float32))

        scores = np.matmul(a_bin * weights[None, :], a_bin.T)
        np.fill_diagonal(scores, 0.0)
        return _from_numpy(scores)


class JaccardSimilarityCoeff:
    """Graph neighborhood overlap Jaccard similarity coefficient matrix."""

    @staticmethod
    def score_matrix(adj: Union[_Tensor, np.ndarray]) -> _Tensor:
        a = ((adj.to_numpy() if isinstance(adj, _Tensor) else np.array(adj)) > 0).copy()
        np.fill_diagonal(a, False)
        n = a.shape[0]

        a_f = a.astype(np.float32)
        intersection = np.matmul(a_f, a_f.T)

        degrees = np.sum(a_f, axis=1, keepdims=True)
        union = degrees + degrees.T - intersection

        jaccard = np.zeros((n, n), dtype=np.float32)
        nonzero = union > 0
        jaccard[nonzero] = intersection[nonzero] / union[nonzero]
        np.fill_diagonal(jaccard, 1.0)
        return _from_numpy(jaccard)
