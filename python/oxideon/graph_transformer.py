"""Graph Transformer with Edge Bias and Spectral Positional Encodings.

Implements Graph Transformer architecture (Dwivedi & Bresson, 2020; Kreuzer et al., 2021):
- GraphTransformerLayer: Full all-pair multi-head attention with structural edge feature bias
- GraphTransformer: Complete model with Laplacian positional encodings and graph-level readout
- compute_laplacian_pe: Computes smallest non-trivial eigenvectors of the normalized graph Laplacian
"""

from typing import Optional, List, Tuple
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, LayerNorm, GELU, Parameter, ModuleList, _check_repeats


def compute_laplacian_pe(adj: _Tensor, k: int = 8) -> _Tensor:
    """Computes k smallest non-trivial eigenvectors of the symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}."""
    a = adj.to_numpy().astype(np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adj must be a square 2D matrix, got shape {a.shape}")
    n = a.shape[0]
    if n == 0 or k <= 0:
        return _from_numpy(np.zeros((n, max(k, 0)), dtype=np.float32))

    deg = np.sum(a, axis=1)
    d_inv_sqrt = np.where(deg > 1e-12, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    l_sym = np.eye(n, dtype=np.float64) - (d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :])
    l_sym = 0.5 * (l_sym + l_sym.T)

    eigvals, eigvecs = np.linalg.eigh(l_sym)
    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]

    # Skip trivial first eigenvector (constant mode) when n > 1
    start = 1 if n > 1 else 0
    selected = eigvecs[:, start : start + k]
    if selected.shape[1] < k:
        pad = np.zeros((n, k - selected.shape[1]), dtype=np.float64)
        selected = np.concatenate([selected, pad], axis=1)

    return _from_numpy(np.ascontiguousarray(selected.astype(np.float32)))


class GraphTransformerLayer(Module):
    """Multi-Head Graph Transformer Layer with structural edge bias."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: Optional[int] = None,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        if node_dim % num_heads != 0:
            raise ValueError(f"node_dim {node_dim} must be divisible by num_heads {num_heads}")

        self.q_proj = Linear(node_dim, node_dim)
        self.k_proj = Linear(node_dim, node_dim)
        self.v_proj = Linear(node_dim, node_dim)
        self.out_proj = Linear(node_dim, node_dim)

        if edge_dim is not None:
            self.edge_proj = Linear(edge_dim, num_heads)
        else:
            self.edge_proj = None

        self.norm1 = LayerNorm(node_dim)
        self.norm2 = LayerNorm(node_dim)

        self.ffn = ModuleList([
            Linear(node_dim, dim_feedforward),
            GELU(),
            Linear(dim_feedforward, node_dim),
        ])

    def forward(
        self,
        h: _Tensor,
        edge_attr: Optional[_Tensor] = None,
        mask: Optional[_Tensor] = None,
    ) -> _Tensor:
        """Args:
            h: Node features [num_nodes, node_dim]
            edge_attr: Optional edge features [num_nodes, num_nodes, edge_dim]
            mask: Optional node attention mask [num_nodes, num_nodes]

        Returns:
            Updated node features [num_nodes, node_dim]
        """
        num_nodes = h.shape[0]
        if num_nodes == 0:
            return h

        # Multi-head self-attention preserving autograd: [N, H, D_h] -> [H, N, D_h]
        q = self.q_proj(h).reshape([num_nodes, self.num_heads, self.head_dim]).transpose(0, 1)
        k = self.k_proj(h).reshape([num_nodes, self.num_heads, self.head_dim]).transpose(0, 1)
        v = self.v_proj(h).reshape([num_nodes, self.num_heads, self.head_dim]).transpose(0, 1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = q.matmul(k.transpose(1, 2)) * scale  # [H, N, N]

        if self.edge_proj is not None and edge_attr is not None:
            # edge_proj(edge_attr): [N, N, H] -> transpose to [H, N, N]
            edge_bias = self.edge_proj(edge_attr).transpose(1, 2).transpose(0, 1)
            scores = scores + edge_bias

        if mask is not None:
            mask_np = mask.to_numpy()
            penalty = np.where(mask_np == 0, -1e9, 0.0).astype(np.float32)
            scores = scores + _from_numpy(penalty.reshape(1, num_nodes, num_nodes))

        attn_weights = scores.softmax(-1)  # [H, N, N]
        out = attn_weights.matmul(v).transpose(0, 1).reshape([num_nodes, self.node_dim])
        attn_out = self.out_proj(out)

        h_res = self.norm1(h + attn_out)

        ff_out = h_res
        for layer in self.ffn:
            ff_out = layer(ff_out)

        return self.norm2(h_res + ff_out)


class GraphTransformer(Module):
    """Complete Graph Transformer model for node or graph property prediction."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
        edge_dim: Optional[int] = None,
        num_layers: int = 3,
        num_heads: int = 4,
        laplacian_pe_dim: int = 8,
    ):
        super().__init__()
        num_layers = _check_repeats("num_layers", num_layers)
        self.laplacian_pe_dim = laplacian_pe_dim
        self.node_embed = Linear(in_dim + laplacian_pe_dim, hidden_dim)

        self.layers = ModuleList([
            GraphTransformerLayer(
                node_dim=hidden_dim,
                edge_dim=edge_dim,
                num_heads=num_heads,
                dim_feedforward=hidden_dim * 2,
            )
            for _ in range(num_layers)
        ])

        self.classifier = Linear(hidden_dim, out_dim)

    def forward(
        self,
        node_features: _Tensor,
        laplacian_pe: Optional[_Tensor] = None,
        edge_attr: Optional[_Tensor] = None,
    ) -> _Tensor:
        """Args:
            node_features: [num_nodes, in_dim]
            laplacian_pe: [num_nodes, laplacian_pe_dim]
            edge_attr: [num_nodes, num_nodes, edge_dim]

        Returns:
            Logits / predictions: [num_nodes, out_dim]
        """
        num_nodes = node_features.shape[0]

        if laplacian_pe is None:
            pe_arr = np.zeros((num_nodes, self.laplacian_pe_dim), dtype=np.float32)
            laplacian_pe = _from_numpy(pe_arr)

        h_in = _Tensor.cat([node_features, laplacian_pe], dim=-1)
        h = self.node_embed(h_in)

        for layer in self.layers:
            h = layer(h, edge_attr=edge_attr)

        return self.classifier(h)


__all__ = ["GraphTransformerLayer", "GraphTransformer", "compute_laplacian_pe"]
