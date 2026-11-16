"""Graph Neural Networks (GNN) module for Oxideon.

Provides graph representations, neighborhood aggregators, and standard GNN layers:
  - GraphData: Geometric graph container (node features, edge_index, targets)
  - degree, add_self_loops: Graph topology utilities
  - GCNConv: Kipf & Welling Graph Convolutional Network layer
  - GATConv: Graph Attention Network layer with multi-head self-attention
  - GraphSAGEConv: Inductive GraphSAGE neighborhood sampling & aggregation
"""

from typing import Optional, Tuple, Union, Any, List
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy, SparseCoo as _SparseCoo
from .nn import Module, Parameter, _check_elements


class GraphData:
    """Graph structure container holding node features and topological connectivity."""

    def __init__(
        self,
        x: Any,
        edge_index: Any,
        edge_attr: Optional[Any] = None,
        y: Optional[Any] = None,
    ):
        if not isinstance(x, _Tensor):
            x = _from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.x = x

        if isinstance(edge_index, _Tensor):
            edge_index = edge_index.to_numpy().astype(np.int64)
        else:
            edge_index = np.asarray(edge_index, dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, num_edges], got {edge_index.shape}")
        self.edge_index = edge_index

        self.edge_attr = edge_attr
        self.y = y

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def num_node_features(self) -> int:
        return self.x.shape[1]

    def is_directed(self) -> bool:
        edges = set(zip(self.edge_index[0].tolist(), self.edge_index[1].tolist()))
        reversed_edges = set(zip(self.edge_index[1].tolist(), self.edge_index[0].tolist()))
        return edges != reversed_edges

    def __repr__(self) -> str:
        return (
            f"GraphData(nodes={self.num_nodes}, edges={self.num_edges}, "
            f"features={self.num_node_features})"
        )


def degree(edge_index: Any, num_nodes: Optional[int] = None) -> np.ndarray:
    """Computes the in-degree of each node in the graph."""
    if isinstance(edge_index, _Tensor):
        edge_index = edge_index.to_numpy().astype(np.int64)
    else:
        edge_index = np.asarray(edge_index, dtype=np.int64)
    target_nodes = edge_index[1]
    if num_nodes is None:
        num_nodes = int(np.max(edge_index)) + 1 if edge_index.size > 0 else 0
    deg = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(deg, target_nodes, 1.0)
    return deg


def add_self_loops(
    edge_index: Any, num_nodes: Optional[int] = None
) -> np.ndarray:
    """Appends self-loops (i, i) to edge_index for all nodes 0..num_nodes-1."""
    if isinstance(edge_index, _Tensor):
        edge_index = edge_index.to_numpy().astype(np.int64)
    else:
        edge_index = np.asarray(edge_index, dtype=np.int64)
    if num_nodes is None:
        num_nodes = int(np.max(edge_index)) + 1 if edge_index.size > 0 else 0
    num_nodes = _check_elements("num_nodes", num_nodes)
    loops = np.stack([np.arange(num_nodes), np.arange(num_nodes)], axis=0)
    new_edge_index = np.concatenate([edge_index, loops], axis=1)
    return new_edge_index


def remove_self_loops(
    edge_index: Any,
    edge_attr: Optional[Any] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Removes self-loops (i, i) from edge_index and optional edge_attr."""
    if isinstance(edge_index, _Tensor):
        edge_index = edge_index.to_numpy().astype(np.int64)
    else:
        edge_index = np.asarray(edge_index, dtype=np.int64)

    mask = edge_index[0] != edge_index[1]
    new_edge_index = edge_index[:, mask]

    if edge_attr is not None:
        if isinstance(edge_attr, _Tensor):
            edge_attr = edge_attr.to_numpy()
        else:
            edge_attr = np.asarray(edge_attr)
        new_edge_attr = edge_attr[mask]
        return new_edge_index, new_edge_attr
    return new_edge_index, None


def to_undirected(
    edge_index: Any,
    num_nodes: Optional[int] = None,
) -> np.ndarray:
    """Converts a directed edge_index into an undirected edge_index (symmetric with duplicates removed)."""
    if isinstance(edge_index, _Tensor):
        edge_index = edge_index.to_numpy().astype(np.int64)
    else:
        edge_index = np.asarray(edge_index, dtype=np.int64)

    if edge_index.shape[1] == 0:
        return edge_index.copy()

    rev_edge_index = np.stack([edge_index[1], edge_index[0]], axis=0)
    all_edges = np.concatenate([edge_index, rev_edge_index], axis=1)

    # Deduplicate edges
    edges_set = set(zip(all_edges[0].tolist(), all_edges[1].tolist()))
    sorted_edges = sorted(edges_set)
    new_src = [e[0] for e in sorted_edges]
    new_dst = [e[1] for e in sorted_edges]
    return np.array([new_src, new_dst], dtype=np.int64)


def subgraph(
    subset: Any,
    edge_index: Any,
    edge_attr: Optional[Any] = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Returns the induced subgraph containing only the nodes specified in subset."""
    if isinstance(edge_index, _Tensor):
        edge_index = edge_index.to_numpy().astype(np.int64)
    else:
        edge_index = np.asarray(edge_index, dtype=np.int64)

    if isinstance(subset, _Tensor):
        subset = subset.to_numpy().astype(np.int64)
    else:
        subset = np.asarray(subset, dtype=np.int64)

    subset_set = set(subset.tolist())
    mask = np.array(
        [s in subset_set and d in subset_set for s, d in zip(edge_index[0], edge_index[1])],
        dtype=bool,
    )
    sub_edge_index = edge_index[:, mask]

    if relabel_nodes:
        node_map = {orig: i for i, orig in enumerate(sorted(subset_set))}
        sub_edge_index = np.array(
            [[node_map[s] for s in sub_edge_index[0]], [node_map[d] for d in sub_edge_index[1]]],
            dtype=np.int64,
        )

    if edge_attr is not None:
        if isinstance(edge_attr, _Tensor):
            edge_attr = edge_attr.to_numpy()
        else:
            edge_attr = np.asarray(edge_attr)
        return sub_edge_index, edge_attr[mask]
    return sub_edge_index, None


def k_hop_subgraph(
    node_idx: Union[int, List[int]],
    num_hops: int,
    edge_index: Any,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the k-hop neighborhood subgraph around specified seed node(s).

    Returns:
        subset_nodes (np.ndarray): Node indices in the k-hop neighborhood.
        sub_edge_index (np.ndarray): Induced subgraph edge indices.
        mapping (np.ndarray): Target seed node(s) mapped to new node indexing if relabel_nodes is True.
        edge_mask (np.ndarray): Boolean mask of edges preserved from original edge_index.
    """
    if isinstance(edge_index, _Tensor):
        edge_index = edge_index.to_numpy().astype(np.int64)
    else:
        edge_index = np.asarray(edge_index, dtype=np.int64)

    if isinstance(node_idx, (int, np.integer)):
        seed_nodes = [int(node_idx)]
    else:
        seed_nodes = [int(n) for n in node_idx]

    visited = set(seed_nodes)
    current_frontier = set(seed_nodes)

    src = edge_index[0]
    dst = edge_index[1]

    # Pre-build adjacency list for fast BFS
    adj: dict = {}
    for s, d in zip(src, dst):
        adj.setdefault(s, []).append(d)
        adj.setdefault(d, []).append(s)

    for _ in range(num_hops):
        next_frontier = set()
        for u in current_frontier:
            for v in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    next_frontier.add(v)
        current_frontier = next_frontier
        if not current_frontier:
            break

    subset_nodes = np.array(sorted(visited), dtype=np.int64)
    visited_set = set(subset_nodes.tolist())

    edge_mask = np.array(
        [s in visited_set and d in visited_set for s, d in zip(src, dst)],
        dtype=bool,
    )
    sub_edges = edge_index[:, edge_mask]

    if relabel_nodes:
        node_map = {orig: i for i, orig in enumerate(subset_nodes)}
        mapping = np.array([node_map[s] for s in seed_nodes], dtype=np.int64)
        sub_edges = np.array(
            [[node_map[s] for s in sub_edges[0]], [node_map[d] for d in sub_edges[1]]],
            dtype=np.int64,
        )
    else:
        mapping = np.array(seed_nodes, dtype=np.int64)

    return subset_nodes, sub_edges, mapping, edge_mask


class GCNConv(Module):
    """Graph Convolutional Network layer (Kipf & Welling, 2017).

    Implements symmetrically normalized sparse message passing:
        X' = D^{-1/2} A_hat D^{-1/2} X W + b
    where A_hat = A + I_N, computed in O(E) sparse memory and operations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        add_self_loops: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be >= 1")
        self.in_features = in_features
        self.out_features = out_features
        self.has_self_loops = add_self_loops

        bound = 1.0 / math.sqrt(in_features)
        w_init = np.random.uniform(-bound, bound, (in_features, out_features)).astype(np.float32)
        self.weight = Parameter(_from_numpy(np.ascontiguousarray(w_init)))

        if bias:
            b_init = np.random.uniform(-bound, bound, (out_features,)).astype(np.float32)
            self.bias = Parameter(_from_numpy(np.ascontiguousarray(b_init)))
        else:
            self.bias = None

    def forward(self, x: _Tensor, edge_index: Any) -> _Tensor:
        """Forward pass over graph node features and edge connectivity in O(E) sparse complexity."""
        if isinstance(edge_index, _Tensor):
            edge_index = edge_index.to_numpy().astype(np.int64)
        else:
            edge_index = np.asarray(edge_index, dtype=np.int64)

        num_nodes = x.shape[0]

        if self.has_self_loops:
            edge_index = add_self_loops(edge_index, num_nodes=num_nodes)

        # 1. Project node features: H = X @ W
        h = x @ self.weight

        # 2. Symmetrically normalized sparse adjacency propagation
        src, dst = edge_index[0], edge_index[1]
        num_edges = len(src)
        deg = np.zeros(num_nodes, dtype=np.float32)
        np.add.at(deg, dst, 1.0)
        deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)

        edge_weight = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        pairs = [(int(dst[i]), int(src[i])) for i in range(num_edges)]
        sp_adj = _SparseCoo(pairs, [float(v) for v in edge_weight], (num_nodes, num_nodes))

        out = sp_adj @ h

        if self.bias is not None:
            out = out + self.bias
        return out


class GATConv(Module):
    """Graph Attention Network layer (Veličković et al., 2018).

    Computes multi-head anisotropic attention weights across graph edges in O(E) sparse execution.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True,
    ):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be >= 1")
        if heads < 1:
            raise ValueError("heads must be >= 1")
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        total_out = out_features * heads
        bound = 1.0 / math.sqrt(in_features)
        w_init = np.random.uniform(-bound, bound, (in_features, total_out)).astype(np.float32)
        self.weight = Parameter(_from_numpy(np.ascontiguousarray(w_init)))

        # Attention vectors: [heads, 2 * out_features]
        a_init = np.random.uniform(-bound, bound, (heads, 2 * out_features)).astype(np.float32)
        self.att = Parameter(_from_numpy(np.ascontiguousarray(a_init)))

        if bias:
            final_dim = total_out if concat else out_features
            b_init = np.zeros(final_dim, dtype=np.float32)
            self.bias = Parameter(_from_numpy(np.ascontiguousarray(b_init)))
        else:
            self.bias = None

    def forward(self, x: _Tensor, edge_index: Any) -> _Tensor:
        if isinstance(edge_index, _Tensor):
            edge_index = edge_index.to_numpy().astype(np.int64)
        else:
            edge_index = np.asarray(edge_index, dtype=np.int64)

        num_nodes = x.shape[0]
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        src, dst = edge_index[0], edge_index[1]
        num_edges = len(src)

        # 1. Project node features
        h = x @ self.weight
        h_np = h.to_numpy().reshape(num_nodes, self.heads, self.out_features)

        # 2. Compute attention coefficients
        att_np = self.att.to_numpy()
        a_src = att_np[:, : self.out_features]
        a_dst = att_np[:, self.out_features :]

        alpha_src = np.sum(h_np * a_src[None, :, :], axis=-1)
        alpha_dst = np.sum(h_np * a_dst[None, :, :], axis=-1)

        edge_alpha = alpha_src[src] + alpha_dst[dst]
        edge_alpha = np.where(edge_alpha >= 0, edge_alpha, edge_alpha * self.negative_slope)

        exp_alpha = np.exp(np.clip(edge_alpha, -20.0, 20.0))
        sum_exp = np.zeros((num_nodes, self.heads), dtype=np.float32)
        np.add.at(sum_exp, dst, exp_alpha)
        norm_alpha = exp_alpha / np.maximum(sum_exp[dst], 1e-12)

        # 3. Differentiable sparse message aggregation per head in O(E) complexity
        head_outs = []
        pairs = [(int(dst[i]), int(src[i])) for i in range(num_edges)]
        for k in range(self.heads):
            h_k = h.narrow(1, k * self.out_features, self.out_features)
            sp_att = _SparseCoo(pairs, [float(v) for v in norm_alpha[:, k]], (num_nodes, num_nodes))
            head_outs.append(sp_att @ h_k)

        if self.concat:
            out = _Tensor.cat(head_outs, dim=-1) if len(head_outs) > 1 else head_outs[0]
        else:
            out = head_outs[0]
            for ho in head_outs[1:]:
                out = out + ho
            out = out * (1.0 / float(self.heads))

        if self.bias is not None:
            out = out + self.bias
        return out


class GraphSAGEConv(Module):
    """Inductive GraphSAGE Convolution layer (Hamilton et al., 2017).

    Aggregates neighbor features and concatenates with target node representation:
        h_N(v) = AGGREGATE({h_u, forall u in N(v)})
        h_v' = W * [h_v || h_N(v)] + b
    Executed with O(E) sparse complexity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggr: str = "mean",
        bias: bool = True,
    ):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be >= 1")
        if aggr not in ("mean", "add", "sum", "max"):
            raise ValueError(f"Unsupported aggr '{aggr}' (choose 'mean', 'add', 'sum', or 'max')")
        self.in_features = in_features
        self.out_features = out_features
        self.aggr = aggr

        bound = 1.0 / math.sqrt(2 * in_features)
        w_init = np.random.uniform(-bound, bound, (2 * in_features, out_features)).astype(np.float32)
        self.weight = Parameter(_from_numpy(np.ascontiguousarray(w_init)))

        if bias:
            b_init = np.zeros(out_features, dtype=np.float32)
            self.bias = Parameter(_from_numpy(np.ascontiguousarray(b_init)))
        else:
            self.bias = None

    def forward(self, x: _Tensor, edge_index: Any) -> _Tensor:
        if isinstance(edge_index, _Tensor):
            edge_index = edge_index.to_numpy().astype(np.int64)
        else:
            edge_index = np.asarray(edge_index, dtype=np.int64)

        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        num_edges = len(src)

        if self.aggr == "max":
            x_np = x.to_numpy()
            neigh_np = np.full_like(x_np, -np.inf)
            if num_edges > 0:
                np.maximum.at(neigh_np, dst, x_np[src])
            neigh_np = np.where(np.isneginf(neigh_np), 0.0, neigh_np)
            neigh = _from_numpy(np.ascontiguousarray(neigh_np, dtype=np.float32))
        else:
            pairs = [(int(dst[i]), int(src[i])) for i in range(num_edges)]
            if self.aggr == "mean":
                deg = np.zeros(num_nodes, dtype=np.float32)
                np.add.at(deg, dst, 1.0)
                deg = np.maximum(deg, 1.0)
                vals = [float(1.0 / deg[dst[i]]) for i in range(num_edges)]
            else:  # "add" or "sum"
                vals = [1.0] * num_edges
            sp_adj = _SparseCoo(pairs, vals, (num_nodes, num_nodes))
            neigh = sp_adj @ x

        combined_tensor = _Tensor.cat([x, neigh], dim=-1)
        out = combined_tensor @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class ChebConv(Module):
    r"""Chebyshev Spectral Graph Convolutional layer (Defferrard et al., NeurIPS 2016).

    Approximates spectral graph convolutions using truncated Chebyshev polynomials up to order K:
        Z^{(0)} = X
        Z^{(1)} = \tilde{L} X
        Z^{(k)} = 2 \tilde{L} Z^{(k-1)} - Z^{(k-2)}
        out = \sum_{k=0}^{K-1} Z^{(k)} W^{(k)} + b
    where \tilde{L} = \frac{2}{\lambda_{\max}} L - I_N is the scaled and normalized graph Laplacian.
    Executed in O(E) sparse operations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        K: int = 3,
        normalization: Optional[str] = "sym",
        bias: bool = True,
    ):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be >= 1")
        if K < 1:
            raise ValueError("K must be >= 1")
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.normalization = normalization

        bound = 1.0 / math.sqrt(in_features)
        self.weights = []
        for k in range(K):
            w_init = np.random.uniform(-bound, bound, (in_features, out_features)).astype(np.float32)
            p = Parameter(_from_numpy(np.ascontiguousarray(w_init)))
            setattr(self, f"weight_{k}", p)
            self.weights.append(p)

        if bias:
            b_init = np.zeros(out_features, dtype=np.float32)
            self.bias = Parameter(_from_numpy(np.ascontiguousarray(b_init)))
        else:
            self.bias = None

    def forward(
        self,
        x: _Tensor,
        edge_index: Any,
        lambda_max: float = 2.0,
    ) -> _Tensor:
        """Forward pass of ChebConv in O(K * E) sparse complexity."""
        if isinstance(edge_index, _Tensor):
            edge_index = edge_index.to_numpy().astype(np.int64)
        else:
            edge_index = np.asarray(edge_index, dtype=np.int64)

        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        num_edges = len(src)

        # Build normalized adjacency A_norm = D^{-1/2} A D^{-1/2}
        deg = np.zeros(num_nodes, dtype=np.float32)
        np.add.at(deg, dst, 1.0)
        deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)

        edge_weight = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        pairs = [(int(dst[i]), int(src[i])) for i in range(num_edges)]
        sp_adj_norm = _SparseCoo(pairs, [float(v) for v in edge_weight], (num_nodes, num_nodes))

        # \tilde{L} X = (2 / lambda_max) * (X - A_norm X) - X
        #              = (2 / lambda_max - 1) X - (2 / lambda_max) (A_norm X)
        scale1 = float(2.0 / lambda_max - 1.0)
        scale2 = float(2.0 / lambda_max)

        def laplacian_op(tensor_in: _Tensor) -> _Tensor:
            ax = sp_adj_norm @ tensor_in
            return (tensor_in * scale1) - (ax * scale2)

        # Chebyshev recurrence
        # T_0(X) = X
        z_prev_prev = x
        out = z_prev_prev @ self.weights[0]

        if self.K > 1:
            # T_1(X) = \tilde{L} X
            z_prev = laplacian_op(x)
            out = out + (z_prev @ self.weights[1])

            for k in range(2, self.K):
                # T_k(X) = 2 \tilde{L} T_{k-1}(X) - T_{k-2}(X)
                z_cur = (laplacian_op(z_prev) * 2.0) - z_prev_prev
                out = out + (z_cur @ self.weights[k])
                z_prev_prev = z_prev
                z_prev = z_cur

        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Scatter Operations & Geometric Pooling Primitives
# ---------------------------------------------------------------------------

def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def scatter_add(
    src: _Tensor,
    index: Any,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> _Tensor:
    """Reduces all values from the src tensor into output at the indices specified in the index tensor along dim."""
    idx_np = _to_np(index).astype(np.int64)
    if dim_size is None:
        dim_size = int(np.max(idx_np)) + 1 if idx_np.size > 0 else 0

    if isinstance(src, _Tensor) and src.requires_grad and dim == 0 and len(src.shape) == 2:
        num_items = src.shape[0]
        pairs = [(int(idx_np[i]), i) for i in range(num_items)]
        vals = [1.0] * num_items
        sp = _SparseCoo(pairs, vals, (dim_size, num_items))
        return sp @ src

    s_np = _to_np(src).astype(np.float32)
    out_shape = list(s_np.shape)
    out_shape[dim] = dim_size
    out_np = np.zeros(out_shape, dtype=np.float32)

    np.add.at(out_np, idx_np, s_np)
    return _from_numpy(np.ascontiguousarray(out_np, dtype=np.float32))


def scatter_mean(
    src: _Tensor,
    index: Any,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> _Tensor:
    """Computes the mean of all values from src reduced into output at specified indices."""
    idx_np = _to_np(index).astype(np.int64)
    if dim_size is None:
        dim_size = int(np.max(idx_np)) + 1 if idx_np.size > 0 else 0

    if isinstance(src, _Tensor) and src.requires_grad and dim == 0 and len(src.shape) == 2:
        num_items = src.shape[0]
        counts = np.zeros(dim_size, dtype=np.float32)
        np.add.at(counts, idx_np, 1.0)
        counts = np.maximum(counts, 1.0)
        pairs = [(int(idx_np[i]), i) for i in range(num_items)]
        vals = [float(1.0 / counts[idx_np[i]]) for i in range(num_items)]
        sp = _SparseCoo(pairs, vals, (dim_size, num_items))
        return sp @ src

    s_np = _to_np(src).astype(np.float32)
    out_shape = list(s_np.shape)
    out_shape[dim] = dim_size
    out_np = np.zeros(out_shape, dtype=np.float32)
    counts = np.zeros(dim_size, dtype=np.float32)

    np.add.at(out_np, idx_np, s_np)
    np.add.at(counts, idx_np, 1.0)
    counts = np.maximum(counts, 1.0)

    for d in range(len(out_shape)):
        if d != dim:
            counts = np.expand_dims(counts, axis=-1)

    out_np = out_np / counts
    return _from_numpy(np.ascontiguousarray(out_np, dtype=np.float32))


def scatter_max(
    src: _Tensor,
    index: Any,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> Tuple[_Tensor, _Tensor]:
    """Computes the maximum of all values from src reduced into output at specified indices."""
    s_np = _to_np(src).astype(np.float32)
    idx_np = _to_np(index).astype(np.int64)

    if dim_size is None:
        dim_size = int(np.max(idx_np)) + 1 if idx_np.size > 0 else 0

    out_shape = list(s_np.shape)
    out_shape[dim] = dim_size
    out_np = np.full(out_shape, -float("inf"), dtype=np.float32)
    arg_np = np.zeros(out_shape, dtype=np.int64)

    for i, idx in enumerate(idx_np):
        mask = s_np[i] > out_np[idx]
        out_np[idx] = np.where(mask, s_np[i], out_np[idx])
        arg_np[idx] = np.where(mask, i, arg_np[idx])

    out_np = np.where(out_np == -float("inf"), 0.0, out_np)
    return (
        _from_numpy(np.ascontiguousarray(out_np, dtype=np.float32)),
        _from_numpy(np.ascontiguousarray(arg_np, dtype=np.int64)),
    )


def global_add_pool(
    x: _Tensor,
    batch: Optional[Any] = None,
    size: Optional[int] = None,
) -> _Tensor:
    """Returns batch-wise graph-level sum aggregations of node features."""
    if batch is None:
        return x.sum(dim=0, keepdim=True)
    return scatter_add(x, batch, dim=0, dim_size=size)


def global_mean_pool(
    x: _Tensor,
    batch: Optional[Any] = None,
    size: Optional[int] = None,
) -> _Tensor:
    """Returns batch-wise graph-level mean aggregations of node features."""
    if batch is None:
        return x.mean(dim=0, keepdim=True)
    return scatter_mean(x, batch, dim=0, dim_size=size)


def global_max_pool(
    x: _Tensor,
    batch: Optional[Any] = None,
    size: Optional[int] = None,
) -> _Tensor:
    """Returns batch-wise graph-level max aggregations of node features."""
    if batch is None:
        return x.max(dim=0, keepdim=True)[0]
    return scatter_max(x, batch, dim=0, dim_size=size)[0]


# ---------------------------------------------------------------------------
# MessagePassing Base Class & High-Order Geometric Layers
# ---------------------------------------------------------------------------

class MessagePassing(Module):
    """Base class for creating Message Passing Graph Neural Networks (Gilmer et al., 2017)."""

    def __init__(self, aggr: str = "add"):
        super().__init__()
        self.aggr = aggr

    def propagate(self, edge_index: Any, x: _Tensor, **kwargs) -> _Tensor:
        if isinstance(edge_index, _Tensor):
            edge_index = edge_index.to_numpy().astype(np.int64)
        else:
            edge_index = np.asarray(edge_index, dtype=np.int64)

        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]
        num_edges = len(src)

        if isinstance(x, _Tensor) and x.requires_grad and len(x.shape) == 2:
            idx_src = _from_numpy(np.ascontiguousarray(src, dtype=np.int64))
            idx_dst = _from_numpy(np.ascontiguousarray(dst, dtype=np.int64))
            x_j = x.index_select(0, idx_src)
            x_i = x.index_select(0, idx_dst)
        else:
            x_j = _from_numpy(np.ascontiguousarray(x.to_numpy()[src]))
            x_i = _from_numpy(np.ascontiguousarray(x.to_numpy()[dst]))

        # 1. Message
        msg = self.message(x_j=x_j, x_i=x_i, **kwargs)

        # 2. Aggregate
        aggr_out = self.aggregate(msg, dst, dim_size=num_nodes)

        # 3. Update
        return self.update(aggr_out, x=x)

    def message(self, x_j: _Tensor, x_i: Optional[_Tensor] = None, **kwargs) -> _Tensor:
        return x_j

    def aggregate(self, inputs: _Tensor, index: Any, dim_size: Optional[int] = None) -> _Tensor:
        if self.aggr == "add":
            return scatter_add(inputs, index, dim=0, dim_size=dim_size)
        elif self.aggr == "mean":
            return scatter_mean(inputs, index, dim=0, dim_size=dim_size)
        elif self.aggr == "max":
            return scatter_max(inputs, index, dim=0, dim_size=dim_size)[0]
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggr}")

    def update(self, inputs: _Tensor, x: Optional[_Tensor] = None) -> _Tensor:
        return inputs


class GINConv(MessagePassing):
    """Graph Isomorphism Network (GIN) layer (Xu et al., ICLR 2019).

    Maximally expressive Weisfeiler-Lehman 1-WL GNN:
        h_v' = MLP((1 + eps) * h_v + sum_{u in N(v)} h_u)
    """

    def __init__(self, nn: Module, eps: float = 0.0, train_eps: bool = False):
        super().__init__(aggr="add")
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = Parameter(_from_numpy(np.array([eps], dtype=np.float32)))
        else:
            self.eps = eps

    def forward(self, x: _Tensor, edge_index: Any) -> _Tensor:
        out = self.propagate(edge_index, x=x)
        if isinstance(self.eps, _Tensor):
            x_weighted = x * (self.eps + 1.0)
        else:
            x_weighted = x * (1.0 + float(self.eps))
        return self.nn(x_weighted + out)


class EdgeConv(MessagePassing):
    """Dynamic Edge Convolution layer (Wang et al., ACM TOG 2019).

    Extracts edge-feature vectors:
        h_v' = MAX_{u in N(v)} MLP(x_v || (x_u - x_v))
    """

    def __init__(self, nn: Module, aggr: str = "max"):
        super().__init__(aggr=aggr)
        self.nn = nn

    def forward(self, x: _Tensor, edge_index: Any) -> _Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: _Tensor, x_i: _Tensor, **kwargs) -> _Tensor:
        diff = x_j - x_i
        cat_t = _Tensor.cat([x_i, diff], dim=-1)
        return self.nn(cat_t)


__all__ = [
    "GraphData",
    "degree",
    "add_self_loops",
    "remove_self_loops",
    "to_undirected",
    "subgraph",
    "k_hop_subgraph",
    "GCNConv",
    "GATConv",
    "GraphSAGEConv",
    "ChebConv",
    "scatter_add",
    "scatter_mean",
    "scatter_max",
    "global_add_pool",
    "global_mean_pool",
    "global_max_pool",
    "MessagePassing",
    "GINConv",
    "EdgeConv",
]
