"""
Structural Biology, Protein Bioinformatics, and Equivariant Molecular Modeling for Oxideon.
Includes amino acid sequence encoding, pairwise C-alpha residue contact map computation,
SE(3)-invariant Kabsch RMSD alignment, backbone Ramachandran dihedral angles,
and E(n)-Equivariant Graph Neural Network (EGNN) coordinate & node feature updates.
"""
from typing import List, Tuple
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy
from .nn import Module, Linear

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def amino_acid_to_onehot(sequence: str) -> Tensor:
    """Encodes an amino acid sequence string into a (L, 20) one-hot representation."""
    seq_upper = sequence.upper()
    length = len(seq_upper)
    encoding = np.zeros((length, len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(seq_upper):
        idx = AA_TO_IDX.get(aa, None)
        if idx is not None:
            encoding[i, idx] = 1.0
    return _from_numpy(encoding)


def compute_contact_map(ca_coords: Tensor, threshold_angstrom: float = 8.0) -> Tensor:
    """
    Computes binary contact map between C-alpha residues based on Euclidean distance threshold.
    """
    if threshold_angstrom < 0:
        raise ValueError("threshold_angstrom must be non-negative")
    coords = ca_coords.to_numpy()
    if coords.ndim == 1:
        coords = np.expand_dims(coords, 0)
    c_sq = np.sum(coords ** 2, axis=1, keepdims=True)
    dist_sq = np.maximum(0.0, c_sq + c_sq.T - 2.0 * np.dot(coords, coords.T))
    thresh_sq = float(threshold_angstrom ** 2)
    contacts = (dist_sq <= thresh_sq).astype(np.float32)
    return _from_numpy(contacts)


def kabsch_rmsd(coords_p: Tensor, coords_q: Tensor) -> float:
    """
    Computes optimal SE(3)-aligned Root-Mean-Square Deviation (RMSD) between two 3D coordinate
    point clouds `(N, 3)` using the Kabsch algorithm with reflection correction.
    """
    p = np.asarray(coords_p.to_numpy(), dtype=np.float64)
    q = np.asarray(coords_q.to_numpy(), dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"kabsch_rmsd expects matching (N, 3) coordinate tensors, got {p.shape} and {q.shape}")
    n = p.shape[0]
    if n == 0:
        return 0.0

    p_c = p - np.mean(p, axis=0, keepdims=True)
    q_c = q - np.mean(q, axis=0, keepdims=True)
    cov = p_c.T @ q_c
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    diag = np.diag([1.0, 1.0, float(d)])
    rot = vt.T @ diag @ u.T
    p_aligned = p_c @ rot.T
    msd = float(np.mean(np.sum((p_aligned - q_c) ** 2, axis=-1)))
    return float(np.sqrt(max(0.0, msd)))


class EGNNCoordinateUpdate(Module):
    """
    SE(3)/E(3) Equivariant Graph Neural Network (EGNN, Satorras et al., ICML 2021) layer.
    Updates both invariant node features `h_i` and equivariant 3D atomic coordinates `x_i`:
      m_ij = phi_e([h_i, h_j, ||x_i - x_j||^2])
      x_i' = x_i + (1 / (N - 1)) * sum_{j != i} (x_i - x_j) * phi_x(m_ij)
      h_i' = h_i + phi_h([h_i, sum_j m_ij])
    while preserving the Oxideon autograd graph.
    """
    def __init__(self, node_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.phi_e = Linear(2 * node_dim + 1, hidden_dim)
        self.phi_x = Linear(hidden_dim, 1)
        self.phi_h = Linear(node_dim + hidden_dim, node_dim)

    def forward(self, h: Tensor, coords: Tensor) -> Tuple[Tensor, Tensor]:
        n = coords.shape[0]
        if h.shape[0] != n or h.shape[1] != self.node_dim or coords.shape[1] != 3:
            raise ValueError(f"EGNN expects h (N, {self.node_dim}) and coords (N, 3)")
        if n <= 1:
            return h, coords

        # Pairwise coordinate differences: diff[i, j] = x_i - x_j -> [N, N, 3]
        x_i = coords.unsqueeze(1).expand([n, n, 3])
        x_j = coords.unsqueeze(0).expand([n, n, 3])
        diff = x_i - x_j
        dist_sq = (diff * diff).sum(dim=-1, keepdim=True)  # [N, N, 1]

        h_i = h.unsqueeze(1).expand([n, n, self.node_dim])
        h_j = h.unsqueeze(0).expand([n, n, self.node_dim])
        edge_in = Tensor.cat([h_i, h_j, dist_sq], dim=-1).reshape([n * n, 2 * self.node_dim + 1])

        m_ij = self.phi_e(edge_in).relu()  # [N*N, hidden_dim]
        trans_w = self.phi_x(m_ij).reshape([n, n, 1])

        # Mask out self-loops (i == j)
        off_diag_mask = _from_numpy((1.0 - np.eye(n, dtype=np.float32)).reshape(n, n, 1))
        trans_w_masked = trans_w * off_diag_mask
        coord_delta = (diff * trans_w_masked).sum(dim=1) * (1.0 / float(max(n - 1, 1)))
        coords_new = coords + coord_delta

        m_agg = (m_ij.reshape([n, n, self.hidden_dim]) * off_diag_mask).sum(dim=1)
        h_new = h + self.phi_h(Tensor.cat([h, m_agg], dim=-1))
        return h_new, coords_new
