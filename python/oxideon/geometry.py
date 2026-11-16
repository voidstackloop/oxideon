"""
3D Point Cloud Processing and Geometric Deep Learning for Oxideon.
Includes PointNet feature encoder, Farthest Point Sampling (FPS), Chamfer Distance, and 3D geometric transformations.
Also includes registration and analysis tools: Umeyama alignment, point-to-point ICP,
voxel downsampling, PCA normal estimation, Earth Mover's distance, Hausdorff distance.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy
from .nn import Module, Linear


class PointNetEncoder(Module):
    """
    PointNet backbone for 3D point cloud feature extraction (Qi et al., CVPR 2017).
    Extracts permutation-invariant global feature representations via symmetric max-pooling.
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, global_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.global_dim = global_dim

        self.conv1 = Linear(in_channels, hidden_dim)
        self.conv2 = Linear(hidden_dim, hidden_dim)
        self.conv3 = Linear(hidden_dim, global_dim)

    def forward(self, points: Tensor) -> Tensor:
        """
        Args:
            points: Tensor of shape (B, N, C) or (N, C)
        Returns:
            global_feat: Tensor of shape (B, global_dim) or (global_dim,)
        """
        orig_shape = list(points.shape)
        single = False
        if len(orig_shape) == 2:
            pts_3d = points.unsqueeze(0)
            single = True
        elif len(orig_shape) == 3:
            pts_3d = points
        else:
            raise ValueError(f"PointNetEncoder expects 2D or 3D points, got shape {orig_shape}")

        B, N, C = pts_3d.shape
        if N == 0:
            res = np.zeros((B, self.global_dim), dtype=np.float32)
            if single:
                return _from_numpy(res.squeeze(0))
            return _from_numpy(res)

        h1 = self.conv1(pts_3d).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()  # (B, N, global_dim)

        # Differentiable symmetric max pooling over point dimension (axis=1)
        h3_np = h3.to_numpy()
        max_idx = np.argmax(h3_np, axis=1, keepdims=True)
        mask = (np.arange(N)[None, :, None] == max_idx).astype(np.float32)
        mask /= np.maximum(np.sum(mask, axis=1, keepdims=True), 1.0)
        global_features = (h3 * _from_numpy(mask)).sum(dim=1)  # (B, global_dim)

        if single:
            return global_features.squeeze(0)
        return global_features


def farthest_point_sampling(points: Tensor, num_samples: int) -> Tensor:
    """
    Farthest Point Sampling (FPS) algorithm for uniform point cloud subsampling.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    pts = points.to_numpy()
    if pts.ndim != 2:
        raise ValueError(f"points must be 2D [N, D], got shape {pts.shape}")
    N, D = pts.shape
    if N == 0:
        raise ValueError("points cannot be empty")
    if num_samples >= N:
        return _from_numpy(np.arange(N, dtype=np.int64))

    selected = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(N, np.inf, dtype=np.float32)

    selected[0] = 0
    curr_pt = pts[0]

    for i in range(1, num_samples):
        dist_to_curr = np.sum((pts - curr_pt) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_curr)

        next_idx = np.argmax(distances)
        selected[i] = next_idx
        curr_pt = pts[next_idx]

    return _from_numpy(selected)


def chamfer_distance(p1: Tensor, p2: Tensor) -> Tensor:
    """
    Computes differentiable symmetric squared Chamfer Distance between two point clouds P1 (N, D) and P2 (M, D).
    CD(P1, P2) = (1/N) * sum_x min_y ||x - y||^2 + (1/M) * sum_y min_x ||y - x||^2
    """
    p1_np = p1.to_numpy()
    p2_np = p2.to_numpy()
    if p1_np.ndim != 2 or p2_np.ndim != 2:
        raise ValueError(f"Point clouds must be 2D, got {p1_np.shape} and {p2_np.shape}")
    if p1_np.shape[1] != p2_np.shape[1]:
        raise ValueError(f"Feature dimension mismatch: {p1_np.shape[1]} vs {p2_np.shape[1]}")
    if len(p1_np) == 0 or len(p2_np) == 0:
        raise ValueError("Point clouds must contain at least 1 point")

    N = len(p1_np)
    M = len(p2_np)
    p1_sq = np.sum(p1_np ** 2, axis=1, keepdims=True)
    p2_sq = np.sum(p2_np ** 2, axis=1, keepdims=True).T
    dist_sq_np = np.maximum(0.0, p1_sq + p2_sq - 2.0 * np.dot(p1_np, p2_np.T))

    # Nearest neighbors via permutation/selection matrices for exact autograd
    idx_1to2 = np.argmin(dist_sq_np, axis=1)  # (N,) in 0..M-1
    idx_2to1 = np.argmin(dist_sq_np, axis=0)  # (M,) in 0..N-1

    sel_1to2 = np.zeros((N, M), dtype=np.float32)
    sel_1to2[np.arange(N), idx_1to2] = 1.0
    p2_matched = _from_numpy(sel_1to2).matmul(p2)  # (N, D)
    diff1 = p1 - p2_matched
    term1 = (diff1 * diff1).sum(dim=1).mean()

    sel_2to1 = np.zeros((M, N), dtype=np.float32)
    sel_2to1[np.arange(M), idx_2to1] = 1.0
    p1_matched = _from_numpy(sel_2to1).matmul(p1)  # (M, D)
    diff2 = p2 - p1_matched
    term2 = (diff2 * diff2).sum(dim=1).mean()

    return term1 + term2


def random_rotation_3d(points: Tensor, seed: Optional[int] = None) -> Tensor:
    """Applies a random 3D SO(3) rotation to a point cloud (N, 3)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=3)
    q = np.array([
        np.sqrt(1.0 - u[0]) * np.sin(2.0 * np.pi * u[1]),
        np.sqrt(1.0 - u[0]) * np.cos(2.0 * np.pi * u[1]),
        np.sqrt(u[0]) * np.sin(2.0 * np.pi * u[2]),
        np.sqrt(u[0]) * np.cos(2.0 * np.pi * u[2])
    ])
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    R = np.array([
        [1.0 - 2.0*(qy**2 + qz**2), 2.0*(qx*qy - qz*qw), 2.0*(qx*qz + qy*qw)],
        [2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx**2 + qz**2), 2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw), 2.0*(qy*qz + qx*qw), 1.0 - 2.0*(qx**2 + qy**2)]
    ], dtype=np.float32)

    return points.matmul(_from_numpy(R.T))


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def _check_cloud(p: np.ndarray, name: str) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"{name} must be [N, 3], got shape {p.shape}")
    if len(p) == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} must be finite")
    return p


def umeyama(source: Any, target: Any, with_scale: bool = True) -> Tensor:
    """Least-squares rigid/similarity alignment of corresponded clouds (Umeyama)."""
    s = _check_cloud(_to_np(source), "source")
    t = _check_cloud(_to_np(target), "target")
    if s.shape != t.shape:
        raise ValueError(f"source/target shapes disagree: {s.shape} vs {t.shape}")
    if len(s) < 3:
        raise ValueError("need at least 3 correspondences")
    mu_s, mu_t = s.mean(axis=0), t.mean(axis=0)
    x, y = s - mu_s, t - mu_t
    var = float(np.mean(np.sum(x**2, axis=1)))
    if var < 1e-12:
        raise ValueError("source points are coincident")
    u, d, vt = np.linalg.svd((x.T @ y) / len(s))
    sm = np.eye(3)
    sm[2, 2] = float(np.linalg.det(vt.T @ u.T))
    r = vt.T @ sm @ u.T
    c = float(np.sum(d * np.diag(sm)) / var) if with_scale else 1.0
    mat = np.eye(4)
    mat[:3, :3] = c * r
    mat[:3, 3] = mu_t - c * (r @ mu_s)
    return _from_numpy(np.ascontiguousarray(mat.astype(np.float32)))


def _apply_transform(pts: np.ndarray, mat: np.ndarray) -> np.ndarray:
    return pts @ mat[:3, :3].T + mat[:3, 3][None, :]


def _nn_indices(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    d2 = np.maximum(
        0.0,
        np.sum(src**2, axis=1, keepdims=True)
        + np.sum(dst**2, axis=1, keepdims=True).T
        - 2.0 * src @ dst.T,
    )
    return np.argmin(d2, axis=1)


def icp(
    source: Any,
    target: Any,
    max_iters: int = 50,
    tol: float = 1e-6,
    init: Optional[Any] = None,
    with_scale: bool = False,
) -> Dict[str, Any]:
    """Point-to-point ICP registration (brute-force NN, Umeyama step)."""
    s = _check_cloud(_to_np(source), "source")
    t = _check_cloud(_to_np(target), "target")
    if int(max_iters) < 1:
        raise ValueError("max_iters must be >= 1")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and >= 0")
    if init is None:
        mat = np.eye(4)
    else:
        mat = _to_np(init).astype(np.float64)
        if mat.shape != (4, 4) or not np.all(np.isfinite(mat)):
            raise ValueError("init must be finite [4, 4]")
        mat = mat.copy()
    prev, converged, it_done = np.inf, False, 0
    for it in range(int(max_iters)):
        s_tr = _apply_transform(s, mat)
        idx = _nn_indices(s_tr, t)
        delta = umeyama(s_tr, t[idx], with_scale).to_numpy().astype(np.float64)
        mat = delta @ mat
        resid = _apply_transform(s, mat) - t[_nn_indices(_apply_transform(s, mat), t)]
        rmse = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
        it_done = it + 1
        if abs(prev - rmse) < float(tol):
            converged = True
            break
        prev = rmse
    s_fin = _apply_transform(s, mat)
    fin_idx = _nn_indices(s_fin, t)
    fin_rmse = float(np.sqrt(np.mean(np.sum((s_fin - t[fin_idx]) ** 2, axis=1))))
    return {
        "transform": _from_numpy(np.ascontiguousarray(mat.astype(np.float32))),
        "rmse": fin_rmse,
        "iterations": it_done,
        "converged": converged,
    }


def voxel_downsample(points: Any, voxel_size: float) -> Tensor:
    """Centroid per occupied voxel; rows ordered by sorted voxel index."""
    p = _check_cloud(_to_np(points), "points")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")
    idx = np.floor(p / float(voxel_size)).astype(np.int64)
    _, inv = np.unique(idx, axis=0, return_inverse=True)
    centroids = np.stack([p[inv == k].mean(axis=0) for k in range(int(inv.max()) + 1)])
    return _from_numpy(np.ascontiguousarray(centroids.astype(np.float32)))


def estimate_normals(
    points: Any, k: int = 10, viewpoint: Optional[Any] = None
) -> Tensor:
    """PCA normals from k nearest neighbors (brute force, small clouds)."""
    p = _check_cloud(_to_np(points), "points")
    if int(k) < 3 or int(k) >= len(p):
        raise ValueError(f"need 3 <= k < N ({len(p)}), got {k}")
    if viewpoint is None:
        vp = None
    else:
        vp = _to_np(viewpoint).astype(np.float64).reshape(-1)
        if vp.shape != (3,) or not np.all(np.isfinite(vp)):
            raise ValueError("viewpoint must be 3 finite elements")
    k = int(k)
    d2 = np.sum(p**2, axis=1, keepdims=True) + np.sum(p**2, axis=1, keepdims=True).T - 2.0 * p @ p.T
    nbr = np.argpartition(np.maximum(d2, 0.0), kth=k, axis=1)[:, 1 : k + 1]
    normals = np.empty_like(p)
    for i in range(len(p)):
        q = p[nbr[i]] - p[nbr[i]].mean(axis=0)
        _, _, vt = np.linalg.svd(q, full_matrices=False)
        n = vt[-1] / max(np.linalg.norm(vt[-1]), 1e-12)
        ref = np.array([0.0, 0.0, 1.0]) if vp is None else vp - p[i]
        if float(n @ ref) < 0:
            n = -n
        normals[i] = n
    return _from_numpy(np.ascontiguousarray(normals.astype(np.float32)))


def earth_mover_distance(p1: Any, p2: Any) -> float:
    """EMD between equal-size clouds = min-mean pairing distance (Hungarian)."""
    a = _check_cloud(_to_np(p1), "p1")
    b = _check_cloud(_to_np(p2), "p2")
    if a.shape != b.shape:
        raise ValueError(f"clouds must share shape for EMD: {a.shape} vs {b.shape}")
    from .vision.ops.tracking import linear_sum_assignment as _lsa

    d = np.sqrt(np.maximum(
        0.0,
        np.sum(a**2, axis=1, keepdims=True)
        + np.sum(b**2, axis=1, keepdims=True).T
        - 2.0 * a @ b.T,
    ))
    rows, cols = (v.to_numpy() for v in _lsa(d))
    return float(d[rows, cols].mean())


def hausdorff_distance(p1: Any, p2: Any) -> float:
    """Symmetric Hausdorff distance (max of directed max-min distances)."""
    a = _check_cloud(_to_np(p1), "p1")
    b = _check_cloud(_to_np(p2), "p2")
    d = np.sqrt(np.maximum(
        0.0,
        np.sum(a**2, axis=1, keepdims=True)
        + np.sum(b**2, axis=1, keepdims=True).T
        - 2.0 * a @ b.T,
    ))
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))
