"""Two-view geometry (essential matrix, triangulation, RANSAC) for Vision."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ...oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def _check_pts(p: np.ndarray, name: str, lo: int = 1) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"{name} must be [N, 2], got shape {p.shape}")
    if len(p) < lo:
        raise ValueError(f"{name} needs at least {lo} points, got {len(p)}")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} must be finite")
    return p


def rodrigues(rotvec: Any) -> _Tensor:
    """Axis-angle vector [3] to rotation matrix [3, 3] (Rodrigues' formula)."""
    v = _to_np(rotvec).astype(np.float64).reshape(-1)
    if v.shape != (3,):
        raise ValueError(f"rotvec must have 3 elements, got shape {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("rotvec must be finite")
    theta = float(np.linalg.norm(v))
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if theta < 1e-12:
        r = np.eye(3) + k
    else:
        k = k / theta
        r = np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)
    return _from_numpy(np.ascontiguousarray(r.astype(np.float32)))


def rodrigues_inv(rotmat: Any) -> _Tensor:
    """Rotation matrix [3, 3] to axis-angle vector [3]."""
    r = _to_np(rotmat).astype(np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"rotmat must be [3, 3], got shape {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("rotmat must be finite")
    if np.linalg.norm(r.T @ r - np.eye(3)) > 1e-4 or abs(np.linalg.det(r) - 1.0) > 1e-4:
        raise ValueError("rotmat must be a proper rotation matrix")
    theta = float(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))
    if theta < 1e-12:
        return _from_numpy(np.zeros(3, dtype=np.float32))
    if abs(theta - np.pi) < 1e-6:
        # OpenCV-style pi axis: signs from off-diagonals (axis sign is free).
        xx = (r[0, 0] + 1.0) / 2.0
        yy = (r[1, 1] + 1.0) / 2.0
        zz = (r[2, 2] + 1.0) / 2.0
        xy = (r[0, 1] + r[1, 0]) / 4.0
        xz = (r[0, 2] + r[2, 0]) / 4.0
        yz = (r[1, 2] + r[2, 1]) / 4.0
        if xx > yy and xx > zz:
            x = np.sqrt(max(xx, 0.0))
            axis = np.array([x, xy / x, xz / x])
        elif yy > zz:
            y = np.sqrt(max(yy, 0.0))
            axis = np.array([xy / y, y, yz / y])
        else:
            z = np.sqrt(max(zz, 0.0))
            axis = np.array([xz / z, yz / z, z])
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return _from_numpy(np.ascontiguousarray((axis * theta).astype(np.float32)))
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    axis = axis / (2.0 * np.sin(theta))
    return _from_numpy(np.ascontiguousarray((axis * theta).astype(np.float32)))


def normalize_by_K(pts: Any, K: Any) -> _Tensor:
    """Undistorted pixels [N, 2] to normalized coords via K^-1 (append 1)."""
    p = _check_pts(_to_np(pts), "pts")
    k = _to_np(K).astype(np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"K must be [3, 3], got shape {k.shape}")
    if not np.all(np.isfinite(k)):
        raise ValueError("K must be finite")
    if abs(np.linalg.det(k)) < 1e-12:
        raise ValueError("K must be invertible")
    homo = np.concatenate([p, np.ones((len(p), 1))], axis=1)
    norm = (np.linalg.inv(k) @ homo.T).T
    return _from_numpy(np.ascontiguousarray(norm[:, :2].astype(np.float32)))


def project_points(
    points: Any, K: Any, R: Optional[Any] = None, t: Optional[Any] = None
) -> Tuple[_Tensor, _Tensor]:
    """Projects world points [N, 3] to pixels (uv [N, 2], depth [N]).

    Camera model: x_cam = R @ x + t, uv ~ K @ x_cam. Points behind the
    camera yield negative depth and are NOT filtered.
    """
    x = _to_np(points).astype(np.float64)
    k = _to_np(K).astype(np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"points must be [N, 3], got shape {x.shape}")
    if len(x) == 0:
        raise ValueError("points must be non-empty")
    if k.shape != (3, 3) or not np.all(np.isfinite(k)):
        raise ValueError("K must be finite [3, 3]")
    if R is None:
        r = np.eye(3)
    else:
        r = _to_np(R).astype(np.float64)
        if r.shape != (3, 3) or not np.all(np.isfinite(r)):
            raise ValueError("R must be finite [3, 3]")
    if t is None:
        tv = np.zeros(3)
    else:
        tv = _to_np(t).astype(np.float64).reshape(-1)
        if tv.shape != (3,) or not np.all(np.isfinite(tv)):
            raise ValueError("t must be 3 finite elements")
    if not np.all(np.isfinite(x)):
        raise ValueError("points must be finite")
    cam = x @ r.T + tv[None, :]
    depth = cam[:, 2]
    z = np.where(depth == 0.0, 1e-12, depth)
    pix = (k @ (cam / z[:, None]).T).T[:, :2]
    return (
        _from_numpy(np.ascontiguousarray(pix.astype(np.float32))),
        _from_numpy(np.ascontiguousarray(depth.astype(np.float32))),
    )


def normalize_points_2d(pts: Any) -> Tuple[_Tensor, _Tensor]:
    """Hartley normalization: centroid at origin, mean radius sqrt(2).

    Returns (normalized [N, 2], T [3, 3]) with x_hat = T @ [x, y, 1].
    """
    p = _check_pts(_to_np(pts), "pts")
    centroid = p.mean(axis=0)
    dist = np.linalg.norm(p - centroid[None, :], axis=1).mean()
    if dist < 1e-12:
        raise ValueError("points are coincident; cannot normalize")
    s = np.sqrt(2.0) / dist
    mat = np.array([[s, 0, -s * centroid[0]], [0, s, -s * centroid[1]], [0, 0, 1.0]])
    homo = np.concatenate([p, np.ones((len(p), 1))], axis=1)
    return (
        _from_numpy(np.ascontiguousarray(((mat @ homo.T).T[:, :2]).astype(np.float32))),
        _from_numpy(np.ascontiguousarray(mat.astype(np.float32))),
    )


def essential_8point(
    pts1: Any,
    pts2: Any,
    K1: Optional[Any] = None,
    K2: Optional[Any] = None,
) -> _Tensor:
    """Normalized 8-point essential matrix from correspondences.

    Args:
        pts1 / pts2: [N, 2] normalized coords (or pixels when K1/K2 given).
        K1 / K2: Optional [3, 3] intrinsics to normalize pixel inputs.

    Returns:
        E [3, 3] with rank 2 and unit Frobenius norm (sign ambiguous).
    """
    p1 = _check_pts(_to_np(pts1), "pts1", 8)
    p2 = _check_pts(_to_np(pts2), "pts2", 8)
    if len(p1) != len(p2):
        raise ValueError("pts1/pts2 must have equal length")
    if K1 is not None:
        p1 = normalize_by_K(p1, K1).to_numpy().astype(np.float64)
    if K2 is not None:
        p2 = normalize_by_K(p2, K2).to_numpy().astype(np.float64)
    q1, t1 = (v.to_numpy().astype(np.float64) for v in normalize_points_2d(p1))
    q2, t2 = (v.to_numpy().astype(np.float64) for v in normalize_points_2d(p2))
    a = np.stack([
        q2[:, 0] * q1[:, 0], q2[:, 0] * q1[:, 1], q2[:, 0],
        q2[:, 1] * q1[:, 0], q2[:, 1] * q1[:, 1], q2[:, 1],
        q1[:, 0], q1[:, 1], np.ones(len(q1)),
    ], axis=1)
    _, _, vt = np.linalg.svd(a)
    e = vt[-1].reshape(3, 3)
    e = t2.T @ e @ t1
    u, s, vt = np.linalg.svd(e)
    e = u @ np.diag([s[0], s[1], 0.0]) @ vt
    n = np.linalg.norm(e)
    if n < 1e-12:
        raise ValueError("degenerate configuration (zero essential matrix)")
    return _from_numpy(np.ascontiguousarray((e / n).astype(np.float32)))


def essential_twisted_pair(E: Any) -> List[Tuple[_Tensor, _Tensor]]:
    """Four (R, t) factorizations of an essential matrix (Horn)."""
    e = _to_np(E).astype(np.float64)
    if e.shape != (3, 3) or not np.all(np.isfinite(e)):
        raise ValueError("E must be finite [3, 3]")
    u, _, vt = np.linalg.svd(e)
    if np.linalg.det(u) < 0:
        u = -u
    if np.linalg.det(vt) < 0:
        vt = -vt
    w = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    r1, r2 = u @ w @ vt, u @ w.T @ vt
    if np.linalg.det(r1) < 0:
        r1 = -r1
    if np.linalg.det(r2) < 0:
        r2 = -r2
    t = u[:, 2]
    return [
        (
            _from_numpy(np.ascontiguousarray(r.astype(np.float32))),
            _from_numpy(np.ascontiguousarray((s * t).astype(np.float32))),
        )
        for r in (r1, r2)
        for s in (1.0, -1.0)
    ]


def triangulate_dlt(pts1: Any, pts2: Any, P1: Any, P2: Any) -> _Tensor:
    """DLT triangulation of [N, 2] correspondences given P [3, 4] matrices."""
    p1 = _check_pts(_to_np(pts1), "pts1")
    p2 = _check_pts(_to_np(pts2), "pts2")
    if len(p1) != len(p2):
        raise ValueError("pts1/pts2 must have equal length")
    m1 = _to_np(P1).astype(np.float64)
    m2 = _to_np(P2).astype(np.float64)
    if m1.shape != (3, 4) or m2.shape != (3, 4):
        raise ValueError("projection matrices must be [3, 4]")
    if not np.all(np.isfinite(m1)) or not np.all(np.isfinite(m2)):
        raise ValueError("projection matrices must be finite")
    out = np.empty((len(p1), 3))
    for i in range(len(p1)):
        a = np.stack([
            p1[i, 1] * m1[2] - m1[1],
            m1[0] - p1[i, 0] * m1[2],
            p2[i, 1] * m2[2] - m2[1],
            m2[0] - p2[i, 0] * m2[2],
        ])
        _, _, vt = np.linalg.svd(a)
        x = vt[-1]
        if abs(x[3]) < 1e-12:
            raise ValueError(f"degenerate triangulation at correspondence {i}")
        out[i] = x[:3] / x[3]
    return _from_numpy(np.ascontiguousarray(out.astype(np.float32)))


def cheirality_count(R: Any, t: Any, pts1: Any, pts2: Any) -> Tuple[int, _Tensor]:
    """Triangulates with P1 = [I|0], P2 = [R|t]; counts points in front of both."""
    r = _to_np(R).astype(np.float64)
    tv = _to_np(t).astype(np.float64).reshape(-1)
    if r.shape != (3, 3) or tv.shape != (3,):
        raise ValueError("R must be [3, 3] and t length 3")
    p1 = _check_pts(_to_np(pts1), "pts1")
    p2 = _check_pts(_to_np(pts2), "pts2")
    if len(p1) != len(p2):
        raise ValueError("pts1/pts2 must have equal length")
    m1 = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    m2 = np.concatenate([r, tv[:, None]], axis=1)
    x = triangulate_dlt(p1, p2, m1, m2).to_numpy().astype(np.float64)
    d1 = x[:, 2]
    d2 = (x @ r.T + tv[None, :])[:, 2]
    return int(np.sum((d1 > 0) & (d2 > 0))), _from_numpy(np.ascontiguousarray(x.astype(np.float32)))


def select_essential_pose(E: Any, pts1: Any, pts2: Any) -> Dict[str, Any]:
    """Cheirality-based pose recovery: returns R, t, in_front count, points."""
    best = None
    for r, t in essential_twisted_pair(E):
        count, x = cheirality_count(r, t, pts1, pts2)
        if best is None or count > best[0]:
            best = (count, r, t, x)
    assert best is not None
    return {"R": best[1], "t": best[2], "in_front": best[0], "points": best[3]}


def sampson_error(pts1: Any, pts2: Any, E: Any) -> _Tensor:
    """Sampson (first-order geometric) error per correspondence."""
    p1 = _check_pts(_to_np(pts1), "pts1")
    p2 = _check_pts(_to_np(pts2), "pts2")
    if len(p1) != len(p2):
        raise ValueError("pts1/pts2 must have equal length")
    e = _to_np(E).astype(np.float64)
    if e.shape != (3, 3) or not np.all(np.isfinite(e)):
        raise ValueError("E must be finite [3, 3]")
    x1 = np.concatenate([p1, np.ones((len(p1), 1))], axis=1)
    x2 = np.concatenate([p2, np.ones((len(p2), 1))], axis=1)
    ex1 = (e @ x1.T).T
    etx2 = (e.T @ x2.T).T
    num = np.sum(x2 * ex1, axis=1) ** 2
    denom = ex1[:, 0] ** 2 + ex1[:, 1] ** 2 + etx2[:, 0] ** 2 + etx2[:, 1] ** 2
    err = num / np.maximum(denom, 1e-24)
    return _from_numpy(np.ascontiguousarray(err.astype(np.float32)))


def ransac_fit(
    n_points: int,
    sample_size: int,
    fit_fn: Callable[[np.ndarray], Any],
    err_fn: Callable[[Any], np.ndarray],
    thresh: float,
    max_iters: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[Any, np.ndarray]:
    """Generic RANSAC loop with deterministic seeding.

    Args:
        n_points: Population size N. sample_size: Minimal sample S (<= N).
        fit_fn: Maps S sampled indices to a model (may raise on degeneracy;
            degenerate samples are skipped).
        err_fn: Maps a model to [N] non-negative errors.
        thresh: Inlier threshold (> 0). max_iters: RANSAC rounds (>= 1).

    Returns:
        (best model refit on... NOTE: returned as fit on its inliers when the
        inlier set admits it, else the raw best sample model; inlier bool mask).
        No model found (all samples degenerate) raises ValueError.
    """
    if int(n_points) < 1 or int(sample_size) < 1 or int(sample_size) > int(n_points):
        raise ValueError("need 1 <= sample_size <= n_points")
    if not np.isfinite(thresh) or thresh <= 0:
        raise ValueError("thresh must be finite and positive")
    if int(max_iters) < 1:
        raise ValueError("max_iters must be >= 1")
    n, s = int(n_points), int(sample_size)
    rng = np.random.default_rng(seed)
    best_model, best_mask, best_n = None, np.zeros(n, dtype=bool), 0
    for _ in range(int(max_iters)):
        idx = rng.choice(n, size=s, replace=False)
        try:
            model = fit_fn(idx)
        except ValueError:
            continue
        err = np.asarray(err_fn(model), dtype=np.float64).reshape(-1)
        if err.shape != (n,) or not np.all(np.isfinite(err)) or np.any(err < 0):
            raise ValueError("err_fn must return [N] finite non-negative errors")
        mask = err <= float(thresh)
        count = int(mask.sum())
        if count > best_n:
            try:
                refit = fit_fn(np.flatnonzero(mask))
                rerr = np.asarray(err_fn(refit), dtype=np.float64).reshape(-1)
                if rerr.shape == (n,) and np.all(np.isfinite(rerr)):
                    rmask = rerr <= float(thresh)
                    if int(rmask.sum()) >= count:
                        best_model, best_mask, best_n = refit, rmask, int(rmask.sum())
                        if best_n == n:
                            break
                        continue
                best_model, best_mask, best_n = refit, mask, count
            except ValueError:
                best_model, best_mask, best_n = model, mask, count
        if best_n == n:
            break
    if best_model is None:
        raise ValueError("RANSAC found no non-degenerate sample")
    return best_model, best_mask


def ransac_essential(
    pts1: Any,
    pts2: Any,
    thresh: float = 1e-3,
    max_iters: int = 500,
    seed: Optional[int] = None,
) -> Tuple[_Tensor, np.ndarray]:
    """Robust essential matrix via RANSAC (normalized coords, Sampson error).

    The threshold is a Sampson (squared, first-order) distance in the input
    coordinate units; use ~1e-3 for normalized coords, ~1-4 for pixels.
    """
    p1 = _check_pts(_to_np(pts1), "pts1", 8)
    p2 = _check_pts(_to_np(pts2), "pts2", 8)
    if len(p1) != len(p2):
        raise ValueError("pts1/pts2 must have equal length")

    def _fit(idx: np.ndarray) -> _Tensor:
        return essential_8point(p1[idx], p2[idx])

    def _err(model: _Tensor) -> np.ndarray:
        return sampson_error(p1, p2, model).to_numpy().astype(np.float64)

    return ransac_fit(len(p1), 8, _fit, _err, thresh, max_iters, seed)


def depth_metrics(pred: Any, gt: Any, valid: Optional[Any] = None) -> Dict[str, float]:
    """Eigen et al. depth metrics over valid pixels (rmse, abs_rel, delta<i>).

    Args:
        pred / gt: [H, W] depths. Valid pixels: gt > 0 (and `valid` mask).
    """
    p = _to_np(pred).astype(np.float64)
    g = _to_np(gt).astype(np.float64)
    if p.ndim != 2 or g.shape != p.shape:
        raise ValueError("pred/gt must be matching [H, W]")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(g)):
        raise ValueError("pred/gt must be finite")
    mask = g > 0
    if valid is not None:
        v = np.asarray(_to_np(valid)).astype(bool)
        if v.shape != p.shape:
            raise ValueError("valid mask must match image shape")
        mask = mask & v
    if not np.any(mask):
        raise ValueError("no valid pixels (need gt > 0)")
    pp, gg = p[mask], g[mask]
    if np.any(pp <= 0):
        raise ValueError("pred must be positive on valid pixels")
    ratio = np.maximum(pp / gg, gg / pp)
    return {
        "rmse": float(np.sqrt(np.mean((pp - gg) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(pp) - np.log(gg)) ** 2))),
        "abs_rel": float(np.mean(np.abs(pp - gg) / gg)),
        "sq_rel": float(np.mean(((pp - gg) ** 2) / gg)),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }


__all__ = [
    "rodrigues",
    "rodrigues_inv",
    "normalize_by_K",
    "project_points",
    "normalize_points_2d",
    "essential_8point",
    "essential_twisted_pair",
    "triangulate_dlt",
    "cheirality_count",
    "select_essential_pose",
    "sampson_error",
    "ransac_fit",
    "ransac_essential",
    "depth_metrics",
]
