"""3D Gaussian splatting forward renderer (Kerbl et al., SIGGRAPH 2023).

CPU reference implementation of EWA splat projection and differentiable front-to-back
rasterization. Uses direct RGB colors (spherical-harmonics degree 0 only).

Conventions: world-to-camera ``view`` matrices map to camera space where the
camera looks down -z; ``depth = -z_cam``. Gaussians with depth below
``near`` are clipped.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def quat_to_matrix(quats: Any) -> _Tensor:
    """Unit-quaternion (w, x, y, z) batches to rotation matrices [N, 3, 3]."""
    q = _to_np(quats).astype(np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quats must be [N, 4], got shape {q.shape}")
    if len(q) == 0:
        raise ValueError("quats must be non-empty")
    if not np.all(np.isfinite(q)):
        raise ValueError("quats must be finite")
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("quats must have non-zero norm")
    w, x, y, z = (q / norms[:, None]).T
    m = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)
    return _from_numpy(np.ascontiguousarray(m.astype(np.float32)))


def project_gaussians(
    means: Any,
    scales: Any,
    quats: Any,
    view: Any,
    focal_x: float,
    focal_y: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near: float = 0.2,
    antialias: float = 0.3,
) -> Dict[str, _Tensor]:
    """Projects 3D Gaussians to 2D splats (EWA covariance projection)."""
    mu = _to_np(means).astype(np.float64)
    sc = _to_np(scales).astype(np.float64)
    v = _to_np(view).astype(np.float64)
    if mu.ndim != 2 or mu.shape[1] != 3:
        raise ValueError(f"means must be [N, 3], got shape {mu.shape}")
    if len(mu) == 0:
        raise ValueError("scene must hold at least one Gaussian")
    if sc.shape != mu.shape:
        raise ValueError(f"scales shape {sc.shape} must match means {mu.shape}")
    if v.shape != (4, 4):
        raise ValueError(f"view must be [4, 4], got shape {v.shape}")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(v)):
        raise ValueError("means and view must be finite")
    if not np.all(np.isfinite(sc)) or np.any(sc <= 0):
        raise ValueError("scales must be finite and positive")
    for name, val in (("focal_x", focal_x), ("focal_y", focal_y)):
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.all(np.isfinite([cx, cy])):
        raise ValueError("cx/cy must be finite")
    if int(width) < 1 or int(height) < 1:
        raise ValueError("width/height must be >= 1")
    if not np.isfinite(near) or near <= 0:
        raise ValueError("near must be finite and positive")
    if not np.isfinite(antialias) or antialias < 0:
        raise ValueError("antialias must be finite and >= 0")
    rot_mats = quat_to_matrix(quats).to_numpy().astype(np.float64)
    if rot_mats.shape[0] != len(mu):
        raise ValueError("quats batch must match means")

    n = len(mu)
    homo = np.concatenate([mu, np.ones((n, 1))], axis=1)
    cam = (v @ homo.T).T[:, :3]
    depth = -cam[:, 2]
    visible = depth >= float(near)
    z = np.maximum(depth, 1e-12)
    uv = np.stack([focal_x * cam[:, 0] / z + cx, focal_y * cam[:, 1] / z + cy], axis=1)

    mats = rot_mats * sc[:, None, :]
    cov3d = mats @ np.transpose(mats, (0, 2, 1))
    w_mat = v[:3, :3]
    cov_cam = w_mat[None, :, :] @ cov3d @ w_mat.T[None, :, :]
    x, y = cam[:, 0], cam[:, 1]
    j00, j11 = focal_x / z, focal_y / z
    j02, j12 = -focal_x * x / z**2, -focal_y * y / z**2
    jac = np.zeros((n, 2, 3))
    jac[:, 0, 0], jac[:, 0, 2] = j00, j02
    jac[:, 1, 1], jac[:, 1, 2] = j11, j12
    cov2d = jac @ cov_cam @ np.transpose(jac, (0, 2, 1))
    cov2d = cov2d + float(antialias) * np.eye(2)[None, :, :]
    return {
        "uv": _from_numpy(np.ascontiguousarray(uv.astype(np.float32))),
        "cov2d": _from_numpy(np.ascontiguousarray(cov2d.astype(np.float32))),
        "depth": _from_numpy(np.ascontiguousarray(depth.astype(np.float32))),
        "visible": _from_numpy(np.ascontiguousarray(visible.astype(np.uint8))),
    }


def rasterize(
    uv: Any,
    cov2d: Any,
    colors: Any,
    opacities: Any,
    depth: Any,
    visible: Any,
    width: int,
    height: int,
    bg: Sequence[float] = (0.0, 0.0, 0.0),
) -> _Tensor:
    """Front-to-back splat rasterization with Tensor autograd support w.r.t. `colors` and `opacities`."""
    p = _to_np(uv).astype(np.float64)
    c2 = _to_np(cov2d).astype(np.float64)
    col = _to_np(colors).astype(np.float64)
    opa = _to_np(opacities).astype(np.float64).reshape(-1)
    dep = _to_np(depth).astype(np.float64).reshape(-1)
    vis = _to_np(visible).astype(bool).reshape(-1)
    n = len(p)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"uv must be [N, 2], got shape {p.shape}")
    if c2.shape != (n, 2, 2):
        raise ValueError("cov2d must be [N, 2, 2] matching uv")
    if col.shape != (n, 3):
        raise ValueError("colors must be [N, 3] matching uv")
    if opa.shape != (n,) or dep.shape != (n,) or vis.shape != (n,):
        raise ValueError("opacities/depth/visible must be length N")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(c2)):
        raise ValueError("uv and cov2d must be finite")
    if not np.all(np.isfinite(col)) or np.any(col < 0) or np.any(col > 1):
        raise ValueError("colors must be finite values in [0, 1]")
    if not np.all(np.isfinite(opa)) or np.any(opa < 0) or np.any(opa > 1):
        raise ValueError("opacities must be finite values in [0, 1]")
    if not np.all(np.isfinite(dep)):
        raise ValueError("depth must be finite")
    if int(width) < 1 or int(height) < 1:
        raise ValueError("width/height must be >= 1")
    b = np.asarray(list(bg), dtype=np.float64).reshape(-1)
    if b.shape != (3,) or not np.all(np.isfinite(b)):
        raise ValueError("bg must be 3 finite values")
    w, h = int(width), int(height)

    needs_grad = (isinstance(colors, _Tensor) and getattr(colors, "requires_grad", False)) or (
        isinstance(opacities, _Tensor) and getattr(opacities, "requires_grad", False)
    )

    order = np.argsort(dep, kind="stable")
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)

    if needs_grad:
        col_t = colors if isinstance(colors, _Tensor) else _from_numpy(np.ascontiguousarray(col.astype(np.float32)))
        opa_t = opacities.reshape([n]) if isinstance(opacities, _Tensor) else _from_numpy(np.ascontiguousarray(opa.astype(np.float32)))
        img_t = _from_numpy(np.zeros((h, w, 3), dtype=np.float32))
        trans_t = _from_numpy(np.ones((h, w, 1), dtype=np.float32))

        for i in order:
            if not vis[i] or opa[i] <= 0:
                continue
            det = c2[i, 0, 0] * c2[i, 1, 1] - c2[i, 0, 1] * c2[i, 1, 0]
            if not np.isfinite(det) or det <= 1e-12:
                continue
            inv = np.array([
                [c2[i, 1, 1], -c2[i, 0, 1]],
                [-c2[i, 1, 0], c2[i, 0, 0]],
            ]) / det
            dx, dy = xs - p[i, 0], ys - p[i, 1]
            mahal = inv[0, 0] * dx * dx + (inv[0, 1] + inv[1, 0]) * dx * dy + inv[1, 1] * dy * dy
            kern = np.where(mahal <= 9.0, np.exp(-0.5 * np.clip(mahal, 0.0, 9.0)), 0.0).astype(np.float32)
            kern_t = _from_numpy(np.ascontiguousarray(kern[:, :, None]))

            # Select i-th opacity and color via one-hot vector
            ei = np.zeros(n, dtype=np.float32)
            ei[i] = 1.0
            ei_t = _from_numpy(ei)
            opa_i = (opa_t * ei_t).sum()
            col_i = (col_t * ei_t.reshape([n, 1])).sum(dim=0).reshape([1, 1, 3])

            alpha_t = (kern_t * opa_i).clamp(0.0, 1.0)
            wgt_t = trans_t * alpha_t
            img_t = img_t + wgt_t * col_i
            trans_t = trans_t * (1.0 - alpha_t)

        bg_t = _from_numpy(np.ascontiguousarray(b.reshape(1, 1, 3).astype(np.float32)))
        return img_t + trans_t * bg_t

    img = np.zeros((h, w, 3), dtype=np.float64)
    trans = np.ones((h, w), dtype=np.float64)
    for i in order:
        if not vis[i] or opa[i] <= 0:
            continue
        if trans.max() <= 1e-6:
            break
        det = c2[i, 0, 0] * c2[i, 1, 1] - c2[i, 0, 1] * c2[i, 1, 0]
        if not np.isfinite(det) or det <= 1e-12:
            continue
        inv = np.array([
            [c2[i, 1, 1], -c2[i, 0, 1]],
            [-c2[i, 1, 0], c2[i, 0, 0]],
        ]) / det
        dx, dy = xs - p[i, 0], ys - p[i, 1]
        mahal = inv[0, 0] * dx * dx + (inv[0, 1] + inv[1, 0]) * dx * dy + inv[1, 1] * dy * dy
        kern = np.where(mahal <= 9.0, np.exp(-0.5 * np.clip(mahal, 0.0, 9.0)), 0.0)
        alpha = np.clip(opa[i] * kern, 0.0, 1.0)
        wgt = trans * alpha
        img += wgt[:, :, None] * col[i][None, None, :]
        trans *= 1.0 - alpha
    img += trans[:, :, None] * b[None, None, :]
    return _from_numpy(np.ascontiguousarray(img.astype(np.float32)))


class GaussianSplatScene:
    """Validated container for a splat scene with a render convenience method."""

    def __init__(
        self,
        means: Any,
        scales: Any,
        quats: Any,
        colors: Any,
        opacities: Any,
    ) -> None:
        mu = _to_np(means).astype(np.float32)
        sc = _to_np(scales).astype(np.float32)
        q = _to_np(quats).astype(np.float32)
        col = _to_np(colors).astype(np.float32)
        opa = _to_np(opacities).astype(np.float32).reshape(-1)
        if mu.ndim != 2 or mu.shape[1] != 3 or len(mu) == 0:
            raise ValueError("means must be non-empty [N, 3]")
        n = len(mu)
        if sc.shape != (n, 3) or not np.all(np.isfinite(sc)) or np.any(sc <= 0):
            raise ValueError("scales must be finite positive [N, 3]")
        if q.shape != (n, 4):
            raise ValueError("quats must be [N, 4]")
        if col.shape != (n, 3) or not np.all(np.isfinite(col)):
            raise ValueError("colors must be finite [N, 3]")
        if np.any(col < 0) or np.any(col > 1):
            raise ValueError("colors must lie in [0, 1]")
        if opa.shape != (n,) or not np.all(np.isfinite(opa)):
            raise ValueError("opacities must be finite length N")
        if np.any(opa < 0) or np.any(opa > 1):
            raise ValueError("opacities must lie in [0, 1]")
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(q)):
            raise ValueError("means and quats must be finite")
        quat_to_matrix(q)
        self.means = _from_numpy(np.ascontiguousarray(mu))
        self.scales = _from_numpy(np.ascontiguousarray(sc))
        self.quats = _from_numpy(np.ascontiguousarray(q))
        self.colors = _from_numpy(np.ascontiguousarray(col))
        self.opacities = _from_numpy(np.ascontiguousarray(opa))

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def render(
        self,
        view: Any,
        focal_x: float,
        focal_y: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        bg: Sequence[float] = (0.0, 0.0, 0.0),
        near: float = 0.2,
        antialias: float = 0.3,
    ) -> _Tensor:
        """Projects and rasterizes the scene for one camera."""
        proj = project_gaussians(
            self.means, self.scales, self.quats, view,
            focal_x, focal_y, cx, cy, width, height, near, antialias,
        )
        return rasterize(
            proj["uv"], proj["cov2d"], self.colors, self.opacities,
            proj["depth"], proj["visible"], width, height, bg,
        )


__all__ = ["quat_to_matrix", "project_gaussians", "rasterize", "GaussianSplatScene"]
