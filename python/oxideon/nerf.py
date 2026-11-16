"""Neural Radiance Fields (Mildenhall et al., ECCV 2020) in native ox ops.

- ``positional_encoding``: Fourier feature mapping of positions/directions.
- ``get_rays``: pinhole-camera ray origins and directions from a c2w pose.
- ``stratified_samples``: stratified depth samples along each ray.
- ``volume_render``: differentiable alpha compositing (density -> rgb/depth) with optional white background.
- ``sample_pdf``: inverse-CDF resampling for the hierarchical fine pass.
- ``NeRFField``: trainable MLP radiance field (density + view-dependent color).
- ``render_rays``: full coarse-to-fine hierarchical ray rendering.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Linear as _Linear, Module as _Module


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def _as_tensor(t: Any, name: str) -> _Tensor:
    if isinstance(t, _Tensor):
        return t
    return _from_numpy(np.ascontiguousarray(np.asarray(t, dtype=np.float32)))


def positional_encoding(x: Any, num_freqs: int, include_input: bool = True) -> _Tensor:
    """Fourier-features ``x`` with frequencies 2^0..2^(F-1)."""
    t = _as_tensor(x, "x")
    if int(num_freqs) < 1:
        raise ValueError(f"num_freqs must be >= 1, got {num_freqs}")
    if t.ndim < 1:
        raise ValueError(f"x must have ndim >= 1, got shape {t.shape}")
    freqs = _from_numpy(
        np.ascontiguousarray((2.0 ** np.arange(int(num_freqs), dtype=np.float32)))
    )
    proj = t.unsqueeze(-1) * freqs  # [..., D, F]
    feats = _Tensor.cat([proj.sin(), proj.cos()], -1).reshape(
        list(t.shape[:-1]) + [t.shape[-1] * 2 * int(num_freqs)]
    )
    if include_input:
        feats = _Tensor.cat([t, feats], -1)
    return feats


def get_rays(
    height: int, width: int, focal: float, pose: Any
) -> Tuple[_Tensor, _Tensor]:
    """Pinhole rays for a camera-to-world pose."""
    if int(height) < 1 or int(width) < 1:
        raise ValueError("height and width must be >= 1")
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError(f"focal must be finite and positive, got {focal}")
    p = _to_np(pose).astype(np.float64)
    if p.shape not in ((3, 4), (4, 4)):
        raise ValueError(f"pose must be [3, 4] or [4, 4], got shape {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("pose must be finite")
    h, w = int(height), int(width)
    i, j = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    dirs_cam = np.stack(
        [(i - w / 2.0 + 0.5) / focal, -(j - h / 2.0 + 0.5) / focal, -np.ones_like(i)],
        axis=-1,
    )
    rot = p[:3, :3]
    t = p[:3, 3]
    dirs = dirs_cam @ rot.T
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("pose produces degenerate ray directions")
    dirs = dirs / np.maximum(norms, 1e-12)
    origins = np.broadcast_to(t, (h, w, 3)).copy()
    return (
        _from_numpy(np.ascontiguousarray(origins.astype(np.float32))),
        _from_numpy(np.ascontiguousarray(dirs.astype(np.float32))),
    )


def stratified_samples(
    near: float,
    far: float,
    n_samples: int,
    n_rays: int,
    perturb: bool = True,
    seed: Optional[int] = None,
) -> _Tensor:
    """Stratified depth samples, one jittered sample per uniform bin."""
    if not np.isfinite(near) or not np.isfinite(far) or not near < far:
        raise ValueError(f"need finite near < far, got {near}, {far}")
    if int(n_samples) < 1 or int(n_rays) < 1:
        raise ValueError("n_samples and n_rays must be >= 1")
    n_samples, n_rays = int(n_samples), int(n_rays)
    edges = np.linspace(near, far, n_samples + 1, dtype=np.float64)
    lower, upper = edges[:-1], edges[1:]
    if perturb:
        u = np.random.default_rng(seed).random((n_rays, n_samples))
        t = lower[None, :] + (upper - lower)[None, :] * u
    else:
        t = np.broadcast_to((lower + upper) / 2.0, (n_rays, n_samples)).copy()
    return _from_numpy(np.ascontiguousarray(t.astype(np.float32)))


def volume_render(
    sigmas: Any, rgbs: Any, t_vals: Any, white_bkgd: bool = False
) -> Dict[str, _Tensor]:
    """Differentiable alpha compositing along rays."""
    s = _as_tensor(sigmas, "sigmas")
    c = _as_tensor(rgbs, "rgbs")
    t = _as_tensor(t_vals, "t_vals")
    if s.ndim != 2:
        raise ValueError(f"sigmas must be [R, S], got shape {s.shape}")
    if c.ndim != 3 or c.shape[:2] != s.shape or c.shape[2] != 3:
        raise ValueError(f"rgbs must be [R, S, 3] matching sigmas, got shape {c.shape}")
    if t.ndim == 1:
        if t.shape[0] != s.shape[1]:
            raise ValueError("t_vals [S] length must match samples")
        t = _Tensor.zeros(list(s.shape)) + t
    elif t.ndim != 2 or list(t.shape) != list(s.shape):
        raise ValueError(f"t_vals must be [S] or [R, S] matching sigmas, got {t.shape}")
    r, _ = s.shape
    density = s.clamp(0.0, 1e10)
    dists = _Tensor.cat(
        [t[:, 1:] - t[:, :-1], _Tensor.full([r, 1], 1e10)], -1
    )
    alpha = 1.0 - (-density * dists).exp()
    trans = (1.0 - alpha + 1e-10).cumprod(-1)
    transmittance = trans / (1.0 - alpha + 1e-10)
    weights = alpha * transmittance
    rgb = (weights.unsqueeze(-1) * c).sum(-2)
    depth = (weights * t).sum(-1)
    acc = weights.sum(-1)
    if white_bkgd:
        rgb = rgb + (1.0 - acc.unsqueeze(-1))
    return {"rgb": rgb, "depth": depth, "acc": acc, "weights": weights}


def sample_pdf(
    t_vals: Any, weights: Any, n_samples: int, seed: Optional[int] = None
) -> _Tensor:
    """Inverse-CDF samples from the coarse piecewise-constant PDF (no grad)."""
    t = _to_np(t_vals).astype(np.float64)
    w = _to_np(weights).astype(np.float64)
    if t.ndim != 2 or w.shape != t.shape:
        raise ValueError("t_vals and weights must be matching [R, S]")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(w)):
        raise ValueError("t_vals and weights must be finite")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if int(n_samples) < 1:
        raise ValueError("n_samples must be >= 1")
    n_samples = int(n_samples)
    pdf = w + 1e-5
    pdf = pdf / pdf.sum(axis=-1, keepdims=True)
    cdf = np.cumsum(pdf, axis=-1)
    cdf = np.concatenate([np.zeros_like(cdf[..., :1]), cdf], axis=-1)
    u = np.random.default_rng(seed).random((t.shape[0], n_samples))
    inds = (
        np.stack(
            [np.searchsorted(cdf[i], u[i], side="right") for i in range(t.shape[0])]
        ).astype(np.int64)
        - 1
    )
    inds = np.clip(inds, 0, t.shape[1] - 1)
    below = np.take_along_axis(t, inds, axis=-1)
    above = np.take_along_axis(t, np.clip(inds + 1, 0, t.shape[1] - 1), axis=-1)
    cdf_below = np.take_along_axis(cdf, inds, axis=-1)
    cdf_above = np.take_along_axis(cdf, np.clip(inds + 1, 0, cdf.shape[1] - 1), axis=-1)
    denom = np.maximum(cdf_above - cdf_below, 1e-10)
    frac = np.clip((u - cdf_below) / denom, 0.0, 1.0)
    samples = below + (above - below) * frac
    return _from_numpy(np.ascontiguousarray(samples.astype(np.float32)))


class NeRFField(_Module):
    """MLP radiance field: points -> density, (points, dirs) -> color."""

    def __init__(
        self,
        depth: int = 8,
        width: int = 256,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        skip_layer: int = 4,
    ) -> None:
        super().__init__()
        if int(depth) < 2 or int(width) < 1:
            raise ValueError("need depth >= 2 and width >= 1")
        if int(pos_freqs) < 1 or int(dir_freqs) < 1:
            raise ValueError("frequency bands must be >= 1")
        if not 0 <= int(skip_layer) < int(depth):
            raise ValueError("skip_layer must satisfy 0 <= skip < depth")
        self.depth = int(depth)
        self.width = int(width)
        self.pos_freqs = int(pos_freqs)
        self.dir_freqs = int(dir_freqs)
        self.skip_layer = int(skip_layer)
        pos_dim = 3 * (2 * self.pos_freqs + 1)
        self.trunk: List[_Linear] = []
        for i in range(self.depth):
            in_dim = pos_dim if i == 0 else self.width
            if 0 < i == self.skip_layer:
                in_dim += pos_dim
            layer = _Linear(in_dim, self.width)
            self.trunk.append(layer)
            self.add_module(f"trunk_{i}", layer)
        self.sigma_head = _Linear(self.width, 1)
        self.add_module("sigma_head", self.sigma_head)
        dir_dim = 3 * (2 * self.dir_freqs + 1)
        color_w = max(1, self.width // 2)
        self.color_hidden = _Linear(self.width + dir_dim, color_w)
        self.add_module("color_hidden", self.color_hidden)
        self.color_head = _Linear(color_w, 3)
        self.add_module("color_head", self.color_head)

    def forward(self, points: Any, dirs: Any) -> Tuple[_Tensor, _Tensor]:
        """Evaluates density and color."""
        p = _as_tensor(points, "points")
        d = _as_tensor(dirs, "dirs")
        if p.ndim < 1 or p.shape[-1] != 3:
            raise ValueError(f"points must be [..., 3], got shape {p.shape}")
        if d.ndim == 1:
            if d.shape[0] != 3:
                raise ValueError(f"dirs must be [..., 3], [R, 3] or [3], got shape {d.shape}")
            d = _Tensor.zeros(list(p.shape)) + d
        elif list(d.shape) == list(p.shape):
            pass
        elif d.ndim == p.ndim - 1 and list(d.shape) == list(p.shape[:-2]) + [3]:
            d = d.unsqueeze(-2) + _Tensor.zeros(list(p.shape))
        else:
            raise ValueError(
                f"dirs shape {d.shape} must be [3], {list(p.shape[:-2]) + [3]} "
                f"or match points {p.shape}"
            )
        h = positional_encoding(p, self.pos_freqs)
        trunk_in = h
        for i, layer in enumerate(self.trunk):
            if 0 < i == self.skip_layer:
                h = _Tensor.cat([h, trunk_in], -1)
            h = layer(h).relu()
        sigma = self.sigma_head(h).squeeze(-1).relu()
        ch = _Tensor.cat([h, positional_encoding(d, self.dir_freqs)], -1)
        rgb = self.color_head(self.color_hidden(ch).relu()).sigmoid()
        return sigma, rgb


def render_rays(
    field: NeRFField,
    origins: Any,
    dirs: Any,
    near: float,
    far: float,
    n_coarse: int = 64,
    n_fine: int = 128,
    perturb: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, _Tensor]:
    """Hierarchical coarse-to-fine rendering of a ray batch (differentiable)."""
    o = _to_np(origins).astype(np.float32)
    d = _to_np(dirs).astype(np.float32)
    if o.ndim != 2 or o.shape[1] != 3 or d.shape != o.shape:
        raise ValueError("origins/dirs must be matching [R, 3]")
    if not np.all(np.isfinite(o)) or not np.all(np.isfinite(d)):
        raise ValueError("origins/dirs must be finite")
    if int(n_coarse) < 1 or int(n_fine) < 1:
        raise ValueError("n_coarse and n_fine must be >= 1")
    rng = np.random.default_rng(seed)
    s1, s2 = int(rng.integers(0, 2**31 - 1)), int(rng.integers(0, 2**31 - 1))
    n_rays = o.shape[0]
    t_coarse = stratified_samples(near, far, n_coarse, n_rays, perturb, s1)
    pts = o[:, None, :] + t_coarse.to_numpy()[:, :, None] * d[:, None, :]
    sig_c, rgb_c = field(_from_numpy(np.ascontiguousarray(pts)), _as_tensor(dirs, "dirs"))
    coarse = volume_render(sig_c, rgb_c, t_coarse)
    t_extra = sample_pdf(t_coarse, coarse["weights"].detach(), n_fine, s2)
    t_all, _ = _Tensor.cat([t_coarse, t_extra], -1).sort(-1)
    pts_all = o[:, None, :] + t_all.to_numpy()[:, :, None] * d[:, None, :]
    sig_f, rgb_f = field(_from_numpy(np.ascontiguousarray(pts_all)), _as_tensor(dirs, "dirs"))
    fine = volume_render(sig_f, rgb_f, t_all)
    return {
        "rgb_coarse": coarse["rgb"],
        "depth_coarse": coarse["depth"],
        "acc_coarse": coarse["acc"],
        "rgb_fine": fine["rgb"],
        "depth_fine": fine["depth"],
        "acc_fine": fine["acc"],
        "weights": fine["weights"],
        "t_vals": t_all,
    }


__all__ = [
    "positional_encoding",
    "get_rays",
    "stratified_samples",
    "volume_render",
    "sample_pdf",
    "NeRFField",
    "render_rays",
]
