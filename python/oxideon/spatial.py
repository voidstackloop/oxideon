"""Geometric deep learning, spatial grid sampling, and Spatial Transformer Networks (STN).

Enables differentiable spatial manipulation and invariant feature extraction:
  - affine_grid: Generates 2D flow field coordinates from affine matrix theta
  - grid_sample: Differentiable bilinear interpolation on 2D feature maps
  - random_affine_matrix: Synthetic affine matrix generator (rotation + translation)
  - rotate_2d: Rotates 2D image tensor
  - SpatialTransformerNetwork: Localization network and differentiable warper (Jaderberg et al., 2015)
"""

from typing import Tuple, Optional, Any, List
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy, cat
from .nn import Module, Linear, Sequential, ReLU, Parameter


def affine_grid(theta: Any, size: Tuple[int, int, int, int]) -> _Tensor:
    """Generates 2D sampling grid from affine transformation theta [N, 2, 3] or [2, 3].

    Size is (N, C, H, W). Output has shape [N, H, W, 2] in normalized [-1, 1] range.
    """
    if len(size) != 4 or any(int(d) < 1 for d in size):
        raise ValueError(f"size must be a 4-tuple of positive integers (N, C, H, W), got {size}")
    n, c, h, w = (int(size[0]), int(size[1]), int(size[2]), int(size[3]))

    th = theta.to_numpy() if isinstance(theta, _Tensor) else np.asarray(theta, dtype=np.float32)
    if th.ndim == 2 and th.shape == (2, 3):
        th = np.broadcast_to(th[None, :, :], (n, 2, 3))
    if th.ndim != 3 or th.shape != (n, 2, 3):
        raise ValueError(f"theta shape must be ({n}, 2, 3) or (2, 3), got {th.shape}")

    # Regular normalized coordinate mesh
    x = np.zeros(1, dtype=np.float32) if w == 1 else np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32) if h == 1 else np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    ones = np.ones_like(xx)
    # Homogeneous grid [3, H * W]
    homo = np.stack([xx.flatten(), yy.flatten(), ones.flatten()], axis=0)

    # Warp coordinates: grid = theta @ homo -> [N, 2, H * W]
    warped = np.einsum("nij,jk->nik", th, homo)
    grid = warped.transpose(0, 2, 1).reshape(n, h, w, 2)
    return _from_numpy(np.ascontiguousarray(grid, dtype=np.float32))


def _reflect_coords(coords: np.ndarray, max_size: int) -> np.ndarray:
    if max_size <= 1:
        return np.zeros_like(coords, dtype=np.int64)
    period = 2 * (max_size - 1)
    c = np.mod(coords, period)
    c = np.where(c >= max_size, period - c, c)
    return np.clip(c, 0, max_size - 1).astype(np.int64)


def grid_sample(
    input_tensor: Any,
    grid: Any,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> _Tensor:
    """Performs 2D spatial sampling using grid coordinates [N, H_out, W_out, 2]."""
    if mode not in ("bilinear", "nearest"):
        raise ValueError(f"Unknown mode '{mode}' (expected 'bilinear' or 'nearest')")
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError(f"Unknown padding_mode '{padding_mode}' (expected 'zeros', 'border', or 'reflection')")

    inp_t = input_tensor if isinstance(input_tensor, _Tensor) else _from_numpy(np.asarray(input_tensor, dtype=np.float32))
    inp = inp_t.to_numpy()
    gr = grid.to_numpy() if isinstance(grid, _Tensor) else np.asarray(grid, dtype=np.float32)

    if inp.ndim != 4:
        raise ValueError(f"input_tensor must be 4D [N, C, H, W], got shape {inp.shape}")
    if gr.ndim != 4 or gr.shape[-1] != 2:
        raise ValueError(f"grid must be 4D [N, H_out, W_out, 2], got shape {gr.shape}")
    if inp.shape[0] != gr.shape[0]:
        raise ValueError(f"Batch size mismatch: input has {inp.shape[0]}, grid has {gr.shape[0]}")

    n, c, h_in, w_in = inp.shape
    if h_in < 1 or w_in < 1:
        raise ValueError(f"Input spatial dimensions must be >= 1, got H={h_in}, W={w_in}")
    _, h_out, w_out, _ = gr.shape

    # Normalize [-1, 1] to pixel coords [0, W-1] and [0, H-1]
    gx = ((gr[..., 0] + 1.0) / 2.0) * (w_in - 1) if w_in > 1 else np.zeros_like(gr[..., 0])
    gy = ((gr[..., 1] + 1.0) / 2.0) * (h_in - 1) if h_in > 1 else np.zeros_like(gr[..., 1])

    # Autograd path: build the explicit interpolation matrix so input grads flow.
    if getattr(inp_t, "requires_grad", False):
        if mode == "nearest":
            taps = [(
                np.rint(gx).astype(np.int64),
                np.rint(gy).astype(np.int64),
                np.ones_like(gx),
            )]
        else:
            x0 = np.floor(gx).astype(np.int64)
            x1 = x0 + 1
            y0 = np.floor(gy).astype(np.int64)
            y1 = y0 + 1
            taps = [
                (x0, y0, (x1 - gx) * (y1 - gy)),
                (x0, y1, (x1 - gx) * (gy - y0)),
                (x1, y0, (gx - x0) * (y1 - gy)),
                (x1, y1, (gx - x0) * (gy - y0)),
            ]
        placed = []
        for xx, yy, w in taps:
            if padding_mode == "zeros":
                valid = (xx >= 0) & (xx < w_in) & (yy >= 0) & (yy < h_in)
                placed.append((
                    np.clip(xx, 0, w_in - 1),
                    np.clip(yy, 0, h_in - 1),
                    np.where(valid, w, 0.0),
                ))
            elif padding_mode == "border":
                placed.append((np.clip(xx, 0, w_in - 1), np.clip(yy, 0, h_in - 1), w))
            else:  # reflection
                placed.append((_reflect_coords(xx, w_in), _reflect_coords(yy, h_in), w))

        batch_outs = []
        for b in range(n):
            M = np.zeros((h_out * w_out, h_in * w_in), dtype=np.float32)
            for y_o in range(h_out):
                for x_o in range(w_out):
                    k = y_o * w_out + x_o
                    for xx, yy, w in placed:
                        M[k, int(yy[b, y_o, x_o]) * w_in + int(xx[b, y_o, x_o])] += w[b, y_o, x_o]

            M_t = _from_numpy(np.ascontiguousarray(M.T))  # [H_in*W_in, H_out*W_out]
            x_b = inp_t[b].reshape(c, h_in * w_in)
            out_b = x_b.matmul(M_t).reshape(c, h_out, w_out)
            batch_outs.append(out_b.unsqueeze(0))

        if len(batch_outs) == 1:
            return batch_outs[0]
        return cat(batch_outs, dim=0)

    out = np.zeros((n, c, h_out, w_out), dtype=np.float32)

    def fetch_pixels(img: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        mask = (x_coords >= 0) & (x_coords < w_in) & (y_coords >= 0) & (y_coords < h_in)
        if padding_mode == "reflection":
            xc = _reflect_coords(x_coords, w_in)
            yc = _reflect_coords(y_coords, h_in)
        else:
            xc = np.clip(x_coords, 0, w_in - 1)
            yc = np.clip(y_coords, 0, h_in - 1)
        val = img[yc, xc]
        if padding_mode == "zeros":
            val = np.where(mask, val, 0.0)
        return val

    if mode == "nearest":
        xn = np.rint(gx).astype(np.int64)
        yn = np.rint(gy).astype(np.int64)
        for b in range(n):
            for ch in range(c):
                out[b, ch] = fetch_pixels(inp[b, ch], xn[b], yn[b])
        return _from_numpy(np.ascontiguousarray(out, dtype=np.float32))

    x0 = np.floor(gx).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(gy).astype(np.int64)
    y1 = y0 + 1

    wa = (x1 - gx) * (y1 - gy)
    wb = (x1 - gx) * (gy - y0)
    wc = (gx - x0) * (y1 - gy)
    wd = (gx - x0) * (gy - y0)

    for b in range(n):
        for ch in range(c):
            img = inp[b, ch]
            out[b, ch] = (
                wa[b] * fetch_pixels(img, x0[b], y0[b])
                + wb[b] * fetch_pixels(img, x0[b], y1[b])
                + wc[b] * fetch_pixels(img, x1[b], y0[b])
                + wd[b] * fetch_pixels(img, x1[b], y1[b])
            )

    return _from_numpy(np.ascontiguousarray(out, dtype=np.float32))


def rotate_2d(image: Any, angle_deg: float) -> _Tensor:
    """Rotates a 2D image tensor [N, C, H, W] or [C, H, W] around its center by angle_deg degrees."""
    t = image if isinstance(image, _Tensor) else _from_numpy(np.asarray(image, dtype=np.float32))
    single = t.ndim == 3
    if single:
        t = t.unsqueeze(0)
    n, c, h, w = t.shape
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    theta = np.array([[[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]]], dtype=np.float32)
    grid = affine_grid(theta, (n, c, h, w))
    sampled = grid_sample(t, grid)
    if single:
        return sampled.squeeze(0)
    return sampled


def random_affine_matrix(
    batch_size: int = 1,
    max_angle_deg: float = 30.0,
    max_trans: float = 0.1,
    seed: Optional[int] = None,
) -> _Tensor:
    """Generates random 2D affine transformation matrices [batch_size, 2, 3]."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if max_angle_deg < 0.0 or max_trans < 0.0:
        raise ValueError(f"max_angle_deg and max_trans must be >= 0, got {max_angle_deg}, {max_trans}")

    rng = np.random.RandomState(seed) if seed is not None else np.random
    angles = np.radians(rng.uniform(-max_angle_deg, max_angle_deg, size=batch_size))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    tx = rng.uniform(-max_trans, max_trans, size=batch_size)
    ty = rng.uniform(-max_trans, max_trans, size=batch_size)

    thetas = np.zeros((batch_size, 2, 3), dtype=np.float32)
    thetas[:, 0, 0] = cos_a
    thetas[:, 0, 1] = -sin_a
    thetas[:, 0, 2] = tx
    thetas[:, 1, 0] = sin_a
    thetas[:, 1, 1] = cos_a
    thetas[:, 1, 2] = ty

    return _from_numpy(thetas)


class SpatialTransformerNetwork(Module):
    """Spatial Transformer Network (STN) mapping inputs through learned geometric rectification."""

    def __init__(self, in_features: int, hidden_dim: int = 32):
        super().__init__()
        if in_features < 1 or hidden_dim < 1:
            raise ValueError(f"in_features and hidden_dim must be >= 1, got {in_features}, {hidden_dim}")
        self.in_features = in_features
        self.localization = Sequential(
            Linear(in_features, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 6),
        )

        # Initialize to identity transform: [ [1, 0, 0], [0, 1, 0] ]
        id_weight = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        last_lin = getattr(self.localization, "2", None) or getattr(self.localization, "_modules", {}).get("2")
        if last_lin is not None and hasattr(last_lin, "bias"):
            last_lin.bias = Parameter(_from_numpy(id_weight))
        if last_lin is not None and hasattr(last_lin, "weight"):
            last_lin.weight = Parameter(_from_numpy(np.zeros_like(last_lin.weight.to_numpy())))

    def forward(self, x: Any) -> _Tensor:
        """Takes image tensor [N, C, H, W] or [C, H, W] and returns spatially transformed tensor."""
        t = x if isinstance(x, _Tensor) else _from_numpy(np.asarray(x, dtype=np.float32))
        single = t.ndim == 3
        if single:
            t = t.unsqueeze(0)
        elif t.ndim != 4:
            raise ValueError(f"SpatialTransformerNetwork expects 3D [C, H, W] or 4D [N, C, H, W] input, got shape {t.shape}")

        n, c, h, w = t.shape
        if c * h * w != self.in_features:
            raise ValueError(f"Flattened feature count C*H*W ({c*h*w}) must match in_features ({self.in_features})")

        flat = t.reshape(n, self.in_features)
        theta = self.localization(flat).to_numpy().reshape(n, 2, 3)

        grid = affine_grid(theta, (n, c, h, w))
        sampled = grid_sample(t, grid)
        if single:
            return sampled.squeeze(0)
        return sampled


__all__ = [
    "affine_grid",
    "grid_sample",
    "random_affine_matrix",
    "rotate_2d",
    "SpatialTransformerNetwork",
]
