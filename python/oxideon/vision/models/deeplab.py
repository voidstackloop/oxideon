"""DeepLabV3-style segmentation head with ASPP (numpy reference) for Vision.

Uses dilated convolutions, which ``ox.nn.Conv2d`` does not support, so the
forward pass runs in numpy. No autograd flows through these modules.
"""

from typing import Any, List, Optional, Sequence, Tuple
import numpy as np

from ...oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def _dilated_conv2d(
    x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray], dilation: int
) -> np.ndarray:
    oc, ic, k, _ = weight.shape
    b, _, h, w = x.shape
    pad = dilation * (k - 1) // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    out = np.zeros((b, oc, h, w), dtype=np.float64)
    for kh in range(k):
        for kw in range(k):
            patch = xp[:, :, kh * dilation : kh * dilation + h, kw * dilation : kw * dilation + w]
            out += np.einsum("oi,bihw->bohw", weight[:, :, kh, kw], patch)
    if bias is not None:
        out += bias[None, :, None, None]
    return out


def _bilinear_upsample(x: np.ndarray, height: int, width: int) -> np.ndarray:
    b, c, h, w = x.shape
    if h == height and w == width:
        return x
    ys = (np.arange(height, dtype=np.float64) + 0.5) * h / height - 0.5
    xs = (np.arange(width, dtype=np.float64) + 0.5) * w / width - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy = np.clip(ys - np.floor(ys), 0.0, 1.0)
    fx = np.clip(xs - np.floor(xs), 0.0, 1.0)
    # x[b, c, y, x]
    a = x[:, :, y0[:, None], x0[None, :]]
    bb = x[:, :, y0[:, None], x1[None, :]]
    cc = x[:, :, y1[:, None], x0[None, :]]
    dd = x[:, :, y1[:, None], x1[None, :]]
    fy = fy[:, None]
    fx = fx[None, :]
    return a * (1 - fy) * (1 - fx) + bb * (1 - fy) * fx + cc * fy * (1 - fx) + dd * fy * fx


class DilatedConv2d:
    """SAME-padded 2-D convolution with dilation (numpy reference, no autograd)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if int(in_channels) < 1 or int(out_channels) < 1:
            raise ValueError("channels must be >= 1")
        if int(kernel_size) < 1 or int(kernel_size) % 2 == 0:
            raise ValueError("kernel_size must be an odd int >= 1")
        if int(dilation) < 1:
            raise ValueError("dilation must be >= 1")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        rng = np.random.default_rng(seed)
        fan_in = in_channels * kernel_size * kernel_size
        bound = float(np.sqrt(6.0 / fan_in))
        w = rng.uniform(-bound, bound, (out_channels, in_channels, kernel_size, kernel_size))
        self.weight = _from_numpy(np.ascontiguousarray(w.astype(np.float32)))
        self.bias = (
            _from_numpy(np.ascontiguousarray(np.zeros(out_channels, dtype=np.float32)))
            if bias
            else None
        )

    def forward(self, x: Any) -> _Tensor:
        """Applies the convolution to [B, C, H, W], returns [B, OC, H, W]."""
        a = np.asarray(_to_np(x), dtype=np.float64)
        if a.ndim != 4 or a.shape[1] != self.in_channels:
            raise ValueError(
                f"expected [B, {self.in_channels}, H, W], got shape {a.shape}"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError("input must be finite")
        w = self.weight.to_numpy().astype(np.float64)
        b = self.bias.to_numpy().astype(np.float64) if self.bias is not None else None
        y = _dilated_conv2d(a, w, b, self.dilation)
        return _from_numpy(np.ascontiguousarray(y.astype(np.float32)))

    def __call__(self, x: Any) -> _Tensor:
        return self.forward(x)

    def parameters(self) -> List[_Tensor]:
        return [self.weight] + ([self.bias] if self.bias is not None else [])


class ASPP:
    """Atrous Spatial Pyramid Pooling: 1x1 + dilated 3x3s + image pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: Sequence[int] = (6, 12, 18),
        seed: Optional[int] = None,
    ) -> None:
        rates = tuple(int(r) for r in rates)
        if not rates or any(r < 1 for r in rates):
            raise ValueError("rates must be a non-empty sequence of ints >= 1")
        if int(out_channels) < 1:
            raise ValueError("out_channels must be >= 1")
        rng = np.random.default_rng(seed)
        seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(len(rates) + 3)]
        self.branch1 = DilatedConv2d(in_channels, out_channels, 1, 1, seed=seeds[0])
        self.branches = [
            DilatedConv2d(in_channels, out_channels, 3, r, seed=s)
            for r, s in zip(rates, seeds[1:])
        ]
        self.pool_proj = DilatedConv2d(in_channels, out_channels, 1, 1, seed=seeds[-2])
        self.project = DilatedConv2d(
            out_channels * (len(rates) + 2), out_channels, 1, 1, seed=seeds[-1]
        )
        self.out_channels = int(out_channels)

    def forward(self, x: Any) -> _Tensor:
        a = np.asarray(_to_np(x), dtype=np.float32)
        if a.ndim != 4:
            raise ValueError(f"expected [B, C, H, W], got shape {a.shape}")
        feats = [self.branch1(a).to_numpy(), *(br(a).to_numpy() for br in self.branches)]
        pooled = a.mean(axis=(2, 3), keepdims=True)
        up = _bilinear_upsample(
            self.pool_proj(pooled).to_numpy().astype(np.float64), a.shape[2], a.shape[3]
        ).astype(np.float32)
        feats.append(up)
        return self.project(np.concatenate(feats, axis=1))

    def __call__(self, x: Any) -> _Tensor:
        return self.forward(x)

    def parameters(self) -> List[_Tensor]:
        ps = self.branch1.parameters() + self.pool_proj.parameters()
        for br in self.branches:
            ps += br.parameters()
        return ps + self.project.parameters()


class DeepLabHead:
    """ASPP + 1x1 classifier producing per-pixel class logits."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        rates: Sequence[int] = (6, 12, 18),
        aspp_channels: int = 256,
        seed: Optional[int] = None,
    ) -> None:
        if int(num_classes) < 2:
            raise ValueError("num_classes must be >= 2")
        rng = np.random.default_rng(seed)
        s1, s2 = int(rng.integers(0, 2**31 - 1)), int(rng.integers(0, 2**31 - 1))
        self.aspp = ASPP(in_channels, aspp_channels, rates, seed=s1)
        self.classifier = DilatedConv2d(aspp_channels, num_classes, 1, 1, seed=s2)
        self.num_classes = int(num_classes)

    def forward(self, x: Any) -> _Tensor:
        """Maps backbone features [B, C, H, W] to logits [B, K, H, W]."""
        return self.classifier(self.aspp(x))

    def __call__(self, x: Any) -> _Tensor:
        return self.forward(x)

    def predict(self, logits: Any) -> _Tensor:
        """Argmax over classes -> int64 [B, H, W] label map."""
        l = np.asarray(_to_np(logits))
        if l.ndim != 4 or l.shape[1] != self.num_classes:
            raise ValueError(
                f"expected [B, {self.num_classes}, H, W], got shape {l.shape}"
            )
        return _from_numpy(np.ascontiguousarray(np.argmax(l, axis=1).astype(np.int64)))

    def parameters(self) -> List[_Tensor]:
        return self.aspp.parameters() + self.classifier.parameters()


__all__ = ["DilatedConv2d", "ASPP", "DeepLabHead"]
