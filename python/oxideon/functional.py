from .special import (
    erf,
    erfc,
    erfinv,
)
"""Comprehensive functional interface for Oxideon.

Mirrors standard deep learning functional APIs across activations, scaled dot-product attention,
linear projections, convolutions, pooling, normalization, spatial transformations,
embeddings, dropout, loss functions, and functional autograd transforms.
"""

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
from ._contig import contig as _contig

from .oxideon import (
    Tensor as _Tensor,
    from_numpy as _from_numpy,
    cat as _native_cat,
    stack as _native_stack,
    cross_entropy as _native_cross_entropy,
    mse_loss as _native_mse_loss,
    l1_loss as _native_l1_loss,
    binary_cross_entropy_with_logits as _native_bce_with_logits,
    huber_loss as _native_huber_loss,
    focal_loss as _native_focal_loss,
    kl_div_loss as _native_kl_div_loss,
    nll_loss as _native_nll_loss,
    cosine_embedding_loss as _native_cosine_embedding_loss,
    triplet_margin_loss as _native_triplet_margin_loss,
    margin_ranking_loss as _native_margin_ranking_loss,
    hinge_embedding_loss as _native_hinge_embedding_loss,
    soft_margin_loss as _native_soft_margin_loss,
    ctc_loss as _native_ctc_loss,
    grad as _native_grad,
    conv2d as _native_conv2d,
    conv1d as _native_conv1d,
)
from .primitives import where as _ox_where


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, _Tensor):
        return x.to_numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _apply_reduction(loss: _Tensor, reduction: str, name: str) -> _Tensor:
    """Applies a torch-style loss reduction, rejecting unknown modes loudly."""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"{name}: reduction must be 'mean', 'sum' or 'none', got {reduction!r}")


def _as_tensor(x: Any) -> _Tensor:
    if isinstance(x, _Tensor):
        return x
    return _from_numpy(_contig(_to_np(x)))


def _diff_pad_nd(x: _Tensor, pads, value: float = 0.0) -> _Tensor:
    """Differentiable zero/constant padding of trailing spatial dims (from dim 2).

    ``pads`` gives (before, after) per spatial dim. Stays in autograd (cat).
    """
    dt = _to_np(x).dtype
    for axis, (pb, pa) in enumerate(pads, start=2):
        if pb == 0 and pa == 0:
            continue
        s = list(x.shape)
        parts = []
        if pb > 0:
            sb = list(s)
            sb[axis] = pb
            parts.append(_from_numpy(np.full(sb, value, dtype=dt)))
        parts.append(x)
        if pa > 0:
            sa = list(s)
            sa[axis] = pa
            parts.append(_from_numpy(np.full(sa, value, dtype=dt)))
        x = _native_cat(parts, dim=axis)
    return x


def _wrap(arr: np.ndarray, like: Optional[_Tensor] = None) -> _Tensor:
    # No dtype coercion: _from_numpy supports float64 natively, and silently
    # narrowing f64 intermediates to f32 loses precision (e.g. in cumsum).
    res = _from_numpy(_contig(arr))
    if like is not None:
        if hasattr(like, "dtype"):
            try:
                res = res.to_dtype(str(like.dtype))
            except Exception:
                pass
        if hasattr(like, "device"):
            try:
                res = res.to(like.device)
            except Exception:
                pass
    return res


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def relu(input: _Tensor) -> _Tensor:
    """Applies rectified linear unit function element-wise: max(0, x)."""
    return input.relu()


def relu6(input: _Tensor) -> _Tensor:
    """Applies ReLU6 element-wise: min(max(0, x), 6)."""
    return input.clamp(0.0, 6.0)


def gelu(input: _Tensor) -> _Tensor:
    """Applies Gaussian Error Linear Units function element-wise."""
    return input.gelu()


def silu(input: _Tensor) -> _Tensor:
    """Applies Sigmoid Linear Unit (SiLU / Swish) function element-wise: x * sigmoid(x)."""
    return input.silu()


def sigmoid(input: _Tensor) -> _Tensor:
    """Applies element-wise Sigmoid function: 1 / (1 + exp(-x))."""
    return input.sigmoid()


def tanh(input: _Tensor) -> _Tensor:
    """Applies element-wise Hyperbolic Tangent function."""
    return input.tanh()


def leaky_relu(input: _Tensor, negative_slope: float = 0.01) -> _Tensor:
    """Applies element-wise LeakyReLU function: max(0, x) + negative_slope * min(0, x)."""
    return input.relu() - ((-input).relu() * float(negative_slope))


def softmax(input: _Tensor, dim: int = -1) -> _Tensor:
    """Applies Softmax function along specified dimension."""
    return input.softmax(dim)


def log_softmax(input: _Tensor, dim: int = -1) -> _Tensor:
    """Applies Log-Softmax function along specified dimension."""
    return input.log_softmax(dim)


def softmin(input: _Tensor, dim: int = -1) -> _Tensor:
    """Applies Softmin function along specified dimension: softmax(-x, dim)."""
    return (-input).softmax(dim)


def softplus(input: _Tensor, beta: float = 1.0, threshold: float = 20.0) -> _Tensor:
    """Applies element-wise Softplus: 1/beta * ln(1 + exp(beta * x))."""
    if beta == 1.0:
        return input.softplus()
    scaled = input * float(beta)
    return (scaled.softplus()) / float(beta)


def softsign(input: _Tensor) -> _Tensor:
    """Applies element-wise Softsign: x / (1 + |x|)."""
    return input.softsign()


def glu(input: _Tensor, dim: int = -1) -> _Tensor:
    """Applies Gated Linear Unit (GLU): splits tensor into halves a and b along dim, returning a * sigmoid(b)."""
    shape = input.shape
    d = dim if dim >= 0 else len(shape) + dim
    if not (0 <= d < len(shape)):
        raise IndexError(f"Dimension out of range (expected to be in range of [{-len(shape)}, {len(shape) - 1}], but got {dim})")
    if shape[d] % 2 != 0:
        raise RuntimeError(f"glu: Halving dimension must be even, but dimension {dim} is {shape[d]}")
    a, b = input.chunk(2, dim=dim)
    return a * b.sigmoid()


def hardswish(input: _Tensor) -> _Tensor:
    """Applies element-wise Hardswish: x * relu6(x + 3) / 6."""
    r6 = (input + 3.0).clamp(0.0, 6.0)
    return (input * r6) / 6.0


def hardsigmoid(input: _Tensor) -> _Tensor:
    """Applies element-wise Hardsigmoid: relu6(x + 3) / 6."""
    return (input + 3.0).clamp(0.0, 6.0) / 6.0




def hardshrink(input: _Tensor, lambd: float = 0.5) -> _Tensor:
    """Applies the Hard Shrinkage function element-wise."""
    mask = (input.abs() > float(lambd)).to_dtype(input.dtype)
    return input * mask


def softshrink(input: _Tensor, lambd: float = 0.5) -> _Tensor:
    """Applies the Soft Shrinkage function element-wise."""
    pos = (input > float(lambd)).to_dtype(input.dtype)
    neg = (input < -float(lambd)).to_dtype(input.dtype)
    return pos * (input - float(lambd)) + neg * (input + float(lambd))


def threshold(input: _Tensor, threshold: float, value: float) -> _Tensor:
    """Thresholds each element of the input Tensor."""
    mask = (input > float(threshold)).to_dtype(input.dtype)
    val_t = _Tensor.full(input.shape, float(value), dtype=input.dtype, device=input.device)
    one_minus_mask = _Tensor.ones_like(mask) - mask
    return input * mask + val_t * one_minus_mask


def logsigmoid(input: _Tensor) -> _Tensor:
    """Applies element-wise Log-Sigmoid: log(1 / (1 + exp(-x))) = -softplus(-x)."""
    return -softplus(-input)


def hardtanh(input: _Tensor, min_val: float = -1.0, max_val: float = 1.0) -> _Tensor:
    """Applies element-wise Hardtanh: clamps to [min_val, max_val]."""
    return input.clamp(min_val, max_val)


def elu(input: _Tensor, alpha: float = 1.0) -> _Tensor:
    """Applies element-wise ELU: max(0, x) + min(0, alpha * (exp(x) - 1))."""
    alpha_f = float(alpha)
    return _ox_where(input > 0.0, input, alpha_f * (input.exp() - 1.0))


def selu(input: _Tensor) -> _Tensor:
    """Applies element-wise Scaled Exponential Linear Unit (SELU)."""
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return scale * _ox_where(input > 0.0, input, alpha * (input.exp() - 1.0))


def celu(input: _Tensor, alpha: float = 1.0) -> _Tensor:
    """Applies element-wise CELU: max(0, x) + min(0, alpha * (exp(x / alpha) - 1))."""
    alpha_f = float(alpha)
    if alpha_f == 0.0:
        raise ZeroDivisionError("celu: alpha cannot be 0")
    return _ox_where(input > 0.0, input, alpha_f * ((input / alpha_f).exp() - 1.0))


def mish(input: _Tensor) -> _Tensor:
    """Applies element-wise Mish: x * tanh(softplus(x))."""
    return input * (input.softplus().tanh())


def tanhshrink(input: _Tensor) -> _Tensor:
    """Applies element-wise Tanhshrink: x - tanh(x)."""
    return input - input.tanh()


def rrelu(input: _Tensor, lower: float = 0.125, upper: float = 0.3333333333333333, training: bool = False) -> _Tensor:
    """Applies Randomized Leaky ReLU."""
    if lower > upper:
        raise ValueError(f"rrelu: lower ({lower}) must be <= upper ({upper})")
    if training:
        a = float(np.random.uniform(lower, upper))
        return leaky_relu(input, negative_slope=a)
    return leaky_relu(input, negative_slope=float(lower + upper) / 2.0)


def prelu(input: _Tensor, weight: _Tensor) -> _Tensor:
    """Applies Parametric ReLU: max(0, x) + weight * min(0, x)."""
    input = _as_tensor(input)
    weight = _as_tensor(weight)
    numel_w = weight.numel()
    if numel_w != 1:
        if input.ndim < 2 or input.shape[1] != numel_w:
            raise ValueError(
                f"prelu: parameter numbers must match channel count or be 1, got {numel_w} for input shape {list(input.shape)}"
            )
        w_shape = [1, numel_w] + [1] * (input.ndim - 2)
        w = weight.reshape(w_shape)
    else:
        w = weight
    return _ox_where(input > 0.0, input, input * w)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

# Largest-magnitude finite negative per dtype, used as the masked-attention
# fill value. finfo.min (not -inf) keeps full-masked softmax rows NaN-free
# (uniform) and stays representable in f16/bf16.
_SDPA_NEG_FILL = {
    "f16": -65504.0,
    "bf16": -3.3895314e38,
    "f32": -3.4028235e38,
    "f64": -1.7976931348623157e308,
}


def _apply_sdpa_keep_mask(scores: _Tensor, keep: _Tensor) -> _Tensor:
    """Applies a 0/1 keep mask to attention scores on the scores' device/dtype."""
    neg = _SDPA_NEG_FILL.get(str(scores.dtype), -1e9)
    k = keep
    try:
        k = k.to(scores.device)
    except Exception:
        pass
    try:
        if k.dtype != scores.dtype:
            k = k.to_dtype(scores.dtype)
    except Exception:
        pass
    return scores * k + (1.0 - k) * neg


def scaled_dot_product_attention(
    query: _Tensor,
    key: _Tensor,
    value: _Tensor,
    attn_mask: Optional[_Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> _Tensor:
    """Computes scaled dot-product attention with support for causal masking, additive/boolean masks, and dropout.

    Mathematically: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V.
    Supports (..., L, d_k) queries, (..., S, d_k) keys, and (..., S, d_v) values.
    """
    # Differentiable native path (stays in the autograd graph).
    scale_factor = (1.0 / math.sqrt(query.shape[-1])) if scale is None else float(scale)
    scores = query.matmul(key.transpose(-2, -1)) * scale_factor

    if is_causal:
        L = query.shape[-2]
        S = key.shape[-2]
        keep = _from_numpy((1.0 - np.triu(np.ones((L, S), dtype=np.float32), k=1)))
        scores = _apply_sdpa_keep_mask(scores, keep)

    if attn_mask is not None:
        m_np = _to_np(attn_mask)
        if m_np.dtype == bool or m_np.dtype == np.bool_:
            keep = _from_numpy(_contig(m_np, dtype=np.float32))
            scores = _apply_sdpa_keep_mask(scores, keep)
        elif isinstance(attn_mask, _Tensor):
            scores = scores + attn_mask
        else:
            scores = scores + _from_numpy(_contig(m_np, dtype=np.float32))

    attn_weights = scores.softmax(-1)

    if dropout_p > 0.0:
        attn_weights = dropout(attn_weights, dropout_p, True)

    return attn_weights.matmul(value)


# ---------------------------------------------------------------------------
# Linear & Projections
# ---------------------------------------------------------------------------

def linear(input: _Tensor, weight: _Tensor, bias: Optional[_Tensor] = None) -> _Tensor:
    """Applies a linear transformation to incoming data: y = x A^T + b."""
    out = input @ weight.t()
    if bias is not None:
        out = out + bias
    return out


def bilinear(input1: _Tensor, input2: _Tensor, weight: _Tensor, bias: Optional[_Tensor] = None) -> _Tensor:
    """Applies a bilinear transformation: y = x1 W x2 + b."""
    input1 = _as_tensor(input1)
    input2 = _as_tensor(input2)
    weight = _as_tensor(weight)
    s1, s2, sw = list(input1.shape), list(input2.shape), list(weight.shape)
    if len(s1) < 2 or len(s2) < 2 or len(sw) != 3:
        raise ValueError(
            f"bilinear expects x1 (*, in1), x2 (*, in2), w (out, in1, in2); got {s1}, {s2}, {sw}"
        )
    if s1[:-1] != s2[:-1]:
        raise ValueError(f"bilinear leading batch shapes must match: {s1[:-1]} vs {s2[:-1]}")
    in1, in2, out = s1[-1], s2[-1], sw[0]
    if sw[1] != in1 or sw[2] != in2:
        raise ValueError(f"bilinear feature dimension mismatch: x1 {s1}, x2 {s2}, w {sw}")
    batch_shape = s1[:-1]
    N = 1
    for d in batch_shape:
        N *= d
    x1_2d = input1.reshape([N, in1])
    x2_2d = input2.reshape([N, in2])
    outer = x1_2d.unsqueeze(-1) * x2_2d.unsqueeze(1)
    w_flat = weight.permute([1, 2, 0]).reshape([in1 * in2, out])
    res = outer.reshape([N, in1 * in2]).matmul(w_flat)
    if bias is not None:
        res = res + _as_tensor(bias)
    return res.reshape(batch_shape + [out])


def _check_dropout_p(p: float, name: str) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name}: dropout probability has to be between 0 and 1, but got {p}")


def dropout(input: _Tensor, p: float = 0.5, training: bool = True) -> _Tensor:
    """Randomly zeroes elements of the input tensor with probability p during training."""
    _check_dropout_p(p, "dropout")
    if not training or p <= 0.0:
        return input
    if p >= 1.0:
        return input * 0.0
    arr = input.to_numpy()
    mask = (np.random.rand(*arr.shape) > p).astype(np.float32) / (1.0 - p)
    mask_tensor = _from_numpy(mask)
    return input * mask_tensor


def _dropout_nd_mask(input: _Tensor, p: float, spatial_rank: int) -> _Tensor:
    """Channel-wise dropout mask broadcast over the trailing spatial dims."""
    shape = list(input.shape)
    mask_shape = shape[:2] + [1] * spatial_rank
    if p >= 1.0:
        mask = np.zeros(mask_shape, dtype=np.float32)
    else:
        mask = (np.random.rand(*mask_shape) > p).astype(np.float32) / (1.0 - p)
    res = _from_numpy(_contig(mask))
    if hasattr(input, "device"):
        try:
            res = res.to(input.device)
        except Exception:
            pass
    return res


def dropout1d(input: _Tensor, p: float = 0.5, training: bool = True) -> _Tensor:
    """Randomly zero out entire channels along 1D features."""
    _check_dropout_p(p, "dropout1d")
    if not training or p <= 0.0:
        return input
    if p >= 1.0:
        return input * 0.0
    if len(input.shape) == 3:
        return input * _dropout_nd_mask(input, p, 1)
    return dropout(input, p, training)


def dropout2d(input: _Tensor, p: float = 0.5, training: bool = True) -> _Tensor:
    """Randomly zero out entire channels along 2D spatial features."""
    _check_dropout_p(p, "dropout2d")
    if not training or p <= 0.0:
        return input
    if p >= 1.0:
        return input * 0.0
    if len(input.shape) == 4:
        return input * _dropout_nd_mask(input, p, 2)
    return dropout(input, p, training)


def dropout3d(input: _Tensor, p: float = 0.5, training: bool = True) -> _Tensor:
    """Randomly zero out entire channels along 3D volumetric features."""
    _check_dropout_p(p, "dropout3d")
    if not training or p <= 0.0:
        return input
    if p >= 1.0:
        return input * 0.0
    if len(input.shape) == 5:
        return input * _dropout_nd_mask(input, p, 3)
    return dropout(input, p, training)


def alpha_dropout(input: _Tensor, p: float = 0.5, training: bool = True) -> _Tensor:
    """Applies Alpha Dropout to preserve self-normalizing properties for SELU networks."""
    _check_dropout_p(p, "alpha_dropout")
    if not training or p <= 0.0:
        return input
    alpha = -1.7580993408473766
    a = (1.0 - p * (1.0 + alpha) ** 2) ** -0.5
    b = -a * alpha * p
    mask_np = (np.random.rand(*input.shape) > p).astype(np.float32)
    mask = _from_numpy(mask_np)
    res = (input * mask + (1.0 - mask) * float(alpha)) * float(a) + float(b)
    return res


def feature_alpha_dropout(input: _Tensor, p: float = 0.5, training: bool = True) -> _Tensor:
    """Randomly masks whole 2D channels with alpha dropout."""
    return alpha_dropout(input, p, training)


# ---------------------------------------------------------------------------
# Convolutions
# ---------------------------------------------------------------------------

def conv1d(
    input: _Tensor,
    weight: _Tensor,
    bias: Optional[_Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> _Tensor:
    """Applies a 1D convolution over an input signal composed of several input planes."""
    return _native_conv1d(input, weight, bias, stride, padding, dilation, groups)


def conv2d(
    input: _Tensor,
    weight: _Tensor,
    bias: Optional[_Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
) -> _Tensor:
    """Applies a 2D convolution over an input image composed of several input planes."""
    s = (stride, stride) if isinstance(stride, int) else tuple(stride)
    p = (padding, padding) if isinstance(padding, int) else tuple(padding)
    d = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    return _native_conv2d(input, weight, bias, s, p, d, groups)


def conv3d(
    input: _Tensor,
    weight: _Tensor,
    bias: Optional[_Tensor] = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: int = 1,
    groups: int = 1,
) -> _Tensor:
    """Applies a 3D convolution over an input volumetric image."""
    s3 = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
    p3 = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
    if (
        isinstance(input, _Tensor)
        and isinstance(weight, _Tensor)
        and hasattr(input, "conv3d")
        and dilation == 1
        and groups == 1
    ):
        return input.conv3d(weight, bias=bias, stride=s3, padding=p3)

    x = _to_np(input)
    w = _to_np(weight)
    if p3 != (0, 0, 0):
        x = np.pad(x, ((0, 0), (0, 0), (p3[0], p3[0]), (p3[1], p3[1]), (p3[2], p3[2])), mode="constant")
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, kD, kH, kW = w.shape
    eff_kd = (kD - 1) * dilation + 1
    eff_kh = (kH - 1) * dilation + 1
    eff_kw = (kW - 1) * dilation + 1

    windows = np.lib.stride_tricks.sliding_window_view(x, (eff_kd, eff_kh, eff_kw), axis=(2, 3, 4))
    windows = windows[:, :, ::s3[0], ::s3[1], ::s3[2], ::dilation, ::dilation, ::dilation]
    res = np.einsum("ncdhwijk,ocijk->nodhw", windows, w)
    if bias is not None:
        res = res + _to_np(bias)[None, :, None, None, None]
    return _wrap(res, like=input)


def conv_transpose1d(
    input: _Tensor,
    weight: _Tensor,
    bias: Optional[_Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
    dilation: int = 1,
) -> _Tensor:
    """Applies a 1D transposed convolution operator over an input signal."""
    # Differentiable masked accumulation (stays in the autograd graph).
    x = _as_tensor(input)
    w = _as_tensor(weight)
    N, C_in, L_in = (int(v) for v in x.shape)
    _, C_out, K = (int(v) for v in w.shape)
    L_out = (L_in - 1) * stride - 2 * padding + (K - 1) * dilation + output_padding + 1
    dt = _to_np(x).dtype
    out = _from_numpy(np.zeros((N, C_out * groups, L_out), dtype=dt))
    for i in range(L_in):
        start = i * stride - padding
        for k in range(K):
            pos = start + k * dilation
            if 0 <= pos < L_out:
                contrib = x[:, :, i].matmul(w[:, :, k])
                mask = np.zeros((L_out,), dtype=dt)
                mask[pos] = 1
                out = out + contrib[:, :, None] * _from_numpy(mask)
    if bias is not None:
        b = bias if isinstance(bias, _Tensor) else _from_numpy(_contig(_to_np(bias)))
        out = out + b.reshape([1, -1, 1])
    return out


def conv_transpose2d(
    input: _Tensor,
    weight: _Tensor,
    bias: Optional[_Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    output_padding: Union[int, Tuple[int, int]] = 0,
    groups: int = 1,
    dilation: int = 1,
) -> _Tensor:
    """Applies a 2D transposed convolution operator over an input image."""
    # Differentiable masked accumulation (stays in the autograd graph).
    s = (stride, stride) if isinstance(stride, int) else stride
    p = (padding, padding) if isinstance(padding, int) else padding
    op = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding

    x = _as_tensor(input)
    w = _as_tensor(weight)
    N, C_in, H_in, W_in = (int(v) for v in x.shape)
    _, C_out, kH, kW = (int(v) for v in w.shape)
    H_out = (H_in - 1) * s[0] - 2 * p[0] + (kH - 1) * dilation + op[0] + 1
    W_out = (W_in - 1) * s[1] - 2 * p[1] + (kW - 1) * dilation + op[1] + 1

    dt = _to_np(x).dtype
    out = _from_numpy(np.zeros((N, C_out * groups, H_out, W_out), dtype=dt))
    for h in range(H_in):
        for w_idx in range(W_in):
            h_start = h * s[0] - p[0]
            w_start = w_idx * s[1] - p[1]
            for kh in range(kH):
                for kw in range(kW):
                    pos_h = h_start + kh * dilation
                    pos_w = w_start + kw * dilation
                    if 0 <= pos_h < H_out and 0 <= pos_w < W_out:
                        contrib = x[:, :, h, w_idx].matmul(w[:, :, kh, kw])
                        mask = np.zeros((H_out, W_out), dtype=dt)
                        mask[pos_h, pos_w] = 1
                        out = out + contrib[:, :, None, None] * _from_numpy(mask)

    if bias is not None:
        b = bias if isinstance(bias, _Tensor) else _from_numpy(_contig(_to_np(bias)))
        out = out + b.reshape([1, -1, 1, 1])
    return out


def conv_transpose3d(
    input: _Tensor,
    weight: _Tensor,
    bias: Optional[_Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
    dilation: int = 1,
) -> _Tensor:
    """Applies a 3D transposed convolution operator over an input volumetric image."""
    # Differentiable masked accumulation (stays in the autograd graph).
    x = _as_tensor(input)
    w = _as_tensor(weight)
    N, C_in, D_in, H_in, W_in = (int(v) for v in x.shape)
    _, C_out, kD, kH, kW = (int(v) for v in w.shape)
    D_out = (D_in - 1) * stride - 2 * padding + (kD - 1) * dilation + output_padding + 1
    H_out = (H_in - 1) * stride - 2 * padding + (kH - 1) * dilation + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + (kW - 1) * dilation + output_padding + 1

    dt = _to_np(x).dtype
    out = _from_numpy(np.zeros((N, C_out * groups, D_out, H_out, W_out), dtype=dt))
    for d in range(D_in):
        for h in range(H_in):
            for w_idx in range(W_in):
                d_pos = d * stride - padding
                h_pos = h * stride - padding
                w_pos = w_idx * stride - padding
                for kd in range(kD):
                    for kh in range(kH):
                        for kw in range(kW):
                            pd = d_pos + kd * dilation
                            ph = h_pos + kh * dilation
                            pw = w_pos + kw * dilation
                            if 0 <= pd < D_out and 0 <= ph < H_out and 0 <= pw < W_out:
                                contrib = x[:, :, d, h, w_idx].matmul(w[:, :, kd, kh, kw])
                                mask = np.zeros((D_out, H_out, W_out), dtype=dt)
                                mask[pd, ph, pw] = 1
                                out = out + contrib[:, :, None, None, None] * _from_numpy(mask)
    if bias is not None:
        b = bias if isinstance(bias, _Tensor) else _from_numpy(_contig(_to_np(bias)))
        out = out + b.reshape([1, -1, 1, 1, 1])
    return out


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def max_pool1d(input: _Tensor, kernel_size: int, stride: Optional[int] = None, padding: int = 0) -> _Tensor:
    """Applies a 1D max pooling over an input signal."""
    # Differentiable window loop (stays in the autograd graph).
    stride = kernel_size if stride is None else stride
    x = _diff_pad_nd(_as_tensor(input), [(padding, padding)], value=float("-inf"))
    n, c, length = (int(v) for v in x.shape)
    out_l = (length - kernel_size) // stride + 1
    cols = []
    for i in range(out_l):
        win = x[:, :, i * stride : i * stride + kernel_size]
        cols.append(win.reshape([n, c, -1]).max(dim=-1).values)
    return _native_stack(cols, dim=-1)


def _pool2d_out_dim(size: int, k: int, s: int, p: int, ceil_mode: bool) -> int:
    if ceil_mode:
        out = (size + 2 * p - k + s - 1) // s + 1
        if (out - 1) * s >= size + p:
            out -= 1
        return max(out, 0)
    return (size + 2 * p - k) // s + 1


def max_pool2d(
    input: _Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
) -> _Tensor:
    """Applies a 2D max pooling over an input image."""
    # Differentiable window loop (stays in the autograd graph).
    k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s = k if stride is None else ((stride, stride) if isinstance(stride, int) else stride)
    p = (padding, padding) if isinstance(padding, int) else padding

    orig = _as_tensor(input)
    n, c, h0, w0 = (int(v) for v in orig.shape)
    kh, kw = k
    oh = _pool2d_out_dim(h0, kh, s[0], p[0], ceil_mode)
    ow = _pool2d_out_dim(w0, kw, s[1], p[1], ceil_mode)
    x = _diff_pad_nd(orig, [(p[0], p[0]), (p[1], p[1])], value=float("-inf"))
    _, _, h, w = (int(v) for v in x.shape)
    rows = []
    for i in range(oh):
        hs, he = i * s[0], min(i * s[0] + kh, h)
        cols = []
        for j in range(ow):
            ws, we = j * s[1], min(j * s[1] + kw, w)
            win = x[:, :, hs:he, ws:we]
            cols.append(win.reshape([n, c, -1]).max(dim=-1).values)
        rows.append(_native_stack(cols, dim=-1))
    return _native_stack(rows, dim=2)


def max_pool3d(
    input: _Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Optional[Union[int, Tuple[int, int, int]]] = None,
    padding: int = 0,
) -> _Tensor:
    """Applies a 3D max pooling over an input volume."""
    # Differentiable window loop (stays in the autograd graph).
    k = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s = k if stride is None else ((stride, stride, stride) if isinstance(stride, int) else stride)

    x = _diff_pad_nd(_as_tensor(input), [(padding, padding)] * 3, value=float("-inf"))
    n, c = int(x.shape[0]), int(x.shape[1])
    kd, kh, kw = k
    od = (int(x.shape[2]) - kd) // s[0] + 1
    oh = (int(x.shape[3]) - kh) // s[1] + 1
    ow = (int(x.shape[4]) - kw) // s[2] + 1
    vols = []
    for d in range(od):
        rows = []
        for i in range(oh):
            cols = []
            for j in range(ow):
                win = x[
                    :, :,
                    d * s[0] : d * s[0] + kd,
                    i * s[1] : i * s[1] + kh,
                    j * s[2] : j * s[2] + kw,
                ]
                cols.append(win.reshape([n, c, -1]).max(dim=-1).values)
            rows.append(_native_stack(cols, dim=-1))
        vols.append(_native_stack(rows, dim=2))
    return _native_stack(vols, dim=2)


def fractional_max_pool2d(
    input: _Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    output_size: Optional[Union[int, Tuple[int, int]]] = None,
    output_ratio: Optional[Union[float, Tuple[float, float]]] = None,
    return_indices: bool = False,
    _random_samples: Optional[_Tensor] = None,
) -> Union[_Tensor, Tuple[_Tensor, _Tensor]]:
    """Applies 2D fractional max pooling over an input signal."""
    x = _as_tensor(input)
    n, c, h, w = (int(v) for v in x.shape)
    k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    kh, kw = k

    if output_size is not None:
        out_s = (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)
        oh, ow = int(out_s[0]), int(out_s[1])
    elif output_ratio is not None:
        out_r = (output_ratio, output_ratio) if isinstance(output_ratio, (int, float)) else tuple(output_ratio)
        oh, ow = int(h * out_r[0]), int(w * out_r[1])
    else:
        raise ValueError("fractional_max_pool2d requires either output_size or output_ratio")

    if oh <= 0 or ow <= 0:
        raise ValueError(f"output_size must be positive, got ({oh}, {ow})")

    h_starts = [int(i * (h - kh) / (oh - 1)) if oh > 1 else 0 for i in range(oh)]
    w_starts = [int(j * (w - kw) / (ow - 1)) if ow > 1 else 0 for j in range(ow)]

    rows = []
    for s_h in h_starts:
        cols = []
        for s_w in w_starts:
            win = x[:, :, s_h : s_h + kh, s_w : s_w + kw]
            cols.append(win.reshape([n, c, -1]).max(dim=-1).values)
        rows.append(_native_stack(cols, dim=-1))
    out = _native_stack(rows, dim=2)
    if return_indices:
        indices = _Tensor.zeros(out.shape, dtype="int64")
        return out, indices
    return out


def fractional_max_pool3d(
    input: _Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    output_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    output_ratio: Optional[Union[float, Tuple[float, float, float]]] = None,
    return_indices: bool = False,
    _random_samples: Optional[_Tensor] = None,
) -> Union[_Tensor, Tuple[_Tensor, _Tensor]]:
    """Applies 3D fractional max pooling over an input signal."""
    x = _as_tensor(input)
    n, c, d, h, w = (int(v) for v in x.shape)
    k = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    kd, kh, kw = k

    if output_size is not None:
        out_s = (output_size, output_size, output_size) if isinstance(output_size, int) else tuple(output_size)
        od, oh, ow = int(out_s[0]), int(out_s[1]), int(out_s[2])
    elif output_ratio is not None:
        out_r = (output_ratio, output_ratio, output_ratio) if isinstance(output_ratio, (int, float)) else tuple(output_ratio)
        od, oh, ow = int(d * out_r[0]), int(h * out_r[1]), int(w * out_r[2])
    else:
        raise ValueError("fractional_max_pool3d requires either output_size or output_ratio")

    if od <= 0 or oh <= 0 or ow <= 0:
        raise ValueError(f"output_size must be positive, got ({od}, {oh}, {ow})")

    d_starts = [int(i * (d - kd) / (od - 1)) if od > 1 else 0 for i in range(od)]
    h_starts = [int(i * (h - kh) / (oh - 1)) if oh > 1 else 0 for i in range(oh)]
    w_starts = [int(j * (w - kw) / (ow - 1)) if ow > 1 else 0 for j in range(ow)]

    vols = []
    for s_d in d_starts:
        rows = []
        for s_h in h_starts:
            cols = []
            for s_w in w_starts:
                win = x[:, :, s_d : s_d + kd, s_h : s_h + kh, s_w : s_w + kw]
                cols.append(win.reshape([n, c, -1]).max(dim=-1).values)
            rows.append(_native_stack(cols, dim=-1))
        vols.append(_native_stack(rows, dim=2))
    out = _native_stack(vols, dim=2)
    if return_indices:
        indices = _Tensor.zeros(out.shape, dtype="int64")
        return out, indices
    return out


def max_unpool1d(
    input: _Tensor,
    indices: _Tensor,
    kernel_size: Union[int, Tuple[int]],
    stride: Optional[Union[int, Tuple[int]]] = None,
    padding: Union[int, Tuple[int]] = 0,
    output_size: Optional[Sequence[int]] = None,
) -> _Tensor:
    """Computes a partial inverse of MaxPool1d."""
    k = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
    s = k if stride is None else ((stride,) if isinstance(stride, int) else tuple(stride))
    p = (padding,) if isinstance(padding, int) else tuple(padding)

    input = _as_tensor(input)
    indices = _as_tensor(indices)
    N, C, L_in = input.shape
    if output_size is not None:
        out_shape = list(output_size)
        L_out = out_shape[-1]
    else:
        L_out = (L_in - 1) * s[0] - 2 * p[0] + k[0]
        out_shape = [N, C, L_out]

    out_flat = _Tensor.zeros((N, C, L_out), dtype=input.dtype, device=input.device)
    src = input.reshape((N, C, -1))
    idx = indices.reshape((N, C, -1))
    unpooled_flat = out_flat.scatter(2, idx, src)
    return unpooled_flat.reshape(out_shape)


def max_unpool2d(
    input: _Tensor,
    indices: _Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    output_size: Optional[Sequence[int]] = None,
) -> _Tensor:
    """Computes a partial inverse of MaxPool2d."""
    k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    s = k if stride is None else ((stride, stride) if isinstance(stride, int) else tuple(stride))
    p = (padding, padding) if isinstance(padding, int) else tuple(padding)

    input = _as_tensor(input)
    indices = _as_tensor(indices)
    N, C, H_in, W_in = input.shape
    if output_size is not None:
        out_shape = list(output_size)
        H_out, W_out = out_shape[-2], out_shape[-1]
    else:
        H_out = (H_in - 1) * s[0] - 2 * p[0] + k[0]
        W_out = (W_in - 1) * s[1] - 2 * p[1] + k[1]
        out_shape = [N, C, H_out, W_out]

    flat_len = H_out * W_out
    out_flat = _Tensor.zeros((N, C, flat_len), dtype=input.dtype, device=input.device)
    src = input.reshape((N, C, -1))
    idx = indices.reshape((N, C, -1))
    unpooled_flat = out_flat.scatter(2, idx, src)
    return unpooled_flat.reshape(out_shape)


def max_unpool3d(
    input: _Tensor,
    indices: _Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Optional[Union[int, Tuple[int, int, int]]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    output_size: Optional[Sequence[int]] = None,
) -> _Tensor:
    """Computes a partial inverse of MaxPool3d."""
    k = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    s = k if stride is None else ((stride, stride, stride) if isinstance(stride, int) else tuple(stride))
    p = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)

    input = _as_tensor(input)
    indices = _as_tensor(indices)
    N, C, D_in, H_in, W_in = input.shape
    if output_size is not None:
        out_shape = list(output_size)
        D_out, H_out, W_out = out_shape[-3], out_shape[-2], out_shape[-1]
    else:
        D_out = (D_in - 1) * s[0] - 2 * p[0] + k[0]
        H_out = (H_in - 1) * s[1] - 2 * p[1] + k[1]
        W_out = (W_in - 1) * s[2] - 2 * p[2] + k[2]
        out_shape = [N, C, D_out, H_out, W_out]

    flat_len = D_out * H_out * W_out
    out_flat = _Tensor.zeros((N, C, flat_len), dtype=input.dtype, device=input.device)
    src = input.reshape((N, C, -1))
    idx = indices.reshape((N, C, -1))
    unpooled_flat = out_flat.scatter(2, idx, src)
    return unpooled_flat.reshape(out_shape)


def avg_pool1d(input: _Tensor, kernel_size: int, stride: Optional[int] = None, padding: int = 0) -> _Tensor:
    """Applies a 1D average pooling over an input signal."""
    # Differentiable window loop (stays in the autograd graph).
    stride = kernel_size if stride is None else stride
    x = _diff_pad_nd(_as_tensor(input), [(padding, padding)])
    n, c, length = (int(v) for v in x.shape)
    out_l = (length - kernel_size) // stride + 1
    cols = []
    for i in range(out_l):
        win = x[:, :, i * stride : i * stride + kernel_size]
        cols.append(win.reshape([n, c, -1]).mean(dim=-1))
    return _native_stack(cols, dim=-1)


def avg_pool2d(
    input: _Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> _Tensor:
    """Applies a 2D average pooling over an input image."""
    # Differentiable window loop (stays in the autograd graph).
    k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s = k if stride is None else ((stride, stride) if isinstance(stride, int) else stride)
    p = (padding, padding) if isinstance(padding, int) else padding

    orig = _as_tensor(input)
    n, c, h0, w0 = (int(v) for v in orig.shape)
    kh, kw = k
    oh = _pool2d_out_dim(h0, kh, s[0], p[0], ceil_mode)
    ow = _pool2d_out_dim(w0, kw, s[1], p[1], ceil_mode)
    src = orig if not count_include_pad else _diff_pad_nd(orig, [(p[0], p[0]), (p[1], p[1])])
    _, _, h, w = (int(v) for v in src.shape)
    rows = []
    for i in range(oh):
        if count_include_pad:
            hs, he = i * s[0], min(i * s[0] + kh, h)
        else:
            hs, he = max(0, i * s[0] - p[0]), min(h0, i * s[0] - p[0] + kh)
        cols = []
        for j in range(ow):
            if count_include_pad:
                ws, we = j * s[1], min(j * s[1] + kw, w)
            else:
                ws, we = max(0, j * s[1] - p[1]), min(w0, j * s[1] - p[1] + kw)
            win = src[:, :, hs:he, ws:we]
            cols.append(win.reshape([n, c, -1]).mean(dim=-1))
        rows.append(_native_stack(cols, dim=-1))
    return _native_stack(rows, dim=2)


def avg_pool3d(
    input: _Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Optional[Union[int, Tuple[int, int, int]]] = None,
    padding: int = 0,
) -> _Tensor:
    """Applies a 3D average pooling over an input volume."""
    # Differentiable window loop (stays in the autograd graph).
    k = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s = k if stride is None else ((stride, stride, stride) if isinstance(stride, int) else stride)

    x = _diff_pad_nd(_as_tensor(input), [(padding, padding)] * 3)
    n, c = int(x.shape[0]), int(x.shape[1])
    kd, kh, kw = k
    od = (int(x.shape[2]) - kd) // s[0] + 1
    oh = (int(x.shape[3]) - kh) // s[1] + 1
    ow = (int(x.shape[4]) - kw) // s[2] + 1
    vols = []
    for d in range(od):
        rows = []
        for i in range(oh):
            cols = []
            for j in range(ow):
                win = x[
                    :, :,
                    d * s[0] : d * s[0] + kd,
                    i * s[1] : i * s[1] + kh,
                    j * s[2] : j * s[2] + kw,
                ]
                cols.append(win.reshape([n, c, -1]).mean(dim=-1))
            rows.append(_native_stack(cols, dim=-1))
        vols.append(_native_stack(rows, dim=2))
    return _native_stack(vols, dim=2)


def adaptive_avg_pool1d(input: _Tensor, output_size: int) -> _Tensor:
    """Applies a 1D adaptive average pooling over an input signal."""
    # Differentiable window loop (stays in the autograd graph).
    x = _as_tensor(input)
    n, c = int(x.shape[0]), int(x.shape[1])
    L_in = int(x.shape[-1])
    L_out = output_size
    cols = []
    for i in range(L_out):
        start = int(math.floor(i * L_in / L_out))
        end = int(math.ceil((i + 1) * L_in / L_out))
        win = x[:, :, start:end]
        cols.append(win.reshape([n, c, -1]).mean(dim=-1))
    return _native_stack(cols, dim=-1)


def adaptive_avg_pool2d(input: _Tensor, output_size: Union[int, Tuple[int, int]]) -> _Tensor:
    """Applies a 2D adaptive average pooling over an input signal."""
    # Differentiable window loop (stays in the autograd graph).
    o = (output_size, output_size) if isinstance(output_size, int) else output_size
    x = _as_tensor(input)
    n, c = int(x.shape[0]), int(x.shape[1])
    H_in, W_in = int(x.shape[-2]), int(x.shape[-1])
    H_out, W_out = o
    rows = []
    for h in range(H_out):
        h_start = int(math.floor(h * H_in / H_out))
        h_end = int(math.ceil((h + 1) * H_in / H_out))
        cols = []
        for w in range(W_out):
            w_start = int(math.floor(w * W_in / W_out))
            w_end = int(math.ceil((w + 1) * W_in / W_out))
            win = x[:, :, h_start:h_end, w_start:w_end]
            cols.append(win.reshape([n, c, -1]).mean(dim=-1))
        rows.append(_native_stack(cols, dim=-1))
    return _native_stack(rows, dim=2)


def adaptive_avg_pool3d(input: _Tensor, output_size: Union[int, Tuple[int, int, int]]) -> _Tensor:
    """Applies a 3D adaptive average pooling over an input volume."""
    # Differentiable window loop (stays in the autograd graph).
    o = (output_size, output_size, output_size) if isinstance(output_size, int) else output_size
    x = _as_tensor(input)
    n, c = int(x.shape[0]), int(x.shape[1])
    D_in, H_in, W_in = int(x.shape[-3]), int(x.shape[-2]), int(x.shape[-1])
    D_out, H_out, W_out = o
    vols = []
    for d in range(D_out):
        d_start = int(math.floor(d * D_in / D_out))
        d_end = int(math.ceil((d + 1) * D_in / D_out))
        rows = []
        for h in range(H_out):
            h_start = int(math.floor(h * H_in / H_out))
            h_end = int(math.ceil((h + 1) * H_in / H_out))
            cols = []
            for w in range(W_out):
                w_start = int(math.floor(w * W_in / W_out))
                w_end = int(math.ceil((w + 1) * W_in / W_out))
                win = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                cols.append(win.reshape([n, c, -1]).mean(dim=-1))
            rows.append(_native_stack(cols, dim=-1))
        vols.append(_native_stack(rows, dim=2))
    return _native_stack(vols, dim=2)


def adaptive_max_pool1d(input: _Tensor, output_size: int) -> _Tensor:
    """Applies a 1D adaptive max pooling over an input signal."""
    if output_size <= 0:
        raise ValueError(f"adaptive_max_pool1d: output_size must be positive, got {output_size}")
    x = _as_tensor(input)
    n, c = int(x.shape[0]), int(x.shape[1])
    L_in = int(x.shape[-1])
    L_out = output_size
    cols = []
    for i in range(L_out):
        start = int(math.floor(i * L_in / L_out))
        end = int(math.ceil((i + 1) * L_in / L_out))
        win = x[:, :, start:end]
        cols.append(win.reshape([n, c, -1]).max(dim=-1).values)
    return _native_stack(cols, dim=-1)


def adaptive_max_pool2d(input: _Tensor, output_size: Union[int, Tuple[int, int]]) -> _Tensor:
    """Applies a 2D adaptive max pooling over an input image."""
    o = (output_size, output_size) if isinstance(output_size, int) else output_size
    if o[0] <= 0 or o[1] <= 0:
        raise ValueError(f"adaptive_max_pool2d: output_size must be positive, got {o}")
    x = _as_tensor(input)
    n, c = int(x.shape[0]), int(x.shape[1])
    H_in, W_in = int(x.shape[-2]), int(x.shape[-1])
    H_out, W_out = o
    rows = []
    for h in range(H_out):
        h_start = int(math.floor(h * H_in / H_out))
        h_end = int(math.ceil((h + 1) * H_in / H_out))
        cols = []
        for w in range(W_out):
            w_start = int(math.floor(w * W_in / W_out))
            w_end = int(math.ceil((w + 1) * W_in / W_out))
            win = x[:, :, h_start:h_end, w_start:w_end]
            cols.append(win.reshape([n, c, -1]).max(dim=-1).values)
        rows.append(_native_stack(cols, dim=-1))
    return _native_stack(rows, dim=2)


def adaptive_max_pool3d(input: _Tensor, output_size: Union[int, Tuple[int, int, int]]) -> _Tensor:
    """Applies a 3D adaptive max pooling over an input volume."""
    o = (output_size, output_size, output_size) if isinstance(output_size, int) else output_size
    if o[0] <= 0 or o[1] <= 0 or o[2] <= 0:
        raise ValueError(f"adaptive_max_pool3d: output_size must be positive, got {o}")
    x = _as_tensor(input)
    n, c = int(x.shape[0]), int(x.shape[1])
    D_in, H_in, W_in = int(x.shape[-3]), int(x.shape[-2]), int(x.shape[-1])
    D_out, H_out, W_out = o
    vols = []
    for d in range(D_out):
        d_start = int(math.floor(d * D_in / D_out))
        d_end = int(math.ceil((d + 1) * D_in / D_out))
        rows = []
        for h in range(H_out):
            h_start = int(math.floor(h * H_in / H_out))
            h_end = int(math.ceil((h + 1) * H_in / H_out))
            cols = []
            for w in range(W_out):
                w_start = int(math.floor(w * W_in / W_out))
                w_end = int(math.ceil((w + 1) * W_in / W_out))
                win = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                cols.append(win.reshape([n, c, -1]).max(dim=-1).values)
            rows.append(_native_stack(cols, dim=-1))
        vols.append(_native_stack(rows, dim=2))
    return _native_stack(vols, dim=2)


def lp_pool1d(input: _Tensor, norm_type: float, kernel_size: int, stride: Optional[int] = None) -> _Tensor:
    """Applies a 1D power-average pooling over an input signal."""
    if kernel_size <= 0:
        raise ValueError(f"lp_pool1d: kernel_size must be positive, got {kernel_size}")
    stride = kernel_size if stride is None else stride
    if stride <= 0:
        raise ValueError(f"lp_pool1d: stride must be positive, got {stride}")
    if norm_type == 0:
        raise ValueError("lp_pool1d: norm_type must be non-zero")
    x = _as_tensor(input)
    L_in = int(x.shape[-1])
    cols = []
    p = float(norm_type)
    for i in range(0, L_in - kernel_size + 1, stride):
        win = x.narrow(2, i, kernel_size)
        cols.append(win.sum(dim=-1) if p == 1.0 else (win.abs().pow(p).sum(dim=-1)).pow(1.0 / p))
    return _native_stack(cols, dim=-1)


def lp_pool2d(
    input: _Tensor,
    norm_type: float,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
) -> _Tensor:
    """Applies a 2D power-average pooling over an input image."""
    k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s = k if stride is None else ((stride, stride) if isinstance(stride, int) else stride)
    if k[0] <= 0 or k[1] <= 0:
        raise ValueError(f"lp_pool2d: kernel_size must be positive, got {k}")
    if s[0] <= 0 or s[1] <= 0:
        raise ValueError(f"lp_pool2d: stride must be positive, got {s}")
    if norm_type == 0:
        raise ValueError("lp_pool2d: norm_type must be non-zero")
    x = _as_tensor(input)
    H_in, W_in = int(x.shape[-2]), int(x.shape[-1])
    p = float(norm_type)
    rows = []
    for h in range(0, H_in - k[0] + 1, s[0]):
        cols = []
        for w in range(0, W_in - k[1] + 1, s[1]):
            win = x.narrow(2, h, k[0]).narrow(3, w, k[1])
            cols.append(win.sum(dim=-1).sum(dim=-1) if p == 1.0 else (win.abs().pow(p).sum(dim=-1).sum(dim=-1)).pow(1.0 / p))
        rows.append(_native_stack(cols, dim=-1))
    return _native_stack(rows, dim=-2)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _update_running_stats(running_mean, running_var, batch_mean_c, batch_var_c, momentum):
    """Exponential-moving-average update of batch/instance running stats in place."""
    if running_mean is None or running_var is None:
        return
    bm = batch_mean_c.detach()
    bv = batch_var_c.detach()
    if hasattr(running_mean, "copy_"):
        running_mean.copy_(running_mean.detach() * (1.0 - momentum) + bm * momentum)
        running_var.copy_(running_var.detach() * (1.0 - momentum) + bv * momentum)


def batch_norm(
    input: _Tensor,
    running_mean: Optional[_Tensor] = None,
    running_var: Optional[_Tensor] = None,
    weight: Optional[_Tensor] = None,
    bias: Optional[_Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> _Tensor:
    """Applies Batch Normalization over a 2D, 3D, 4D, or 5D input."""
    shape = list(input.shape)
    if len(shape) < 2:
        raise ValueError(f"batch_norm expects at least 2D input, got shape {shape}")
    c = shape[1]
    bshape = [1, c] + [1] * (len(shape) - 2)

    if training or running_mean is None or running_var is None:
        # Native differentiable path: flatten all but the channel dim.
        flat = input.transpose(0, 1).reshape([c, -1])
        count = flat.shape[-1]
        mean = flat.mean(dim=-1, keepdim=True)
        var = ((flat - mean) * (flat - mean)).mean(dim=-1, keepdim=True)
        if running_mean is not None and running_var is not None:
            unbiased = var * (count / (count - 1)) if count > 1 else var
            _update_running_stats(
                running_mean, running_var,
                mean.reshape([c]), unbiased.reshape([c]), momentum,
            )
        mean_b = mean.reshape(bshape)
        var_b = var.reshape(bshape)
    else:
        mean_b = running_mean.reshape(bshape)
        var_b = running_var.reshape(bshape)

    norm_x = (input - mean_b) / (var_b + eps).sqrt()
    if weight is not None:
        norm_x = norm_x * _as_tensor(weight).reshape(bshape)
    if bias is not None:
        norm_x = norm_x + _as_tensor(bias).reshape(bshape)
    return norm_x


def layer_norm(
    input: _Tensor,
    normalized_shape: Union[int, Sequence[int]],
    weight: Optional[_Tensor] = None,
    bias: Optional[_Tensor] = None,
    eps: float = 1e-5,
) -> _Tensor:
    """Applies Layer Normalization over the specified trailing dimensions."""
    # Differentiable native path (stays in the autograd graph).
    shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    k = len(shape)
    n = 1
    for d in shape:
        n *= d
    flat = input.reshape([-1, n])
    mean = flat.mean(dim=-1, keepdim=True)
    var = ((flat - mean) * (flat - mean)).mean(dim=-1, keepdim=True)
    normed = (flat - mean) / (var + eps).sqrt()
    norm_x = normed.reshape(list(input.shape))
    if weight is not None:
        w = weight if isinstance(weight, _Tensor) else _from_numpy(_contig(_to_np(weight)))
        norm_x = norm_x * w
    if bias is not None:
        b = bias if isinstance(bias, _Tensor) else _from_numpy(_contig(_to_np(bias)))
        norm_x = norm_x + b
    return norm_x


def group_norm(
    input: _Tensor,
    num_groups: int,
    weight: Optional[_Tensor] = None,
    bias: Optional[_Tensor] = None,
    eps: float = 1e-5,
) -> _Tensor:
    """Applies Group Normalization over mini-batches of inputs."""
    shape = list(input.shape)
    if len(shape) < 2:
        raise ValueError(f"group_norm expects at least 2D input, got shape {shape}")
    if num_groups < 1:
        raise ValueError(f"group_norm num_groups must be positive, got {num_groups}")
    n, c = shape[0], shape[1]
    if c % num_groups != 0:
        raise ValueError(f"Number of channels {c} must be divisible by num_groups {num_groups}")
    # Native differentiable path.
    grouped = input.reshape([n, num_groups, -1])
    mean = grouped.mean(dim=-1, keepdim=True)
    var = ((grouped - mean) * (grouped - mean)).mean(dim=-1, keepdim=True)
    res = ((grouped - mean) / (var + eps).sqrt()).reshape(shape)
    bshape = [1, c] + [1] * (len(shape) - 2)
    if weight is not None:
        res = res * _as_tensor(weight).reshape(bshape)
    if bias is not None:
        res = res + _as_tensor(bias).reshape(bshape)
    return res


def instance_norm(
    input: _Tensor,
    running_mean: Optional[_Tensor] = None,
    running_var: Optional[_Tensor] = None,
    weight: Optional[_Tensor] = None,
    bias: Optional[_Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> _Tensor:
    """Applies Instance Normalization over each channel in each data sample."""
    shape = list(input.shape)
    if len(shape) < 3:
        raise ValueError(f"instance_norm expects at least 3D input, got shape {shape}")
    n, c = shape[0], shape[1]
    if use_input_stats:
        # Native differentiable path: normalize each (sample, channel) instance.
        flat = input.reshape([n * c, -1])
        count = flat.shape[-1]
        mean = flat.mean(dim=-1, keepdim=True)
        var = ((flat - mean) * (flat - mean)).mean(dim=-1, keepdim=True)
        if running_mean is not None and running_var is not None:
            unbiased = var * (count / (count - 1)) if count > 1 else var
            per_channel_mean = mean.reshape([n, c]).mean(dim=0)
            per_channel_var = unbiased.reshape([n, c]).mean(dim=0)
            _update_running_stats(
                running_mean, running_var, per_channel_mean, per_channel_var, momentum,
            )
        tail = [1] * (len(shape) - 2)
        mean_b = mean.reshape([n, c] + tail)
        var_b = var.reshape([n, c] + tail)
    else:
        if running_mean is None or running_var is None:
            raise ValueError("instance_norm: running stats are required when use_input_stats=False")
        bshape = [1, c] + [1] * (len(shape) - 2)
        mean_b = running_mean.reshape(bshape)
        var_b = running_var.reshape(bshape)
    norm_x = (input - mean_b) / (var_b + eps).sqrt()
    bshape = [1, c] + [1] * (len(shape) - 2)
    if weight is not None:
        norm_x = norm_x * _as_tensor(weight).reshape(bshape)
    if bias is not None:
        norm_x = norm_x + _as_tensor(bias).reshape(bshape)
    return norm_x


def local_response_norm(input: _Tensor, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0) -> _Tensor:
    """Applies Local Response Normalization over an input image."""
    if size < 1:
        raise ValueError(f"local_response_norm: size must be greater than 0, but got: {size}")
    x = _as_tensor(input)
    if x.ndim < 3:
        raise ValueError(f"local_response_norm: input must have at least 3 dimensions, got {x.ndim}")
    C = int(x.shape[1])
    sq = x * x
    sq_channels = [sq.narrow(1, c, 1) for c in range(C)]
    scales = []
    for c in range(C):
        c_start = max(0, c - size // 2)
        c_end = min(C - 1, c + (size - 1) // 2)
        cur_sum = sq_channels[c_start]
        for ci in range(c_start + 1, c_end + 1):
            cur_sum = cur_sum + sq_channels[ci]
        scale_c = (cur_sum * (alpha / size) + k).pow(beta)
        scales.append(scale_c)
    scale = _native_cat(scales, dim=1)
    return x / scale


def normalize(input: _Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12, out: Optional[_Tensor] = None) -> _Tensor:
    """Performs L_p normalization of inputs over specified dimension."""
    x = _as_tensor(input)
    norm = x.norm(p, dim=dim, keepdim=True).clamp(min=eps)
    res = x / norm
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


# ---------------------------------------------------------------------------
# Spatial & Vision Transforms
# ---------------------------------------------------------------------------

def pad(input: _Tensor, pad: Sequence[int], mode: str = "constant", value: float = 0.0) -> _Tensor:
    """Pads tensor with pad tuple (pad_left, pad_right, pad_top, pad_bottom, ...) over trailing dimensions."""
    if len(pad) % 2 != 0:
        raise ValueError(f"pad: length must be even, got {len(pad)}")
    if any(p < 0 for p in pad):
        raise ValueError("pad: padding must be non-negative")
    if mode not in ("constant", "reflect", "replicate", "circular"):
        raise ValueError(f"pad: unknown mode {mode!r}")
    cur = _as_tensor(input)
    ndim = cur.ndim
    num_pairs = len(pad) // 2
    if num_pairs > ndim:
        raise ValueError(f"pad: padding length {len(pad)} exceeds tensor rank {ndim}")
    for i in range(num_pairs):
        pl = pad[2 * i]
        pr = pad[2 * i + 1]
        axis = ndim - 1 - i
        dim_len = cur.shape[axis]
        if pl == 0 and pr == 0:
            continue
        if mode == "constant":
            parts = []
            if pl > 0:
                s_l = list(cur.shape)
                s_l[axis] = pl
                parts.append(_from_numpy(np.full(s_l, value, dtype=np.float32)).to_dtype(cur.dtype).to(cur.device))
            parts.append(cur)
            if pr > 0:
                s_r = list(cur.shape)
                s_r[axis] = pr
                parts.append(_from_numpy(np.full(s_r, value, dtype=np.float32)).to_dtype(cur.dtype).to(cur.device))
            cur = _native_cat(parts, dim=axis)
        elif mode == "replicate":
            idx = [0] * pl + list(range(dim_len)) + [dim_len - 1] * pr
            cur = cur.index_select(axis, _from_numpy(np.array(idx, dtype=np.int64)))
        elif mode == "reflect":
            if pl >= dim_len or pr >= dim_len:
                raise ValueError("pad: padding size cannot be greater than or equal to dimension size for reflect")
            idx_l = list(range(pl, 0, -1))
            idx_r = list(range(dim_len - 2, dim_len - 2 - pr, -1))
            idx = idx_l + list(range(dim_len)) + idx_r
            cur = cur.index_select(axis, _from_numpy(np.array(idx, dtype=np.int64)))
        elif mode == "circular":
            idx = [(j) % dim_len for j in range(-pl, dim_len + pr)]
            cur = cur.index_select(axis, _from_numpy(np.array(idx, dtype=np.int64)))
    return cur


def pixel_shuffle(input: _Tensor, upscale_factor: int) -> _Tensor:
    """Rearranges elements in a tensor of shape (N, C * r^2, H, W) to (N, C, H * r, W * r)."""
    r = int(upscale_factor)
    if r <= 0:
        raise ValueError(f"pixel_shuffle: upscale_factor must be positive, got {r}")
    x = _as_tensor(input)
    if x.ndim != 4:
        raise ValueError(f"pixel_shuffle: input must be 4D (N, C, H, W), got {x.ndim}D")
    N, C_r2, H, W = x.shape
    r2 = r * r
    if C_r2 % r2 != 0:
        raise ValueError(f"pixel_shuffle: channels ({C_r2}) must be divisible by {r}^2={r2}")
    C = C_r2 // r2
    reshaped = x.reshape([N, C, r, r, H, W])
    transposed = reshaped.permute([0, 1, 4, 2, 5, 3])
    return transposed.reshape([N, C, H * r, W * r])


def pixel_unshuffle(input: _Tensor, downscale_factor: int) -> _Tensor:
    """Rearranges elements in a tensor of shape (N, C, H * r, W * r) to (N, C * r^2, H, W)."""
    r = int(downscale_factor)
    if r <= 0:
        raise ValueError(f"pixel_unshuffle: downscale_factor must be positive, got {r}")
    x = _as_tensor(input)
    if x.ndim != 4:
        raise ValueError(f"pixel_unshuffle: input must be 4D (N, C, H, W), got {x.ndim}D")
    N, C, H_r, W_r = x.shape
    if H_r % r != 0 or W_r % r != 0:
        raise ValueError(f"pixel_unshuffle: spatial dims ({H_r}, {W_r}) must be divisible by {r}")
    H = H_r // r
    W = W_r // r
    reshaped = x.reshape([N, C, H, r, W, r])
    transposed = reshaped.permute([0, 1, 3, 5, 2, 4])
    return transposed.reshape([N, C * r * r, H, W])


def channel_shuffle(input: _Tensor, groups: int) -> _Tensor:
    """Rearranges channels in a tensor of shape (N, C, H, W) into (N, groups, C // groups, H, W) then transposed."""
    g = int(groups)
    if g <= 0:
        raise ValueError(f"channel_shuffle: groups must be positive, got {g}")
    x = _as_tensor(input)
    if x.ndim != 4:
        raise ValueError(f"channel_shuffle: input must be 4D (N, C, H, W), got {x.ndim}D")
    N, C, H, W = x.shape
    if C % g != 0:
        raise ValueError(f"channel_shuffle: channels ({C}) must be divisible by groups ({g})")
    channels_per_group = C // g
    return x.reshape([N, g, channels_per_group, H, W]).permute([0, 2, 1, 3, 4]).reshape([N, C, H, W])


def interpolate(
    input: _Tensor,
    size: Optional[Union[int, Sequence[int]]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
) -> _Tensor:
    """Down/up samples the input to either the given size or the given scale_factor."""
    ndim = input.ndim
    expect = max(ndim - 2, 0)
    if size is not None:
        target_spatial = [int(size)] * expect if isinstance(size, int) else [int(v) for v in size]
        size_arg = target_spatial
        scale_arg = None
    elif scale_factor is not None:
        scale_arg = [float(scale_factor)] * expect if isinstance(scale_factor, (int, float)) else [float(v) for v in scale_factor]
        target_spatial = [int(input.shape[i + 2] * scale_arg[i]) for i in range(expect)]
        size_arg = None
    else:
        raise ValueError("Either size or scale_factor must be defined")

    if len(target_spatial) != expect or expect == 0:
        raise ValueError(f"interpolate: expected {expect} spatial size(s) for a {ndim}D input, got {list(target_spatial)}")
    if any(v <= 0 for v in target_spatial):
        raise ValueError(f"interpolate: spatial sizes must be positive, got {target_spatial}")

    if isinstance(input, _Tensor) and hasattr(input, "interpolate") and mode in (
        "nearest", "nearest-exact", "linear", "bilinear", "trilinear", "bicubic", "area"
    ):
        return input.interpolate(
            size=size_arg,
            scale_factor=scale_arg,
            mode=mode,
            align_corners=bool(align_corners) if align_corners is not None else False,
        )

    arr = _to_np(input)
    if mode == "nearest":
        if ndim == 3:
            idx_x = (np.arange(target_spatial[0]) * arr.shape[2] // target_spatial[0]).astype(np.int64)
            return input.index_select(2, _from_numpy(idx_x))
        elif ndim == 4:
            idx_y = (np.arange(target_spatial[0]) * arr.shape[2] // target_spatial[0]).astype(np.int64)
            idx_x = (np.arange(target_spatial[1]) * arr.shape[3] // target_spatial[1]).astype(np.int64)
            return input.index_select(2, _from_numpy(idx_y)).index_select(3, _from_numpy(idx_x))
        else:
            idx_z = (np.arange(target_spatial[0]) * arr.shape[2] // target_spatial[0]).astype(np.int64)
            idx_y = (np.arange(target_spatial[1]) * arr.shape[3] // target_spatial[1]).astype(np.int64)
            idx_x = (np.arange(target_spatial[2]) * arr.shape[4] // target_spatial[2]).astype(np.int64)
            return input.index_select(2, _from_numpy(idx_z)).index_select(3, _from_numpy(idx_y)).index_select(4, _from_numpy(idx_x))
    return interpolate(input, size=target_spatial, mode="nearest")


def unfold(
    input: _Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
) -> _Tensor:
    """Extracts sliding local blocks from a batched input tensor (im2col)."""
    x = _as_tensor(input)
    if x.ndim != 4:
        raise ValueError(f"unfold: input must be 4D (N, C, H, W), got {x.ndim}D")
    k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    d = (dilation, dilation) if isinstance(dilation, int) else dilation
    p = (padding, padding) if isinstance(padding, int) else padding
    s = (stride, stride) if isinstance(stride, int) else stride

    if k[0] <= 0 or k[1] <= 0:
        raise ValueError(f"unfold: kernel_size must be positive, got {k}")
    if d[0] <= 0 or d[1] <= 0:
        raise ValueError(f"unfold: dilation must be positive, got {d}")
    if s[0] <= 0 or s[1] <= 0:
        raise ValueError(f"unfold: stride must be positive, got {s}")
    if p[0] < 0 or p[1] < 0:
        raise ValueError(f"unfold: padding must be non-negative, got {p}")

    # Pad if requested (pad tuple order for F.pad is (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom))
    if p[0] > 0 or p[1] > 0:
        x_pad = pad(x, (p[1], p[1], p[0], p[0]), mode="constant", value=0.0)
    else:
        x_pad = x

    N, C, H, W = x_pad.shape
    eff_kh = (k[0] - 1) * d[0] + 1
    eff_kw = (k[1] - 1) * d[1] + 1
    if H < eff_kh or W < eff_kw:
        raise ValueError(f"unfold: calculated output shape is negative for input {list(x.shape)} and kernel {k}")

    out_h = (H - eff_kh) // s[0] + 1
    out_w = (W - eff_kw) // s[1] + 1

    patches = []
    for di in range(k[0]):
        for dj in range(k[1]):
            h_idx = _from_numpy(np.array([di * d[0] + i * s[0] for i in range(out_h)], dtype=np.int64))
            w_idx = _from_numpy(np.array([dj * d[1] + j * s[1] for j in range(out_w)], dtype=np.int64))
            patch = x_pad.index_select(2, h_idx).index_select(3, w_idx)
            patches.append(patch.reshape([N, C, 1, out_h * out_w]))

    all_patches = _native_cat(patches, dim=2)
    return all_patches.reshape([N, C * k[0] * k[1], out_h * out_w])


def fold(
    input: _Tensor,
    output_size: Union[int, Tuple[int, int]],
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
) -> _Tensor:
    """Combines an array of sliding local blocks into a large containing tensor (col2im)."""
    out_s = (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)
    k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    d = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    p = (padding, padding) if isinstance(padding, int) else tuple(padding)
    s = (stride, stride) if isinstance(stride, int) else tuple(stride)
    if s[0] <= 0 or s[1] <= 0 or d[0] <= 0 or d[1] <= 0 or k[0] <= 0 or k[1] <= 0:
        raise ValueError(f"fold: kernel_size, stride and dilation must be positive, got {k}, {s}, {d}")
    if out_s[0] <= 0 or out_s[1] <= 0:
        raise ValueError(f"fold: output_size must be positive, got {out_s}")
    if isinstance(input, _Tensor) and hasattr(input, "fold_2d"):
        return input.fold_2d(out_s, k, dilation=d, padding=p, stride=s)
    arr = _to_np(input)
    N, C_block, L = arr.shape
    kH, kW = k
    C = C_block // (kH * kW)
    H_out, W_out = out_s
    padded_H = H_out + 2 * p[0]
    padded_W = W_out + 2 * p[1]

    res = np.zeros((N, C, padded_H, padded_W), dtype=arr.dtype)
    H_blocks = (padded_H - (kH - 1) * d[0] - 1) // s[0] + 1
    W_blocks = (padded_W - (kW - 1) * d[1] - 1) // s[1] + 1

    blocks = arr.reshape(N, C, kH, kW, H_blocks, W_blocks)
    for h in range(H_blocks):
        for w in range(W_blocks):
            h_start = h * s[0]
            w_start = w * s[1]
            for kh in range(kH):
                for kw in range(kW):
                    res[:, :, h_start + kh * d[0], w_start + kw * d[1]] += blocks[:, :, kh, kw, h, w]

    if p[0] > 0 or p[1] > 0:
        res = res[:, :, p[0] : p[0] + H_out, p[1] : p[1] + W_out]
    return _wrap(res, like=input)


def grid_sample(
    input: _Tensor,
    grid: _Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> _Tensor:
    """Given an input and a flow-field grid, computes the output using input pixel values and pixel locations from grid."""
    if (
        isinstance(input, _Tensor)
        and isinstance(grid, _Tensor)
        and hasattr(input, "grid_sample")
        and mode == "bilinear"
        and padding_mode == "zeros"
    ):
        return input.grid_sample(grid, align_corners=bool(align_corners))
    arr = _to_np(input)
    g = _to_np(grid)
    N, C, H_in, W_in = arr.shape
    _, H_out, W_out, _ = g.shape

    if align_corners:
        x = ((g[..., 0] + 1) / 2) * (W_in - 1)
        y = ((g[..., 1] + 1) / 2) * (H_in - 1)
    else:
        x = ((g[..., 0] + 1) * W_in - 1) / 2
        y = ((g[..., 1] + 1) * H_in - 1) / 2

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    valid_mask00 = (x0 >= 0) & (x0 < W_in) & (y0 >= 0) & (y0 < H_in)
    valid_mask01 = (x1 >= 0) & (x1 < W_in) & (y0 >= 0) & (y0 < H_in)
    valid_mask10 = (x0 >= 0) & (x0 < W_in) & (y1 >= 0) & (y1 < H_in)
    valid_mask11 = (x1 >= 0) & (x1 < W_in) & (y1 >= 0) & (y1 < H_in)

    x0_c = np.clip(x0, 0, W_in - 1)
    x1_c = np.clip(x1, 0, W_in - 1)
    y0_c = np.clip(y0, 0, H_in - 1)
    y1_c = np.clip(y1, 0, H_in - 1)

    wa = ((x1 - x) * (y1 - y))[:, None, :, :]
    wb = ((x - x0) * (y1 - y))[:, None, :, :]
    wc = ((x1 - x) * (y - y0))[:, None, :, :]
    wd = ((x - x0) * (y - y0))[:, None, :, :]

    out = np.zeros((N, C, H_out, W_out), dtype=arr.dtype)
    for n in range(N):
        Ia = np.where(valid_mask00[n, None, :, :], arr[n, :, y0_c[n], x0_c[n]], 0.0)
        Ib = np.where(valid_mask01[n, None, :, :], arr[n, :, y0_c[n], x1_c[n]], 0.0)
        Ic = np.where(valid_mask10[n, None, :, :], arr[n, :, y1_c[n], x0_c[n]], 0.0)
        Id = np.where(valid_mask11[n, None, :, :], arr[n, :, y1_c[n], x1_c[n]], 0.0)
        out[n] = (wa[n] * Ia + wb[n] * Ib + wc[n] * Ic + wd[n] * Id)

    return _wrap(out, like=input)


def affine_grid(theta: _Tensor, size: Sequence[int], align_corners: bool = False) -> _Tensor:
    """Generates a 2D flow field given a batch of affine matrices theta of shape (N, 2, 3)."""
    if isinstance(theta, _Tensor) and hasattr(_Tensor, "affine_grid") and len(size) == 4:
        return _Tensor.affine_grid(theta, [int(v) for v in size], align_corners=bool(align_corners))
    t = _to_np(theta)
    N, H, W = size[0], size[2], size[3]
    if align_corners:
        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
    else:
        x = (2 * np.arange(W) + 1) / W - 1
        y = (2 * np.arange(H) + 1) / H - 1
    xx, yy = np.meshgrid(x, y)
    ones = np.ones_like(xx)
    coords = np.stack([xx, yy, ones], axis=0).reshape(3, -1)  # (3, H*W)

    grid = np.matmul(t, coords).reshape(N, 2, H, W).transpose(0, 2, 3, 1)  # (N, H, W, 2)
    return _wrap(grid, like=theta)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embedding(
    input: _Tensor,
    weight: _Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> _Tensor:
    """A simple lookup table that looks up embeddings in a fixed dictionary and size."""
    # Differentiable path via one-hot matmul (stays in the autograd graph).
    # max_norm/norm_type/scale_grad_by_freq/sparse match torch defaults (no-ops here,
    # as in the previous implementation).
    idx_np = _to_np(input).astype(int)
    w = weight if isinstance(weight, _Tensor) else _from_numpy(_contig(_to_np(weight)))
    oh = one_hot(_from_numpy(_contig(idx_np)), num_classes=w.shape[0])
    oh_f = oh.to_dtype("f32") if str(oh.to_numpy().dtype) != "float32" else oh
    res = oh_f.matmul(w)
    if padding_idx is not None:
        mask = _from_numpy(_contig((idx_np != padding_idx).astype(np.float32)))
        res = res * mask.unsqueeze(-1)
    return res


def embedding_bag(
    input: _Tensor,
    weight: _Tensor,
    offsets: Optional[_Tensor] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    mode: str = "mean",
    sparse: bool = False,
    per_sample_weights: Optional[_Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: Optional[int] = None,
) -> _Tensor:
    """Computes sums or means of bags of embeddings without instantiating intermediate embeddings."""
    if mode not in ("sum", "mean", "max"):
        raise ValueError(f"embedding_bag: unknown mode {mode!r}, must be 'sum', 'mean', or 'max'")
    input = _as_tensor(input)
    weight = _as_tensor(weight)

    if offsets is None:
        if input.ndim != 2:
            raise ValueError(f"embedding_bag: input must be 2D when offsets is None, got {input.ndim}D")
        emb = embedding(input, weight, padding_idx=padding_idx)
        if mode == "sum":
            return emb.sum(dim=1)
        elif mode == "mean":
            return emb.mean(dim=1)
        else:
            return emb.max(dim=1).values

    off = _to_np(offsets).astype(int)
    idx = _to_np(input).astype(int)
    num_bags = len(off) - 1 if include_last_offset else len(off)
    bags = []
    for b in range(num_bags):
        start = off[b]
        end = off[b + 1] if b + 1 < len(off) else len(idx)
        bag_indices = _from_numpy(_contig(idx[start:end]))
        bag_emb = embedding(bag_indices, weight, padding_idx=padding_idx)
        if per_sample_weights is not None:
            psw = _as_tensor(per_sample_weights).narrow(0, start, end - start)
            bag_emb = bag_emb * psw.unsqueeze(-1)
        if mode == "sum":
            bags.append(bag_emb.sum(dim=0))
        elif mode == "mean":
            bags.append(bag_emb.mean(dim=0))
        else:
            bags.append(bag_emb.max(dim=0).values)
    return _native_stack(bags, dim=0)


def one_hot(tensor: _Tensor, num_classes: int = -1) -> _Tensor:
    """Takes LongTensor with index values of shape (*) and returns a tensor of shape (*, num_classes) with one-hot values."""
    arr = _to_np(tensor).astype(int)
    if num_classes > 0:
        c = num_classes
    else:
        if arr.size == 0:
            raise ValueError("one_hot: cannot infer num_classes from an empty tensor (pass num_classes explicitly)")
        c = int(np.max(arr)) + 1
    # eye[arr] would silently wrap out-of-range ids (NumPy negative indexing);
    # torch raises, so validate first.
    if arr.size:
        lo, hi = int(np.min(arr)), int(np.max(arr))
        if lo < 0 or hi >= c:
            raise ValueError(f"one_hot: indices out of range [0, {c}) (got min {lo}, max {hi})")
    eye = np.eye(c, dtype=np.int64)
    res = eye[arr]
    return _wrap(res, like=tensor)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def mse_loss(input: _Tensor, target: _Tensor, reduction: str = "mean") -> _Tensor:
    """Computes Mean Squared Error (MSE) loss with full autograd preservation across reduction modes."""
    diff = input - target
    loss = diff * diff
    return _apply_reduction(loss, reduction, "mse_loss")


def l1_loss(input: _Tensor, target: _Tensor, reduction: str = "mean") -> _Tensor:
    """Computes L1 (Mean Absolute Error) loss with full autograd preservation across reduction modes."""
    diff = input - target
    loss = diff.abs()
    return _apply_reduction(loss, reduction, "l1_loss")


def cross_entropy(
    input: _Tensor,
    target: Any,
    weight: Optional[_Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> _Tensor:
    """Cross entropy loss supporting PyTorch-compatible reductions, ignore_index, weights, soft targets, and label smoothing."""
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"cross_entropy: reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError(f"cross_entropy: label_smoothing must be in [0.0, 1.0), got {label_smoothing}")
    if target is None or isinstance(target, (str, bytes)):
        raise TypeError(f"cross_entropy: target must be a Tensor or sequence of numbers, got {type(target).__name__}")
    if not isinstance(target, _Tensor):
        t_probe = np.asarray(target)
        if t_probe.dtype.kind not in ("i", "u", "f", "b"):
            raise TypeError(f"cross_entropy: target elements must be numeric, got dtype {t_probe.dtype}")

    tgt_is_float = (
        (isinstance(target, _Tensor) and target.dtype in ("f16", "bf16", "f32", "f64"))
        or (isinstance(target, np.ndarray) and np.issubdtype(target.dtype, np.floating))
    )

    # 1. Empty batch handling
    if input.shape[0] == 0 or input.numel() == 0:
        out_dt = np.float64 if input.dtype == "f64" else np.float32
        if reduction == "none":
            if tgt_is_float and input.ndim >= 2:
                t_shape = [input.shape[0]] + list(input.shape[2:])
            else:
                t_shape = list(target.shape) if isinstance(target, _Tensor) else []
            return _from_numpy(np.zeros(t_shape, dtype=out_dt)).to(input.device)
        if reduction == "mean":
            return _from_numpy(np.array(float("nan"), dtype=out_dt)).to(input.device)
        return _from_numpy(np.array(0.0, dtype=out_dt)).to(input.device)

    # 2. Unbatched 1D logits [C] -> unsqueeze to [1, C]
    if input.ndim == 1:
        if tgt_is_float:
            tgt_t = _as_tensor(target).to(input.device).unsqueeze(0)
        elif isinstance(target, int):
            tgt_t = _from_numpy(np.array([target], dtype=np.int64)).to(input.device)
        elif isinstance(target, _Tensor):
            if target.numel() != 1:
                raise RuntimeError(f"cross_entropy: unbatched 1D logits [C={input.shape[0]}] expects 0D or 1-element target, got {target.shape}")
            tgt_t = target.reshape([1])
        else:
            t_arr = np.asarray(target, dtype=np.int64)
            if t_arr.size != 1:
                raise RuntimeError(f"cross_entropy: unbatched 1D logits [C={input.shape[0]}] expects 0D or 1-element target, got {t_arr.size} elements")
            tgt_t = _from_numpy(t_arr.reshape([1])).to(input.device)
        res = cross_entropy(
            input.unsqueeze(0),
            tgt_t,
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        return res.squeeze(0) if reduction == "none" else res

    # 3. K-D segmentation logits [N, C, d1, d2, ...]
    if input.ndim > 2:
        c = input.shape[1]
        perm = [0] + list(range(2, input.ndim)) + [1]
        inp_flat = input.permute(perm).reshape([-1, c])
        tgt_raw = target if isinstance(target, _Tensor) else _from_numpy(
            np.asarray(target, dtype=np.float32 if tgt_is_float else np.int64)
        ).to(input.device)
        if tgt_is_float:
            if list(tgt_raw.shape) != list(input.shape):
                raise RuntimeError(f"cross_entropy: soft target shape {tgt_raw.shape} does not match input shape {input.shape}")
            out_shape = [input.shape[0]] + list(input.shape[2:])
            tgt_flat = tgt_raw.permute(perm).reshape([-1, c])
        else:
            out_shape = list(tgt_raw.shape)
            if tgt_raw.numel() != inp_flat.shape[0]:
                raise RuntimeError(f"cross_entropy: target elements ({tgt_raw.numel()}) does not match input spatial elements ({inp_flat.shape[0]})")
            tgt_flat = tgt_raw.reshape([-1])
        res = cross_entropy(
            inp_flat,
            tgt_flat,
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        return res.reshape(out_shape) if reduction == "none" else res

    # 4. Standard 2D [N, C]
    num_classes = input.shape[1]
    log_probs = input.log_softmax(-1)
    w_t = _as_tensor(weight).to(input.device) if weight is not None else None

    if tgt_is_float:
        t_prob = _as_tensor(target).to(input.device)
        if list(t_prob.shape) != list(input.shape):
            raise RuntimeError(
                f"cross_entropy: target shape {list(t_prob.shape)} must match input shape {list(input.shape)}"
            )
        if float(label_smoothing) > 0.0:
            ls = float(label_smoothing)
            t_prob = t_prob * (1.0 - ls) + (ls / float(num_classes))
        loss_terms = -(t_prob * log_probs)
        if w_t is not None:
            loss_terms = loss_terms * w_t.reshape([1, num_classes])
        loss_t = loss_terms.sum(dim=-1)
        return _apply_reduction(loss_t, reduction, "cross_entropy")

    tgt_t = target if isinstance(target, _Tensor) else _from_numpy(np.asarray(target, dtype=np.int64)).to(input.device)
    if tgt_t.ndim != 1 or tgt_t.shape[0] != input.shape[0]:
        raise RuntimeError(
            f"cross_entropy: expected target shape [{input.shape[0]}], got {list(tgt_t.shape)}"
        )
    valid = tgt_t != int(ignore_index)
    if np.any(_to_np(valid & (tgt_t < 0))):
        raise ValueError("cross_entropy: class index must be >= 0 (unless equal to ignore_index)")
    safe_tgt = _ox_where(valid, tgt_t, 0)
    nll_picked = -log_probs.gather(1, safe_tgt.unsqueeze(1)).squeeze(1)

    if w_t is not None:
        w_elem = w_t.gather(0, safe_tgt)
        nll_picked = nll_picked * w_elem
        denom = _ox_where(valid, w_elem, 0.0).sum()
    else:
        denom = valid.to_dtype(input.dtype).sum()

    if float(label_smoothing) > 0.0:
        ls = float(label_smoothing)
        lp_weighted = log_probs * w_t.reshape([1, num_classes]) if w_t is not None else log_probs
        smooth = -lp_weighted.sum(dim=-1) / float(num_classes)
        per_sample = (1.0 - ls) * nll_picked + ls * smooth
    else:
        per_sample = nll_picked

    per_sample = _ox_where(valid, per_sample, 0.0)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.sum() / denom


def binary_cross_entropy_with_logits(
    input: _Tensor,
    target: _Tensor,
    weight: Optional[_Tensor] = None,
    reduction: str = "mean",
    pos_weight: Optional[_Tensor] = None,
) -> _Tensor:
    """Binary cross entropy with logits supporting PyTorch-compatible reduction modes."""
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"binary_cross_entropy_with_logits: reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
    if reduction == "mean" and weight is None and pos_weight is None:
        return _native_bce_with_logits(input, target)
    if reduction == "sum" and weight is None and pos_weight is None:
        return _native_bce_with_logits(input, target) * float(input.numel())

    if pos_weight is None:
        loss = input.relu() - input * target + (-input.abs()).softplus()
    else:
        log_weight = 1.0 + (pos_weight - 1.0) * target
        loss = (1.0 - target) * input + log_weight * ((-input.abs()).softplus() + (-input).relu())
    if weight is not None:
        loss = loss * weight
    return _apply_reduction(loss, reduction, "binary_cross_entropy_with_logits")


def huber_loss(
    input: _Tensor,
    target: _Tensor,
    reduction: Union[str, float] = "mean",
    delta: float = 1.0,
) -> _Tensor:
    """Computes Huber loss with transition threshold `delta` and PyTorch-compatible `reduction`."""
    if isinstance(reduction, (int, float)):
        delta = float(reduction)
        reduction = "mean"
    if delta <= 0.0 or not math.isfinite(delta):
        raise ValueError(f"huber_loss requires a positive finite delta, got {delta}")
    if reduction == "mean":
        return _native_huber_loss(input, target, delta=float(delta))
    if reduction == "sum":
        return _native_huber_loss(input, target, delta=float(delta)) * float(input.numel())
    if reduction != "none":
        raise ValueError(f"huber_loss: reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
    diff = (input - target).abs()
    d = float(delta)
    quad = 0.5 * (diff * diff)
    lin = d * (diff - 0.5 * d)
    return _ox_where(diff <= d, quad, lin)


def _kl_div_impl(
    input: _Tensor,
    target: _Tensor,
    reduction: str = "mean",
    log_target: bool = False,
) -> _Tensor:
    if reduction not in ("mean", "batchmean", "sum", "none"):
        raise ValueError(
            f"kl_div: reduction must be 'mean', 'batchmean', 'sum' or 'none', got {reduction!r}"
        )
    if log_target:
        loss = target.exp() * (target - input)
    else:
        pos = target > 0
        safe_t = _ox_where(pos, target, 1.0)
        loss = _ox_where(pos, target * (safe_t.ln() - input), 0.0)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "batchmean":
        batch = float(input.shape[0]) if input.ndim > 0 and input.shape[0] > 0 else 1.0
        return loss.sum() / batch
    return loss.mean()


def kl_div(
    input: _Tensor,
    target: _Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    log_target: bool = False,
) -> _Tensor:
    """Computes the Kullback-Leibler divergence loss matching `torch.nn.functional.kl_div`."""
    if size_average is False or reduce is False:
        reduction = "sum" if reduce is True else ("none" if reduce is False else "sum")
    return _kl_div_impl(input, target, reduction=reduction, log_target=log_target)


def kl_div_loss(
    log_probs: _Tensor,
    target_probs: _Tensor,
    reduction: str = "batchmean",
    log_target: bool = False,
) -> _Tensor:
    """KL divergence matching the native `loss::kl_div_loss` convention (`batchmean` by default)."""
    if reduction == "batchmean" and not log_target:
        return _native_kl_div_loss(log_probs, target_probs)
    return _kl_div_impl(log_probs, target_probs, reduction=reduction, log_target=log_target)


def nll_loss(
    input: _Tensor,
    target: Any,
    weight: Optional[_Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> _Tensor:
    """Negative log likelihood loss supporting `reduction` ('mean', 'sum', 'none'), `weight`, and `ignore_index`."""
    if reduce is False:
        reduction = "none"
    elif size_average is False:
        reduction = "sum"
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"nll_loss: reduction must be 'mean', 'sum' or 'none', got {reduction!r}")

    if input.ndim == 2 and weight is None and reduction in ("mean", "sum"):
        t_arr = target.to_numpy() if isinstance(target, _Tensor) else np.asarray(target)
        if not np.any(t_arr == int(ignore_index)):
            base = _native_nll_loss(input, target)
            return base if reduction == "mean" else base * float(input.shape[0])

    if input.shape[0] == 0 or input.numel() == 0:
        if reduction == "none":
            t_shape = target.shape if isinstance(target, _Tensor) else ()
            return _from_numpy(np.zeros(t_shape, dtype=np.float32)).to(input.device)
        val = float("nan") if reduction == "mean" else 0.0
        return _from_numpy(np.array(val, dtype=np.float32)).to(input.device)

    if input.ndim == 1:
        tgt_1d = target.reshape([1]) if isinstance(target, _Tensor) else _from_numpy(np.asarray(target, dtype=np.int64).reshape([1])).to(input.device)
        res = nll_loss(input.unsqueeze(0), tgt_1d, weight=weight, ignore_index=ignore_index, reduction=reduction)
        return res.squeeze(0) if reduction == "none" else res

    if input.ndim > 2:
        c = input.shape[1]
        perm = [0] + list(range(2, input.ndim)) + [1]
        inp_flat = input.permute(perm).reshape([-1, c])
        tgt_t = target if isinstance(target, _Tensor) else _from_numpy(np.asarray(target, dtype=np.int64)).to(input.device)
        tgt_shape = list(tgt_t.shape)
        res = nll_loss(inp_flat, tgt_t.reshape([-1]), weight=weight, ignore_index=ignore_index, reduction=reduction)
        return res.reshape(tgt_shape) if reduction == "none" else res

    tgt_t = target if isinstance(target, _Tensor) else _from_numpy(np.asarray(target, dtype=np.int64)).to(input.device)
    valid = tgt_t != int(ignore_index)
    if np.any(_to_np(valid & (tgt_t < 0))):
        raise ValueError("cross_entropy: class index must be >= 0 (unless equal to ignore_index)")
    safe_tgt = _ox_where(valid, tgt_t, 0)
    picked = -input.gather(1, safe_tgt.unsqueeze(1)).squeeze(1)
    picked = _ox_where(valid, picked, 0.0)
    if weight is not None:
        w_elem = _as_tensor(weight).to(input.device).gather(0, safe_tgt)
        w_elem = _ox_where(valid, w_elem, 0.0)
        picked = picked * w_elem
        denom = w_elem.sum()
    else:
        denom = valid.to_dtype(input.dtype).sum()
    if reduction == "none":
        return picked
    if reduction == "sum":
        return picked.sum()
    return picked.sum() / denom


focal_loss = _native_focal_loss


def cosine_embedding_loss(
    input1: _Tensor,
    input2: _Tensor,
    target: _Tensor,
    margin: float = 0.0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> _Tensor:
    """Cosine embedding loss supporting margin and reduction ('none', 'mean', 'sum')."""
    a, b, y = _as_tensor(input1), _as_tensor(input2), _as_tensor(target)
    dim = 0 if a.ndim == 1 else 1
    num = (a * b).sum(dim=dim)
    denom = (a.pow(2.0).sum(dim=dim).sqrt() * b.pow(2.0).sum(dim=dim).sqrt()).clamp_min(1e-12)
    cos = num / denom
    pos_loss = 1.0 - cos
    neg_loss = (cos - float(margin)).clamp_min(0.0)
    loss = _ox_where(y > 0.0, pos_loss, neg_loss)
    return _apply_reduction(loss, reduction, "cosine_embedding_loss")


def triplet_margin_loss(
    anchor: _Tensor,
    positive: _Tensor,
    negative: _Tensor,
    margin: float = 1.0,
    p: float = 2.0,
    eps: float = 1e-6,
    swap: bool = False,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> _Tensor:
    """Triplet margin loss supporting p-norm, eps, swap, margin, and reduction."""
    a, pos, neg = _as_tensor(anchor), _as_tensor(positive), _as_tensor(negative)
    was_1d = a.ndim == 1
    if was_1d:
        a, pos, neg = a.unsqueeze(0), pos.unsqueeze(0), neg.unsqueeze(0)
    d_pos = pairwise_distance(a, pos, p=float(p), eps=float(eps), keepdim=False)
    d_neg = pairwise_distance(a, neg, p=float(p), eps=float(eps), keepdim=False)
    if swap:
        d_swap = pairwise_distance(pos, neg, p=float(p), eps=float(eps), keepdim=False)
        d_neg = _ox_where(d_swap < d_neg, d_swap, d_neg)
    loss = (d_pos - d_neg + float(margin)).clamp_min(0.0)
    if was_1d:
        loss = loss.squeeze(0)
    return _apply_reduction(loss, reduction, "triplet_margin_loss")


def margin_ranking_loss(
    input1: _Tensor,
    input2: _Tensor,
    target: _Tensor,
    margin: float = 0.0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> _Tensor:
    """Margin ranking loss: max(0, -y * (x1 - x2) + margin)."""
    x1, x2, y = _as_tensor(input1), _as_tensor(input2), _as_tensor(target)
    loss = (-y * (x1 - x2) + float(margin)).clamp_min(0.0)
    return _apply_reduction(loss, reduction, "margin_ranking_loss")


def hinge_embedding_loss(
    input: _Tensor,
    target: _Tensor,
    margin: float = 1.0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> _Tensor:
    """Hinge embedding loss: x if y == 1 else max(0, margin - x)."""
    x, y = _as_tensor(input), _as_tensor(target)
    neg = (float(margin) - x).clamp_min(0.0)
    loss = _ox_where(y > 0.0, x, neg)
    return _apply_reduction(loss, reduction, "hinge_embedding_loss")


def soft_margin_loss(
    input: _Tensor,
    target: _Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> _Tensor:
    """Soft margin loss: log(1 + exp(-y * x))."""
    x, y = _as_tensor(input), _as_tensor(target)
    loss = (-y * x).softplus()
    return _apply_reduction(loss, reduction, "soft_margin_loss")


def smooth_l1_loss(input: _Tensor, target: _Tensor, beta: float = 1.0, reduction: str = "mean") -> _Tensor:
    """Computes Smooth L1 (Huber) loss: 0.5 * (x - y)^2 / beta if |x - y| < beta else |x - y| - 0.5 * beta."""
    if beta <= 0:
        raise ValueError(f"smooth_l1_loss: beta must be positive, got {beta}")
    diff = input - target
    abs_diff = diff.abs()
    beta_f = float(beta)
    loss = _ox_where(abs_diff < beta_f, 0.5 * (diff * diff) / beta_f, abs_diff - 0.5 * beta_f)
    return _apply_reduction(loss, reduction, "smooth_l1_loss")


def binary_cross_entropy(
    input: _Tensor,
    target: _Tensor,
    weight: Optional[_Tensor] = None,
    reduction: str = "mean",
    eps: float = 1e-7,
) -> _Tensor:
    """Computes Binary Cross Entropy between input probabilities and targets with numerical clamping."""
    input_clamped = input.clamp(eps, 1.0 - eps)
    comp_clamped = (1.0 - input_clamped).clamp(eps, 1.0)
    loss = -((target * input_clamped.ln()) + ((1.0 - target) * comp_clamped.ln()))
    if weight is not None:
        loss = loss * weight
    return _apply_reduction(loss, reduction, "binary_cross_entropy")


def poisson_nll_loss(
    input: _Tensor,
    target: _Tensor,
    log_input: bool = True,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> _Tensor:
    """Poisson negative log likelihood loss."""
    if log_input:
        loss = input.exp() - target * input
    else:
        loss = input - target * (input + eps).ln()
    return _apply_reduction(loss, reduction, "poisson_nll_loss")


def gaussian_nll_loss(
    input: _Tensor,
    target: _Tensor,
    var: _Tensor,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> _Tensor:
    """Gaussian negative log likelihood loss."""
    var_clamped = var.clamp(eps, 1e12)
    loss = 0.5 * (var_clamped.ln() + ((input - target).square() / var_clamped))
    if full:
        loss = loss + 0.5 * math.log(2.0 * math.pi)
    return _apply_reduction(loss, reduction, "gaussian_nll_loss")


ctc_loss = _native_ctc_loss


def cosine_similarity(x1: _Tensor, x2: _Tensor, dim: int = -1, eps: float = 1e-8) -> _Tensor:
    """Computes cosine similarity between x1 and x2 along dim."""
    dot = (x1 * x2).sum(dim)
    norm1 = (x1.square().sum(dim)).sqrt()
    norm2 = (x2.square().sum(dim)).sqrt()
    denom = (norm1 * norm2).clamp(eps, 1e9)
    return dot / denom


def pairwise_distance(x1: _Tensor, x2: _Tensor, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> _Tensor:
    """Computes the batchwise pairwise distance between vectors v1, v2 using the p-norm."""
    diff = _as_tensor(x1) - _as_tensor(x2)
    p_val = float(p)
    eps_val = float(eps)
    dist = (diff.abs().pow(p_val).sum(dim=-1, keepdim=keepdim) + eps_val ** p_val).pow(1.0 / p_val)
    return dist


def pdist(input: _Tensor, p: float = 2.0) -> _Tensor:
    """Computes the p-norm distance between every pair of row vectors in the input matrix."""
    return input.pdist(p) if hasattr(input, "pdist") else _as_tensor(input).pdist(p)


def cdist(x1: _Tensor, x2: _Tensor, p: float = 2.0) -> _Tensor:
    """Computes batched pairwise distance between vectors in x1 and x2 under the p-norm."""
    t1 = _as_tensor(x1)
    t2 = _as_tensor(x2)
    return t1.cdist(t2, p=p)


def gumbel_softmax(logits: _Tensor, tau: float = 1.0, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> _Tensor:
    """Samples from the Gumbel-Softmax distribution and optionally discretizes."""
    logits = _as_tensor(logits)
    # Gumbel noise: -log(-log(U + eps) + eps) where U ~ Uniform(0, 1)
    u = np.random.uniform(eps, 1.0 - eps, size=list(logits.shape)).astype(np.float32)
    g = -np.log(-np.log(u))
    g_noise = _from_numpy(_contig(g))
    y_soft = ((logits + g_noise) / float(tau)).softmax(dim)
    if not hard:
        return y_soft
    # Straight-through estimator
    y_hard_np = np.zeros(list(y_soft.shape), dtype=np.float32)
    idx = y_soft.argmax(dim=dim).to_numpy()
    np.put_along_axis(y_hard_np, idx[..., None], 1.0, axis=dim)
    y_hard = _from_numpy(_contig(y_hard_np))
    return (y_hard - y_soft).detach() + y_soft


def _finfo_eps(dtype: Any) -> float:
    tag = str(dtype)
    if tag in ("f64", "float64"):
        return float(np.finfo(np.float64).eps)
    if tag in ("f16", "float16"):
        return float(np.finfo(np.float16).eps)
    if tag in ("bf16", "bfloat16"):
        return 0.0078125
    return float(np.finfo(np.float32).eps)


def rms_norm(
    input: _Tensor,
    normalized_shape: Union[int, Sequence[int]],
    weight: Optional[_Tensor] = None,
    eps: Optional[float] = None,
) -> _Tensor:
    """Applies Root Mean Square Layer Normalization."""
    x = _as_tensor(input)
    eps_val = _finfo_eps(x.dtype) if eps is None else float(eps)
    norm_dims = 1 if isinstance(normalized_shape, int) else len(normalized_shape)
    if norm_dims == 1:
        variance = x.pow(2.0).mean(dim=-1, keepdim=True)
    else:
        axes = tuple(range(x.ndim - norm_dims, x.ndim))
        variance = x.pow(2.0).mean(dim=axes, keepdim=True)
    r = x / (variance + eps_val).sqrt()
    if weight is not None:
        r = r * _as_tensor(weight)
    return r


def unflatten(input: _Tensor, dim: int, sizes: Sequence[int]) -> _Tensor:
    """Unflattens a dimension of the input tensor to a specified shape."""
    shape = list(input.shape)
    d = dim if dim >= 0 else len(shape) + dim
    if not (0 <= d < len(shape)):
        raise IndexError(f"Dimension out of range (expected to be in range of [{-len(shape)}, {len(shape) - 1}], but got {dim})")
    prod = 1
    for s in sizes:
        prod *= s
    if prod != shape[d]:
        raise RuntimeError(f"Provided sizes {list(sizes)} don't multiply up to the size of dim {d} ({shape[d]}) in the input tensor")
    new_shape = shape[:d] + list(sizes) + shape[d + 1:]
    return input.reshape(new_shape)


def multi_head_attention_forward(
    query: _Tensor,
    key: _Tensor,
    value: _Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[_Tensor],
    in_proj_bias: Optional[_Tensor],
    bias_k: Optional[_Tensor] = None,
    bias_v: Optional[_Tensor] = None,
    add_zero_attn: bool = False,
    dropout_p: float = 0.0,
    out_proj_weight: Optional[_Tensor] = None,
    out_proj_bias: Optional[_Tensor] = None,
    training: bool = True,
    key_padding_mask: Optional[_Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[_Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[_Tensor] = None,
    k_proj_weight: Optional[_Tensor] = None,
    v_proj_weight: Optional[_Tensor] = None,
    static_k: Optional[_Tensor] = None,
    static_v: Optional[_Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[_Tensor, Optional[_Tensor]]:
    """Functional Multi-Head Attention forward function matching torch.nn.functional.multi_head_attention_forward."""
    tgt_len, bsz, embed_dim = query.shape
    src_len = key.shape[0]
    head_dim = embed_dim // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    if in_proj_weight is not None:
        w_q, w_k, w_v = in_proj_weight.chunk(3, dim=0)
        b_q = b_k = b_v = None
        if in_proj_bias is not None:
            b_q, b_k, b_v = in_proj_bias.chunk(3, dim=0)
        q = linear(query, w_q, b_q)
        k = linear(key, w_k, b_k)
        v = linear(value, w_v, b_v)
    else:
        q = linear(query, q_proj_weight) if q_proj_weight is not None else query
        k = linear(key, k_proj_weight) if k_proj_weight is not None else key
        v = linear(value, v_proj_weight) if v_proj_weight is not None else value

    q = q.transpose(0, 1).reshape([bsz, tgt_len, num_heads, head_dim]).transpose(1, 2)
    k = k.transpose(0, 1).reshape([bsz, src_len, num_heads, head_dim]).transpose(1, 2)
    v = v.transpose(0, 1).reshape([bsz, src_len, num_heads, head_dim]).transpose(1, 2)

    scores = q.matmul(k.transpose(-2, -1)) * scale

    if is_causal:
        keep_np = 1.0 - np.triu(np.ones((tgt_len, src_len), dtype=np.float32), k=1)
        keep = _from_numpy(keep_np)
        scores = scores * keep + (1.0 - keep) * -1e9

    if attn_mask is not None:
        if isinstance(attn_mask, _Tensor) and attn_mask.dtype == bool:
            keep = attn_mask.to_dtype(scores.dtype)
            scores = scores * keep + (1.0 - keep) * -1e9
        elif isinstance(attn_mask, _Tensor):
            scores = scores + attn_mask
        else:
            m_np = np.asarray(attn_mask)
            if m_np.dtype == bool or m_np.dtype == np.bool_:
                keep = _from_numpy(_contig(m_np, dtype=np.float32))
                scores = scores * keep + (1.0 - keep) * -1e9
            else:
                scores = scores + _from_numpy(_contig(m_np, dtype=np.float32))

    attn_weights = scores.softmax(-1)
    if dropout_p > 0.0 and training:
        attn_weights = dropout(attn_weights, p=dropout_p, training=True)

    context = attn_weights.matmul(v)
    context = context.transpose(1, 2).reshape([bsz, tgt_len, embed_dim]).transpose(0, 1)

    if out_proj_weight is not None:
        out = linear(context, out_proj_weight, out_proj_bias)
    else:
        out = context

    if need_weights:
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)
        return out, attn_weights
    else:
        return out, None


# ---------------------------------------------------------------------------
# Autograd Functional Transforms
# ---------------------------------------------------------------------------

def vjp(func: Callable, inputs: _Tensor, v: Optional[_Tensor] = None) -> Tuple[_Tensor, Callable]:
    """Vector-Jacobian Product: computes output and backward pullback closure."""
    if isinstance(inputs, _Tensor):
        x = inputs.detach().requires_grad_(True)
        y = func(x)

        def pullback(cotangent=None):
            if cotangent is None:
                cotangent = _Tensor.ones_like(y)
            proj = (y * cotangent).sum()
            grads = _native_grad([proj], [x], create_graph=False)
            return grads[0]

        return y, pullback
    raise TypeError("inputs must be an Oxideon Tensor")


def jvp(func: Callable, inputs: _Tensor, v: _Tensor) -> Tuple[_Tensor, _Tensor]:
    """Jacobian-Vector Product: computes (f(x), J(x) * v)."""
    x = inputs.detach().requires_grad_(True)
    y = func(x)

    u = _Tensor.zeros_like(y).requires_grad_(True)
    proj = (y * u).sum()
    vjp_val = _native_grad([proj], [x], create_graph=True)[0]
    prod = (vjp_val * v).sum()
    jvp_val = _native_grad([prod], [u], create_graph=False)[0]
    return y, jvp_val


def jacobian(func: Callable, inputs: _Tensor) -> _Tensor:
    """Computes the Jacobian matrix of a function func with respect to inputs."""
    x = inputs.detach().requires_grad_(True)
    y = func(x)
    y_flat = y.reshape(-1)
    rows = []
    for i in range(len(y_flat)):
        target = y_flat[i]
        g = _native_grad([target], [x], create_graph=False, retain_graph=True)[0]
        rows.append(_to_np(g).reshape(-1))
    jac_np = np.stack(rows, axis=0).reshape(*y.shape, *x.shape)
    return _wrap(jac_np, like=inputs)


def hessian(func: Callable, inputs: _Tensor) -> _Tensor:
    """Computes the Hessian matrix of a scalar function func with respect to inputs."""
    def grad_func(x):
        y = func(x)
        return _native_grad([y], [x], create_graph=True)[0]
    return jacobian(grad_func, inputs)


__all__ = [
    # Activations
    "relu",
    "relu6",
    "gelu",
    "silu",
    "sigmoid",
    "tanh",
    "leaky_relu",
    "softmax",
    "log_softmax",
    "softmin",
    "softplus",
    "softsign",
    "glu",
    "hardswish",
    "hardsigmoid",
    "hardtanh",
    "elu",
    "selu",
    "celu",
    "mish",
    "tanhshrink",
    "rrelu",
    "prelu",
    # Attention
    "scaled_dot_product_attention",
    "multi_head_attention_forward",
    # Projections & Convolutions
    "linear",
    "bilinear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    # Pooling
    "max_pool1d",
    "max_pool2d",
    "fractional_max_pool2d",
    "fractional_max_pool3d",
    "max_unpool1d",
    "max_unpool2d",
    "max_unpool3d",
    "max_pool3d",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "lp_pool1d",
    "lp_pool2d",
    # Normalizations
    "batch_norm",
    "layer_norm",
    "group_norm",
    "instance_norm",
    "local_response_norm",
    "normalize",
    "rms_norm",
    # Vision & Spatial
    "pad",
    "pixel_shuffle",
    "pixel_unshuffle",
    "channel_shuffle",
    "interpolate",
    "unfold",
    "fold",
    "grid_sample",
        "hardshrink",
    "softshrink",
    "threshold",
    "logsigmoid",
    "gumbel_softmax",
    "erf",
    "erfc",
    "erfinv",
    "affine_grid",
    # Embeddings
    "embedding",
    "embedding_bag",
    "one_hot",
    "unflatten",
    # Dropout
    "dropout",
    "dropout1d",
    "dropout2d",
    "dropout3d",
    "alpha_dropout",
    "feature_alpha_dropout",
    # Losses
    "mse_loss",
    "l1_loss",
    "smooth_l1_loss",
    "cross_entropy",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "huber_loss",
    "focal_loss",
    "kl_div_loss",
    "kl_div",
    "nll_loss",
    "poisson_nll_loss",
    "gaussian_nll_loss",
    "ctc_loss",
    "cosine_similarity",
    "pairwise_distance",
    "pdist",
    "cdist",
    "cosine_embedding_loss",
    "triplet_margin_loss",
    "margin_ranking_loss",
    "hinge_embedding_loss",
    "soft_margin_loss",
    # Functional Autograd
    "vjp",
    "jvp",
    "jacobian",
    "hessian",
]


# ---------------------------------------------------------------------------------------------------------------------
# Max pooling with indices, dilation and ceil_mode; more losses; in-place activations; deprecated upsample aliases
# ---------------------------------------------------------------------------------------------------------------------

def _ntuple(v: Any, n: int, name: str) -> Tuple[int, ...]:
    if isinstance(v, (list, tuple)):
        if len(v) != n:
            raise ValueError(f"{name} must be an int or a sequence of {n} ints, got {v}")
        return tuple(int(x) for x in v)
    return (int(v),) * n


def _max_pool_nd(input: _Tensor, nd: int, kernel_size, stride, padding, dilation, ceil_mode: bool, name: str):
    """Argument handling for max pooling; the windows, maxima and indices are computed by the Rust kernel."""
    from .oxideon import _max_pool_nd as _rs_max_pool_nd
    x = _as_tensor(input)
    if len(x.shape) not in (nd + 1, nd + 2):
        raise ValueError(f"{name}: expected a {nd + 1}-D or {nd + 2}-D input, got shape {tuple(x.shape)}")
    k = _ntuple(kernel_size, nd, "kernel_size")
    s = k if stride is None or (isinstance(stride, (list, tuple)) and len(stride) == 0) else _ntuple(stride, nd, "stride")
    return _rs_max_pool_nd(x, list(k), list(s), list(_ntuple(padding, nd, "padding")), list(_ntuple(dilation, nd, "dilation")), bool(ceil_mode))


def _adaptive_max_pool_nd(input: _Tensor, nd: int, output_size, name: str):
    """Argument handling for adaptive max pooling (None keeps a dimension's size); the pooling runs in Rust."""
    from .oxideon import _adaptive_max_pool_nd as _rs_adaptive_max_pool_nd
    x = _as_tensor(input)
    spatial = tuple(int(v) for v in x.shape[len(x.shape) - nd:])
    if isinstance(output_size, (list, tuple)):
        o = tuple(spatial[i] if v is None else int(v) for i, v in enumerate(output_size))
    else:
        o = _ntuple(output_size, nd, "output_size")
    if len(o) != nd or any(v <= 0 for v in o):
        raise ValueError(f"{name}: output_size must be {nd} positive sizes, got {output_size}")
    return _rs_adaptive_max_pool_nd(x, list(o))


def max_pool1d_with_indices(input: _Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode: bool = False,
                            return_indices: bool = False) -> Tuple[_Tensor, _Tensor]:
    """1-D max pooling returning (values, indices); indices point into the flattened input length."""
    return _max_pool_nd(input, 1, kernel_size, stride, padding, dilation, ceil_mode, "max_pool1d")


def max_pool2d_with_indices(input: _Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode: bool = False,
                            return_indices: bool = False) -> Tuple[_Tensor, _Tensor]:
    """2-D max pooling returning (values, indices); indices point into the flattened H * W plane (as MaxUnpool2d expects)."""
    return _max_pool_nd(input, 2, kernel_size, stride, padding, dilation, ceil_mode, "max_pool2d")


def max_pool3d_with_indices(input: _Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode: bool = False,
                            return_indices: bool = False) -> Tuple[_Tensor, _Tensor]:
    """3-D max pooling returning (values, indices into the flattened D * H * W volume)."""
    return _max_pool_nd(input, 3, kernel_size, stride, padding, dilation, ceil_mode, "max_pool3d")


def adaptive_max_pool1d_with_indices(input: _Tensor, output_size, return_indices: bool = False) -> Tuple[_Tensor, _Tensor]:
    """Adaptive 1-D max pooling returning (values, indices)."""
    return _adaptive_max_pool_nd(input, 1, output_size, "adaptive_max_pool1d")


def adaptive_max_pool2d_with_indices(input: _Tensor, output_size, return_indices: bool = False) -> Tuple[_Tensor, _Tensor]:
    """Adaptive 2-D max pooling returning (values, indices)."""
    return _adaptive_max_pool_nd(input, 2, output_size, "adaptive_max_pool2d")


def adaptive_max_pool3d_with_indices(input: _Tensor, output_size, return_indices: bool = False) -> Tuple[_Tensor, _Tensor]:
    """Adaptive 3-D max pooling returning (values, indices)."""
    return _adaptive_max_pool_nd(input, 3, output_size, "adaptive_max_pool3d")


def _pool_front(with_indices):
    def pool(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        vals, idx = with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)
        return (vals, idx) if return_indices else vals
    return pool


def _adaptive_front(with_indices):
    def pool(input, output_size, return_indices=False):
        vals, idx = with_indices(input, output_size)
        return (vals, idx) if return_indices else vals
    return pool


max_pool1d = _pool_front(max_pool1d_with_indices)
max_pool2d = _pool_front(max_pool2d_with_indices)
max_pool3d = _pool_front(max_pool3d_with_indices)
adaptive_max_pool1d = _adaptive_front(adaptive_max_pool1d_with_indices)
adaptive_max_pool2d = _adaptive_front(adaptive_max_pool2d_with_indices)
adaptive_max_pool3d = _adaptive_front(adaptive_max_pool3d_with_indices)
for _f, _nd in ((max_pool1d, 1), (max_pool2d, 2), (max_pool3d, 3)):
    _f.__name__ = f"max_pool{_nd}d"
    _f.__doc__ = (f"{_nd}-D max pooling (kernel_size, stride, padding, dilation, ceil_mode as in PyTorch); with "
                  "return_indices=True also the indices into the flattened spatial input.")
for _f, _nd in ((adaptive_max_pool1d, 1), (adaptive_max_pool2d, 2), (adaptive_max_pool3d, 3)):
    _f.__name__ = f"adaptive_max_pool{_nd}d"
    _f.__doc__ = f"Adaptive {_nd}-D max pooling to output_size; with return_indices=True also the indices."


def lp_pool3d(input: _Tensor, norm_type: float, kernel_size, stride=None, ceil_mode: bool = False) -> _Tensor:
    """3-D power-average pooling: (sum over the window of x ** p) ** (1 / p)."""
    if ceil_mode:
        raise NotImplementedError("lp_pool3d: ceil_mode is not supported")
    k = _ntuple(kernel_size, 3, "kernel_size")
    s = k if stride is None else _ntuple(stride, 3, "stride")
    out = avg_pool3d(_as_tensor(input).pow(float(norm_type)), k, s)
    return (out.sign() * out.abs().relu() * float(np.prod(k))).pow(1.0 / norm_type)


def _reduce(loss: _Tensor, reduction: str, name: str) -> _Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"{name}: reduction must be 'none', 'mean' or 'sum', got {reduction!r}")


def multi_margin_loss(input: _Tensor, target: _Tensor, p: int = 1, margin: float = 1.0, weight: Optional[_Tensor] = None,
                      size_average=None, reduce=None, reduction: str = "mean") -> _Tensor:
    """Multi-class hinge loss: for each sample, sum over the wrong classes j of max(0, margin - x[y] + x[j]) ** p,
    times weight[y], divided by the number of classes."""
    if p not in (1, 2):
        raise ValueError("multi_margin_loss: only p = 1 and p = 2 are supported")
    x = _as_tensor(input)
    unbatched = len(x.shape) == 1
    if unbatched:
        x = x.reshape([1, -1])
    n, c = int(x.shape[0]), int(x.shape[1])
    y = target.detach().to_numpy().astype(np.int64).reshape(-1) if isinstance(target, _Tensor) else np.asarray(target, np.int64).reshape(-1)
    xy = x.gather(1, _from_numpy(_contig(y.reshape(n, 1))))
    m = (x - xy + float(margin)).relu()
    if p == 2:
        m = m * m
    not_y = np.ones((n, c), dtype=np.float32)
    not_y[np.arange(n), y] = 0.0
    m = m * _from_numpy(not_y)
    if weight is not None:
        w = weight.detach().to_numpy().reshape(-1)[y].reshape(n, 1).astype(np.float32)
        m = m * _from_numpy(_contig(w))
    loss = m.sum(1) * (1.0 / c)
    if unbatched:
        loss = loss.reshape([])
    return _reduce(loss, reduction, "multi_margin_loss")


def multilabel_margin_loss(input: _Tensor, target: _Tensor, size_average=None, reduce=None, reduction: str = "mean") -> _Tensor:
    """Multi-label hinge loss. `target` holds class indices per sample, ended by the first -1; the loss sums
    max(0, 1 - (x[j] - x[i])) over listed classes j and unlisted classes i, divided by the number of classes."""
    x = _as_tensor(input)
    unbatched = len(x.shape) == 1
    if unbatched:
        x = x.reshape([1, -1])
    n, c = int(x.shape[0]), int(x.shape[1])
    t = (target.detach().to_numpy() if isinstance(target, _Tensor) else np.asarray(target)).astype(np.int64).reshape(n, c)
    # the labels of a row are its entries before the first -1
    active = np.cumprod(t >= 0, axis=1).astype(bool)
    rows = np.broadcast_to(np.arange(n)[:, None], t.shape)[active]
    cols = t[active]
    listed = np.zeros((n, c), dtype=np.float32)
    np.add.at(listed, (rows, cols), 1.0)
    is_target = (listed > 0).astype(np.float32)
    mask = listed[:, :, None] * (1.0 - is_target)[:, None, :]  # (n, j, i)
    diff = x.reshape([n, c, 1]) - x.reshape([n, 1, c])
    hinge = (1.0 - diff).relu() * _from_numpy(_contig(mask))
    loss = hinge.sum(2).sum(1) * (1.0 / c)
    if unbatched:
        loss = loss.reshape([])
    return _reduce(loss, reduction, "multilabel_margin_loss")


def multilabel_soft_margin_loss(input: _Tensor, target: _Tensor, weight: Optional[_Tensor] = None, size_average=None,
                                reduce=None, reduction: str = "mean") -> _Tensor:
    """Multi-label one-versus-all logistic loss: the mean over classes of -(y log sigmoid(x) + (1 - y) log sigmoid(-x))."""
    x = _as_tensor(input)
    y = _as_tensor(target)
    loss = (y * x.log_sigmoid() + (1.0 - y) * (-x).log_sigmoid()) * -1.0
    if weight is not None:
        loss = loss * weight
    c = int(x.shape[-1])
    loss = loss.sum(len(x.shape) - 1) * (1.0 / c)
    return _reduce(loss, reduction, "multilabel_soft_margin_loss")


def triplet_margin_with_distance_loss(anchor: _Tensor, positive: _Tensor, negative: _Tensor, *,
                                      distance_function: Optional[Callable[[_Tensor, _Tensor], _Tensor]] = None,
                                      margin: float = 1.0, swap: bool = False, reduction: str = "mean") -> _Tensor:
    """Triplet loss max(0, d(a, p) - d(a, n) + margin) with any distance function (pairwise Euclidean by default);
    `swap=True` uses min(d(a, n), d(p, n)) as the negative distance."""
    if margin <= 0:
        raise ValueError(f"triplet_margin_with_distance_loss: margin must be greater than 0, got {margin}")
    dist = distance_function if distance_function is not None else pairwise_distance
    dp = dist(anchor, positive)
    dn = dist(anchor, negative)
    if swap:
        ds = dist(positive, negative)
        dn = _ox_where(ds < dn, ds, dn)
    return _reduce((dp - dn + float(margin)).relu(), reduction, "triplet_margin_with_distance_loss")


def linear_cross_entropy(input: _Tensor, linear_weight: _Tensor, target: Any, *, linear_bias: Optional[_Tensor] = None,
                         weight: Optional[_Tensor] = None, reduction: str = "mean", ignore_index: int = -100,
                         label_smoothing: float = 0.0, **_options: Any) -> _Tensor:
    """Cross entropy of `input @ linear_weight.T + linear_bias` against `target` (the projection to class logits and
    the loss in one call, as for a language-model head)."""
    logits = input.matmul(linear_weight.transpose(0, 1))
    if linear_bias is not None:
        logits = logits + linear_bias
    return cross_entropy(logits, target, weight=weight, reduction=reduction, ignore_index=ignore_index,
                         label_smoothing=label_smoothing)


def _functional_inplace(name: str, fn: Callable[..., _Tensor]) -> Callable[..., _Tensor]:
    def inplace(input: _Tensor, *args: Any, **kwargs: Any) -> _Tensor:
        from .tensor_methods import _check_inplace, _write  # the same rules as Tensor's in-place methods
        _check_inplace(input, name)
        return _write(input, fn(input, *args, **kwargs), name)
    inplace.__name__ = name
    inplace.__doc__ = f"In-place `{name[:-1]}`: overwrites `input` with the result and returns it."
    return inplace


relu_ = _functional_inplace("relu_", lambda x: relu(x))
elu_ = _functional_inplace("elu_", lambda x, alpha=1.0: elu(x, alpha))
celu_ = _functional_inplace("celu_", lambda x, alpha=1.0: celu(x, alpha))
selu_ = _functional_inplace("selu_", lambda x: selu(x))
leaky_relu_ = _functional_inplace("leaky_relu_", lambda x, negative_slope=0.01: leaky_relu(x, negative_slope))
hardtanh_ = _functional_inplace("hardtanh_", lambda x, min_val=-1.0, max_val=1.0: hardtanh(x, min_val, max_val))
threshold_ = _functional_inplace("threshold_", lambda x, threshold_value, value: threshold(x, threshold_value, value))
rrelu_ = _functional_inplace("rrelu_", lambda x, lower=0.125, upper=1.0 / 3, training=False: rrelu(x, lower, upper, training))


def upsample(input: _Tensor, size=None, scale_factor=None, mode: str = "nearest", align_corners=None) -> _Tensor:
    """Deprecated alias of `interpolate`."""
    return interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


def upsample_nearest(input: _Tensor, size=None, scale_factor=None) -> _Tensor:
    """Deprecated: `interpolate(..., mode="nearest")`."""
    return interpolate(input, size=size, scale_factor=scale_factor, mode="nearest")


def upsample_bilinear(input: _Tensor, size=None, scale_factor=None) -> _Tensor:
    """Deprecated: `interpolate(..., mode="bilinear", align_corners=True)` (note align_corners=True, as in PyTorch)."""
    return interpolate(input, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=True)


__all__ += [
    "max_pool1d_with_indices", "max_pool2d_with_indices", "max_pool3d_with_indices",
    "adaptive_max_pool1d_with_indices", "adaptive_max_pool2d_with_indices", "adaptive_max_pool3d_with_indices",
    "lp_pool3d", "multi_margin_loss", "multilabel_margin_loss", "multilabel_soft_margin_loss",
    "triplet_margin_with_distance_loss", "linear_cross_entropy", "relu_", "elu_", "celu_", "selu_", "leaky_relu_",
    "hardtanh_", "threshold_", "rrelu_", "upsample", "upsample_nearest", "upsample_bilinear",
]
