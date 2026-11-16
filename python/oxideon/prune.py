"""Model pruning, structured/unstructured sparsity, and compression for Oxideon.

Enables reducing parameter counts and computational requirements via magnitude-based
and structured pruning:
  - prune_l1_unstructured: Prunes parameters with lowest L1 magnitude
  - prune_random_unstructured: Prunes parameters uniformly at random
  - prune_structured: Prunes entire channels / neurons along a tensor dimension
  - remove_pruning: Permanently bakes the mask into parameter values
  - prune_wanda: Per-row pruning by weight magnitude times input norms
  - prune_sparsegpt: Exact sequential pruning with Hessian compensation
  - prune_nm_structured: N:M structured sparsity per row
  - block_influence / select_layers_to_drop / drop_sequential_blocks: depth pruning
  - get_sparsity: Measures sparsity ratio across modules or tensors
  - global_unstructured_pruning: Prunes across multiple layers with a global threshold
"""

from typing import Union, Any, List, Dict, Tuple, Optional, Sequence
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter, Sequential

# In-memory mask storage for arbitrary modules (including PyO3 Rust structs)
_PRUNING_MASKS: Dict[Tuple[int, str], np.ndarray] = {}


def get_sparsity(obj: Any) -> float:
    """Calculates the proportion of zero weights (0.0 = dense, 1.0 = fully sparse)."""
    if isinstance(obj, _Tensor):
        arr = obj.to_numpy()
        return float(np.mean(arr == 0))
    if isinstance(obj, Parameter):
        return get_sparsity(obj.data)
    if isinstance(obj, Module):
        total_zeros = 0
        total_elems = 0
        for p in obj.parameters():
            arr = p.to_numpy()
            total_zeros += int(np.sum(arr == 0))
            total_elems += arr.size
        return float(total_zeros / max(total_elems, 1))
    arr = np.asarray(obj)
    return float(np.mean(arr == 0))


def get_pruning_mask(module: Any, name: str = "weight") -> Optional[np.ndarray]:
    """Retrieves the active pruning mask for a module parameter."""
    key = (id(module), name)
    if key in _PRUNING_MASKS:
        return _PRUNING_MASKS[key]
    return getattr(module, f"{name}_mask", None)


def has_pruning_mask(module: Any, name: str = "weight") -> bool:
    """Returns True if the module parameter currently has an active pruning mask."""
    return (id(module), name) in _PRUNING_MASKS or hasattr(module, f"{name}_mask")


def _store_pruning_mask(module: Any, name: str, mask: np.ndarray, arr: np.ndarray) -> Any:
    pruned_arr = arr * mask
    new_param = Parameter(_from_numpy(np.ascontiguousarray(pruned_arr, dtype=np.float32)))
    setattr(module, name, new_param)
    _PRUNING_MASKS[(id(module), name)] = mask
    try:
        setattr(module, f"{name}_mask", mask)
    except (AttributeError, TypeError):
        pass
    return module


def prune_l1_unstructured(module: Any, name: str = "weight", amount: float = 0.3) -> Any:
    """Prunes lowest L1 magnitude elements in the specified parameter."""
    param = getattr(module, name, None)
    if param is None:
        raise AttributeError(f"Module {module} has no parameter named '{name}'")

    arr = param.to_numpy().copy()
    flat = np.abs(arr).flatten()
    k = int(len(flat) * amount)
    if k >= len(flat):
        mask = np.zeros_like(arr, dtype=np.float32)
    elif k > 0:
        pruned_indices = np.argsort(flat)[:k]
        flat_mask = np.ones_like(flat, dtype=np.float32)
        flat_mask[pruned_indices] = 0.0
        mask = flat_mask.reshape(arr.shape)
    else:
        mask = np.ones_like(arr, dtype=np.float32)

    return _store_pruning_mask(module, name, mask, arr)


def prune_random_unstructured(
    module: Any, name: str = "weight", amount: float = 0.3, seed: Optional[int] = None
) -> Any:
    """Prunes elements uniformly at random in the specified parameter."""
    if not (0.0 <= amount <= 1.0):
        raise ValueError(f"amount must be between 0 and 1, got {amount}")
    param = getattr(module, name, None)
    if param is None:
        raise AttributeError(f"Module {module} has no parameter named '{name}'")

    arr = param.to_numpy().copy()
    rng = np.random.default_rng(seed) if seed is not None else np.random
    mask = (rng.random(arr.shape) >= amount).astype(np.float32)

    return _store_pruning_mask(module, name, mask, arr)


def prune_structured(
    module: Any, name: str = "weight", amount: float = 0.3, dim: int = 0
) -> Any:
    """Prunes entire channels / neurons along dimension `dim` based on L1 norm."""
    if not (0.0 <= amount <= 1.0):
        raise ValueError(f"amount must be between 0 and 1, got {amount}")
    param = getattr(module, name, None)
    if param is None:
        raise AttributeError(f"Module {module} has no parameter named '{name}'")

    arr = param.to_numpy().copy()
    axes = tuple(i for i in range(arr.ndim) if i != dim)
    norms = np.sum(np.abs(arr), axis=axes)

    num_channels = arr.shape[dim]
    k = int(num_channels * amount)
    if k > 0:
        sorted_indices = np.argsort(norms)
        pruned_indices = sorted_indices[:k]
        mask = np.ones(num_channels, dtype=np.float32)
        mask[pruned_indices] = 0.0

        shape = [1] * arr.ndim
        shape[dim] = num_channels
        full_mask = mask.reshape(shape)
    else:
        full_mask = np.ones_like(arr, dtype=np.float32)

    return _store_pruning_mask(module, name, full_mask, arr)


def global_unstructured_pruning(
    parameters: Sequence[Tuple[Any, str]],
    amount: float = 0.3,
    method: str = "l1",
    seed: Optional[int] = None,
) -> None:
    """Applies global unstructured pruning across multiple parameters by pooling their scores."""
    if not (0.0 <= amount <= 1.0):
        raise ValueError(f"amount must be between 0 and 1, got {amount}")

    all_scores = []
    shapes = []
    arrays = []

    for mod, name in parameters:
        param = getattr(mod, name, None)
        if param is None:
            raise AttributeError(f"Module {mod} has no parameter named '{name}'")
        arr = param.to_numpy().copy()
        arrays.append(arr)
        shapes.append(arr.shape)
        if method == "l1":
            all_scores.append(np.abs(arr).flatten())
        elif method == "random":
            rng = np.random.default_rng(seed) if seed is not None else np.random
            all_scores.append(rng.random(arr.size).astype(np.float32))
        else:
            raise ValueError(f"Unsupported method '{method}', choose 'l1' or 'random'")

    flat_scores = np.concatenate(all_scores)
    k = int(len(flat_scores) * amount)
    if k == 0:
        threshold = -float("inf")
    else:
        threshold = float(np.partition(flat_scores, k - 1)[k - 1])

    offset = 0
    for (mod, name), arr, shape in zip(parameters, arrays, shapes):
        size = arr.size
        scores = flat_scores[offset : offset + size].reshape(shape)
        offset += size
        mask = (scores > threshold).astype(np.float32)
        _store_pruning_mask(mod, name, mask, arr)


def remove_pruning(module: Any, name: str = "weight") -> Any:
    """Removes the pruning mask metadata while preserving sparse weights."""
    key = (id(module), name)
    if key in _PRUNING_MASKS:
        del _PRUNING_MASKS[key]
    mask_name = f"{name}_mask"
    if hasattr(module, mask_name):
        try:
            delattr(module, mask_name)
        except (AttributeError, TypeError):
            pass
    return module


def _linear_weight(module: Any, name: str) -> np.ndarray:
    param = getattr(module, name, None)
    if param is None:
        raise AttributeError(f"Module {module} has no parameter named '{name}'")
    arr = np.asarray(param.to_numpy(), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"'{name}' must be a 2-D [out, in] matrix, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"'{name}' must be finite")
    return arr


def _calibration_inputs(activations: Any, in_dim: int, name: str = "activations") -> np.ndarray:
    mat = np.asarray(
        activations.to_numpy() if isinstance(activations, _Tensor) else activations,
        dtype=np.float64,
    )
    if mat.ndim != 2 or mat.shape[0] < 1 or mat.shape[1] != in_dim:
        raise ValueError(f"{name} must be [samples, {in_dim}] with at least one sample")
    if not np.all(np.isfinite(mat)):
        raise ValueError(f"{name} must be finite")
    return mat


def prune_wanda(
    module: Any, name: str = "weight", activations: Any = None, amount: float = 0.5
) -> Any:
    """Wanda pruning: per-row bottom-``amount`` by |W| times input norms (Sun et al., 2024)."""
    if activations is None:
        raise ValueError("prune_wanda needs calibration activations [samples, in]")
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount = {amount} must sit within [0, 1]")
    arr = _linear_weight(module, name)
    out_dim, in_dim = arr.shape
    norms = np.linalg.norm(_calibration_inputs(activations, in_dim), axis=0)
    metric = np.abs(arr) * norms[None, :]
    keep = in_dim - int(in_dim * amount)
    mask = np.zeros_like(arr, dtype=np.float32)
    if keep > 0:
        top = np.argsort(-metric, axis=1, kind="stable")[:, :keep]
        rows = np.arange(out_dim)[:, None]
        mask[rows, top] = 1.0
    return _store_pruning_mask(module, name, mask, arr)


def prune_sparsegpt(
    module: Any,
    name: str = "weight",
    activations: Any = None,
    amount: float = 0.5,
    blocksize: int = 128,
    damping: float = 0.01,
) -> Any:
    """Exact sequential SparseGPT with Hessian compensation (Frantar & Alistarh, 2023)."""
    if activations is None:
        raise ValueError("prune_sparsegpt needs calibration activations [samples, in]")
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount = {amount} must sit within [0, 1]")
    if not isinstance(blocksize, (int, np.integer)) or blocksize < 1:
        raise ValueError("blocksize must be a positive integer")
    if damping < 0:
        raise ValueError("damping must be non-negative")
    arr = _linear_weight(module, name)
    out_dim, in_dim = arr.shape
    mat = _calibration_inputs(activations, in_dim)
    hess = mat.T @ mat
    if damping:
        hess = hess + damping * float(np.mean(np.diag(hess))) * np.eye(in_dim)
    try:
        chol = np.linalg.cholesky(hess)
        eye = np.eye(in_dim)
        hinv = np.linalg.solve(chol.T, np.linalg.solve(chol, eye))
    except np.linalg.LinAlgError:
        raise ValueError("activation Hessian is not positive-definite; increase damping")
    diag = np.diag(hinv)
    if (diag <= 0).any():
        raise ValueError("activation Hessian inverse has non-positive diagonal")
    scores = arr**2 / diag[None, :]
    keep = in_dim - int(in_dim * amount)
    mask = np.zeros_like(arr, dtype=np.float32)
    if keep > 0:
        top = np.argsort(-scores, axis=1, kind="stable")[:, :keep]
        rows = np.arange(out_dim)[:, None]
        mask[rows, top] = 1.0
    if keep == 0 or keep == in_dim:
        return _store_pruning_mask(module, name, mask, arr)
    comp = arr.copy()
    for start in range(0, in_dim, blocksize):
        stop = min(start + blocksize, in_dim)
        width = stop - start
        block = comp[:, start:stop]
        block_mask = mask[:, start:stop].astype(np.float64)
        hinv_block = hinv[start:stop, start:stop]
        err = np.zeros((out_dim, width))
        for j in range(width):
            col = start + j
            keep_col = block_mask[:, j]
            quantized = block[:, j] * keep_col
            err[:, j] = (block[:, j] - quantized) / hinv[col, col]
            block[:, j] = quantized
            block[:, j + 1 :] = block[:, j + 1 :] - np.outer(err[:, j], hinv_block[j, j + 1 :])
        comp[:, start:stop] = block
        comp[:, stop:] = comp[:, stop:] - err @ hinv[start:stop, stop:]
    return _store_pruning_mask(module, name, mask, comp)


def prune_nm_structured(
    module: Any, name: str = "weight", n: int = 2, m: int = 4, activations: Optional[Any] = None
) -> Any:
    """N:M structured sparsity per row (e.g. 2:4 for NVIDIA Ampere tensor cores).
    Supports optional Wanda-style calibration activations weighting.
    """
    if n < 1 or m < 1 or n > m:
        raise ValueError(f"must satisfy 1 <= n <= m, got n={n}, m={m}")
    arr = _linear_weight(module, name)
    out_dim, in_dim = arr.shape
    if in_dim % m != 0:
        raise ValueError(f"input dimension {in_dim} must be divisible by m={m}")
    if n == m:
        mask = np.ones((out_dim, in_dim), dtype=np.float32)
        return _store_pruning_mask(module, name, mask, arr)

    importance = np.abs(arr)
    if activations is not None:
        act_np = activations.to_numpy() if isinstance(activations, _Tensor) else np.asarray(activations, dtype=np.float32)
        if act_np.ndim == 2:
            if act_np.shape[1] != in_dim:
                raise ValueError(f"activations shape {act_np.shape} column count does not match in_dim {in_dim}")
            col_norms = np.linalg.norm(act_np, axis=0)
        elif act_np.ndim == 1:
            if len(act_np) != in_dim:
                raise ValueError(f"activations length {len(act_np)} does not match in_dim {in_dim}")
            col_norms = np.abs(act_np)
        else:
            raise ValueError(f"unsupported activations dimensionality {act_np.ndim}")
        importance = importance * col_norms[None, :]

    blocks = in_dim // m
    reshaped = importance.reshape(out_dim * blocks, m)
    top_indices = np.argsort(-reshaped, axis=1, kind="stable")[:, :n]
    mask_reshaped = np.zeros_like(reshaped, dtype=np.float32)
    rows = np.arange(out_dim * blocks)[:, None]
    mask_reshaped[rows, top_indices] = 1.0
    mask = mask_reshaped.reshape(out_dim, in_dim)
    return _store_pruning_mask(module, name, mask, arr)


def block_influence(
    block: Any, activations: Any, metric: str = "cosine"
) -> float:
    """Measures representation distance introduced by an intermediate block.
    `block` may be either a Module or an input Tensor representation to compare against `activations`.
    """
    if isinstance(block, (_Tensor, np.ndarray)):
        x = block if isinstance(block, _Tensor) else _from_numpy(np.asarray(block, dtype=np.float32))
        y = activations if isinstance(activations, _Tensor) else _from_numpy(np.asarray(activations, dtype=np.float32))
    else:
        x = activations if isinstance(activations, _Tensor) else _from_numpy(np.asarray(activations, dtype=np.float32))
        y = block(x)

    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch in block_influence: x has {x.shape}, y has {y.shape}")

    x_np = x.to_numpy().reshape(x.shape[0], -1)
    y_np = y.to_numpy().reshape(y.shape[0], -1)
    if metric == "cosine":
        dot = np.sum(x_np * y_np, axis=1)
        norm_x = np.linalg.norm(x_np, axis=1)
        norm_y = np.linalg.norm(y_np, axis=1)
        sim = dot / np.maximum(norm_x * norm_y, 1e-12)
        return float(np.mean(1.0 - sim))
    elif metric == "l2":
        return float(np.mean(np.linalg.norm(y_np - x_np, axis=1)))
    else:
        raise ValueError(f"Unsupported metric '{metric}'")


def select_layers_to_drop(influences: Sequence[float], n_drop: int) -> List[int]:
    """Selects indices of blocks with lowest influence to drop."""
    vals = list(influences)
    if len(vals) == 0:
        raise ValueError("influences cannot be empty")
    if any(math.isnan(x) for x in vals):
        raise ValueError("influences contains NaN")
    if n_drop == 0:
        return []
    if n_drop < 1 or n_drop >= len(vals):
        raise ValueError(f"n_drop ({n_drop}) must be in [0, {len(vals)-1}]")
    order = np.argsort(vals, kind="stable")[: int(n_drop)]
    return sorted(int(i) for i in order)


def drop_sequential_blocks(model: Any, indices: Any) -> Any:
    """Returns a new Sequential without the blocks at ``indices`` (keeps >= 1)."""
    if not isinstance(model, Sequential):
        raise ValueError("drop_sequential_blocks needs an nn.Sequential model")
    n = len(model)
    normed = []
    for i in list(indices):
        if not isinstance(i, (int, np.integer)):
            raise ValueError(f"index {i!r} is not an integer")
        j = int(i) + n if int(i) < 0 else int(i)
        if not 0 <= j < n:
            raise ValueError(f"index {i} out of range for {n} blocks")
        normed.append(j)
    keep = [k for k in range(n) if k not in set(normed)]
    if not keep:
        raise ValueError("dropping every block is refused; keep at least one")
    return Sequential([model[k] for k in keep])


__all__ = [
    "get_sparsity",
    "get_pruning_mask",
    "has_pruning_mask",
    "prune_l1_unstructured",
    "prune_random_unstructured",
    "prune_structured",
    "global_unstructured_pruning",
    "remove_pruning",
    "prune_wanda",
    "prune_sparsegpt",
    "prune_nm_structured",
    "block_influence",
    "select_layers_to_drop",
    "drop_sequential_blocks",
]
