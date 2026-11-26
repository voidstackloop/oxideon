"""High-level deep learning, training, parameterization, tree-mapping, and workflow utilities."""

import math
import os
import random
import sys
import time
import types
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from . import data
from . import checkpoint as _checkpoint_mod
from . import rnn_utils as rnn
from . import treemap as treemap
from . import nn_utils as _nn_utils
from . import utils_ext as _utils_ext
from . import workflow_utils as _workflow_utils

__path__ = []  # Exposes 'from oxideon.utils.<submod> import ...'

from .oxideon import manual_seed as _manual_seed, cat as _native_cat
from .primitives import no_grad as _no_grad


def parameters_to_vector(parameters):
    """Concatenates an iterable of parameter tensors into a single flattened 1-D vector."""
    vecs = [p.reshape([-1]) for p in parameters]
    if not vecs:
        from .oxideon import zeros
        return zeros([0])
    return _native_cat(vecs, dim=0)


def vector_to_parameters(vec, parameters):
    """Copies slices of a flattened 1-D vector back into `parameters` in-place."""
    pointer = 0
    with _no_grad():
        for p in parameters:
            num_param = p.numel()
            p.copy_(vec[pointer : pointer + num_param].reshape(list(p.shape)))
            pointer += num_param


def set_seed(seed: int = 42):
    """Sets random seeds deterministically across Oxideon, NumPy, and Python's `random` module."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _manual_seed(seed)


set_random_seed = set_seed


def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> float:
    """Clips the joint Lp norm of gradients across `parameters` to at most `max_norm`."""
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    param_list = []
    grads = []
    for p in parameters:
        g = p.grad() if hasattr(p, "grad") and callable(p.grad) else getattr(p, "grad", None)
        if g is not None:
            param_list.append(p)
            grads.append(g)

    if not grads:
        return 0.0

    if norm_type == float("inf"):
        total = max(float(g.abs().max().item()) for g in grads)
    elif norm_type <= 0.0 or math.isnan(norm_type):
        raise ValueError(f"clip_grad_norm_ requires norm_type > 0 or inf, got {norm_type}")
    else:
        total = 0.0
        for g in grads:
            total += float(g.abs().powf(norm_type).sum().item())
        total = total ** (1.0 / norm_type)

    if error_if_nonfinite and (math.isnan(total) or math.isinf(total)):
        raise RuntimeError(f"The total norm of order {norm_type} for gradients is non-finite ({total}).")

    clip_coef = max_norm / (total + 1e-6)
    if clip_coef < 1.0:
        for p, g in zip(param_list, grads):
            scaled_g = g.mul_scalar(clip_coef)
            if hasattr(p, "set_grad"):
                p.set_grad(scaled_g)

    return total


def clip_grad_value_(parameters, clip_value: float):
    """Clips gradient values of an iterable of parameters elementwise to `[-clip_value, clip_value]`."""
    clip_value = float(clip_value)
    for p in parameters:
        g = p.grad() if hasattr(p, "grad") and callable(p.grad) else getattr(p, "grad", None)
        if g is not None and hasattr(g, "clamp"):
            clamped = g.clamp(-clip_value, clip_value)
            if hasattr(p, "set_grad"):
                p.set_grad(clamped)


_DTYPE_NUM_BYTES = {
    "f32": 4, "f64": 8, "f16": 2, "bf16": 2,
    "i32": 4, "i64": 8, "u8": 1, "bool": 1,
}


def model_summary(model, input_size=None) -> str:
    """Generates a tabular architectural summary of an Oxideon `nn.Module`."""
    del input_size
    header = f"{'Layer (type)':<32} {'Param Shapes':<26} {'Param #':<12} {'Trainable':<10}"
    separator = "=" * len(header)
    subsep = "-" * len(header)
    lines = [separator, header, separator]

    total_params = 0
    total_bytes = 0
    trainable_params = 0
    non_trainable_params = 0
    seen_params = set()

    def walk(mod, prefix=""):
        for name, submod in getattr(mod, "_modules", {}).items():
            full_name = f"{prefix}.{name}" if prefix else name
            cls_name = submod.__class__.__name__
            try:
                mod_params = list(submod.parameters(recurse=False))
            except TypeError:
                mod_params = list(submod.parameters()) if hasattr(submod, "parameters") else []

            if mod_params:
                shapes_str = ", ".join(str(p.shape) for p in mod_params)
                num = sum(p.numel() for p in mod_params)
                is_train = any(getattr(p, "requires_grad", True) for p in mod_params)
                lines.append(f"{full_name + ' (' + cls_name + ')':<32} {shapes_str:<26} {num:<12} {str(is_train):<10}")
            else:
                lines.append(f"{full_name + ' (' + cls_name + ')':<32} {'--':<26} {'0':<12} {'--':<10}")
            walk(submod, full_name)

    walk(model)

    for p in model.parameters():
        p_id = id(p)
        if p_id not in seen_params:
            seen_params.add(p_id)
            n = p.numel()
            total_params += n
            total_bytes += n * _DTYPE_NUM_BYTES.get(str(getattr(p, "dtype", "f32")), 4)
            if getattr(p, "requires_grad", True):
                trainable_params += n
            else:
                non_trainable_params += n

    lines.append(subsep)
    param_bytes = total_bytes
    size_str = f"{param_bytes / 1024:.2f} KB" if param_bytes < 1024 * 1024 else f"{param_bytes / (1024 * 1024):.2f} MB"

    lines.append(f"Total params: {total_params:,}")
    lines.append(f"Trainable params: {trainable_params:,}")
    lines.append(f"Non-trainable params: {non_trainable_params:,}")
    lines.append(f"Estimated Model Size: {size_str}")
    lines.append(separator)
    return "\n".join(lines)


class BenchmarkTimer:
    """High-resolution wall-clock benchmarking context manager."""

    def __init__(self, description: str = "Benchmark", warmup: int = 1, iterations: int = 10):
        self.description = description
        self.warmup = warmup
        self.iterations = iterations
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start

    @property
    def ms(self) -> float:
        return self.elapsed * 1000.0

    def __repr__(self):
        return f"<BenchmarkTimer '{self.description}': {self.ms:.3f} ms>"


class SummaryWriter:
    """Structured experiment metric, histogram, image, hyperparameter, and graph event writer."""

    def __init__(self, log_dir: str = "runs"):
        self.log_dir = log_dir
        self.events: List[Dict[str, Any]] = []
        self.scalars: Dict[str, List[Tuple[int, float, float]]] = {}
        os.makedirs(log_dir, exist_ok=True)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0, walltime: Optional[float] = None):
        if hasattr(scalar_value, "item"):
            scalar_value = scalar_value.item()
        val = float(scalar_value)
        ts = time.time() if walltime is None else float(walltime)

        if tag not in self.scalars:
            self.scalars[tag] = []
        self.scalars[tag].append((global_step, val, ts))

        self.events.append({
            "type": "scalar",
            "tag": tag,
            "step": global_step,
            "value": val,
            "walltime": ts,
        })

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int = 0, walltime: Optional[float] = None):
        for sub_tag, val in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{sub_tag}"
            self.add_scalar(full_tag, val, global_step=global_step, walltime=walltime)

    def add_histogram(
        self,
        tag: str,
        values: Any,
        global_step: int = 0,
        bins: int = 30,
        walltime: Optional[float] = None,
    ) -> None:
        arr = values.detach().to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
        flat = arr.astype(np.float64).reshape(-1)
        counts, edges = np.histogram(flat, bins=int(bins)) if flat.size > 0 else (np.array([], dtype=np.int64), np.array([], dtype=np.float64))
        ts = time.time() if walltime is None else float(walltime)
        self.events.append({
            "type": "histogram",
            "tag": tag,
            "step": global_step,
            "counts": counts.tolist(),
            "bin_edges": edges.tolist(),
            "min": float(np.min(flat)) if flat.size else 0.0,
            "max": float(np.max(flat)) if flat.size else 0.0,
            "mean": float(np.mean(flat)) if flat.size else 0.0,
            "std": float(np.std(flat)) if flat.size else 0.0,
            "walltime": ts,
        })

    def add_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: int = 0,
        dataformats: str = "CHW",
        walltime: Optional[float] = None,
    ) -> None:
        arr = img_tensor.detach().to_numpy() if hasattr(img_tensor, "to_numpy") else np.asarray(img_tensor)
        fmt = dataformats.upper()
        if fmt == "HWC" and arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
        elif fmt == "HW" and arr.ndim == 2:
            arr = arr[None, :, :]
        ts = time.time() if walltime is None else float(walltime)
        self.events.append({
            "type": "image",
            "tag": tag,
            "step": global_step,
            "shape": list(arr.shape),
            "min": float(np.min(arr)) if arr.size else 0.0,
            "max": float(np.max(arr)) if arr.size else 0.0,
            "walltime": ts,
        })

    def add_images(
        self,
        tag: str,
        img_tensor: Any,
        global_step: int = 0,
        dataformats: str = "NCHW",
        walltime: Optional[float] = None,
    ) -> None:
        arr = img_tensor.detach().to_numpy() if hasattr(img_tensor, "to_numpy") else np.asarray(img_tensor)
        fmt = dataformats.upper()
        if fmt == "NHWC" and arr.ndim == 4:
            arr = np.transpose(arr, (0, 3, 1, 2))
        ts = time.time() if walltime is None else float(walltime)
        self.events.append({
            "type": "images",
            "tag": tag,
            "step": global_step,
            "shape": list(arr.shape),
            "count": int(arr.shape[0]) if arr.ndim >= 1 else 0,
            "walltime": ts,
        })

    def add_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, float],
        global_step: int = 0,
    ) -> None:
        clean_metrics = {
            k: float(v.item() if hasattr(v, "item") else v) for k, v in metric_dict.items()
        }
        self.events.append({
            "type": "hparams",
            "step": global_step,
            "hparams": dict(hparam_dict),
            "metrics": clean_metrics,
            "walltime": time.time(),
        })
        for mk, mv in clean_metrics.items():
            self.add_scalar(mk, mv, global_step=global_step)

    def add_graph(self, model: Any, input_to_model: Any = None) -> None:
        del input_to_model
        summary = model_summary(model) if hasattr(model, "parameters") else repr(model)
        self.events.append({
            "type": "graph",
            "model_class": type(model).__name__,
            "summary": summary,
            "walltime": time.time(),
        })

    def add_text(self, tag: str, text_string: str, global_step: int = 0):
        self.events.append({
            "type": "text",
            "tag": tag,
            "step": global_step,
            "text": str(text_string),
            "walltime": time.time(),
        })

    def get_history(self, tag: Optional[str] = None):
        if tag is not None:
            return self.scalars.get(tag, [])
        return self.scalars

    def export_json(self, filepath: Optional[str] = None) -> str:
        import json
        out_path = filepath or os.path.join(self.log_dir, "events.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2)
        return out_path

    def flush(self):
        pass

    def close(self):
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def count_parameters(model: Any, trainable_only: bool = False) -> int:
    """Returns the total number of parameters in an Oxideon model or collection."""
    params = model.parameters() if hasattr(model, "parameters") else []
    if trainable_only:
        return sum(p.numel() for p in params if getattr(p, "requires_grad", True))
    return sum(p.numel() for p in params)


def freeze(model: Any) -> None:
    """Disables gradient tracking across all parameters in a model."""
    params = model.parameters() if hasattr(model, "parameters") else []
    for p in params:
        if hasattr(p, "requires_grad_"):
            p.requires_grad_(False)
        elif hasattr(p, "requires_grad"):
            p.requires_grad = False


def unfreeze(model: Any) -> None:
    """Enables gradient tracking across all parameters in a model."""
    params = model.parameters() if hasattr(model, "parameters") else []
    for p in params:
        if hasattr(p, "requires_grad_"):
            p.requires_grad_(True)
        elif hasattr(p, "requires_grad"):
            p.requires_grad = True


def get_device(tensor_or_model: Any) -> str:
    """Returns device name ('cpu', 'cuda', etc.) of a tensor or module."""
    if hasattr(tensor_or_model, "device"):
        dev = tensor_or_model.device
        return str(dev() if callable(dev) else dev)
    if hasattr(tensor_or_model, "parameters"):
        for p in tensor_or_model.parameters():
            if hasattr(p, "device"):
                dev = p.device
                return str(dev() if callable(dev) else dev)
    return "cpu"


# Re-exports from new Oxideon utility modules
from .checkpoint import checkpoint  # noqa: E402
from .treemap import (  # noqa: E402
    TreeSchema,
    LeafSchema,
    register_tree_node,
    unregister_tree_node,
    is_nested,
    tree_is_leaf,
    tree_flatten,
    tree_unflatten,
    tree_leaves,
    tree_schema,
    tree_map,
    tree_map_,
    tree_map_only,
    tree_all,
    tree_any,
    tree_all_only,
    tree_any_only,
    broadcast_tree_prefix,
    flatten_nested,
    assert_same_structure,
    pack_sequence_as,
    map_structure,
)
from .nn_utils import (  # noqa: E402
    total_norm,
    weight_norm,
    remove_weight_norm,
    spectral_norm,
    remove_spectral_norm,
    fuse_conv_batchnorm_weights,
    fuse_conv_batchnorm,
    fuse_linear_batchnorm_weights,
    fuse_linear_batchnorm,
    fast_init,
    apply_with_weights,
    bind_reparam,
    unbind_reparam,
    has_reparam,
    cache_reparam,
    ln_structured_prune,
    mask_prune,
    identity_prune,
    is_pruned,
    fusion,
    reparam,
    prune,
)
from .utils_ext import (  # noqa: E402
    HookHandle,
    ephemeral_hook,
    RuntimeInfo,
    collect_runtime_info,
    format_runtime_info,
    FlopProfiler,
    count_flops,
    fetch_weights_url,
    TimingReport,
    MicroTimer,
    TimingComparison,
    model_to_dot,
    model_to_mermaid,
    freeze_module_types,
    freeze_by_prefix,
    MemoryFootprint,
    estimate_memory_footprint,
    hooks,
    sysinfo,
    flops,
    dlpack,
    weights_cache,
    timing,
    graph_viz,
)
from .workflow_utils import (  # noqa: E402
    one_hot_encode,
    axis_normalize,
    pad_sequences,
    pack_batch,
    unpack_batch,
    compute_class_weight,
    compute_sample_weight,
    sliding_window_dataset,
    split_dataset,
    image_folder_dataset,
    text_folder_dataset,
    read_wav_tensor,
    save_wav_tensor,
    audio_folder_dataset,
    csv_dataset,
    jsonl_dataset,
    VocabularyIndex,
    build_vocabulary,
    ProgressBar,
    fetch_artifact,
    verify_checksum,
    SerializationScope,
    serialization_scope,
    register_serializable,
    get_registered_objects,
    get_registered_name,
    get_registered_object,
    serialize_config,
    deserialize_config,
    CheckpointVault,
    ShadowWeightEMA,
)
from .serialization import (  # noqa: E402
    save_sharded,
    load_sharded,
)
from .testing import (  # noqa: E402
    assert_finite,
    assert_shapes,
    gradgradcheck,
)

events = types.ModuleType("oxideon.utils.events")
events.SummaryWriter = SummaryWriter  # type: ignore[attr-defined]
events.__all__ = ["SummaryWriter"]  # type: ignore[attr-defined]

# Submodule registrations under `oxideon.utils.*` and `oxideon.nn.utils.*`
sys.modules["oxideon.utils.data"] = data
sys.modules["oxideon.utils.checkpoint"] = _checkpoint_mod
sys.modules["oxideon.utils.rnn"] = rnn
sys.modules["oxideon.utils.treemap"] = treemap
sys.modules["oxideon.utils.hooks"] = hooks
sys.modules["oxideon.utils.sysinfo"] = sysinfo
sys.modules["oxideon.utils.flops"] = flops
sys.modules["oxideon.utils.dlpack"] = dlpack
sys.modules["oxideon.utils.weights_cache"] = weights_cache
sys.modules["oxideon.utils.timing"] = timing
sys.modules["oxideon.utils.graph_viz"] = graph_viz
sys.modules["oxideon.utils.events"] = events

sys.modules["oxideon.nn.utils"] = sys.modules[__name__]
sys.modules["oxideon.nn.utils.rnn"] = rnn
sys.modules["oxideon.nn.utils.fusion"] = fusion
sys.modules["oxideon.nn.utils.reparam"] = reparam
sys.modules["oxideon.nn.utils.prune"] = prune


__all__ = [
    # Core utilities
    "parameters_to_vector",
    "vector_to_parameters",
    "set_seed",
    "set_random_seed",
    "clip_grad_norm_",
    "clip_grad_value_",
    "total_norm",
    "model_summary",
    "BenchmarkTimer",
    "SummaryWriter",
    "count_parameters",
    "freeze",
    "unfreeze",
    "freeze_module_types",
    "freeze_by_prefix",
    "get_device",
    "data",
    "checkpoint",
    # Tree mapping & nested container utilities
    "TreeSchema",
    "LeafSchema",
    "register_tree_node",
    "unregister_tree_node",
    "is_nested",
    "tree_is_leaf",
    "tree_flatten",
    "tree_unflatten",
    "tree_leaves",
    "tree_schema",
    "tree_map",
    "tree_map_",
    "tree_map_only",
    "tree_all",
    "tree_any",
    "tree_all_only",
    "tree_any_only",
    "broadcast_tree_prefix",
    "flatten_nested",
    "assert_same_structure",
    "pack_sequence_as",
    "map_structure",
    # Neural parameterization, fusion & pruning
    "weight_norm",
    "remove_weight_norm",
    "spectral_norm",
    "remove_spectral_norm",
    "fuse_conv_batchnorm_weights",
    "fuse_conv_batchnorm",
    "fuse_linear_batchnorm_weights",
    "fuse_linear_batchnorm",
    "fast_init",
    "apply_with_weights",
    "bind_reparam",
    "unbind_reparam",
    "has_reparam",
    "cache_reparam",
    "ln_structured_prune",
    "mask_prune",
    "identity_prune",
    "is_pruned",
    # Runtime, FLOP profiling, graph rendering, hooks & microtiming
    "HookHandle",
    "ephemeral_hook",
    "RuntimeInfo",
    "collect_runtime_info",
    "format_runtime_info",
    "FlopProfiler",
    "count_flops",
    "fetch_weights_url",
    "TimingReport",
    "MicroTimer",
    "TimingComparison",
    "model_to_dot",
    "model_to_mermaid",
    "MemoryFootprint",
    "estimate_memory_footprint",
    # Data encoding, folder/window loaders, vocabulary, progress, caching, serialization & EMA
    "one_hot_encode",
    "axis_normalize",
    "pad_sequences",
    "pack_batch",
    "unpack_batch",
    "compute_class_weight",
    "compute_sample_weight",
    "sliding_window_dataset",
    "split_dataset",
    "image_folder_dataset",
    "text_folder_dataset",
    "read_wav_tensor",
    "save_wav_tensor",
    "audio_folder_dataset",
    "csv_dataset",
    "jsonl_dataset",
    "VocabularyIndex",
    "build_vocabulary",
    "ProgressBar",
    "fetch_artifact",
    "verify_checksum",
    "SerializationScope",
    "serialization_scope",
    "register_serializable",
    "get_registered_objects",
    "get_registered_name",
    "get_registered_object",
    "serialize_config",
    "deserialize_config",
    "CheckpointVault",
    "ShadowWeightEMA",
    "save_sharded",
    "load_sharded",
    # Numerical & shape contract assertions
    "assert_finite",
    "assert_shapes",
    "gradgradcheck",
]
