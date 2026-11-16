"""High-level deep learning utilities: model summary, gradient clipping, seeding, and benchmarking."""

import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from . import data
from . import checkpoint as checkpoint
from . import rnn_utils as rnn
__path__ = []  # Exposes 'from oxideon.utils.data import ...'
sys.modules["oxideon.utils.data"] = data
sys.modules["oxideon.utils.checkpoint"] = checkpoint
sys.modules["oxideon.utils.rnn"] = rnn
sys.modules["oxideon.nn.utils.rnn"] = rnn

from .oxideon import manual_seed as _manual_seed


def set_seed(seed: int = 42):
    """Sets random seeds deterministically across Oxideon, NumPy, and Python standard random."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _manual_seed(seed)


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together as a single vector.
    Gradients are modified in-place or via parameter set_grad.
    """
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
        total_norm = max(float(g.abs().max().item()) for g in grads)
    elif norm_type <= 0.0:
        raise ValueError(f"clip_grad_norm_ requires norm_type > 0 or inf, got {norm_type}")
    else:
        # True Lp norm over all gradient elements (torch semantics): summing
        # per-tensor L2 norms is only correct for p == 2.
        total_norm = 0.0
        for g in grads:
            total_norm += float(g.abs().powf(norm_type).sum().item())
        total_norm = total_norm ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p, g in zip(param_list, grads):
            scaled_g = g.mul_scalar(clip_coef)
            if hasattr(p, "set_grad"):
                p.set_grad(scaled_g)

    return total_norm


def clip_grad_value_(parameters, clip_value: float):
    """Clips gradient values of an iterable of parameters to `[-clip_value, clip_value]`."""
    clip_value = float(clip_value)
    for p in parameters:
        g = p.grad() if hasattr(p, "grad") and callable(p.grad) else getattr(p, "grad", None)
        if g is not None and hasattr(g, "clamp"):
            clamped = g.clamp(-clip_value, clip_value)
            if hasattr(p, "set_grad"):
                p.set_grad(clamped)


# Native dtype tag -> bytes per element for model_summary size estimates.
_DTYPE_NUM_BYTES = {
    "f32": 4, "f64": 8, "f16": 2, "bf16": 2,
    "i32": 4, "i64": 8, "u8": 1, "bool": 1,
}


def model_summary(model, input_size=None) -> str:
    """Generates a detailed tabular architectural summary of an Oxideon `nn.Module`.

    Displays:
      - Layer names & classes
      - Parameter counts (weights, biases)
      - Trainable vs frozen status
      - Estimated memory size
      - Total parameters breakdown
    """
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
    """High-resolution precision benchmarking timer context manager."""

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
    """Lightweight experiment tracking and event writer (TensorBoard / JSON compatible)."""

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


from . import data

from .checkpoint import checkpoint


__all__ = [
    "set_seed",
    "clip_grad_norm_",
    "clip_grad_value_",
    "model_summary",
    "BenchmarkTimer",
    "SummaryWriter",
    "count_parameters",
    "freeze",
    "unfreeze",
    "get_device",
    "data",
    "checkpoint",
]
