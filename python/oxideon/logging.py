"""
Comprehensive Logging and Debugging Framework for Oxideon.
Provides structured logging, real-time training process tracking,
tensor health inspection (NaN/Inf checks, gradient norms), and global debug switches.
"""
from contextlib import contextmanager
from enum import IntEnum
import datetime
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


_GLOBAL_DEBUG: bool = False
_GLOBAL_LOG_LEVEL: LogLevel = LogLevel.INFO

# ANSI styling codes
_COLORS = {
    LogLevel.DEBUG: "\033[36m",     # Cyan
    LogLevel.INFO: "\033[32m",      # Green
    LogLevel.WARNING: "\033[33m",   # Yellow
    LogLevel.ERROR: "\033[31m",     # Red
    LogLevel.CRITICAL: "\033[41m",  # Red Background
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def is_debug() -> bool:
    """Returns True if global Oxideon debug mode is active."""
    return _GLOBAL_DEBUG


def set_debug(enabled: bool = True):
    """Enables or disables global debug mode across Oxideon."""
    global _GLOBAL_DEBUG, _GLOBAL_LOG_LEVEL
    _GLOBAL_DEBUG = bool(enabled)
    if _GLOBAL_DEBUG:
        _GLOBAL_LOG_LEVEL = LogLevel.DEBUG
    else:
        _GLOBAL_LOG_LEVEL = LogLevel.INFO


def enable_debug():
    """Convenience alias to enable debug mode."""
    set_debug(True)


def disable_debug():
    """Convenience alias to disable debug mode."""
    set_debug(False)


@contextmanager
def debug_mode(enabled: bool = True):
    """Context manager for locally toggling debug mode."""
    prev = is_debug()
    set_debug(enabled)
    try:
        yield
    finally:
        set_debug(prev)


def set_log_level(level: Union[str, int, LogLevel]):
    """Sets the minimum logging threshold."""
    global _GLOBAL_LOG_LEVEL
    if isinstance(level, str):
        level_map = {
            "debug": LogLevel.DEBUG,
            "info": LogLevel.INFO,
            "warning": LogLevel.WARNING,
            "warn": LogLevel.WARNING,
            "error": LogLevel.ERROR,
            "critical": LogLevel.CRITICAL,
        }
        _GLOBAL_LOG_LEVEL = level_map.get(level.lower(), LogLevel.INFO)
    elif isinstance(level, (int, LogLevel)):
        _GLOBAL_LOG_LEVEL = LogLevel(int(level))


class OxideonLogger:
    """High-performance structured logger with formatted console and stream output."""
    def __init__(self, name: str = "oxideon", stream=None):
        self.name = name
        self.stream = stream or sys.stdout
        self._step_start_time = time.perf_counter()

    def _log(self, level: LogLevel, msg: str, extra: Optional[Dict[str, Any]] = None):
        if level < _GLOBAL_LOG_LEVEL:
            return

        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        color = _COLORS.get(level, "")
        lvl_name = level.name.ljust(8)
        header = f"{_DIM}[{ts}]{_RESET} {color}{_BOLD}[{lvl_name}]{_RESET} {_BOLD}[{self.name}]{_RESET}"

        full_msg = f"{header} {msg}"
        if extra:
            extra_str = " | ".join([f"{k}={v}" for k, v in extra.items()])
            full_msg += f" {_DIM}({extra_str}){_RESET}"

        print(full_msg, file=self.stream, flush=True)

    def debug(self, msg: str, **kwargs):
        self._log(LogLevel.DEBUG, msg, kwargs if kwargs else None)

    def info(self, msg: str, **kwargs):
        self._log(LogLevel.INFO, msg, kwargs if kwargs else None)

    def warning(self, msg: str, **kwargs):
        self._log(LogLevel.WARNING, msg, kwargs if kwargs else None)

    def error(self, msg: str, **kwargs):
        self._log(LogLevel.ERROR, msg, kwargs if kwargs else None)

    def critical(self, msg: str, **kwargs):
        self._log(LogLevel.CRITICAL, msg, kwargs if kwargs else None)

    def log_training_step(
        self,
        epoch: int,
        step: int,
        total_steps: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        samples_per_sec: Optional[float] = None,
        lr: Optional[float] = None,
    ):
        """Logs a formatted real-time training step with throughput and loss."""
        pct = (step / max(total_steps, 1)) * 100.0
        bar_len = 16
        filled = int(round(bar_len * (step / max(total_steps, 1))))
        bar = "=" * filled + (">" if filled < bar_len else "") + "." * (bar_len - filled - (1 if filled < bar_len else 0))

        extra_parts = [f"loss={loss:.4f}"]
        if lr is not None:
            extra_parts.append(f"lr={lr:.2e}")
        if samples_per_sec is not None:
            extra_parts.append(f"{samples_per_sec:.1f} samples/s")
        if metrics:
            for k, v in metrics.items():
                extra_parts.append(f"{k}={v:.4f}")

        details = " | ".join(extra_parts)
        msg = f"Epoch {epoch:02d} [{bar}] {step}/{total_steps} ({pct:5.1f}%) | {details}"
        self.info(msg)

    def log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_acc: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        elapsed_sec: float = 0.0,
        samples_per_sec: Optional[float] = None,
    ):
        """Prints a prominent visual summary table for an epoch."""
        sep = "-" * 70
        title = f" EPOCH {epoch:02d}/{total_epochs:02d} SUMMARY (Elapsed: {elapsed_sec:.2f}s) "
        header = title.center(70, "=")

        lines = [
            header,
            f"  * Train Loss : {train_loss:.5f}" + (f"  |  Train Acc: {train_acc*100.0:.2f}%" if train_acc is not None else ""),
        ]
        if val_loss is not None:
            val_line = f"  * Val Loss   : {val_loss:.5f}"
            if val_acc is not None:
                val_line += f"  |  Val Acc  : {val_acc*100.0:.2f}%"
            lines.append(val_line)

        if samples_per_sec is not None:
            lines.append(f"  * Throughput : {samples_per_sec:.1f} samples/second")

        lines.append(sep)
        for line in lines:
            print(line, file=self.stream, flush=True)

    def inspect_tensor(self, tensor: Any, name: str = "tensor", check_nan: bool = True) -> Dict[str, Any]:
        """Detailed statistical diagnosis and NaN/Inf audit of a tensor."""
        arr = tensor.to_numpy() if hasattr(tensor, "to_numpy") else np.asarray(tensor)
        shape = list(arr.shape)
        dtype = str(arr.dtype)
        num_elements = int(arr.size)

        is_num = np.issubdtype(arr.dtype, np.number)
        has_nan = bool(np.isnan(arr).any()) if is_num else False
        has_inf = bool(np.isinf(arr).any()) if is_num else False

        stats = {
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "size": num_elements,
            "has_nan": has_nan,
            "has_inf": has_inf,
        }

        if num_elements > 0 and is_num:
            finite_vals = arr[np.isfinite(arr)]
            if finite_vals.size > 0:
                stats["min"] = float(np.min(finite_vals))
                stats["max"] = float(np.max(finite_vals))
                stats["mean"] = float(np.mean(finite_vals))
                stats["std"] = float(np.std(finite_vals))
                stats["norm_l2"] = float(np.linalg.norm(finite_vals))
            else:
                stats["min"] = float("nan")
                stats["max"] = float("nan")
                stats["mean"] = float("nan")
                stats["std"] = float("nan")
                stats["norm_l2"] = float("nan")

        if check_nan and (has_nan or has_inf):
            self.error(f"[NaN/Inf Alert] Tensor '{name}' contains abnormal values: has_nan={has_nan}, has_inf={has_inf}!")
        elif is_debug():
            stat_msg = f"shape={shape} dtype={dtype} min={stats.get('min', 0.0):.4f} max={stats.get('max', 0.0):.4f} mean={stats.get('mean', 0.0):.4f} std={stats.get('std', 0.0):.4f}"
            self.debug(f"[Tensor Audit] '{name}': {stat_msg}")

        return stats

    def inspect_gradients(self, model: Any) -> Dict[str, Dict[str, Any]]:
        """Inspects gradient tensors across all parameters of a model."""
        report: Dict[str, Dict[str, Any]] = {}
        named_params: List[Tuple[str, Any]] = []
        if hasattr(model, "named_parameters") and callable(model.named_parameters):
            named_params = list(model.named_parameters())
        else:
            for attr in ("weight", "bias", "gamma", "beta"):
                val = getattr(model, attr, None)
                if val is not None and hasattr(val, "to_numpy"):
                    named_params.append((attr, val))

        for name, param in named_params:
            grad = getattr(param, "grad", None)
            if grad is not None:
                report[name] = self.inspect_tensor(grad, name=f"{name}.grad")
        return report


_DEFAULT_LOGGER = OxideonLogger("oxideon")


def get_logger(name: str = "oxideon") -> OxideonLogger:
    return OxideonLogger(name)


def debug_tensor(tensor: Any, name: str = "tensor") -> Dict[str, Any]:
    """Inspects a tensor using the default logger."""
    return _DEFAULT_LOGGER.inspect_tensor(tensor, name=name)


def inspect_gradients(model: Any) -> Dict[str, Dict[str, Any]]:
    """Inspects all parameter gradients of a model using the default logger."""
    return _DEFAULT_LOGGER.inspect_gradients(model)


def check_nan(tensor: Any, name: str = "tensor") -> bool:
    """Returns True if tensor contains NaN or Inf values; raises FloatingPointError if debug is active."""
    arr = tensor.to_numpy() if hasattr(tensor, "to_numpy") else np.asarray(tensor)
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())
    if has_nan or has_inf:
        msg = f"Tensor '{name}' contains NaN={has_nan} or Inf={has_inf}!"
        _DEFAULT_LOGGER.error(msg)
        if is_debug():
            raise FloatingPointError(msg)
        return True
    return False


__all__ = [
    "LogLevel",
    "is_debug",
    "set_debug",
    "enable_debug",
    "disable_debug",
    "debug_mode",
    "set_log_level",
    "OxideonLogger",
    "get_logger",
    "debug_tensor",
    "inspect_gradients",
    "check_nan",
]
