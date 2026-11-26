"""Extended runtime, profiling, hook, timing, and interchange submodules for `oxideon.utils`:
- `hooks`: `HookHandle`, `ephemeral_hook`
- `sysinfo`: `RuntimeInfo`, `collect_runtime_info`, `format_runtime_info`
- `flops`: `FlopProfiler`, `count_flops`
- `dlpack`: `to_dlpack`, `from_dlpack`
- `weights_cache`: `fetch_weights_url`
- `timing`: `MicroTimer`, `TimingReport`, `TimingComparison`
"""

from dataclasses import dataclass
import itertools
import os
import platform
import sys
import timeit as _py_timeit
import types
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .oxideon import Tensor as _Tensor


# ---------------------------------------------------------------------------
# `oxideon.utils.hooks`
# ---------------------------------------------------------------------------

class HookHandle:
    """Handle capable of detaching a registered callback from one or more hook registries."""

    _next_id = itertools.count()

    def __init__(self, hooks_dict: Dict[int, Any], *, extra_dict: Optional[Any] = None) -> None:
        self.id = next(self._next_id)
        self.hooks_dict = hooks_dict
        self.extra_dict = extra_dict

    def remove(self) -> None:
        if self.hooks_dict is not None:
            self.hooks_dict.pop(self.id, None)
        if self.extra_dict is not None:
            if isinstance(self.extra_dict, (list, tuple)):
                for d in self.extra_dict:
                    if d is not None:
                        d.pop(self.id, None)
            else:
                self.extra_dict.pop(self.id, None)

    def __enter__(self) -> "HookHandle":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove()


def ephemeral_hook(f: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator marking a callback hook as transient and excluded from serialization."""
    f.__oxideon_unserializable__ = True  # type: ignore[attr-defined]
    return f


hooks = types.ModuleType("oxideon.utils.hooks")
hooks.HookHandle = HookHandle  # type: ignore[attr-defined]
hooks.ephemeral_hook = ephemeral_hook  # type: ignore[attr-defined]
hooks.__all__ = ["HookHandle", "ephemeral_hook"]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# `oxideon.utils.sysinfo`
# ---------------------------------------------------------------------------

@dataclass
class RuntimeInfo:
    oxideon_version: str
    python_version: str
    python_platform: str
    os: str
    cpu_architecture: str
    cpu_count: int
    numpy_version: str
    cuda_available: bool
    cuda_device_count: int
    mps_available: bool


def collect_runtime_info() -> RuntimeInfo:
    """Collects runtime, OS, CPU, and accelerator availability metadata."""
    import oxideon as ox

    cuda_avail = bool(ox.cuda.is_available()) if hasattr(ox, "cuda") else False
    cuda_cnt = int(ox.cuda.device_count()) if cuda_avail and hasattr(ox.cuda, "device_count") else 0
    mps_avail = bool(ox.mps.is_available()) if hasattr(ox, "mps") and hasattr(ox.mps, "is_available") else False

    return RuntimeInfo(
        oxideon_version=str(getattr(ox, "__version__", "unknown")),
        python_version=sys.version.replace("\n", " "),
        python_platform=platform.platform(),
        os=f"{platform.system()} {platform.release()}",
        cpu_architecture=platform.machine() or "unknown",
        cpu_count=os.cpu_count() or 1,
        numpy_version=np.__version__,
        cuda_available=cuda_avail,
        cuda_device_count=cuda_cnt,
        mps_available=mps_avail,
    )


def format_runtime_info() -> str:
    """Formats runtime, OS, CPU, and accelerator diagnostics into a human-readable summary."""
    env = collect_runtime_info()
    lines = [
        "Oxideon Runtime Environment",
        f"Oxideon version: {env.oxideon_version}",
        f"OS: {env.os} ({env.cpu_architecture})",
        f"Platform: {env.python_platform}",
        f"Python version: {env.python_version}",
        f"CPU logical cores: {env.cpu_count}",
        f"NumPy version: {env.numpy_version}",
        f"CUDA available: {env.cuda_available} (devices: {env.cuda_device_count})",
        f"MPS available: {env.mps_available}",
    ]
    return "\n".join(lines)


sysinfo = types.ModuleType("oxideon.utils.sysinfo")
sysinfo.RuntimeInfo = RuntimeInfo  # type: ignore[attr-defined]
sysinfo.collect_runtime_info = collect_runtime_info  # type: ignore[attr-defined]
sysinfo.format_runtime_info = format_runtime_info  # type: ignore[attr-defined]
sysinfo.__all__ = ["RuntimeInfo", "collect_runtime_info", "format_runtime_info"]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# `oxideon.utils.flops`
# ---------------------------------------------------------------------------

def _estimate_module_flops(module: Any, inputs: Tuple[Any, ...], output: Any) -> int:
    cls_name = type(module).__name__
    inp = inputs[0] if inputs and isinstance(inputs[0], _Tensor) else None
    out = output[0] if isinstance(output, tuple) and output and isinstance(output[0], _Tensor) else (
        output if isinstance(output, _Tensor) else None
    )

    if inp is None or out is None:
        return 0

    w = getattr(module, "weight", None)
    if cls_name in ("Linear", "SpectralNormLinear") and isinstance(w, _Tensor) and len(w.shape) == 2:
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        batch_elems = int(out.numel()) // max(out_f, 1)
        flops = 2 * batch_elems * in_f * out_f
        if getattr(module, "bias", None) is not None:
            flops += batch_elems * out_f
        return flops

    if ("Conv" in cls_name) and isinstance(w, _Tensor) and len(w.shape) >= 3:
        out_elems = int(out.numel())
        kernel_ops = int(np.prod(list(w.shape[1:])))
        flops = 2 * out_elems * kernel_ops
        if getattr(module, "bias", None) is not None:
            flops += out_elems
        return flops

    if cls_name == "MultiHeadAttention" and hasattr(module, "embed_dim"):
        d = int(module.embed_dim)
        tokens = int(inp.numel()) // max(d, 1)
        seq_len = int(inp.shape[1]) if len(inp.shape) >= 3 and getattr(module, "batch_first", False) else int(inp.shape[0])
        batch = max(tokens // max(seq_len, 1), 1)
        return 8 * tokens * d * d + 4 * batch * seq_len * seq_len * d

    if "Norm" in cls_name:
        return 5 * int(out.numel())

    return 0


class FlopProfiler:
    """Context manager measuring per-layer and total forward floating-point operations."""

    def __init__(self, mods: Optional[Union[Any, Sequence[Any]]] = None, depth: int = 2, display: bool = False) -> None:
        if mods is None:
            self._mods: List[Any] = []
        elif isinstance(mods, (list, tuple)):
            self._mods = list(mods)
        else:
            self._mods = [mods]
        self.depth = depth
        self.display = display
        self.flop_counts: Dict[str, int] = {}
        self._total_flops: int = 0
        self._restorers: List[Callable[[], None]] = []

    def register(self, model: Any) -> "FlopProfiler":
        self._mods.append(model)
        return self

    def _instrument(self, root: Any, root_prefix: str = "") -> None:
        named = list(root.named_modules()) if hasattr(root, "named_modules") else [("", root)]
        for name, mod in named:
            full_name = f"{root_prefix}.{name}".strip(".") if root_prefix else (name or type(mod).__name__)
            has_children = bool(getattr(mod, "_modules", None))
            if has_children and type(mod).__name__ != "MultiHeadAttention":
                continue

            if hasattr(mod, "register_forward_hook"):
                def _make_hook(m_name: str):
                    def _hook(m: Any, args: Any, out: Any) -> None:
                        fl = _estimate_module_flops(m, args, out)
                        if fl > 0:
                            self._record(m_name, fl)
                    return _hook

                handle = mod.register_forward_hook(_make_hook(full_name))
                self._restorers.append(handle.remove)

        for parent_prefix, parent in named:
            children = getattr(parent, "_modules", None)
            if not children:
                continue
            for ch_name, ch in list(children.items()):
                if ch is None or hasattr(ch, "register_forward_hook"):
                    continue
                full_ch = f"{parent_prefix}.{ch_name}".strip(".")

                class _NativeFlopTap:
                    def __init__(self, target: Any, m_name: str, counter: "FlopProfiler") -> None:
                        object.__setattr__(self, "_target", target)
                        object.__setattr__(self, "_m_name", m_name)
                        object.__setattr__(self, "_counter", counter)

                    def __call__(self, *a: Any, **kw: Any) -> Any:
                        t = object.__getattribute__(self, "_target")
                        out = t(*a, **kw)
                        fl = _estimate_module_flops(t, a, out)
                        if fl > 0:
                            object.__getattribute__(self, "_counter")._record(
                                object.__getattribute__(self, "_m_name"), fl
                            )
                        return out

                    def __getattr__(self, item: str) -> Any:
                        return getattr(object.__getattribute__(self, "_target"), item)

                tap = _NativeFlopTap(ch, full_ch, self)
                had_attr = parent.__dict__.get(ch_name) is ch
                children[ch_name] = tap
                if had_attr:
                    object.__setattr__(parent, ch_name, tap)

                def _restore(p=parent, cn=ch_name, orig=ch, ha=had_attr) -> None:
                    p._modules[cn] = orig
                    if ha:
                        object.__setattr__(p, cn, orig)

                self._restorers.append(_restore)

    def _record(self, full_name: str, flops: int) -> None:
        self._total_flops += flops
        self.flop_counts["Global"] = self.flop_counts.get("Global", 0) + flops
        parts = full_name.split(".")
        for d in range(1, min(len(parts), self.depth) + 1):
            key = ".".join(parts[:d])
            self.flop_counts[key] = self.flop_counts.get(key, 0) + flops

    def get_total_flops(self) -> int:
        return self._total_flops

    def get_flop_counts(self) -> Dict[str, int]:
        return dict(self.flop_counts)

    def get_table(self, depth: Optional[int] = None) -> str:
        d_lim = self.depth if depth is None else depth
        header = f"{'Module':<36} | {'FLOPs':>16} | {'% Total':>9}"
        sep = "-" * len(header)
        lines = [sep, header, sep]
        total = max(self._total_flops, 1)
        for key, val in self.flop_counts.items():
            if key != "Global" and key.count(".") >= d_lim:
                continue
            pct = 100.0 * val / total
            lines.append(f"{key:<36} | {val:>16,d} | {pct:>8.2f}%")
        lines.append(sep)
        return "\n".join(lines)

    def __enter__(self) -> "FlopProfiler":
        self.flop_counts.clear()
        self._total_flops = 0
        for m in self._mods:
            self._instrument(m)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for restore in reversed(self._restorers):
            restore()
        self._restorers.clear()
        if self.display:
            print(self.get_table())


def count_flops(model: Any, *args: Any, **kwargs: Any) -> int:
    """Runs `model(*args, **kwargs)` inside `FlopProfiler` and returns total forward FLOPs."""
    from .primitives import no_grad

    if not hasattr(model, "_modules") or not getattr(model, "_modules", None):
        with no_grad():
            out = model(*args, **kwargs)
        return _estimate_module_flops(model, args, out)

    with FlopProfiler(model) as counter, no_grad():
        model(*args, **kwargs)
    return counter.get_total_flops()


flops = types.ModuleType("oxideon.utils.flops")
flops.FlopProfiler = FlopProfiler  # type: ignore[attr-defined]
flops.count_flops = count_flops  # type: ignore[attr-defined]
flops.__all__ = ["FlopProfiler", "count_flops"]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# `oxideon.utils.dlpack` & `oxideon.utils.weights_cache`
# ---------------------------------------------------------------------------

def _build_dlpack_module() -> types.ModuleType:
    mod = types.ModuleType("oxideon.utils.dlpack")

    def _to_dlpack(tensor: Any) -> Any:
        from .interop import to_dlpack
        return to_dlpack(tensor)

    def _from_dlpack(ext_tensor: Any, device: str = "cpu") -> _Tensor:
        from .interop import from_dlpack
        return from_dlpack(ext_tensor, device=device)

    mod.to_dlpack = _to_dlpack  # type: ignore[attr-defined]
    mod.from_dlpack = _from_dlpack  # type: ignore[attr-defined]
    mod.__all__ = ["to_dlpack", "from_dlpack"]  # type: ignore[attr-defined]
    return mod


def fetch_weights_url(
    url: str,
    model_dir: Optional[str] = None,
    map_location: Optional[str] = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None,
) -> Any:
    """Downloads, caches, and loads a state dictionary from a remote URL."""
    from .hub import load_state_dict_from_url

    return load_state_dict_from_url(
        url,
        model_dir=model_dir,
        map_location=map_location,
        progress=progress,
        check_hash=check_hash,
        file_name=file_name,
    )


def _build_weights_cache_module() -> types.ModuleType:
    mod = types.ModuleType("oxideon.utils.weights_cache")
    mod.fetch_weights_url = fetch_weights_url  # type: ignore[attr-defined]
    mod.__all__ = ["fetch_weights_url"]  # type: ignore[attr-defined]
    return mod


dlpack = _build_dlpack_module()
weights_cache = _build_weights_cache_module()


# ---------------------------------------------------------------------------
# `oxideon.utils.timing` (`MicroTimer` / `TimingReport` / `TimingComparison`)
# ---------------------------------------------------------------------------

class TimingReport:
    """Statistical summary of repeated microbenchmark runs."""

    def __init__(
        self,
        number_per_run: int,
        raw_times: Sequence[float],
        label: Optional[str] = None,
        sub_label: Optional[str] = None,
        description: Optional[str] = None,
        num_threads: int = 1,
    ) -> None:
        if number_per_run < 1:
            raise ValueError("number_per_run must be >= 1")
        if not raw_times:
            raise ValueError("raw_times must be non-empty")
        self.number_per_run = int(number_per_run)
        self.raw_times = [float(t) for t in raw_times]
        self.times = [t / self.number_per_run for t in self.raw_times]
        self.label = label or "Benchmark"
        self.sub_label = sub_label
        self.description = description
        self.num_threads = int(num_threads)

    @property
    def mean(self) -> float:
        return float(np.mean(self.times))

    @property
    def median(self) -> float:
        return float(np.median(self.times))

    @property
    def iqr(self) -> float:
        q75, q25 = np.percentile(self.times, [75, 25])
        return float(q75 - q25)

    def __repr__(self) -> str:
        med_us = self.median * 1e6
        iqr_us = self.iqr * 1e6
        desc = f" ({self.description})" if self.description else ""
        sub = f" [{self.sub_label}]" if self.sub_label else ""
        return (
            f"<TimingReport {self.label}{sub}{desc}: "
            f"median={med_us:.2f} us, iqr={iqr_us:.2f} us "
            f"({len(self.raw_times)} runs, {self.number_per_run} iters/run)>"
        )


class MicroTimer:
    """Sub-microsecond statement and callable microbenchmark timer with adaptive block sizing."""

    def __init__(
        self,
        stmt: Union[str, Callable[[], Any]] = "pass",
        setup: Union[str, Callable[[], Any]] = "pass",
        globals: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
        sub_label: Optional[str] = None,
        description: Optional[str] = None,
        num_threads: int = 1,
    ) -> None:
        self.stmt = stmt
        self.setup = setup
        self.globals = dict(globals or {})
        self.label = label
        self.sub_label = sub_label
        self.description = description
        self.num_threads = num_threads
        self._timer = _py_timeit.Timer(stmt=stmt, setup=setup, globals=self.globals)

    def run(self, iterations: int = 10000) -> TimingReport:
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        self._timer.timeit(number=1)
        elapsed = self._timer.timeit(number=iterations)
        return TimingReport(
            number_per_run=iterations,
            raw_times=[elapsed],
            label=self.label,
            sub_label=self.sub_label,
            description=self.description,
            num_threads=self.num_threads,
        )

    def autorange(
        self,
        callback: Optional[Callable[[int, float], None]] = None,
        min_run_time: float = 0.2,
    ) -> TimingReport:
        self._timer.timeit(number=1)
        number = 1
        while True:
            t0 = self._timer.timeit(number=number)
            if t0 >= min_run_time * 0.1 or number >= 1_000_000:
                break
            number *= 10

        raw_times: List[float] = []
        total = 0.0
        while total < min_run_time or len(raw_times) < 2:
            dt = self._timer.timeit(number=number)
            raw_times.append(dt)
            total += dt
            if callback is not None:
                callback(number, dt)
            if len(raw_times) >= 100:
                break

        return TimingReport(
            number_per_run=number,
            raw_times=raw_times,
            label=self.label,
            sub_label=self.sub_label,
            description=self.description,
            num_threads=self.num_threads,
        )


class TimingComparison:
    """Formats a collection of `TimingReport` results into a side-by-side comparison table."""

    def __init__(self, results: Sequence[TimingReport]) -> None:
        self.results = list(results)

    def format(self) -> str:
        lines = [f"{'Label':<24} | {'Sub-label':<18} | {'Description':<18} | {'Median (us)':>12}"]
        sep = "-" * len(lines[0])
        lines.insert(0, sep)
        lines.insert(2, sep)
        for m in self.results:
            lines.append(
                f"{(m.label or ''):<24} | {(m.sub_label or ''):<18} | "
                f"{(m.description or ''):<18} | {m.median * 1e6:>12.2f}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def print(self) -> None:
        print(self.format())

    def __str__(self) -> str:
        return self.format()


def _build_timing_submodule() -> types.ModuleType:
    mod = types.ModuleType("oxideon.utils.timing")
    mod.MicroTimer = MicroTimer  # type: ignore[attr-defined]
    mod.TimingReport = TimingReport  # type: ignore[attr-defined]
    mod.TimingComparison = TimingComparison  # type: ignore[attr-defined]

    def _bench(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        from .benchmark import benchmark as _bench_fn
        return _bench_fn(fn, *args, **kwargs)

    def _prof(model: Any, sample_input: Any, num_runs: int = 10) -> Dict[str, Any]:
        from .benchmark import profile as _prof_fn
        return _prof_fn(model, sample_input, num_runs=num_runs)

    mod.benchmark = _bench  # type: ignore[attr-defined]
    mod.profile = _prof  # type: ignore[attr-defined]
    mod.__all__ = [  # type: ignore[attr-defined]
        "MicroTimer",
        "TimingReport",
        "TimingComparison",
        "benchmark",
        "profile",
    ]
    return mod


timing = _build_timing_submodule()


# ---------------------------------------------------------------------------
# Model Architecture Exporters (`model_to_dot`, `model_to_mermaid`) & Selective Freezing
# ---------------------------------------------------------------------------

def _iter_leaf_and_container_nodes(model: Any) -> List[Tuple[str, Any]]:
    nodes: List[Tuple[str, Any]] = []
    visited = set()

    def _walk(prefix: str, mod: Any) -> None:
        mid = id(mod)
        if mid in visited:
            return
        visited.add(mid)
        children = getattr(mod, "_modules", None)
        if isinstance(children, dict) and children:
            for cname, child in children.items():
                if child is None:
                    continue
                sub_name = f"{prefix}.{cname}" if prefix else str(cname)
                _walk(sub_name, child)
        else:
            nodes.append((prefix or type(mod).__name__, mod))

    _walk("", model)
    return nodes


def _module_param_stats(mod: Any) -> Tuple[int, bool]:
    total = 0
    any_trainable = False
    params_fn = getattr(mod, "parameters", None)
    if callable(params_fn):
        for p in params_fn():
            n = int(np.prod(p.shape)) if getattr(p, "shape", None) else 1
            total += n
            if getattr(p, "requires_grad", False):
                any_trainable = True
    else:
        for attr in ("weight", "bias"):
            p = getattr(mod, attr, None)
            if isinstance(p, _Tensor):
                n = int(np.prod(p.shape)) if p.shape else 1
                total += n
                if getattr(p, "requires_grad", False):
                    any_trainable = True
    return total, any_trainable


def _trace_layer_execution(model: Any, sample_input: Any) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    from .primitives import no_grad

    leaves = _iter_leaf_and_container_nodes(model)
    records: List[Dict[str, Any]] = []
    restorers: List[Callable[[], None]] = []

    def _shape_str(val: Any) -> str:
        if isinstance(val, _Tensor):
            return str(tuple(val.shape))
        if isinstance(val, (tuple, list)) and val:
            return ", ".join(_shape_str(v) for v in val[:3])
        return "?"

    if sample_input is None:
        for idx, (name, mod) in enumerate(leaves):
            n_params, trainable = _module_param_stats(mod)
            records.append(
                {
                    "id": idx,
                    "name": name,
                    "type": type(mod).__name__,
                    "in_shape": None,
                    "out_shape": None,
                    "params": n_params,
                    "trainable": trainable,
                }
            )
        edges = [(i, i + 1) for i in range(max(len(records) - 1, 0))]
        return records, edges

    # Instrument parent `_modules` slots to capture dynamic execution order and tensor shapes
    def _instrument(parent: Any, prefix: str = "") -> None:
        children = getattr(parent, "_modules", None)
        if not isinstance(children, dict):
            return
        for cname, child in list(children.items()):
            if child is None:
                continue
            full = f"{prefix}.{cname}" if prefix else str(cname)
            sub_children = getattr(child, "_modules", None)
            if isinstance(sub_children, dict) and sub_children:
                _instrument(child, full)
            else:
                orig_child = child
                has_attr = cname in getattr(parent, "__dict__", {})

                class _TraceProxy:
                    def __init__(self, target: Any, mod_name: str) -> None:
                        self._target = target
                        self._mod_name = mod_name

                    def __call__(self, *a: Any, **kw: Any) -> Any:
                        out = self._target(*a, **kw)
                        n_p, tr = _module_param_stats(self._target)
                        in_s = _shape_str(a[0] if len(a) == 1 else a)
                        out_s = _shape_str(out)
                        records.append(
                            {
                                "id": len(records),
                                "name": self._mod_name,
                                "type": type(self._target).__name__,
                                "in_shape": in_s,
                                "out_shape": out_s,
                                "params": n_p,
                                "trainable": tr,
                            }
                        )
                        return out

                    def __getattr__(self, item: str) -> Any:
                        return getattr(self._target, item)

                proxy = _TraceProxy(orig_child, full)
                children[cname] = proxy
                if has_attr:
                    object.__setattr__(parent, cname, proxy)

                def _restore(p=parent, k=cname, orig=orig_child, ha=has_attr) -> None:
                    p._modules[k] = orig
                    if ha:
                        object.__setattr__(p, k, orig)

                restorers.append(_restore)

    _instrument(model)
    try:
        with no_grad():
            if isinstance(sample_input, tuple):
                out = model(*sample_input)
            elif isinstance(sample_input, dict):
                out = model(**sample_input)
            else:
                out = model(sample_input)
        if not records:
            n_p, tr = _module_param_stats(model)
            records.append(
                {
                    "id": 0,
                    "name": type(model).__name__,
                    "type": type(model).__name__,
                    "in_shape": _shape_str(sample_input),
                    "out_shape": _shape_str(out),
                    "params": n_p,
                    "trainable": tr,
                }
            )
    finally:
        for r in reversed(restorers):
            r()

    edges = [(i, i + 1) for i in range(max(len(records) - 1, 0))]
    return records, edges


def model_to_dot(
    model: Any,
    sample_input: Any = None,
    show_shapes: bool = True,
    show_trainable: bool = True,
    rankdir: str = "TB",
) -> str:
    """Renders a model's layer topology (and optional tensor shapes) as a Graphviz DOT string."""
    if rankdir not in ("TB", "LR", "BT", "RL"):
        raise ValueError(f"rankdir must be one of TB, LR, BT, RL; got {rankdir!r}")
    records, edges = _trace_layer_execution(model, sample_input)

    lines = [
        "digraph Model {",
        f'  rankdir="{rankdir}";',
        '  node [shape=record, fontname="Helvetica", fontsize=10];',
    ]
    for rec in records:
        parts = [f"{rec['name']} ({rec['type']})"]
        if show_shapes and rec["in_shape"] is not None:
            parts.append(f"in: {rec['in_shape']} | out: {rec['out_shape']}")
        if rec["params"] > 0:
            parts.append(f"params: {rec['params']:,d}")
        if show_trainable and rec["params"] > 0:
            parts.append("trainable" if rec["trainable"] else "frozen")
        escaped = " | ".join(p.replace('"', '\\"').replace("<", "\\<").replace(">", "\\>") for p in parts)
        lines.append(f'  n{rec["id"]} [label="{{ {escaped} }}"];')

    for src_id, dst_id in edges:
        lines.append(f"  n{src_id} -> n{dst_id};")
    lines.append("}")
    return "\n".join(lines)


def model_to_mermaid(
    model: Any,
    sample_input: Any = None,
    show_shapes: bool = True,
    direction: str = "TD",
) -> str:
    """Renders a model's layer topology (and optional tensor shapes) as a Mermaid flowchart string."""
    if direction not in ("TD", "TB", "LR", "RL", "BT"):
        raise ValueError(f"direction must be one of TD, TB, LR, RL, BT; got {direction!r}")
    records, edges = _trace_layer_execution(model, sample_input)

    lines = [f"flowchart {direction}"]
    for rec in records:
        label = f"{rec['name']}: {rec['type']}"
        if show_shapes and rec["in_shape"] is not None:
            label += f" ({rec['in_shape']} -> {rec['out_shape']})"
        safe_label = label.replace('"', "'")
        lines.append(f'    n{rec["id"]}["{safe_label}"]')
    for src_id, dst_id in edges:
        lines.append(f"    n{src_id} --> n{dst_id}")
    return "\n".join(lines)


def freeze_module_types(
    model: Any,
    module_types: Union[type, Tuple[type, ...]],
    freeze: bool = True,
) -> int:
    """Sets `requires_grad_(not freeze)` on all parameters inside submodules matching `module_types`.

    Returns the number of parameter tensors updated.
    """
    types_tuple = module_types if isinstance(module_types, tuple) else (module_types,)
    req_grad = not bool(freeze)
    updated = 0
    seen_params = set()

    def _apply_mod(mod: Any) -> None:
        nonlocal updated
        if isinstance(mod, types_tuple):
            params_fn = getattr(mod, "parameters", None)
            if callable(params_fn):
                for p in params_fn():
                    if id(p) not in seen_params:
                        seen_params.add(id(p))
                        p.requires_grad_(req_grad)
                        updated += 1
            else:
                for attr in ("weight", "bias"):
                    p = getattr(mod, attr, None)
                    if isinstance(p, _Tensor) and id(p) not in seen_params:
                        seen_params.add(id(p))
                        p.requires_grad_(req_grad)
                        updated += 1
        children = getattr(mod, "_modules", None)
        if isinstance(children, dict):
            for child in children.values():
                if child is not None:
                    _apply_mod(child)

    _apply_mod(model)
    return updated


def freeze_by_prefix(
    model: Any,
    prefixes: Union[str, Sequence[str]],
    freeze: bool = True,
) -> int:
    """Sets `requires_grad_(not freeze)` on named parameters matching one or more name prefixes.

    Returns the number of parameter tensors updated.
    """
    pref_tuple = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
    req_grad = not bool(freeze)
    updated = 0
    named_params_fn = getattr(model, "named_parameters", None)
    if not callable(named_params_fn):
        raise TypeError(f"Model of type {type(model).__name__} does not implement named_parameters()")
    for name, param in named_params_fn():
        if any(name == pf or name.startswith(pf) for pf in pref_tuple):
            param.requires_grad_(req_grad)
            updated += 1
    return updated


graph_viz = types.ModuleType("oxideon.utils.graph_viz")
graph_viz.model_to_dot = model_to_dot  # type: ignore[attr-defined]
graph_viz.model_to_mermaid = model_to_mermaid  # type: ignore[attr-defined]
graph_viz.__all__ = ["model_to_dot", "model_to_mermaid"]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Model Memory Footprint Estimator (`MemoryFootprint`, `estimate_memory_footprint`)
# ---------------------------------------------------------------------------

_MEM_DTYPE_BYTES = {
    "bool": 1,
    "u8": 1,
    "uint8": 1,
    "i8": 1,
    "int8": 1,
    "f16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "i16": 2,
    "int16": 2,
    "f32": 4,
    "float32": 4,
    "i32": 4,
    "int32": 4,
    "f64": 8,
    "float64": 8,
    "i64": 8,
    "int64": 8,
}


def _tensor_tree_bytes(obj: Any) -> int:
    from .treemap import tree_leaves

    total = 0
    seen = set()
    for leaf in tree_leaves(obj):
        if isinstance(leaf, _Tensor) and id(leaf) not in seen:
            seen.add(id(leaf))
            numel = 1
            for d in getattr(leaf, "shape", ()):
                numel *= int(d)
            dt = str(getattr(leaf, "dtype", "f32")).lower()
            total += numel * _MEM_DTYPE_BYTES.get(dt, 4)
    return total


def _format_byte_count(n: int) -> str:
    val = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(val) < 1024.0 or unit == "TiB":
            return f"{int(val)} B" if unit == "B" else f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{val:.2f} TiB"


class MemoryFootprint:
    """Breakdown of parameter, buffer, gradient, optimizer-state, and activation memory usage."""

    def __init__(
        self,
        parameter_bytes: int,
        trainable_parameter_bytes: int,
        buffer_bytes: int,
        gradient_bytes: int,
        optimizer_state_bytes: int,
        activation_bytes: int,
        peak_layer_activation_bytes: int,
        per_layer_activation_bytes: Dict[str, int],
    ) -> None:
        self.parameter_bytes = int(parameter_bytes)
        self.trainable_parameter_bytes = int(trainable_parameter_bytes)
        self.buffer_bytes = int(buffer_bytes)
        self.gradient_bytes = int(gradient_bytes)
        self.optimizer_state_bytes = int(optimizer_state_bytes)
        self.activation_bytes = int(activation_bytes)
        self.peak_layer_activation_bytes = int(peak_layer_activation_bytes)
        self.per_layer_activation_bytes = dict(per_layer_activation_bytes)
        self.total_inference_bytes = (
            self.parameter_bytes + self.buffer_bytes + self.peak_layer_activation_bytes
        )
        self.total_training_bytes = (
            self.parameter_bytes
            + self.buffer_bytes
            + self.gradient_bytes
            + self.optimizer_state_bytes
            + self.activation_bytes
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_bytes": self.parameter_bytes,
            "trainable_parameter_bytes": self.trainable_parameter_bytes,
            "buffer_bytes": self.buffer_bytes,
            "gradient_bytes": self.gradient_bytes,
            "optimizer_state_bytes": self.optimizer_state_bytes,
            "activation_bytes": self.activation_bytes,
            "peak_layer_activation_bytes": self.peak_layer_activation_bytes,
            "total_inference_bytes": self.total_inference_bytes,
            "total_training_bytes": self.total_training_bytes,
            "per_layer_activation_bytes": dict(self.per_layer_activation_bytes),
        }

    def summary(self) -> str:
        rows = [
            ("Parameters (total)", self.parameter_bytes),
            ("Parameters (trainable)", self.trainable_parameter_bytes),
            ("Buffers", self.buffer_bytes),
            ("Gradients", self.gradient_bytes),
            ("Optimizer states", self.optimizer_state_bytes),
            ("Activations (cumulative)", self.activation_bytes),
            ("Total inference estimate", self.total_inference_bytes),
            ("Total training estimate", self.total_training_bytes),
        ]
        sep = "-" * 52
        lines = [sep, f"{'Category':<28} | {'Memory':>21}", sep]
        for label, b in rows:
            lines.append(f"{label:<28} | {_format_byte_count(b):>21}")
        lines.append(sep)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MemoryFootprint(params={_format_byte_count(self.parameter_bytes)}, "
            f"inference={_format_byte_count(self.total_inference_bytes)}, "
            f"training={_format_byte_count(self.total_training_bytes)})"
        )


def estimate_memory_footprint(
    model: Any,
    sample_input: Any = None,
    optimizer_states: int = 2,
    backward: bool = True,
) -> MemoryFootprint:
    """Estimates model RAM/VRAM usage across parameters, buffers, gradients, optimizer states, and activations."""
    if optimizer_states < 0:
        raise ValueError(f"optimizer_states must be >= 0, got {optimizer_states}")
    from .primitives import no_grad

    param_bytes = 0
    trainable_bytes = 0
    seen_ids = set()

    params_fn = getattr(model, "parameters", None)
    if callable(params_fn):
        for p in params_fn():
            if id(p) in seen_ids:
                continue
            seen_ids.add(id(p))
            nb = _tensor_tree_bytes(p)
            param_bytes += nb
            if getattr(p, "requires_grad", False):
                trainable_bytes += nb
    else:
        for attr in ("weight", "bias"):
            p = getattr(model, attr, None)
            if isinstance(p, _Tensor) and id(p) not in seen_ids:
                seen_ids.add(id(p))
                nb = _tensor_tree_bytes(p)
                param_bytes += nb
                if getattr(p, "requires_grad", False):
                    trainable_bytes += nb

    buf_bytes = 0
    buffers_fn = getattr(model, "buffers", None)
    if callable(buffers_fn):
        for b in buffers_fn():
            if isinstance(b, _Tensor) and id(b) not in seen_ids:
                seen_ids.add(id(b))
                buf_bytes += _tensor_tree_bytes(b)

    per_layer_act: Dict[str, int] = {}
    act_total = 0
    peak_act = 0

    if sample_input is not None:
        inp_bytes = _tensor_tree_bytes(sample_input)
        restorers: List[Callable[[], None]] = []

        def _instrument(parent: Any, prefix: str = "") -> None:
            children = getattr(parent, "_modules", None)
            if not isinstance(children, dict):
                return
            for cname, child in list(children.items()):
                if child is None:
                    continue
                full = f"{prefix}.{cname}" if prefix else str(cname)
                sub = getattr(child, "_modules", None)
                if isinstance(sub, dict) and sub:
                    _instrument(child, full)
                else:
                    orig = child
                    has_attr = cname in getattr(parent, "__dict__", {})

                    class _MemProxy:
                        def __init__(self, target: Any, name: str) -> None:
                            self._target = target
                            self._name = name

                        def __call__(self, *a: Any, **kw: Any) -> Any:
                            out = self._target(*a, **kw)
                            per_layer_act[self._name] = per_layer_act.get(self._name, 0) + _tensor_tree_bytes(out)
                            return out

                        def __getattr__(self, item: str) -> Any:
                            return getattr(self._target, item)

                    proxy = _MemProxy(orig, full)
                    children[cname] = proxy
                    if has_attr:
                        object.__setattr__(parent, cname, proxy)

                    def _restore(p=parent, k=cname, o=orig, ha=has_attr) -> None:
                        p._modules[k] = o
                        if ha:
                            object.__setattr__(p, k, o)

                    restorers.append(_restore)

        _instrument(model)
        try:
            with no_grad():
                if isinstance(sample_input, tuple):
                    out = model(*sample_input)
                elif isinstance(sample_input, dict):
                    out = model(**sample_input)
                else:
                    out = model(sample_input)
            if not per_layer_act:
                per_layer_act[type(model).__name__] = _tensor_tree_bytes(out)
        finally:
            for r in reversed(restorers):
                r()

        act_total = inp_bytes + sum(per_layer_act.values())
        peak_act = inp_bytes + max(per_layer_act.values(), default=0)

    grad_bytes = trainable_bytes if backward else 0
    opt_bytes = trainable_bytes * int(optimizer_states) if backward else 0

    return MemoryFootprint(
        parameter_bytes=param_bytes,
        trainable_parameter_bytes=trainable_bytes,
        buffer_bytes=buf_bytes,
        gradient_bytes=grad_bytes,
        optimizer_state_bytes=opt_bytes,
        activation_bytes=act_total,
        peak_layer_activation_bytes=peak_act,
        per_layer_activation_bytes=per_layer_act,
    )


__all__ = [
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
    "freeze_module_types",
    "freeze_by_prefix",
    "MemoryFootprint",
    "estimate_memory_footprint",
    "hooks",
    "sysinfo",
    "flops",
    "dlpack",
    "weights_cache",
    "timing",
    "graph_viz",
]
