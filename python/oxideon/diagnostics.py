"""Model and training diagnostics.

- `detect_anomaly(model)`: names the layer whose forward pass, or backward pass, first produces NaN or Inf.
- `precision_audit(model, *inputs)`: runs the model next to a float64 copy and reports the error each layer adds.
- `estimate_memory(model, *inputs, optimizer=...)`: parameter, gradient, optimizer-state and activation memory.
- `lr_find(model, loss_fn, data)`: the learning-rate range test, with a suggested learning rate.
- `check_determinism(fn)`: runs `fn` several times with the same seeds and reports the first value that differs.
- `diff_checkpoints(a, b)`: compares two state dicts (or models, or saved files) tensor by tensor.
- `TrainingMonitor(model)`: per-step gradient norms, update-to-weight ratios and activation statistics, with warnings.
- `hessian_top_eigenvalue` / `hessian_trace`: the sharpness of the loss at the current weights.

Layers are observed by temporarily wrapping every child module (native layers such as `Linear` included) in a probe
during the call; the model is restored afterwards. The granularity is therefore a module: an operation written
directly inside a `forward` is attributed to that module.
"""

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy

__all__ = [
    "AnomalyError", "detect_anomaly", "precision_audit", "PrecisionReport", "estimate_memory", "MemoryEstimate",
    "lr_find", "LRFinderResult", "check_determinism", "DeterminismReport", "diff_checkpoints", "CheckpointDiff",
    "TrainingMonitor", "hessian_top_eigenvalue", "hessian_trace",
]

_BYTES = {"bool": 1, "u8": 1, "i8": 1, "i16": 2, "f16": 2, "bf16": 2, "i32": 4, "u32": 4, "f32": 4, "i64": 8, "u64": 8, "f64": 8}


def _np(t: Any) -> np.ndarray:
    return t.detach().to_numpy() if isinstance(t, _Tensor) else np.asarray(t)


def _is_float(t: Any) -> bool:
    return isinstance(t, _Tensor) and t.dtype in ("f16", "bf16", "f32", "f64")


def _nbytes(t: _Tensor) -> int:
    return int(t.numel()) * _BYTES.get(t.dtype, 4)


def _tensors(obj: Any) -> List[_Tensor]:
    if isinstance(obj, _Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for o in obj for t in _tensors(o)]
    if isinstance(obj, dict):
        return [t for o in obj.values() for t in _tensors(o)]
    return []


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def _params(model: Any) -> List[Tuple[str, _Tensor]]:
    if hasattr(model, "named_parameters"):
        return list(model.named_parameters())
    return [(str(i), p) for i, p in enumerate(model.parameters())]


# ------------------------------------------------------------------------------------------------------------ probes


class _Probe:
    """Stands in for a child module: calls it and reports the call to `hook(probe, args, out) -> out`."""

    def __init__(self, name: str, target: Any, pre: Optional[Callable], post: Callable):
        self.__dict__.update(_name=name, _target=target, _pre=pre, _post=post)

    def __call__(self, *args, **kwargs):
        if self._pre is not None:
            args = self._pre(self._name, self._target, args)
        out = self._target(*args, **kwargs)
        return self._post(self._name, self._target, args, out)

    def __getattr__(self, item):
        return getattr(self.__dict__["_target"], item)


@contextmanager
def _probed(model: Any, post: Callable, pre: Optional[Callable] = None) -> Iterator[None]:
    """Wraps every child module of `model` (recursively) in a `_Probe` for the duration of the block."""
    if not hasattr(model, "named_modules"):
        yield
        return
    modules = list(model.named_modules())
    replaced = []
    try:
        for prefix, parent in modules:
            children = getattr(parent, "_modules", None)
            if not children:
                continue
            for child_name, child in list(children.items()):
                if child is None or isinstance(child, _Probe):
                    continue
                full = f"{prefix}.{child_name}" if prefix else child_name
                probe = _Probe(full, child, pre, post)
                had_attr = parent.__dict__.get(child_name) is child
                children[child_name] = probe
                if had_attr:
                    object.__setattr__(parent, child_name, probe)
                replaced.append((parent, child_name, child, had_attr))
        yield
    finally:
        for parent, child_name, child, had_attr in reversed(replaced):
            parent._modules[child_name] = child
            if had_attr:
                object.__setattr__(parent, child_name, child)


def _leaf_names(model: Any) -> set:
    """Names of modules without children (the layers that do the computing)."""
    out = set()
    for name, m in model.named_modules() if hasattr(model, "named_modules") else []:
        if not getattr(m, "_modules", None):
            out.add(name)
    return out


# ----------------------------------------------------------------------------------------------------------- anomaly


class AnomalyError(RuntimeError):
    """Raised by `detect_anomaly` with the module where NaN / Inf first appeared."""

    def __init__(self, message: str, module: str, phase: str):
        super().__init__(message)
        self.module, self.phase = module, phase


def _stats(t: _Tensor) -> str:
    a = _np(t).astype(np.float64)
    bad = ~np.isfinite(a)
    fin = a[~bad]
    rng = f"finite range [{fin.min():.4g}, {fin.max():.4g}]" if fin.size else "no finite values"
    return f"shape {tuple(t.shape)}, {int(np.isnan(a).sum())} NaN, {int(np.isinf(a).sum())} Inf, {rng}"


def _all_finite(ts: Sequence[_Tensor]) -> bool:
    return all(np.isfinite(_np(t)).all() for t in ts if _is_float(t))


class _AnomalyWatch:
    def __init__(self, model: Any, check_backward: bool):
        self.model = model
        self.check_backward = check_backward
        self.calls: List[Tuple[str, Any, List[_Tensor], List[_Tensor]]] = []  # name, module, input taps, output taps
        self.report: Optional[str] = None

    def _tap(self, t: Any) -> Tuple[Any, Optional[_Tensor]]:
        if not (self.check_backward and _is_float(t)):
            return t, None
        from .primitives import zeros
        z = zeros(list(t.shape), dtype=t.dtype).requires_grad_(True)
        return t + z, z

    def pre(self, name, module, args):
        new_args, taps = [], []
        for a in args:
            a2, z = self._tap(a)
            new_args.append(a2)
            if z is not None:
                taps.append(z)
        self._pending = taps
        return tuple(new_args)

    def post(self, name, module, args, out):
        ins = [a for a in args if _is_float(a)]
        outs = [t for t in _tensors(out) if _is_float(t)]
        if not _all_finite(outs) and _all_finite(ins):
            bad = next(t for t in outs if not np.isfinite(_np(t)).all())
            detail = "; ".join(f"input {i}: {_stats(a)}" for i, a in enumerate(ins)) or "no tensor inputs"
            msg = (f"forward: {type(module).__name__} '{name}' produced non-finite values from finite inputs\n"
                   f"  output: {_stats(bad)}\n  {detail}")
            self.report = msg
            raise AnomalyError(msg, name, "forward")
        out_taps = []
        if self.check_backward and isinstance(out, _Tensor) and _is_float(out):
            out, z = self._tap(out)
            out_taps.append(z)
        self.calls.append((name, module, getattr(self, "_pending", []), out_taps))
        return out

    def analyse_backward(self) -> None:
        def finite_grad(zs):
            return all(z.grad is None or np.isfinite(_np(z.grad)).all() for z in zs)

        if not self.calls or all(all(z.grad is None for z in outs) for _, _, _, outs in self.calls):
            return  # no backward pass ran
        culprits = []
        for idx, (name, module, in_taps, out_taps) in enumerate(self.calls):
            params = [p for _, p in _params(module)] if hasattr(module, "parameters") else []
            params_bad = any(p.grad is not None and not np.isfinite(_np(p.grad)).all() for p in params)
            if finite_grad(out_taps) and (not finite_grad(in_taps) or params_bad):
                culprits.append((idx, name, module, in_taps, params_bad))
        # keep the innermost: drop a module whose child is also a culprit
        names = [c[1] for c in culprits]
        inner = [c for c in culprits if not any(n != c[1] and n.startswith(c[1] + ".") for n in names)]
        if inner:
            idx, name, module, in_taps, params_bad = inner[-1]
            what = "parameter gradients" if params_bad and finite_grad(in_taps) else "input gradient"
            msg = (f"backward: {type(module).__name__} '{name}' turned a finite output gradient into a non-finite {what}")
            self.report = msg
            raise AnomalyError(msg, name, "backward")
        last = self.calls[-1]
        if not finite_grad(last[3]):
            msg = ("backward: the gradient reaching the model's output is already non-finite: the NaN / Inf comes from "
                   "the loss (or other code after the model)")
            self.report = msg
            raise AnomalyError(msg, "<loss>", "backward")


@contextmanager
def detect_anomaly(model: Any, check_backward: bool = True) -> Iterator[_AnomalyWatch]:
    """Finds where NaN or Inf first appears.

        with ox.diagnostics.detect_anomaly(model):
            loss = loss_fn(model(x), y)
            loss.backward()

    The forward pass raises `AnomalyError` at the first module that turns finite inputs into non-finite outputs. When the
    block ends after a backward pass, the module whose backward turned a finite gradient into a non-finite one is
    reported the same way (or the loss, when the gradient reaching the model is already bad). Parameters that already
    hold NaN / Inf are reported up front. The extra bookkeeping makes the block slower; use it to debug, not to train.
    """
    bad = [n for n, p in _params(model) if _is_float(p) and not np.isfinite(_np(p)).all()]
    if bad:
        raise AnomalyError(f"parameters already contain NaN / Inf before the forward pass: {', '.join(bad)}", bad[0], "parameters")
    watch = _AnomalyWatch(model, check_backward)
    with _probed(model, watch.post, watch.pre):
        yield watch
    if check_backward:
        watch.analyse_backward()


# --------------------------------------------------------------------------------------------------- precision audit


@dataclass
class PrecisionRow:
    name: str
    kind: str
    shape: Tuple[int, ...]
    total_error: float   # error of this module's float32 output, through the whole model so far (units of 2^-24)
    local_error: float   # error this module adds on its own, from exact (float64) inputs
    input_error: float   # error already present in its input


@dataclass
class PrecisionReport:
    """Per-module float32 error against a float64 run of the same model, in units of 2^-24 relative to the largest
    output magnitude (1.0 means half an ulp at the scale of the output)."""

    rows: List[PrecisionRow]
    output_error: float

    def worst(self, n: int = 5) -> List[PrecisionRow]:
        """The modules that add the most error on their own."""
        return sorted(self.rows, key=lambda r: -r.local_error)[:n]

    def __str__(self) -> str:
        lines = [f"{'module':<32}{'type':<18}{'output shape':<18}{'input err':>11}{'local err':>11}{'total err':>11}"]
        for r in self.rows:
            lines.append(f"{r.name[:31]:<32}{r.kind[:17]:<18}{str(r.shape)[:17]:<18}{r.input_error:>11.2f}{r.local_error:>11.2f}{r.total_error:>11.2f}")
        lines.append(f"model output error: {self.output_error:.2f} (units of 2^-24 of the largest output value)")
        w = self.worst(1)
        if w:
            lines.append(f"largest local error: {w[0].name} ({w[0].kind}), {w[0].local_error:.2f}")
        return "\n".join(lines)

    __repr__ = __str__


def _err(a: Any, ref: Any) -> float:
    x, r = _np(a).astype(np.float64), _np(ref).astype(np.float64)
    if x.shape != r.shape or r.size == 0:
        return float("nan")
    scale = np.abs(r).max()
    if not np.isfinite(scale) or scale == 0:
        return float(np.abs(x - r).max() * 2 ** 24) if scale == 0 else float("nan")
    return float(np.abs(x - r).max() / scale * 2 ** 24)


def _float_attrs(module: Any) -> List[Tuple[str, _Tensor]]:
    """The float32 tensors a module holds (parameters and buffers), by attribute name."""
    seen, out = set(), []
    names = list(getattr(module, "_parameters", {}).keys()) + list(getattr(module, "_buffers", {}).keys())
    if not hasattr(module, "_modules"):  # a native layer: its tensors are plain attributes
        names += [n for n in dir(module) if not n.startswith("_")]
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        try:
            v = getattr(module, n)
        except Exception:
            continue
        if isinstance(v, _Tensor) and v.dtype == "f32":
            out.append((n, v))
    return out


@contextmanager
def _as_float64(model: Any) -> Iterator[None]:
    """Temporarily replaces every float32 parameter / buffer of `model` with a float64 copy (restored exactly)."""
    swapped = []
    try:
        mods = list(model.named_modules()) if hasattr(model, "named_modules") else [("", model)]
        for _, m in mods:
            for n, v in _float_attrs(m):
                d = v.detach().double()
                if n in getattr(m, "_parameters", {}):
                    m._parameters[n] = d
                    object.__setattr__(m, n, d)
                elif n in getattr(m, "_buffers", {}):
                    m._buffers[n] = d
                    object.__setattr__(m, n, d)
                else:
                    setattr(m, n, d)
                swapped.append((m, n, v))
        yield
    finally:
        for m, n, v in reversed(swapped):
            if n in getattr(m, "_parameters", {}):
                m._parameters[n] = v
                object.__setattr__(m, n, v)
            elif n in getattr(m, "_buffers", {}):
                m._buffers[n] = v
                object.__setattr__(m, n, v)
            else:
                setattr(m, n, v)


def precision_audit(model: Any, *inputs: Any) -> PrecisionReport:
    """Runs `model(*inputs)` in float32 and in float64 and reports, for every module, the error of its float32 output
    (`total`), the error it adds on its own when given the exact float64 input (`local`), and the error already in its
    input. Use it to find the layer that loses accuracy (for example a softmax over large logits, or a long sum).

    The model runs in eval mode and without gradients; its parameters are restored bit for bit afterwards."""
    from .primitives import no_grad

    def record(store):
        def post(name, module, args, out):
            store.append((name, module, [a for a in args], out))
            return out
        return post

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    try:
        with no_grad():
            run32: List[Tuple[str, Any, list, Any]] = []
            with _probed(model, record(run32)):
                out32 = model(*inputs)
            run64: List[Tuple[str, Any, list, Any]] = []
            ins64 = [x.detach().double() if _is_float(x) else x for x in inputs]
            with _as_float64(model):
                with _probed(model, record(run64)):
                    out64 = model(*ins64)
            if len(run32) != len(run64):
                raise RuntimeError("the float32 and float64 runs called different modules; the model is not deterministic")
            rows = []
            for (name, module, a32, o32), (_, _, a64, o64) in zip(run32, run64):
                t32, t64 = _tensors(o32), _tensors(o64)
                if not t32 or not _is_float(t32[0]):
                    continue
                try:
                    local_out = module(*[a.detach().float() if _is_float(a) else a for a in a64])
                    local = _err(_tensors(local_out)[0], t64[0])
                except Exception:
                    local = float("nan")
                fin32 = [a for a in a32 if _is_float(a)]
                fin64 = [a for a in a64 if _is_float(a)]
                in_err = max([_err(x, y) for x, y in zip(fin32, fin64)], default=0.0)
                rows.append(PrecisionRow(name, type(module).__name__, tuple(t32[0].shape), _err(t32[0], t64[0]), local, in_err))
        return PrecisionReport(rows, _err(_tensors(out32)[0], _tensors(out64)[0]))
    finally:
        if was_training and hasattr(model, "train"):
            model.train()


# ---------------------------------------------------------------------------------------------------- memory estimate


_OPT_STATES = {"sgd": 0, "momentum": 1, "nesterov": 1, "adam": 2, "adamw": 2, "nadam": 2, "radam": 2, "lamb": 2,
               "rmsprop": 1, "adagrad": 1, "adadelta": 2, "lion": 1, "none": 0, None: 0}


@dataclass
class MemoryEstimate:
    """Bytes needed to train (or run) a model; `total` is their sum."""

    parameters: int
    gradients: int
    optimizer_state: int
    activations: int
    per_module: Dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.parameters + self.gradients + self.optimizer_state + self.activations

    def __str__(self) -> str:
        lines = [f"parameters       {_human(self.parameters):>12}", f"gradients        {_human(self.gradients):>12}",
                 f"optimizer state  {_human(self.optimizer_state):>12}", f"activations      {_human(self.activations):>12}",
                 f"total            {_human(self.total):>12}"]
        top = sorted(self.per_module.items(), key=lambda kv: -kv[1])[:5]
        if top:
            lines.append("largest activations: " + ", ".join(f"{k or '<model>'} {_human(v)}" for k, v in top))
        return "\n".join(lines)

    __repr__ = __str__


def estimate_memory(model: Any, *inputs: Any, optimizer: Optional[str] = "adam", training: bool = True) -> MemoryEstimate:
    """Estimates the memory a training step needs at the batch size of `inputs`: parameters, their gradients, the
    optimizer state (`optimizer` one of sgd, momentum, adam, adamw, rmsprop, adagrad, lion, ...; per element, in the
    parameter dtype), and the activations kept for the backward pass (the outputs of every computing layer, plus the
    inputs). Activations are measured by running the model once without gradients; the true figure also depends on
    which intermediate results each operation keeps, so treat it as an estimate. `training=False` counts parameters
    and the largest single activation only (inference)."""
    from .primitives import no_grad

    key = optimizer.lower() if isinstance(optimizer, str) else optimizer
    if key not in _OPT_STATES:
        raise ValueError(f"unknown optimizer {optimizer!r}; use one of {', '.join(k for k in _OPT_STATES if k)}")
    params = [p for _, p in _params(model)]
    seen, uniq = set(), []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            uniq.append(p)
    p_bytes = sum(_nbytes(p) for p in uniq)
    trainable = [p for p in uniq if getattr(p, "requires_grad", False)]
    g_bytes = sum(_nbytes(p) for p in trainable) if training else 0
    o_bytes = _OPT_STATES[key] * sum(_nbytes(p) for p in trainable) if training else 0

    leaves = _leaf_names(model)
    per_module: Dict[str, int] = {}

    def post(name, module, args, out):
        if name in leaves:
            per_module[name] = per_module.get(name, 0) + sum(_nbytes(t) for t in _tensors(out))
        return out

    with no_grad():
        with _probed(model, post):
            out = model(*inputs)
    in_bytes = sum(_nbytes(t) for t in inputs if isinstance(t, _Tensor))
    if not per_module:  # a single native layer: count its output
        per_module[""] = sum(_nbytes(t) for t in _tensors(out))
    act = in_bytes + sum(per_module.values()) if training else max(per_module.values(), default=0) + in_bytes
    return MemoryEstimate(p_bytes, g_bytes, o_bytes, act, per_module)


# ------------------------------------------------------------------------------------------------------- LR finder


@dataclass
class LRFinderResult:
    """The learning rates tried and the smoothed losses. `suggestion` is one tenth of the learning rate with the lowest
    loss (the classic rule); `valley` is a more cautious pick from the longest stretch of falling loss."""

    lrs: List[float]
    losses: List[float]
    suggestion: Optional[float]
    valley: Optional[float] = None

    def __str__(self) -> str:
        best = int(np.nanargmin(self.losses)) if self.losses else -1
        s = f"tried {len(self.lrs)} learning rates from {self.lrs[0]:.2e} to {self.lrs[-1]:.2e}" if self.lrs else "no steps"
        if best >= 0:
            s += f"; lowest loss {self.losses[best]:.4g} at lr {self.lrs[best]:.2e}"
        if self.suggestion:
            s += f"; suggested lr {self.suggestion:.2e}"
        if self.valley:
            s += f" (cautious: {self.valley:.2e})"
        return s

    __repr__ = __str__


def _snapshot(model: Any) -> List[Tuple[_Tensor, np.ndarray]]:
    return [(p, _np(p).copy()) for _, p in _params(model)]


def _restore(snap: List[Tuple[_Tensor, np.ndarray]]) -> None:
    from .primitives import no_grad
    with no_grad():
        for p, a in snap:
            p.assign(_from_numpy(np.ascontiguousarray(a)))
            p.set_grad(None)


def lr_find(model: Any, loss_fn: Callable[[Any, Any], _Tensor], data: Iterable[Tuple[Any, Any]],
            optimizer: Optional[Callable[[List[_Tensor], float], Any]] = None, start_lr: float = 1e-7, end_lr: float = 10.0,
            num_iter: int = 100, smooth: float = 0.05, diverge: float = 4.0) -> LRFinderResult:
    """The learning-rate range test (Smith, 2017): trains for up to `num_iter` steps while the learning rate grows
    exponentially from `start_lr` to `end_lr`, stops once the loss exceeds `diverge` times its best value, and suggests
    one tenth of the learning rate with the lowest smoothed loss (plus a more cautious `valley` pick). `data` yields (input, target) batches (it is cycled);
    `optimizer(params, lr)` builds the optimizer (Adam by default). The model's weights are restored afterwards."""
    from . import optim

    if num_iter < 2 or not 0 < start_lr < end_lr:
        raise ValueError("need num_iter >= 2 and 0 < start_lr < end_lr")
    make = optimizer or (lambda ps, lr: optim.Adam(ps, lr=lr))
    snap = _snapshot(model)
    params = [p for _, p in _params(model) if getattr(p, "requires_grad", False)]
    opt = make(params, start_lr)
    gamma = (end_lr / start_lr) ** (1.0 / (num_iter - 1))
    lrs, losses = [], []
    avg, best = None, float("inf")
    batches = iter(data)
    cache: List[Tuple[Any, Any]] = []
    try:
        for i in range(num_iter):
            lr = start_lr * gamma ** i
            try:
                batch = next(batches)
                cache.append(batch)
            except StopIteration:
                if not cache:
                    raise ValueError("data yielded no batches")
                batch = cache[i % len(cache)]
            if hasattr(opt, "set_learning_rate"):
                opt.set_learning_rate(lr)
            else:
                opt.lr = lr
            x, y = batch
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            value = float(loss.item())
            avg = value if avg is None else smooth * value + (1 - smooth) * avg
            smoothed = avg / (1 - (1 - smooth) ** (i + 1))
            lrs.append(lr)
            losses.append(smoothed)
            if not math.isfinite(value) or smoothed > diverge * best:
                break
            best = min(best, smoothed)
    finally:
        _restore(snap)
    lowest = float(lrs[int(np.nanargmin(losses))]) / 10 if losses else None
    return LRFinderResult(lrs, losses, lowest, _valley(lrs, losses))


def _valley(lrs: List[float], losses: List[float]) -> Optional[float]:
    """The "valley" rule (as in fastai): find the longest run of steps over which the loss keeps falling (the longest
    decreasing subsequence) and take the learning rate about half-way into its first two thirds. Steepest-slope rules
    are fooled by batch noise while the loss is still flat; this one is not."""
    n = len(losses)
    if n < 5:
        return None
    ls = np.asarray(losses, dtype=np.float64)
    lds = [1] * n
    max_start = max_end = 0
    for i in range(1, n):
        for j in range(i):
            if ls[i] < ls[j] and lds[i] < lds[j] + 1:
                lds[i] = lds[j] + 1
        if lds[max_end] < lds[i]:
            max_end = i
            max_start = max_end - lds[max_end]
    sections = (max_end - max_start) / 3
    return float(lrs[max_start + int(sections) + int(sections / 2)])


# -------------------------------------------------------------------------------------------------------- determinism


@dataclass
class DeterminismReport:
    """Whether every run of the function gave bit-identical results, and where the first difference is."""

    deterministic: bool
    runs: int
    values_compared: int
    first_difference: Optional[str] = None
    max_abs_difference: float = 0.0
    differing_elements: int = 0

    def __bool__(self) -> bool:
        return self.deterministic

    def __str__(self) -> str:
        if self.deterministic:
            return f"deterministic: {self.runs} runs, {self.values_compared} values bit-identical"
        return (f"NOT deterministic: '{self.first_difference}' differs ({self.differing_elements} elements, "
                f"max abs difference {self.max_abs_difference:.3g})")

    __repr__ = __str__


def _flatten(obj: Any, path: str = "") -> List[Tuple[str, np.ndarray]]:
    if hasattr(obj, "state_dict") and callable(obj.state_dict) and not isinstance(obj, dict):
        obj = obj.state_dict()
    if isinstance(obj, _Tensor):
        return [(path or "<output>", _np(obj))]
    if isinstance(obj, np.ndarray):
        return [(path or "<output>", obj)]
    if isinstance(obj, dict):
        return [kv for k in obj for kv in _flatten(obj[k], f"{path}.{k}" if path else str(k))]
    if isinstance(obj, (list, tuple)):
        return [kv for i, v in enumerate(obj) for kv in _flatten(v, f"{path}[{i}]")]
    if isinstance(obj, (int, float, bool, np.number)):
        return [(path or "<output>", np.asarray(obj))]
    return []


def _seed_all(seed: int) -> None:
    from .oxideon import manual_seed
    manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_determinism(fn: Callable[[], Any], runs: int = 3, seed: int = 0) -> DeterminismReport:
    """Runs `fn()` `runs` times, each after seeding oxideon, NumPy and `random` with `seed`, and compares the results
    bit for bit. `fn` may return a tensor, an array, a number, a model (its state dict is compared) or any nesting of
    lists / tuples / dicts of those; for example a function that builds a model, trains it a few steps and returns it."""
    if runs < 2:
        raise ValueError("runs must be at least 2")
    _seed_all(seed)
    first = _flatten(fn())
    for _ in range(runs - 1):
        _seed_all(seed)
        other = _flatten(fn())
        if [k for k, _ in other] != [k for k, _ in first]:
            return DeterminismReport(False, runs, len(first), "the structure of the result (keys / lengths) changed")
        for (k, a), (_, b) in zip(first, other):
            if a.shape != b.shape:
                return DeterminismReport(False, runs, len(first), k + " (shape)")
            same = (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind in "fc" else (a == b)
            if not np.all(same):
                diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
                return DeterminismReport(False, runs, len(first), k, float(np.nanmax(diff)), int((~same).sum()))
    return DeterminismReport(True, runs, len(first))


# --------------------------------------------------------------------------------------------------- checkpoint diff


@dataclass
class CheckpointDiff:
    """Tensor-by-tensor comparison of two state dicts."""

    only_in_a: List[str]
    only_in_b: List[str]
    shape_mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]
    dtype_mismatch: List[Tuple[str, str, str]]
    changed: List[Tuple[str, float, float]]  # key, relative change ||b - a|| / ||a||, max |b - a|
    unchanged: List[str]

    @property
    def identical(self) -> bool:
        return not (self.only_in_a or self.only_in_b or self.shape_mismatch or self.dtype_mismatch or self.changed)

    def __str__(self) -> str:
        if self.identical:
            return f"identical ({len(self.unchanged)} tensors)"
        lines = [f"{len(self.changed)} changed, {len(self.unchanged)} unchanged, {len(self.only_in_a)} only in a, "
                 f"{len(self.only_in_b)} only in b, {len(self.shape_mismatch)} shape and {len(self.dtype_mismatch)} dtype mismatches"]
        for k in self.only_in_a:
            lines.append(f"  - {k} (only in a)")
        for k in self.only_in_b:
            lines.append(f"  + {k} (only in b)")
        for k, sa, sb in self.shape_mismatch:
            lines.append(f"  ! {k}: shape {sa} -> {sb}")
        for k, da, db in self.dtype_mismatch:
            lines.append(f"  ! {k}: dtype {da} -> {db}")
        for k, rel, mx in self.changed[:20]:
            lines.append(f"  ~ {k}: relative change {rel:.3e}, max abs {mx:.3e}")
        if len(self.changed) > 20:
            lines.append(f"  ... {len(self.changed) - 20} more changed")
        return "\n".join(lines)

    __repr__ = __str__


def _state(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        from . import load
        obj = load(obj)
    if hasattr(obj, "state_dict") and callable(obj.state_dict) and not isinstance(obj, dict):
        obj = obj.state_dict()
    if not isinstance(obj, dict):
        raise TypeError("expected a state dict, a model or a path to a saved one")
    return obj


def diff_checkpoints(a: Any, b: Any, atol: float = 0.0, rtol: float = 0.0) -> CheckpointDiff:
    """Compares two checkpoints (state dicts, models, or paths of saved ones). A tensor counts as changed when some
    element differs by more than `atol + rtol * |a|`; `changed` is sorted by relative change, largest first."""
    sa, sb = _state(a), _state(b)
    only_a = sorted(k for k in sa if k not in sb)
    only_b = sorted(k for k in sb if k not in sa)
    shape_mm, dtype_mm, changed, same = [], [], [], []
    for k in (k for k in sa if k in sb):
        x, y = _np(sa[k]), _np(sb[k])
        if x.shape != y.shape:
            shape_mm.append((k, x.shape, y.shape))
            continue
        if x.dtype != y.dtype:
            dtype_mm.append((k, str(x.dtype), str(y.dtype)))
        xf, yf = x.astype(np.float64), y.astype(np.float64)
        d = np.abs(yf - xf)
        if np.any(d > atol + rtol * np.abs(xf)) or np.any(np.isnan(xf) != np.isnan(yf)):
            norm = float(np.linalg.norm(xf.ravel()))
            rel = float(np.linalg.norm((yf - xf).ravel()) / norm) if norm > 0 else float("inf")
            changed.append((k, rel, float(np.nanmax(d)) if d.size else 0.0))
        else:
            same.append(k)
    changed.sort(key=lambda r: -r[1] if math.isfinite(r[1]) else -float("inf"))
    return CheckpointDiff(only_a, only_b, shape_mm, dtype_mm, changed, same)


# ---------------------------------------------------------------------------------------------- training monitor


class TrainingMonitor:
    """Watches training health. Call `step()` once per iteration after `optimizer.step()` (and before `zero_grad()`):

        monitor = ox.diagnostics.TrainingMonitor(model)
        for x, y in data:
            with monitor.watch():          # optional: activation statistics of every layer
                loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            monitor.step(loss)
            opt.zero_grad()
        print(monitor.report())

    Recorded per step: the loss, every parameter's gradient norm, weight norm and update-to-weight ratio
    (|w_t - w_t-1| / |w_t-1|, healthy around 1e-3), and per layer the activation mean, standard deviation, the share of
    units that were zero for the whole batch (dead ReLUs) and of non-finite values. `report()` turns them into warnings.
    """

    def __init__(self, model: Any):
        self.model = model
        self.history: List[Dict[str, Any]] = []
        self._prev = {n: _np(p).astype(np.float64).copy() for n, p in _params(model)}
        self._acts: Dict[str, Dict[str, float]] = {}

    @contextmanager
    def watch(self) -> Iterator[None]:
        def post(name, module, args, out):
            ts = [t for t in _tensors(out) if _is_float(t)]
            if ts:
                a = _np(ts[0]).astype(np.float64)
                finite = a[np.isfinite(a)]
                flat = a.reshape(-1, a.shape[-1]) if a.ndim >= 2 else a.reshape(1, -1)
                dead = float(np.mean(np.all(flat == 0, axis=0))) if flat.size else 0.0
                self._acts[name] = {"kind": type(module).__name__, "mean": float(finite.mean()) if finite.size else float("nan"),
                                    "std": float(finite.std()) if finite.size else float("nan"), "dead": dead,
                                    "nonfinite": float(1 - finite.size / max(a.size, 1))}
            return out
        with _probed(self.model, post):
            yield

    def step(self, loss: Any = None) -> Dict[str, Any]:
        rec: Dict[str, Any] = {"step": len(self.history), "loss": float(_np(loss)) if loss is not None else None, "params": {}}
        for n, p in _params(self.model):
            w = _np(p).astype(np.float64)
            g = p.grad
            gn = float(np.linalg.norm(_np(g).ravel())) if g is not None else None
            wn = float(np.linalg.norm(w.ravel()))
            prev = self._prev.get(n)
            upd = None
            if prev is not None and prev.shape == w.shape:
                pn = float(np.linalg.norm(prev.ravel()))
                upd = float(np.linalg.norm((w - prev).ravel()) / pn) if pn > 0 else None
            self._prev[n] = w.copy()
            rec["params"][n] = {"grad_norm": gn, "weight_norm": wn, "update_ratio": upd}
        rec["activations"] = dict(self._acts)
        self._acts = {}
        self.history.append(rec)
        return rec

    def warnings(self, step: int = -1) -> List[str]:
        if not self.history:
            return []
        rec = self.history[step]
        out = []
        if rec["loss"] is not None and not math.isfinite(rec["loss"]):
            out.append(f"loss is {rec['loss']}")
        for n, s in rec["params"].items():
            g, u = s["grad_norm"], s["update_ratio"]
            if g is not None and not math.isfinite(g):
                out.append(f"{n}: non-finite gradient")
            elif g is not None and g < 1e-8 and s["weight_norm"] > 0:
                out.append(f"{n}: vanishing gradient (norm {g:.1e})")
            elif g is not None and g > 1e3:
                out.append(f"{n}: exploding gradient (norm {g:.1e})")
            if u is not None and u > 1e-1:
                out.append(f"{n}: update is {u:.1e} of the weights (too large; ~1e-3 is typical)")
            elif u is not None and 0 < u < 1e-6:
                out.append(f"{n}: update is {u:.1e} of the weights (barely moving)")
        for n, a in rec["activations"].items():
            if a["nonfinite"] > 0:
                out.append(f"{n} ({a['kind']}): {a['nonfinite']:.1%} non-finite activations")
            if "relu" in a["kind"].lower() and a["dead"] > 0.5:
                out.append(f"{n} ({a['kind']}): {a['dead']:.0%} of units dead (zero for the whole batch)")
        if len(self.history) >= 10:
            losses = [r["loss"] for r in self.history[-10:] if r["loss"] is not None]
            if len(losses) == 10 and all(math.isfinite(v) for v in losses) and losses[-1] > 1.5 * min(losses):
                out.append("loss rose by more than 50% over the last 10 steps")
        return out

    def report(self) -> str:
        if not self.history:
            return "no steps recorded"
        rec = self.history[-1]
        lines = [f"step {rec['step']}" + (f", loss {rec['loss']:.4g}" if rec["loss"] is not None else "")]
        lines.append(f"{'parameter':<36}{'grad norm':>12}{'weight norm':>13}{'update/weight':>15}")
        for n, s in rec["params"].items():
            g = "-" if s["grad_norm"] is None else f"{s['grad_norm']:.3e}"
            u = "-" if s["update_ratio"] is None else f"{s['update_ratio']:.3e}"
            lines.append(f"{n[:35]:<36}{g:>12}{s['weight_norm']:>13.3e}{u:>15}")
        if rec["activations"]:
            lines.append(f"{'layer':<36}{'mean':>10}{'std':>10}{'dead':>8}")
            for n, a in rec["activations"].items():
                lines.append(f"{n[:35]:<36}{a['mean']:>10.3g}{a['std']:>10.3g}{a['dead']:>8.0%}")
        w = self.warnings()
        lines.append("warnings: " + ("none" if not w else ""))
        lines += ["  " + s for s in w]
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------- sharpness


def _loss_and_params(model, loss_fn, inputs, targets):
    params = [p for _, p in _params(model) if getattr(p, "requires_grad", False)]
    if not params:
        raise ValueError("the model has no parameters that require grad")
    loss = loss_fn(model(inputs), targets) if targets is not None else loss_fn(model(inputs))
    return loss, params


def _hvp(model, loss_fn, inputs, targets, vs):
    from .primitives import _AutogradModule as autograd
    loss, params = _loss_and_params(model, loss_fn, inputs, targets)
    grads = autograd.grad(loss, params, create_graph=True)
    dot = None
    for g, v in zip(grads, vs):
        term = (g * v).sum()
        dot = term if dot is None else dot + term
    hv = autograd.grad(dot, params, allow_unused=True)
    return [h if h is not None else v * 0 for h, v in zip(hv, vs)]


def _vdot(a, b) -> float:
    return float(sum(float((x * y).sum().item()) for x, y in zip(a, b)))


def hessian_top_eigenvalue(model: Any, loss_fn: Callable, inputs: Any, targets: Any = None, iters: int = 30,
                           tol: float = 1e-4, seed: int = 0) -> float:
    """The largest-magnitude eigenvalue of the Hessian of the loss with respect to the parameters, by power iteration
    on Hessian-vector products (no Hessian is formed). A large value means a sharp minimum, which tends to go with
    worse generalization and a lower stable learning rate (roughly lr < 2 / eigenvalue for SGD)."""
    rng = np.random.default_rng(seed)
    _, params = _loss_and_params(model, loss_fn, inputs, targets)
    v = [_from_numpy(rng.standard_normal(tuple(p.shape)).astype(_np(p).dtype)) for p in params]
    norm = math.sqrt(_vdot(v, v))
    v = [x / norm for x in v]
    eig = 0.0
    for _ in range(iters):
        hv = _hvp(model, loss_fn, inputs, targets, v)
        new = _vdot(v, hv)
        n = math.sqrt(_vdot(hv, hv))
        if n == 0:
            return 0.0
        v = [h.detach() / n for h in hv]
        if abs(new - eig) <= tol * max(abs(new), 1e-12):
            return new
        eig = new
    return eig


def hessian_trace(model: Any, loss_fn: Callable, inputs: Any, targets: Any = None, samples: int = 50, seed: int = 0) -> float:
    """Hutchinson's estimate of the trace of the Hessian of the loss (the mean of v^T H v over random +-1 vectors v):
    the average curvature, a second sharpness measure next to `hessian_top_eigenvalue`."""
    rng = np.random.default_rng(seed)
    _, params = _loss_and_params(model, loss_fn, inputs, targets)
    total = 0.0
    for _ in range(samples):
        v = [_from_numpy(rng.choice([-1.0, 1.0], size=tuple(p.shape)).astype(_np(p).dtype)) for p in params]
        total += _vdot(v, _hvp(model, loss_fn, inputs, targets, v))
    return total / samples
