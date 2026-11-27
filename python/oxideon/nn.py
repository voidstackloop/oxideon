"""PyTorch-style class-based neural network module architecture for Oxideon.

Enables full OOP subclassing and inheritance (`class MyModel(ox.nn.Module): ...`)
with automatic parameter tracking, state_dict serialization, training/eval modes,
device placement, forward pre/post hooks, and modular containers (Sequential,
ModuleList, ModuleDict).
"""

from collections import OrderedDict
import copy
import itertools
import math
import operator
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from . import functional
__path__ = []  # Exposes 'import oxideon.nn.functional as F'
sys.modules["oxideon.nn.functional"] = functional

from . import (
    Tensor as _Tensor,
    Linear,
    SpectralNormLinear,
    Conv2d as _NativeConv2d,
    Conv3d as _NativeConv3d,
    ConvLstmCell,
    Dropout,
    DropPath,
    Embedding,
    LayerNorm as _NativeLayerNorm,
    Maxout,
    Snake,
    Swish,
    Upsample2d,
    ReLU,
    MLP,
    Sequential as _NativeSequential,
    from_numpy as _from_numpy,
    cat as _native_cat,
    stack as _native_stack,
    fused_rmsnorm as _fused_rmsnorm,
    fused_add_rmsnorm as _fused_add_rmsnorm,
    fused_add_layernorm as _fused_add_layernorm,
    fused_bias_activation as _fused_bias_activation,
    fused_silu_and_mul as _fused_silu_and_mul,
    fused_rotary_qk as _fused_rotary_qk,
    fused_layernorm as _fused_layernorm,
    barlow_twins_loss,
    ctc_loss,
    dpo_loss,
    focal_loss,
    grpo_loss,
    info_nce_loss,
    ppo_clipped_loss,
    siglip_loss,
    simpo_loss,
    vicreg_loss,
)
from .serialization import save_file, load_file, save, load
from .primitives import no_grad as _no_grad
from . import functional
from . import utils
from . import rnn_utils as _rnn_utils
utils.rnn = _rnn_utils

mse_loss = functional.mse_loss
l1_loss = functional.l1_loss
cross_entropy = functional.cross_entropy
binary_cross_entropy_with_logits = functional.binary_cross_entropy_with_logits
huber_loss = functional.huber_loss
kl_div_loss = functional.kl_div_loss
nll_loss = functional.nll_loss
cosine_embedding_loss = functional.cosine_embedding_loss
triplet_margin_loss = functional.triplet_margin_loss
margin_ranking_loss = functional.margin_ranking_loss
hinge_embedding_loss = functional.hinge_embedding_loss
soft_margin_loss = functional.soft_margin_loss
from . import init

F = functional

# ---------------------------------------------------------------------------
# Repeat-count guard
# ---------------------------------------------------------------------------

# Most repeated structural units (layers, blocks) a constructor accepts. Mirrors
# `oxideon::limits::MAX_REPEATS` so an absurd count fails fast instead of hanging.
_MAX_REPEATS = 1 << 16


def _check_repeats(what: str, n: int) -> int:
    if isinstance(n, bool):
        return n
    try:
        v = operator.index(n)
    except TypeError:
        return n
    if v > _MAX_REPEATS:
        raise ValueError(f"{what} = {n} is above the limit of {_MAX_REPEATS}")
    return n


# Mirrors oxideon_core::limits::MAX_ELEMENTS: most elements a single user-sized
# buffer may hold. Larger requests are refused up front instead of corner
# cases like numpy arange silently wrapping to an empty array past 2**63 - 1.
_MAX_ELEMENTS = 1 << 31


def _check_elements(what: str, n: int) -> int:
    if isinstance(n, bool):
        return n
    try:
        v = operator.index(n)
    except TypeError:
        return n
    if v > _MAX_ELEMENTS:
        raise ValueError(f"{what} = {n} is above the limit of {_MAX_ELEMENTS}")
    return n


def _check_positive(what: str, n: int) -> int:
    n = _check_elements(what, n)
    try:
        v = operator.index(n)
    except TypeError:
        return n
    if v < 1:
        raise ValueError(f"{what} = {n} must be at least 1")
    return n


def _check_non_negative(what: str, n: int) -> int:
    n = _check_elements(what, n)
    try:
        v = operator.index(n)
    except TypeError:
        return n
    if v < 0:
        raise ValueError(f"{what} = {n} must be non-negative")
    return n


# ---------------------------------------------------------------------------
# Parameter Class
# ---------------------------------------------------------------------------

class _ParameterMeta(type):
    """Metaclass allowing `isinstance(p, Parameter)` to recognize any `requires_grad` Tensor."""
    def __instancecheck__(cls, instance):
        return isinstance(instance, _Tensor) and getattr(instance, "requires_grad", False)


class Parameter(metaclass=_ParameterMeta):
    """A learnable neural network parameter wrapping an Oxideon `Tensor` with `requires_grad=True`."""
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            raise ValueError("Parameter requires a Tensor or array-like data.")
        if not isinstance(data, _Tensor):
            data = _from_numpy(np.asarray(data, dtype=np.float32))
        if requires_grad and not getattr(data, "requires_grad", False):
            if hasattr(data, "requires_grad_"):
                data = data.requires_grad_(True)
            else:
                data.requires_grad = True
        return data


# ---------------------------------------------------------------------------
# Hook Handle
# ---------------------------------------------------------------------------

class _HookHandle:
    _counter = itertools.count()

    def __init__(self, hooks_dict):
        self.id = next(self._counter)
        self.hooks_dict = hooks_dict

    def remove(self):
        self.hooks_dict.pop(self.id, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


# ---------------------------------------------------------------------------
# Base Module Class
# ---------------------------------------------------------------------------

class Module:
    """Base class for all neural network modules in Oxideon.

    Models should inherit from this class and override `forward()`.
    Submodules and parameters assigned as attributes are automatically discovered
    and registered for optimizer tracking.

    Example:
    ```python
    import oxideon as ox
    import oxideon.nn as nn

    class MLPClassifier(nn.Module):
        def __init__(self, in_features, hidden, out_features):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden)
            self.fc2 = nn.Linear(hidden, out_features)

        def forward(self, x):
            return self.fc2(self.fc1(x).relu())

    model = MLPClassifier(784, 128, 10)
    optimizer = ox.optim.Adam(model.parameters(), lr=1e-3)
    out = model(x)
    ```
    """

    def __init__(self):
        self._modules = OrderedDict()
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._non_persistent_buffers_set = set()
        self.training = True

    def add_module(self, name: str, module: Optional[Any]) -> None:
        """Adds a child module to the current module."""
        if not isinstance(name, str):
            raise TypeError(f"module name should be a string. Got {type(name).__name__}")
        if hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        self._modules[name] = module

    def __setattr__(self, name, value):
        if name.startswith("_") or not hasattr(self, "_modules"):
            super().__setattr__(name, value)
            return

        # 1. Module or native Rust layer with parameters() and callable
        if isinstance(value, Module) or (hasattr(value, "parameters") and callable(value)):
            self._parameters.pop(name, None)
            self._modules[name] = value
        # 2. Tensor attribute: always a parameter, frozen or not. Gating on
        # requires_grad silently dropped frozen weights from parameters() and
        # state_dict(), corrupting save/load.
        elif isinstance(value, _Tensor):
            self._modules.pop(name, None)
            self._buffers.pop(name, None)
            self._parameters[name] = value
        # 3. List or tuple of modules -> auto ModuleList
        elif isinstance(value, (list, tuple)) and value:
            if all(isinstance(v, Module) or (hasattr(v, "parameters") and callable(v)) for v in value):
                value = ModuleList(value)
                self._modules[name] = value
        # 4. Dict of modules -> auto ModuleDict
        elif isinstance(value, dict) and value:
            if all(isinstance(v, Module) or (hasattr(v, "parameters") and callable(v)) for v in value.values()):
                value = ModuleDict(value)
                self._modules[name] = value

        super().__setattr__(name, value)

    def __delattr__(self, name):
        self._modules.pop(name, None)
        self._parameters.pop(name, None)
        self._buffers.pop(name, None)
        super().__delattr__(name)

    def register_parameter(self, name, param):
        """Adds a parameter to the module."""
        if param is not None:
            if not isinstance(param, _Tensor):
                param = Parameter(param)
            elif not getattr(param, "requires_grad", False):
                param = param.requires_grad_(True)
            self._parameters[name] = param
            super().__setattr__(name, param)
        else:
            self._parameters.pop(name, None)
            super().__setattr__(name, None)

    def register_buffer(self, name, tensor, persistent=True):
        """Adds a buffer to the module (optionally non-persistent so omitted from state_dict)."""
        if tensor is not None and not isinstance(tensor, _Tensor):
            tensor = _from_numpy(np.asarray(tensor))
        self._buffers[name] = tensor
        if not hasattr(self, "_non_persistent_buffers_set"):
            self._non_persistent_buffers_set = set()
        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)
        super().__setattr__(name, tensor)

    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call."""
        raise NotImplementedError(
            f"Module [{type(self).__name__}] must implement the 'forward(*args, **kwargs)' method."
        )

    def __call__(self, *args, **kwargs):
        for hook in list(self._forward_pre_hooks.values()):
            hook(self, args)
        out = self.forward(*args, **kwargs)
        for hook in list(self._forward_hooks.values()):
            hook_result = hook(self, args, out)
            if hook_result is not None:
                out = hook_result
        return out

    def parameters(self, recurse=True):
        """Returns a list of all module parameters."""
        return [p for _, p in self.named_parameters(recurse=recurse)]

    def named_parameters(self, prefix="", recurse=True):
        """Returns an iterator over module parameters, yielding both name and parameter."""
        for name, p in self._parameters.items():
            if p is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, p
        if recurse:
            for mod_name, child in self.named_children():
                full_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                if isinstance(child, Module):
                    yield from child.named_parameters(prefix=full_prefix, recurse=True)
                elif hasattr(child, "named_parameters"):
                    yield from child.named_parameters(prefix=f"{full_prefix}.")
                elif hasattr(child, "parameters"):
                    for i, p in enumerate(child.parameters()):
                        p_name = f"{full_prefix}.weight" if i == 0 else f"{full_prefix}.bias" if i == 1 else f"{full_prefix}.param_{i}"
                        yield p_name, p

    def buffers(self, recurse=True):
        """Returns an iterator over module buffers."""
        for b in self._buffers.values():
            if b is not None:
                yield b
        if recurse:
            for child in self.children():
                if isinstance(child, Module):
                    yield from child.buffers(recurse=True)

    def named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        """Returns an iterator over module buffers, yielding both name and buffer."""
        for name, b in self._buffers.items():
            if b is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, b
        if recurse:
            for mod_name, child in self.named_children():
                full_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                if isinstance(child, Module):
                    yield from child.named_buffers(prefix=full_prefix, recurse=True, remove_duplicate=remove_duplicate)

    def _named_persistent_buffers(self, prefix=""):
        non_p = getattr(self, "_non_persistent_buffers_set", ())
        for name, b in self._buffers.items():
            if b is not None and name not in non_p:
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, b
        for mod_name, child in self.named_children():
            full_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            if isinstance(child, Module):
                yield from child._named_persistent_buffers(prefix=full_prefix)

    def children(self):
        """Returns an iterator over immediate child modules."""
        for m in self._modules.values():
            yield m

    def named_children(self):
        """Returns an iterator over immediate child modules, yielding both name and module."""
        for name, m in self._modules.items():
            yield name, m

    def modules(self):
        """Returns an iterator over all modules in the network."""
        yield self
        for child in self.children():
            if isinstance(child, Module):
                yield from child.modules()
            else:
                yield child

    def named_modules(self, memo=None, prefix=""):
        """Returns an iterator over all modules in the network, yielding name and module."""
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, child in self.named_children():
                sub_prefix = f"{prefix}.{name}" if prefix else name
                if isinstance(child, Module):
                    yield from child.named_modules(memo, prefix=sub_prefix)
                else:
                    yield sub_prefix, child

    def train(self, mode=True):
        """Sets the module in training mode."""
        self.training = mode
        for child in self.children():
            if hasattr(child, "train") and callable(child.train):
                child.train(mode)
            elif hasattr(child, "training"):
                child.training = mode
        return self

    def eval(self):
        """Sets the module in evaluation mode."""
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True):
        """Changes whether autograd records operations on this module's parameters; returns self."""
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def _cast(self, dtype: str):
        for m in self.modules():
            if isinstance(m, Module):
                for name, p in list(m._parameters.items()):
                    if p is not None and p.dtype != dtype:
                        q = Parameter(p.detach().to_dtype(dtype))
                        q.requires_grad_(p.requires_grad)
                        m._parameters[name] = q
                        if name in m.__dict__:
                            m.__dict__[name] = q
                for name, b in list(m._buffers.items()):
                    if isinstance(b, _Tensor) and b.dtype in ("f16", "bf16", "f32", "f64") and b.dtype != dtype:
                        m._buffers[name] = b.to_dtype(dtype)
        return self

    def float(self):
        """Casts the floating-point parameters and buffers of Python modules to float32 (native layers stay float32)."""
        return self._cast("f32")

    def double(self):
        """Casts the floating-point parameters and buffers of Python modules to float64 (native layers stay float32)."""
        return self._cast("f64")

    def half(self):
        """Casts the floating-point parameters and buffers of Python modules to float16 (native layers stay float32)."""
        return self._cast("f16")

    def get_submodule(self, target: str):
        """Returns the submodule at the dotted path `target` ('' is the module itself)."""
        mod = self
        for part in [t for t in target.split(".") if t]:
            mod = dict(mod.named_children()).get(part) if not hasattr(mod, part) else getattr(mod, part)
            if mod is None:
                raise AttributeError(f"no submodule {target!r}")
        return mod

    def get_parameter(self, target: str):
        """Returns the parameter at the dotted path `target`."""
        for name, p in self.named_parameters():
            if name == target:
                return p
        raise AttributeError(f"no parameter {target!r}")

    def zero_grad(self, set_to_none=False):
        """Sets gradients of all model parameters to zero (or None with `set_to_none`)."""
        for p in self.parameters():
            if set_to_none and hasattr(p, "grad"):
                p.grad = None
            elif hasattr(p, "zero_grad"):
                p.zero_grad()
            elif hasattr(p, "grad"):
                if p.grad is not None:
                    if hasattr(p.grad, "zero_"):
                        p.grad.zero_()
                    else:
                        p.grad = None
        return self

    @staticmethod
    def _move_in_place(tensor, device, dtype):
        """Moves/casts `tensor` onto `device`/`dtype`, preserving object identity
        (so live optimizers keep working) and the `requires_grad` flag."""
        moved = tensor
        if device is not None and hasattr(moved, "to_device"):
            moved = moved.to_device(device)
        if dtype is not None and hasattr(moved, "to_dtype"):
            moved = moved.to_dtype(dtype)
        if moved is tensor:
            return tensor
        if hasattr(tensor, "retarget"):
            # Native in-place retarget: dtype/device FOLLOW the source
            # (assign() would cast back and silently no-op the move).
            tensor.retarget(moved)
            return tensor
        # Fallback for tensor-likes without retarget(): swap objects but keep
        # the flag and move any stored gradient alongside. Callers holding the
        # old object (e.g. optimizers) must be recreated.
        if getattr(tensor, "requires_grad", False) and not getattr(moved, "requires_grad", False):
            moved = moved.requires_grad_(True)
        g = getattr(tensor, "grad", None)
        if g is not None:
            if device is not None and hasattr(g, "to_device"):
                g = g.to_device(device)
            if dtype is not None and hasattr(g, "to_dtype"):
                g = g.to_dtype(dtype)
            try:
                moved.grad = g
            except AttributeError:
                pass
        return moved

    def to(self, device=None, dtype=None):
        """Moves and/or casts the parameters and buffers."""
        for name, p in list(self._parameters.items()):
            if p is not None:
                new_p = self._move_in_place(p, device, dtype)
                if new_p is not p:
                    self._parameters[name] = new_p
                    super().__setattr__(name, new_p)

        for name, b in list(self._buffers.items()):
            if b is not None:
                new_b = self._move_in_place(b, device, dtype)
                if new_b is not b:
                    self._buffers[name] = new_b
                    super().__setattr__(name, new_b)

        for child in self.children():
            if hasattr(child, "to"):
                child.to(device=device, dtype=dtype)
        return self

    def cpu(self):
        """Moves all model parameters and buffers to the CPU."""
        return self.to(device="cpu")

    def gpu(self):
        """Moves all model parameters and buffers to the GPU."""
        return self.to(device="gpu")

    def cuda(self):
        """Alias for `gpu()`."""
        return self.to(device="gpu")

    def state_dict(self, destination=None, prefix=""):
        """Returns a dictionary containing a whole state of the module."""
        if destination is None:
            destination = OrderedDict()
        for name, p in self.named_parameters(prefix=prefix):
            destination[name] = p.clone() if hasattr(p, "clone") else p
        for name, b in self._named_persistent_buffers(prefix=prefix):
            destination[name] = b.clone() if hasattr(b, "clone") else b
        return destination

    def _resolve_key(self, key):
        """Resolves a dotted state_dict key to `(holder, leaf_name)`."""
        parts = key.split(".")
        obj = self
        for part in parts[:-1]:
            if isinstance(obj, (ModuleList, Sequential)):
                try:
                    obj = obj[int(part)]
                except (ValueError, IndexError):
                    obj = getattr(obj, part)
            elif isinstance(obj, ModuleDict) and part in obj:
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        return obj, parts[-1]

    @staticmethod
    def _copy_state_value(target, val, key, owner_name):
        """Copies `val` into `target` in place (preserving object identity so
        live optimizers keep working). Returns True when copied in place."""
        if isinstance(val, _Tensor) and target is not None and hasattr(target, "copy_"):
            try:
                if list(target.shape) != list(val.shape):
                    raise RuntimeError(
                        f"shape mismatch (expected {list(target.shape)}, got {list(val.shape)})"
                    )
                with _no_grad():
                    target.copy_(val)
            except Exception as e:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {owner_name}:\n"
                    f"\tWhile copying '{key}': {e}"
                ) from e
            return True
        return False

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from `state_dict` into this module and its descendants."""
        missing_keys = []
        unexpected_keys = list(state_dict.keys())
        current_params = dict(self.named_parameters())
        current_buffers = dict(self._named_persistent_buffers())

        for key, target_param in current_params.items():
            if key in state_dict:
                unexpected_keys.remove(key)
                val = state_dict[key]
                if not self._copy_state_value(target_param, val, key, type(self).__name__):
                    obj, leaf = self._resolve_key(key)
                    if isinstance(val, _Tensor):
                        # Preserve the target's frozen/training flag instead of
                        # forcing requires_grad on (which would unfreeze).
                        want = bool(getattr(target_param, "requires_grad", False))
                        if bool(getattr(val, "requires_grad", False)) != want:
                            val = val.clone().requires_grad_(want) if hasattr(val, "clone") else val
                    setattr(obj, leaf, val)
            else:
                missing_keys.append(key)

        for key, target_buf in current_buffers.items():
            if key in state_dict:
                if key in unexpected_keys:
                    unexpected_keys.remove(key)
                val = state_dict[key]
                if not self._copy_state_value(target_buf, val, key, type(self).__name__):
                    obj, leaf = self._resolve_key(key)
                    if isinstance(obj, Module):
                        obj.register_buffer(leaf, val)
                    else:
                        setattr(obj, leaf, val)

        if strict and (missing_keys or unexpected_keys):
            err_msg = ""
            if missing_keys:
                err_msg += f"Missing key(s) in state_dict: {missing_keys}. "
            if unexpected_keys:
                err_msg += f"Unexpected key(s) in state_dict: {unexpected_keys}."
            raise RuntimeError(f"Error(s) in loading state_dict for {type(self).__name__}:\n\t{err_msg}")
        return {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

    def apply(self, fn):
        """Applies `fn` recursively to every submodule (as returned by `.modules()`)."""
        for module in self.modules():
            fn(module)
        return self

    def freeze(self):
        """Freezes all parameters so gradients are not computed during backward."""
        for p in self.parameters():
            if hasattr(p, "requires_grad_"):
                p.requires_grad_(False)
            elif hasattr(p, "requires_grad"):
                p.requires_grad = False
        return self

    def unfreeze(self):
        """Unfreezes all parameters so gradients are computed during backward."""
        for p in self.parameters():
            if hasattr(p, "requires_grad_"):
                p.requires_grad_(True)
            elif hasattr(p, "requires_grad"):
                p.requires_grad = True
        return self

    def register_forward_hook(self, hook):
        """Registers a forward hook on the module: `hook(module, input, output) -> None or output`."""
        handle = _HookHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle

    def register_forward_pre_hook(self, hook):
        """Registers a forward pre-hook on the module: `hook(module, input) -> None`."""
        handle = _HookHandle(self._forward_pre_hooks)
        self._forward_pre_hooks[handle.id] = hook
        return handle

    def save_file(self, path, metadata=None):
        """Saves the module's state dictionary to a `.safetensors` file."""
        from .serialization import save_file
        save_file(self, path, metadata=metadata)
        return self

    def load_file(self, path, strict=True, device=None):
        """Loads weights from a `.safetensors` file into this module."""
        from .serialization import load_file
        load_file(path, device=device, model=self, strict=strict)
        return self

    def save(self, path, metadata=None):
        """Alias for `save_file`."""
        return self.save_file(path, metadata=metadata)

    def load(self, path, strict=True, device=None):
        """Alias for `load_file`."""
        return self.load_file(path, strict=strict, device=device)

    def __repr__(self):
        lines = [f"{type(self).__name__}("]
        for name, child in self.named_children():
            child_str = repr(child)
            child_str = "\n  ".join(child_str.split("\n"))
            lines.append(f"  ({name}): {child_str}")
        for name, p in self._parameters.items():
            if p is not None:
                lines.append(f"  (param {name}): Parameter(shape={getattr(p, 'shape', [])}, dtype={getattr(p, 'dtype', 'unknown')})")
        if len(lines) == 1:
            return f"{type(self).__name__}()"
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Container Modules: Sequential, ModuleList, ModuleDict
# ---------------------------------------------------------------------------

class Sequential(Module):
    """A sequential container that passes input through modules in the order they were added."""

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            modules = args[0]
        elif len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
            return
        else:
            modules = args

        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)

    def add_module(self, name, module):
        self._modules[str(name)] = module
        super().__setattr__(str(name), module)

    def append(self, module):
        self.add_module(str(len(self._modules)), module)
        return self

    def add_linear(self, in_features, out_features, bias=True):
        return self.append(Linear(in_features, out_features, bias=bias))

    def add_relu(self):
        return self.append(ReLU())

    def add_dropout(self, p=0.5):
        if not (0.0 <= p < 1.0):
            raise ValueError(f"dropout p must be in [0, 1), got {p}")
        return self.append(Dropout(p=p))

    def add_layernorm(self, normalized_shape, eps=1e-5):
        return self.append(LayerNorm(normalized_shape, eps=eps))

    def add_gelu(self):
        return self.append(GELU())

    def add_silu(self):
        return self.append(SiLU())

    def add_sigmoid(self):
        return self.append(Sigmoid())

    def add_tanh(self):
        return self.append(Tanh())

    def add_softmax(self, dim=-1):
        return self.append(Softmax(dim=dim))

    def add_flatten(self, start_dim=1, end_dim=-1):
        return self.append(Flatten(start_dim=start_dim, end_dim=end_dim))

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        return list(self._modules.values())[idx]

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class ModuleList(Module):
    """Holds submodules in a list, automatically registering their parameters."""

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ModuleList(list(self._modules.values())[idx])
        return list(self._modules.values())[idx]

    def __setitem__(self, idx, module):
        key = str(idx)
        self._modules[key] = module
        super().__setattr__(key, module)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def append(self, module):
        self.__setitem__(len(self._modules), module)
        return self

    def extend(self, modules):
        for m in modules:
            self.append(m)
        return self


class ModuleDict(Module):
    """Holds submodules in a dictionary, automatically registering their parameters."""

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self._modules[key] = module
        super().__setattr__(key, module)

    def __delitem__(self, key):
        del self._modules[key]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __contains__(self, key):
        return key in self._modules

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def update(self, modules):
        if isinstance(modules, dict):
            for k, v in modules.items():
                self[k] = v
        elif isinstance(modules, (list, tuple)):
            for k, v in modules:
                self[k] = v
        return self


class ParameterList(Module):
    """Holds parameters in a list, automatically registering them with Module."""

    def __init__(self, values=None):
        super().__init__()
        if values is not None:
            self.extend(values)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ParameterList(list(self._parameters.values())[idx])
        return list(self._parameters.values())[idx]

    def __setitem__(self, idx, param):
        key = str(idx)
        p = param if isinstance(param, Parameter) else Parameter(param)
        self.register_parameter(key, p)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def append(self, param):
        self.__setitem__(len(self._parameters), param)
        return self

    def extend(self, values):
        for v in values:
            self.append(v)
        return self


class ParameterDict(Module):
    """Holds parameters in a dictionary, automatically registering them with Module."""

    def __init__(self, parameters=None):
        super().__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key):
        return self._parameters[key]

    def __setitem__(self, key, param):
        p = param if isinstance(param, Parameter) else Parameter(param)
        self.register_parameter(key, p)

    def __delitem__(self, key):
        del self._parameters[key]

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters)

    def __contains__(self, key):
        return key in self._parameters

    def keys(self):
        return self._parameters.keys()

    def values(self):
        return self._parameters.values()

    def items(self):
        return self._parameters.items()

    def update(self, parameters):
        if isinstance(parameters, dict):
            for k, v in parameters.items():
                self[k] = v
        elif isinstance(parameters, (list, tuple)):
            for k, v in parameters:
                self[k] = v
        return self


# ---------------------------------------------------------------------------
# Class-Based Loss Function Modules
# ---------------------------------------------------------------------------

class _Loss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def _reduce_native_mean(self, value, count, name):
        """Honors torch-style `reduction` for native kernels that always mean-reduce.

        `count` is the number of elements the native mean was taken over, so
        ``'sum' == mean * count``. ``'none'`` is rejected: the native kernel
        only returns the reduced scalar, so per-element output is unavailable.
        """
        if self.reduction == "mean":
            return value
        if self.reduction == "sum":
            return value * count
        raise ValueError(
            f"{name}: reduction must be 'mean' or 'sum' (native kernel returns "
            f"a reduced scalar; 'none' is unsupported), got {self.reduction!r}"
        )

    def _require_mean_reduction(self, name):
        """Raises unless `reduction == 'mean'`.

        For losses whose native mean has no unambiguous element count to
        rescale by (broadcast-dependent), a loud error beats silently
        ignoring the requested reduction.
        """
        if self.reduction != "mean":
            raise ValueError(
                f"{name}: only reduction='mean' is supported, got {self.reduction!r}"
            )


class MSELoss(_Loss):
    """Mean Squared Error loss as a module."""
    def forward(self, input, target):
        return self._reduce_native_mean(mse_loss(input, target), input.numel(), "MSELoss")


class CrossEntropyLoss(_Loss):
    """Cross Entropy loss with logits as a module."""
    def forward(self, input, target):
        return self._reduce_native_mean(
            cross_entropy(input, target), input.shape[0], "CrossEntropyLoss"
        )


class L1Loss(_Loss):
    """L1 / Mean Absolute Error loss as a module."""
    def forward(self, input, target):
        return self._reduce_native_mean(l1_loss(input, target), input.numel(), "L1Loss")


class BCEWithLogitsLoss(_Loss):
    """Binary Cross Entropy with logits loss as a module."""
    def forward(self, input, target):
        return self._reduce_native_mean(
            binary_cross_entropy_with_logits(input, target), input.numel(), "BCEWithLogitsLoss"
        )


class HuberLoss(_Loss):
    """Huber / Smooth L1 loss as a module."""
    def __init__(self, delta=1.0, reduction="mean"):
        super().__init__(reduction=reduction)
        self.delta = delta

    def forward(self, input, target):
        return self._reduce_native_mean(
            huber_loss(input, target, delta=self.delta), input.numel(), "HuberLoss"
        )


class NLLLoss(_Loss):
    """Negative Log Likelihood loss as a module."""
    def forward(self, input, target):
        return self._reduce_native_mean(nll_loss(input, target), input.shape[0], "NLLLoss")


class KLDivLoss(_Loss):
    """Kullback-Leibler Divergence loss as a module."""
    def forward(self, input, target):
        return self._reduce_native_mean(
            kl_div_loss(input, target), input.shape[0], "KLDivLoss"
        )


class FocalLoss(_Loss):
    """Focal loss for class-imbalanced classification."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        return focal_loss(input, target, gamma=self.gamma, alpha=self.alpha)


class CosineEmbeddingLoss(_Loss):
    """Cosine Embedding loss module."""
    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.margin = margin

    def forward(self, input1, input2, target):
        self._require_mean_reduction("CosineEmbeddingLoss")
        return cosine_embedding_loss(input1, input2, target, margin=self.margin)


class TripletMarginLoss(_Loss):
    """Triplet Margin loss module."""
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.margin = margin

    def forward(self, anchor, positive, negative):
        self._require_mean_reduction("TripletMarginLoss")
        return triplet_margin_loss(anchor, positive, negative, margin=self.margin)


class MarginRankingLoss(_Loss):
    """Margin Ranking loss module."""
    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.margin = margin

    def forward(self, input1, input2, target):
        self._require_mean_reduction("MarginRankingLoss")
        if hasattr(target, "ndim") and hasattr(input1, "ndim") and target.ndim < input1.ndim:
            shape = list(target.shape) + [1] * (input1.ndim - target.ndim)
            target = target.reshape(shape)
        return margin_ranking_loss(input1, input2, target, margin=self.margin)


class HingeEmbeddingLoss(_Loss):
    """Hinge Embedding loss module."""
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.margin = margin

    def forward(self, input, target):
        self._require_mean_reduction("HingeEmbeddingLoss")
        if hasattr(target, "ndim") and hasattr(input, "ndim") and target.ndim < input.ndim:
            shape = list(target.shape) + [1] * (input.ndim - target.ndim)
            target = target.reshape(shape)
        return hinge_embedding_loss(input, target, margin=self.margin)


class SoftMarginLoss(_Loss):
    """Soft Margin loss module."""
    def forward(self, input, target):
        return soft_margin_loss(input, target)


class BCELoss(_Loss):
    """Binary Cross Entropy Loss module."""
    def __init__(self, weight=None, size_average=None, reduce=None, reduction="mean"):
        super().__init__(reduction=reduction)
        self.weight = weight

    def forward(self, input, target):
        from . import functional as F
        if self.weight is None:
            return F.binary_cross_entropy(input, target, reduction=self.reduction)
        # torch applies `weight` elementwise BEFORE reduction; weighting the
        # reduced scalar is only equivalent for scalar weights.
        unreduced = F.binary_cross_entropy(input, target, reduction="none")
        return F._apply_reduction(unreduced * self.weight, self.reduction, "binary_cross_entropy")


class SmoothL1Loss(_Loss):
    """Smooth L1 (Huber) Loss module."""
    def __init__(self, size_average=None, reduce=None, reduction="mean", beta: float = 1.0):
        super().__init__(reduction=reduction)
        self.beta = beta

    def forward(self, input, target):
        from . import functional as F
        return F.smooth_l1_loss(input, target, beta=self.beta, reduction=self.reduction)


class PoissonNLLLoss(_Loss):
    """Poisson Negative Log Likelihood Loss module."""
    def __init__(self, log_input: bool = True, full: bool = False, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps

    def forward(self, input, target):
        from . import functional as F
        return F.poisson_nll_loss(input, target, log_input=self.log_input, full=self.full, eps=self.eps, reduction=self.reduction)


class GaussianNLLLoss(_Loss):
    """Gaussian Negative Log Likelihood Loss module."""
    def __init__(self, full: bool = False, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.full = full
        self.eps = eps

    def forward(self, input, target, var):
        from . import functional as F
        return F.gaussian_nll_loss(input, target, var, full=self.full, eps=self.eps, reduction=self.reduction)


class CTCLoss(_Loss):
    """Connectionist Temporal Classification Loss module."""
    def __init__(self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = False):
        super().__init__(reduction=reduction)
        self.blank = blank
        self.zero_infinity = zero_infinity

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        from . import functional as F
        shape = list(log_probs.shape)
        if len(shape) != 3:
            raise ValueError(f"CTCLoss: log_probs must be 3D [T, N, C], got shape {shape}")
        T, N, C = (int(v) for v in shape)
        if not 0 <= self.blank < C:
            raise ValueError(f"CTCLoss: blank must be in [0, {C}), got {self.blank}")

        def _as_1d_int(x: Any, label: str) -> np.ndarray:
            arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x)
            if arr.dtype.kind not in "iu":
                raise ValueError(f"CTCLoss: {label} must be integer type, got {arr.dtype}")
            return arr.astype(np.int64).reshape(-1)

        ilen = _as_1d_int(input_lengths, "input_lengths")
        tlen = _as_1d_int(target_lengths, "target_lengths")
        if len(ilen) != N or len(tlen) != N:
            raise ValueError(f"CTCLoss: expected {N} input/target lengths, got {len(ilen)}/{len(tlen)}")
        if not bool(np.all(ilen == T)):
            raise ValueError(
                f"CTCLoss: native kernel requires full input lengths (all == T={T}), got {ilen.tolist()}"
            )
        if not bool(np.all(tlen == tlen[0])):
            raise ValueError(f"CTCLoss: native kernel requires equal target lengths, got {tlen.tolist()}")
        S = int(tlen[0])
        t_flat = _as_1d_int(targets, "targets")
        if t_flat.shape[0] != N * S:
            raise ValueError(f"CTCLoss: targets have {t_flat.shape[0]} elements, expected {N * S}")
        tgt_ns = _from_numpy(np.ascontiguousarray(t_flat.reshape(N, S)))
        out = F.ctc_loss(log_probs, tgt_ns, blank=self.blank, zero_infinity=self.zero_infinity)
        # The native kernel mean-reduces over all N*S target positions (torch
        # CTCLoss 'mean' semantics for equal lengths), so 'sum' rescales by N*S.
        return self._reduce_native_mean(out, N * S, "CTCLoss")


# ---------------------------------------------------------------------------
# Utility & Activation Layer Modules
# ---------------------------------------------------------------------------

class Identity(Module):
    """A placeholder identity operator that returns the input unchanged."""
    def forward(self, input):
        return input


class Flatten(Module):
    """Flattens a contiguous range of dimensions into a 2D tensor."""
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        shape = list(input.shape)
        ndim = len(shape)
        start = self.start_dim if self.start_dim >= 0 else ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else ndim + self.end_dim
        if not 0 <= start < ndim or not 0 <= end < ndim:
            raise ValueError(
                f"Flatten: start_dim={self.start_dim} and end_dim={self.end_dim} "
                f"out of range for {ndim}D input"
            )
        if start > end:
            raise ValueError(
                f"Flatten: start_dim ({self.start_dim}) cannot come after end_dim ({self.end_dim})"
            )
        new_shape = shape[:start] + [int(np.prod(shape[start:end + 1]))] + shape[end + 1:]
        return input.reshape(new_shape)


class LazyLinear(Module):
    """A linear transformation with lazy initialization of the input features."""
    def __init__(self, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.in_features = None
        self.out_features = out_features
        self._has_bias = bias
        self.weight = None
        self.bias = None

    def initialize_parameters(self, input):
        in_features = input.shape[-1]
        self.in_features = in_features
        bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0
        w = _from_numpy(np.random.uniform(-bound, bound, size=(self.out_features, in_features)).astype(np.float32)).requires_grad_(True)
        self.register_parameter("weight", w)
        if self._has_bias:
            b = _from_numpy(np.random.uniform(-bound, bound, size=(self.out_features,)).astype(np.float32)).requires_grad_(True)
            self.register_parameter("bias", b)

    def forward(self, input):
        if self.weight is None:
            self.initialize_parameters(input)
        return functional.linear(input, self.weight, self.bias)


class LazyConv2d(Module):
    """A 2D convolution with lazy initialization of the in_channels."""
    def __init__(
        self,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_channels = None
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.groups = groups
        self._has_bias = bias
        self.padding_mode = padding_mode
        self.weight = None
        self.bias = None

    def initialize_parameters(self, input):
        in_channels = input.shape[1]
        self.in_channels = in_channels
        kH, kW = self.kernel_size
        fan_in = (in_channels // self.groups) * kH * kW
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        w = _from_numpy(np.random.uniform(-bound, bound, size=(self.out_channels, in_channels // self.groups, kH, kW)).astype(np.float32)).requires_grad_(True)
        self.register_parameter("weight", w)
        if self._has_bias:
            b = _from_numpy(np.random.uniform(-bound, bound, size=(self.out_channels,)).astype(np.float32)).requires_grad_(True)
            self.register_parameter("bias", b)

    def forward(self, input):
        if self.weight is None:
            self.initialize_parameters(input)
        return functional.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Unflatten(Module):
    """Unflattens a dimension of the input tensor to a specified shape."""
    def __init__(self, dim: int, unflattened_size: Sequence[int]):
        super().__init__()
        self.dim = dim
        self.unflattened_size = list(unflattened_size)

    def forward(self, input):
        from . import functional as F
        return F.unflatten(input, self.dim, self.unflattened_size)


class LayerNorm(Module):
    """Layer normalisation over the last `len(normalized_shape)` dimensions."""
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = (int(normalized_shape),) if isinstance(normalized_shape, (int, np.integer)) else tuple(int(v) for v in normalized_shape)
        if not self.normalized_shape or any(d <= 0 for d in self.normalized_shape):
            raise ValueError(f"LayerNorm: normalized_shape must be positive integers, got {normalized_shape!r}")
        self.eps, self.elementwise_affine = eps, elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(_Tensor.ones(list(self.normalized_shape)))
            self.bias = Parameter(_Tensor.zeros(list(self.normalized_shape))) if bias else None
        else:
            self.weight = None
            self.bias = None

    def forward(self, input):
        return functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


class GELU(Module):
    """Gaussian Error Linear Units activation function."""
    def __init__(self, approximate: str = "none"):
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(f"GELU: approximate must be 'none' or 'tanh', got {approximate!r}")
        self.approximate = approximate

    def forward(self, input):
        return functional.gelu(input, approximate=self.approximate)


class SiLU(Module):
    """Sigmoid Linear Unit (SiLU / Swish) activation: x * sigmoid(x)."""
    def forward(self, input):
        return input.silu()


class Sigmoid(Module):
    """Applies element-wise Sigmoid: 1 / (1 + exp(-x))."""
    def forward(self, input):
        return input.sigmoid()


class Tanh(Module):
    """Applies element-wise Hyperbolic Tangent: tanh(x)."""
    def forward(self, input):
        return input.tanh()


class LeakyReLU(Module):
    """Applies element-wise LeakyReLU: max(0, x) + negative_slope * min(0, x)."""
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        pos = input.relu()
        neg = (-input).relu()
        return pos - (neg * self.negative_slope)


class Softmax(Module):
    """Applies the Softmax function rescaling input elements to sum to 1."""
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        dim = self.dim
        if dim < 0 and hasattr(input, "shape") and input.shape:
            dim = len(input.shape) + dim
        return input.softmax(dim)


class LogSoftmax(Module):
    """Applies the log(Softmax(x)) function to an n-dimensional input Tensor."""
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        dim = self.dim
        if dim < 0 and hasattr(input, "shape") and input.shape:
            dim = len(input.shape) + dim
        return input.log_softmax(dim)


class Softmax2d(Module):
    """Applies Softmax over features to each spatial location."""
    def __init__(self):
        super().__init__()

    def forward(self, input: _Tensor) -> _Tensor:
        if input.ndim == 4:
            return F.softmax(input, dim=1)
        elif input.ndim == 3:
            return F.softmax(input, dim=0)
        else:
            raise ValueError(f"Softmax2d: input must be 3D or 4D, got {input.ndim}D")


class Softplus(Module):
    """Applies element-wise Softplus: `1/beta * log(1 + exp(beta * x))` (linear above `threshold`)."""
    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.beta, self.threshold = beta, threshold

    def forward(self, input):
        return functional.softplus(input, self.beta, self.threshold)


class Hardshrink(Module):
    """Hard shrinkage: x where |x| > lambd, else 0."""
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, input):
        return functional.hardshrink(input, self.lambd)


class Softshrink(Module):
    """Soft shrinkage: x - sign(x) * lambd where |x| > lambd, else 0."""
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, input):
        return functional.softshrink(input, self.lambd)


class LogSigmoid(Module):
    """Applies element-wise `log(sigmoid(x))`."""
    def forward(self, input):
        return functional.logsigmoid(input)

class Softsign(Module):
    """Applies element-wise Softsign: x / (1 + |x|)."""
    def forward(self, input):
        return input.softsign()


class ReLU6(Module):
    """Applies element-wise ReLU6: min(max(0, x), 6)."""
    def forward(self, input):
        from . import functional as F
        return F.relu6(input)


class Hardswish(Module):
    """Applies element-wise Hardswish."""
    def forward(self, input):
        from . import functional as F
        return F.hardswish(input)


class Hardsigmoid(Module):
    """Applies element-wise Hardsigmoid."""
    def forward(self, input):
        from . import functional as F
        return F.hardsigmoid(input)


class Hardtanh(Module):
    """Applies element-wise Hardtanh."""
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input):
        return input.clamp(self.min_val, self.max_val)


class ELU(Module):
    """Applies element-wise ELU."""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        from . import functional as F
        return F.elu(input, alpha=self.alpha)


class SELU(Module):
    """Applies element-wise SELU."""
    def forward(self, input):
        from . import functional as F
        return F.selu(input)


class CELU(Module):
    """Applies element-wise CELU."""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        from . import functional as F
        return F.celu(input, alpha=self.alpha)


class Mish(Module):
    """Applies element-wise Mish: x * tanh(softplus(x))."""
    def forward(self, input):
        from . import functional as F
        return F.mish(input)


class GLU(Module):
    """Applies Gated Linear Unit (GLU)."""
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        from . import functional as F
        return F.glu(input, dim=self.dim)


class PReLU(Module):
    """Applies Parametric ReLU."""
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        if num_parameters < 0:
            raise ValueError(f"PReLU: num_parameters must be >= 0, got {num_parameters}")
        self.num_parameters = num_parameters
        self.weight = Parameter(_Tensor.full([num_parameters], init))

    def forward(self, input):
        from . import functional as F
        return F.prelu(input, self.weight)


class RReLU(Module):
    """Applies Randomized LeakyReLU."""
    def __init__(self, lower: float = 0.125, upper: float = 0.3333333333333333):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, input):
        from . import functional as F
        return F.rrelu(input, lower=self.lower, upper=self.upper, training=self.training)


class Tanhshrink(Module):
    """Applies element-wise Tanhshrink: x - tanh(x)."""
    def forward(self, input):
        return input - input.tanh()


class Softmin(Module):
    """Applies Softmin."""
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return (-input).softmax(self.dim)


class MultiheadAttention(Module):
    """Multi-Head Attention module supporting arbitrary sequence lengths, causal masking, and batch_first."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True, batch_first: bool = False):
        super().__init__()
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by a positive num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.batch_first = batch_first
        self.dropout_p = dropout

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key=None,
        value=None,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if key is None:
            key = query
        if value is None:
            value = query

        # If 2D (L, E)
        is_2d = len(query.shape) == 2
        if is_2d:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            batch_first = True
        else:
            batch_first = self.batch_first

        if not batch_first:
            # (L, B, E) -> (B, L, E)
            q_in = query.transpose(0, 1)
            k_in = key.transpose(0, 1)
            v_in = value.transpose(0, 1)
        else:
            q_in, k_in, v_in = query, key, value

        B, L, E = q_in.shape[0], q_in.shape[1], q_in.shape[2]
        S = k_in.shape[1]

        q = self.q_proj(q_in).reshape([B, L, self.num_heads, self.head_dim]).transpose(1, 2)
        k = self.k_proj(k_in).reshape([B, S, self.num_heads, self.head_dim]).transpose(1, 2)
        v = self.v_proj(v_in).reshape([B, S, self.num_heads, self.head_dim]).transpose(1, 2)

        # Differentiable native path (stays in the autograd graph).
        scores = q.matmul(k.transpose(-2, -1)) * self.scale
        if is_causal:
            keep = _from_numpy((1.0 - np.triu(np.ones((L, S), dtype=np.float32), k=1)))
            scores = scores * keep + (1.0 - keep) * -1e9
        if attn_mask is not None:
            mask_np = attn_mask.to_numpy() if isinstance(attn_mask, _Tensor) else np.asarray(attn_mask)
            if mask_np.dtype == bool or mask_np.dtype == np.bool_:
                # In PyTorch MHA, bool attn_mask True entries are masked out (-inf).
                keep = _from_numpy(np.ascontiguousarray(np.where(mask_np, 0.0, 1.0), dtype=np.float32))
                scores = scores * keep + (1.0 - keep) * -1e9
            elif isinstance(attn_mask, _Tensor):
                scores = scores + attn_mask
            else:
                scores = scores + _from_numpy(np.ascontiguousarray(mask_np, dtype=np.float32))
        if key_padding_mask is not None:
            kpm_t = key_padding_mask if isinstance(key_padding_mask, _Tensor) else _from_numpy(np.asarray(key_padding_mask))
            if is_2d and kpm_t.ndim == 1:
                kpm_t = kpm_t.unsqueeze(0)
            if kpm_t.dtype == "bool" or (not isinstance(key_padding_mask, _Tensor) and np.asarray(key_padding_mask).dtype == np.bool_):
                pad_f = kpm_t.to_dtype(scores.dtype).reshape([B, 1, 1, S])
                scores = scores + pad_f * -1e9
            else:
                scores = scores + kpm_t.to_dtype(scores.dtype).reshape([B, 1, 1, S])

        weights = scores.softmax(-1)
        if self.training and self.dropout_p > 0.0:
            weights = functional.dropout(weights, p=self.dropout_p, training=True)

        attn_out = weights.matmul(v).transpose(1, 2).reshape([B, L, E])
        out = self.out_proj(attn_out)

        attn_weights = weights.mean(dim=1) if average_attn_weights else weights

        if not batch_first:
            out = out.transpose(0, 1)
        if is_2d:
            out = out.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        return (out, attn_weights) if need_weights else (out, None)


# ---------------------------------------------------------------------------
# Recurrent Neural Network Layers
# ---------------------------------------------------------------------------

class LSTMCell(Module):
    """Long Short-Term Memory (LSTM) cell."""
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        k = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Parameter(_Tensor.from_numpy(np.random.uniform(-k, k, (4 * hidden_size, input_size)).astype(np.float32)))
        self.weight_hh = Parameter(_Tensor.from_numpy(np.random.uniform(-k, k, (4 * hidden_size, hidden_size)).astype(np.float32)))
        if bias:
            self.bias_ih = Parameter(_Tensor.zeros([4 * hidden_size]))
            self.bias_hh = Parameter(_Tensor.zeros([4 * hidden_size]))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x, hx=None):
        if hx is None:
            batch_size = x.shape[0] if len(x.shape) > 1 else 1
            h = _Tensor.zeros([batch_size, self.hidden_size])
            c = _Tensor.zeros([batch_size, self.hidden_size])
        else:
            h, c = hx

        gates = x @ self.weight_ih.t() + h @ self.weight_hh.t()
        if self.bias:
            gates = gates + self.bias_ih + self.bias_hh

        # Differentiable gate path (stays in the autograd graph).
        H = self.hidden_size
        i = gates[..., :H].sigmoid()
        f = gates[..., H:2*H].sigmoid()
        g = gates[..., 2*H:3*H].tanh()
        o = gates[..., 3*H:].sigmoid()

        c_new = f * c + i * g
        h_new = o * c_new.tanh()

        return h_new, c_new


class GRUCell(Module):
    """Gated Recurrent Unit (GRU) cell."""
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        k = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Parameter(_Tensor.from_numpy(np.random.uniform(-k, k, (3 * hidden_size, input_size)).astype(np.float32)))
        self.weight_hh = Parameter(_Tensor.from_numpy(np.random.uniform(-k, k, (3 * hidden_size, hidden_size)).astype(np.float32)))
        if bias:
            self.bias_ih = Parameter(_Tensor.zeros([3 * hidden_size]))
            self.bias_hh = Parameter(_Tensor.zeros([3 * hidden_size]))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x, hx=None):
        if hx is None:
            batch_size = x.shape[0] if len(x.shape) > 1 else 1
            h = _Tensor.zeros([batch_size, self.hidden_size])
        else:
            h = hx

        x_gates = x @ self.weight_ih.t()
        h_gates = h @ self.weight_hh.t()
        if self.bias:
            x_gates = x_gates + self.bias_ih
            h_gates = h_gates + self.bias_hh

        # Differentiable gate path (stays in the autograd graph).
        H = self.hidden_size

        r = (x_gates[..., :H] + h_gates[..., :H]).sigmoid()
        z = (x_gates[..., H:2*H] + h_gates[..., H:2*H]).sigmoid()
        n = (x_gates[..., 2*H:] + r * h_gates[..., 2*H:]).tanh()

        h_next = (1.0 - z) * n + z * h
        return h_next


class RNNBase(Module):
    """Base class for recurrent modules (RNN, LSTM, GRU)."""
    def __init__(self, mode: str, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional


class LSTM(RNNBase):
    """Multi-layer Long Short-Term Memory (LSTM) network."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = True):
        super().__init__("LSTM", input_size, hidden_size, num_layers, bias, batch_first)

        num_layers = _check_repeats("num_layers", num_layers)
        self.cells = ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias)
            for i in range(num_layers)
        ])

    def forward(self, x, hx=None):
        x_seq = x if self.batch_first else x.transpose(0, 1)
        batch_size, seq_len, _ = (int(v) for v in x_seq.shape)
        if hx is None:
            h_prev = [_Tensor.zeros([batch_size, self.hidden_size], dtype=x_seq.dtype, device=x_seq.device) for _ in range(self.num_layers)]
            c_prev = [_Tensor.zeros([batch_size, self.hidden_size], dtype=x_seq.dtype, device=x_seq.device) for _ in range(self.num_layers)]
        else:
            h0, c0 = hx
            h_prev = [h0[l] for l in range(self.num_layers)] if isinstance(h0, _Tensor) else list(h0)
            c_prev = [c0[l] for l in range(self.num_layers)] if isinstance(c0, _Tensor) else list(c0)

        layer_inputs = [x_seq[:, t, :] for t in range(seq_len)]
        for l, cell in enumerate(self.cells):
            next_layer_inputs = []
            h, c = h_prev[l], c_prev[l]
            for t in range(seq_len):
                h, c = cell(layer_inputs[t], (h, c))
                next_layer_inputs.append(h)
            layer_inputs = next_layer_inputs
            h_prev[l] = h
            c_prev[l] = c

        out = _Tensor.stack(layer_inputs, dim=1)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, (_Tensor.stack(h_prev, dim=0), _Tensor.stack(c_prev, dim=0))


class GRU(RNNBase):
    """Multi-layer Gated Recurrent Unit (GRU) network."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = True):
        super().__init__("GRU", input_size, hidden_size, num_layers, bias, batch_first)

        num_layers = _check_repeats("num_layers", num_layers)
        self.cells = ModuleList([
            GRUCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias)
            for i in range(num_layers)
        ])

    def forward(self, x, hx=None):
        x_seq = x if self.batch_first else x.transpose(0, 1)
        batch_size, seq_len, _ = (int(v) for v in x_seq.shape)
        if hx is None:
            h_prev = [_Tensor.zeros([batch_size, self.hidden_size], dtype=x_seq.dtype, device=x_seq.device) for _ in range(self.num_layers)]
        else:
            h_prev = [hx[l] for l in range(self.num_layers)] if isinstance(hx, _Tensor) else list(hx)

        layer_inputs = [x_seq[:, t, :] for t in range(seq_len)]
        for l, cell in enumerate(self.cells):
            next_layer_inputs = []
            h = h_prev[l]
            for t in range(seq_len):
                h = cell(layer_inputs[t], h)
                next_layer_inputs.append(h)
            layer_inputs = next_layer_inputs
            h_prev[l] = h

        out = _Tensor.stack(layer_inputs, dim=1)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, _Tensor.stack(h_prev, dim=0)


# ---------------------------------------------------------------------------
# Normalization Layers
# ---------------------------------------------------------------------------

class RMSNorm(Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""
    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int], None] = None,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        *,
        dim: Union[int, Sequence[int], None] = None,
    ):
        super().__init__()
        shape_arg = normalized_shape if normalized_shape is not None else dim
        if shape_arg is None:
            raise ValueError("RMSNorm requires normalized_shape (or dim)")
        if isinstance(shape_arg, int):
            self.normalized_shape = (shape_arg,)
        else:
            self.normalized_shape = tuple(shape_arg)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(_Tensor.ones(list(self.normalized_shape)))
        else:
            self.weight = None

    def forward(self, x):
        return functional.rms_norm(x, self.normalized_shape, weight=self.weight, eps=self.eps)


class Threshold(Module):
    """Applies the Threshold function element-wise: y = x if x > threshold else value."""
    def __init__(self, threshold: float, value: float):
        super().__init__()
        self.threshold = float(threshold)
        self.value = float(value)

    def forward(self, input: _Tensor) -> _Tensor:
        return functional.threshold(input, self.threshold, self.value)


def _batch_norm_train_stats(x, num_features):
    """Differentiable batch mean/var over all dims but 1 (mirrors F.batch_norm).

    Returns (mean_b, var_b, mean_c, unbiased_var_c): broadcastable Tensors for
    the norm path plus flat [C] numpy stats for the running update (torch
    parity: biased var normalizes, unbiased var feeds the running average).
    """
    c = num_features
    flat = x.transpose(0, 1).reshape([c, -1])
    count = int(flat.shape[-1])
    mean = flat.mean(dim=-1, keepdim=True)
    var = ((flat - mean) * (flat - mean)).mean(dim=-1, keepdim=True)
    bshape = [1, c] + [1] * (len(x.shape) - 2)
    mean_c = mean.reshape([c]).to_numpy()
    var_c = var.reshape([c]).to_numpy()
    if count > 1:
        var_c = var_c * (count / (count - 1))
    return mean.reshape(bshape), var.reshape(bshape), mean_c, var_c


class _BatchNormNd(Module):
    """Batch normalisation with PyTorch's state: `running_mean`, `running_var` and `num_batches_tracked` are buffers (they are in
    `state_dict()` and move with the module); `momentum=None` uses a cumulative moving average."""
    _ndims = (2, 3)

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: Optional[float] = 0.1, affine: bool = True,
                 track_running_stats: bool = True, device=None, dtype=None):
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.affine, self.track_running_stats = affine, track_running_stats
        self._init_state(num_features)

    def _init_state(self, n: int) -> None:
        self.num_features = n
        if self.affine:
            self.weight = Parameter(_Tensor.ones([n]))
            self.bias = Parameter(_Tensor.zeros([n]))
        else:
            self.weight = None
            self.bias = None
        if self.track_running_stats:
            self.register_buffer("running_mean", _Tensor.zeros([n]))
            self.register_buffer("running_var", _Tensor.ones([n]))
            self.register_buffer("num_batches_tracked", _from_numpy(np.array(0, dtype=np.int64)))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x):
        if len(x.shape) not in self._ndims:
            dims = " or ".join(f"{d}D" for d in self._ndims)
            raise ValueError(f"{type(self).__name__}: expected {dims} input (got {len(x.shape)}D input)")
        if int(x.shape[1]) != self.num_features:
            raise RuntimeError(f"{type(self).__name__}: expected {self.num_features} channels, got {int(x.shape[1])}")
        momentum = self.momentum
        if self.training and self.track_running_stats:
            self.num_batches_tracked.assign(self.num_batches_tracked + 1)
            if momentum is None:
                momentum = 1.0 / float(self.num_batches_tracked.item())
        use_batch_stats = self.training or not self.track_running_stats
        return functional.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, use_batch_stats,
                                     0.1 if momentum is None else momentum, self.eps)

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}"


class _LazyBatchNormMixin:
    def __init__(self, eps: float = 1e-5, momentum: Optional[float] = 0.1, affine: bool = True, track_running_stats: bool = True):
        Module.__init__(self)
        self.num_features, self.eps, self.momentum = None, eps, momentum
        self.affine, self.track_running_stats = affine, track_running_stats
        self.weight = None
        self.bias = None

    def initialize_parameters(self, input):
        self._init_state(int(input.shape[1]))

    def forward(self, x):
        if self.num_features is None:
            self.initialize_parameters(x)
        return super().forward(x)


class BatchNorm1d(_BatchNormNd):
    """1D Batch Normalization over 2D or 3D input."""
    _ndims = (2, 3)

class BatchNorm2d(_BatchNormNd):
    """2D Batch Normalization over 4D NCHW input."""
    _ndims = (4,)

class LazyBatchNorm1d(_LazyBatchNormMixin, BatchNorm1d):
    """1D Batch Normalization with lazy initialization of num_features."""

class LazyBatchNorm2d(_LazyBatchNormMixin, BatchNorm2d):
    """2D Batch Normalization with lazy initialization of num_features."""

class SyncBatchNorm(BatchNorm2d):
    """Applies Batch Normalization synchronized across multiple processes."""
    @classmethod
    def convert_sync_batchnorm(cls, module: Module) -> Module:
        return module


class GroupNorm(Module):
    """Group Normalization over channels."""
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if num_groups <= 0 or num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by a positive num_groups ({num_groups})")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = Parameter(_Tensor.ones([num_channels]))
            self.bias = Parameter(_Tensor.zeros([num_channels]))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        from . import functional as F
        return F.group_norm(x, self.num_groups, weight=self.weight, bias=self.bias, eps=self.eps)


# ---------------------------------------------------------------------------
# Vision Pooling & Resampling
# ---------------------------------------------------------------------------

class MaxPool2d(Module):
    """2D Max Pooling over input planes."""
    def __init__(self, kernel_size: Union[int, Tuple[int, int]] = 2, stride: Optional[Union[int, Tuple[int, int]]] = None, padding: Union[int, Tuple[int, int]] = 0, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = self.kernel_size if stride is None else ((stride, stride) if isinstance(stride, int) else tuple(stride))
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return functional.max_pool2d(x, self.kernel_size, self.stride, self.padding, ceil_mode=self.ceil_mode)


class FractionalMaxPool2d(Module):
    """Applies a 2D fractional max pooling over an input signal."""
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        output_size: Optional[Union[int, Tuple[int, int]]] = None,
        output_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        return_indices: bool = False,
        _random_samples: Optional[_Tensor] = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.output_ratio = output_ratio
        self.return_indices = return_indices
        self._random_samples = _random_samples

    def forward(self, input: _Tensor):
        return functional.fractional_max_pool2d(
            input,
            self.kernel_size,
            self.output_size,
            self.output_ratio,
            self.return_indices,
            self._random_samples,
        )


class FractionalMaxPool3d(Module):
    """Applies a 3D fractional max pooling over an input signal."""
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        output_size: Optional[Union[int, Tuple[int, int, int]]] = None,
        output_ratio: Optional[Union[float, Tuple[float, float, float]]] = None,
        return_indices: bool = False,
        _random_samples: Optional[_Tensor] = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.output_ratio = output_ratio
        self.return_indices = return_indices
        self._random_samples = _random_samples

    def forward(self, input: _Tensor):
        return functional.fractional_max_pool3d(
            input,
            self.kernel_size,
            self.output_size,
            self.output_ratio,
            self.return_indices,
            self._random_samples,
        )


class MaxUnpool1d(Module):
    """Computes a partial inverse of MaxPool1d."""
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Optional[Union[int, Tuple[int]]] = None,
        padding: Union[int, Tuple[int]] = 0,
    ):
        super().__init__()
        self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = self.kernel_size if stride is None else ((stride,) if isinstance(stride, int) else tuple(stride))
        self.padding = (padding,) if isinstance(padding, int) else tuple(padding)

    def forward(self, input, indices, output_size=None):
        return functional.max_unpool1d(input, indices, self.kernel_size, self.stride, self.padding, output_size)


class MaxUnpool2d(Module):
    """Computes a partial inverse of MaxPool2d."""
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = self.kernel_size if stride is None else ((stride, stride) if isinstance(stride, int) else tuple(stride))
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)

    def forward(self, input, indices, output_size=None):
        return functional.max_unpool2d(input, indices, self.kernel_size, self.stride, self.padding, output_size)


class MaxUnpool3d(Module):
    """Computes a partial inverse of MaxPool3d."""
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = self.kernel_size if stride is None else ((stride, stride, stride) if isinstance(stride, int) else tuple(stride))
        self.padding = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)

    def forward(self, input, indices, output_size=None):
        return functional.max_unpool3d(input, indices, self.kernel_size, self.stride, self.padding, output_size)


class AvgPool2d(Module):
    """2D Average Pooling over input planes."""
    def __init__(self, kernel_size: Union[int, Tuple[int, int]] = 2, stride: Optional[Union[int, Tuple[int, int]]] = None, padding: Union[int, Tuple[int, int]] = 0, ceil_mode: bool = False, count_include_pad: bool = True):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = self.kernel_size if stride is None else ((stride, stride) if isinstance(stride, int) else tuple(stride))
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x):
        return functional.avg_pool2d(x, self.kernel_size, self.stride, self.padding, ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad)


class AdaptiveAvgPool2d(Module):
    """Applies a 2D adaptive average pooling over an input signal."""
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)

    def forward(self, x):
        return functional.adaptive_avg_pool2d(x, self.output_size)


class Upsample(Module):
    """Upsamples 1-D, 2-D or 3-D data (`mode`: nearest, nearest-exact, linear, bilinear, bicubic, trilinear, area)."""
    def __init__(self, size=None, scale_factor=None, mode: str = "nearest", align_corners=None, recompute_scale_factor=None):
        super().__init__()
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        self.size, self.scale_factor, self.mode = size, scale_factor, mode
        self.align_corners, self.recompute_scale_factor = align_corners, recompute_scale_factor

    def forward(self, x):
        return functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                      align_corners=self.align_corners)

class UpsamplingNearest2d(Module):
    """Applies a 2D nearest neighbor upsampling to an input signal composed of several input channels."""
    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, input):
        from . import functional as F
        return F.interpolate(input, size=self.size, scale_factor=self.scale_factor, mode="nearest")


class UpsamplingBilinear2d(Module):
    """Applies a 2D bilinear upsampling to an input signal composed of several input channels."""
    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, input):
        from . import functional as F
        return F.interpolate(input, size=self.size, scale_factor=self.scale_factor, mode="bilinear", align_corners=True)


class PixelShuffle(Module):
    """Rearranges elements in a tensor of shape (N, C * r^2, H, W) to (N, C, H * r, W * r)."""
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        from . import functional as F
        return F.pixel_shuffle(input, self.upscale_factor)


class PixelUnshuffle(Module):
    """Rearranges elements in a tensor of shape (N, C, H * r, W * r) to (N, C * r^2, H, W)."""
    def __init__(self, downscale_factor: int):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        from . import functional as F
        return F.pixel_unshuffle(input, self.downscale_factor)


class Unfold(Module):
    """Extracts sliding local blocks from a batched input tensor (im2col)."""
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], dilation: int = 1, padding: int = 0, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        from . import functional as F
        return F.unfold(input, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)


class Fold(Module):
    """Combines an array of sliding local blocks into a large containing tensor (col2im)."""
    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]],
        kernel_size: Union[int, Tuple[int, int]],
        dilation: int = 1,
        padding: int = 0,
        stride: int = 1,
    ):
        super().__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        from . import functional as F
        return F.fold(input, self.output_size, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)


class Dropout1d(Module):
    """Randomly zero out entire channels along 1D features."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        from . import functional as F
        return F.dropout1d(input, self.p, training=self.training)


class Dropout2d(Module):
    """Randomly zero out entire channels along 2D spatial features."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        from . import functional as F
        return F.dropout2d(input, self.p, training=self.training)


class Dropout3d(Module):
    """Randomly zero out entire channels along 3D volumetric features."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        from . import functional as F
        return F.dropout3d(input, self.p, training=self.training)


class AlphaDropout(Module):
    """Applies Alpha Dropout over the input tensor."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        from . import functional as F
        return F.alpha_dropout(input, self.p, training=self.training)


class FeatureAlphaDropout(Module):
    """Applies Alpha Dropout over entire channels."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        from . import functional as F
        return F.feature_alpha_dropout(input, self.p, training=self.training)


class EmbeddingBag(Module):
    """Computes sums or means of bags of embeddings without instantiating intermediate embeddings."""
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.mode = mode
        self.sparse = sparse
        w = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        self.weight = Parameter(_from_numpy(w))

    def forward(self, input, offsets=None, per_sample_weights=None):
        from . import functional as F
        return F.embedding_bag(
            input,
            self.weight,
            offsets=offsets,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            mode=self.mode,
            sparse=self.sparse,
            per_sample_weights=per_sample_weights,
        )


class BatchNorm3d(_BatchNormNd):
    """3D Batch Normalization over 5D NCDHW input."""
    _ndims = (5,)

class LazyBatchNorm3d(_LazyBatchNormMixin, BatchNorm3d):
    """3D Batch Normalization with lazy initialization of num_features."""

class InstanceNorm1d(Module):
    """Applies 1D Instance Normalization."""
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = False, track_running_stats: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Parameter(_Tensor.ones([num_features]))
            self.bias = Parameter(_Tensor.zeros([num_features]))
        else:
            self.weight = None
            self.bias = None
        if track_running_stats:
            self.running_mean = _Tensor.zeros([num_features])
            self.running_var = _Tensor.ones([num_features])
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        from . import functional as F
        return F.instance_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)


class InstanceNorm2d(Module):
    """Applies 2D Instance Normalization."""
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = False, track_running_stats: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Parameter(_Tensor.ones([num_features]))
            self.bias = Parameter(_Tensor.zeros([num_features]))
        else:
            self.weight = None
            self.bias = None
        if track_running_stats:
            self.running_mean = _Tensor.zeros([num_features])
            self.running_var = _Tensor.ones([num_features])
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        from . import functional as F
        return F.instance_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)


class InstanceNorm3d(Module):
    """Applies 3D Instance Normalization."""
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = False, track_running_stats: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Parameter(_Tensor.ones([num_features]))
            self.bias = Parameter(_Tensor.zeros([num_features]))
        else:
            self.weight = None
            self.bias = None
        if track_running_stats:
            self.running_mean = _Tensor.zeros([num_features])
            self.running_var = _Tensor.ones([num_features])
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        from . import functional as F
        return F.instance_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)


class LazyInstanceNorm1d(InstanceNorm1d):
    """1D Instance Normalization with lazy initialization of num_features."""
    def __init__(self, eps: float = 1e-5, momentum: float = 0.1, affine: bool = False):
        Module.__init__(self)
        self.num_features = None
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.weight = None
        self.bias = None

    def initialize_parameters(self, input):
        num_features = input.shape[1]
        self.num_features = num_features
        if self.affine:
            self.register_parameter("weight", Parameter(_Tensor.ones([num_features])))
            self.register_parameter("bias", Parameter(_Tensor.zeros([num_features])))

    def forward(self, x):
        if self.num_features is None:
            self.initialize_parameters(x)
        return super().forward(x)


class LazyInstanceNorm2d(InstanceNorm2d):
    """2D Instance Normalization with lazy initialization of num_features."""
    def __init__(self, eps: float = 1e-5, momentum: float = 0.1, affine: bool = False):
        Module.__init__(self)
        self.num_features = None
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.weight = None
        self.bias = None

    def initialize_parameters(self, input):
        num_features = input.shape[1]
        self.num_features = num_features
        if self.affine:
            self.register_parameter("weight", Parameter(_Tensor.ones([num_features])))
            self.register_parameter("bias", Parameter(_Tensor.zeros([num_features])))

    def forward(self, x):
        if self.num_features is None:
            self.initialize_parameters(x)
        return super().forward(x)


class LazyInstanceNorm3d(InstanceNorm3d):
    """3D Instance Normalization with lazy initialization of num_features."""
    def __init__(self, eps: float = 1e-5, momentum: float = 0.1, affine: bool = False):
        Module.__init__(self)
        self.num_features = None
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.weight = None
        self.bias = None

    def initialize_parameters(self, input):
        num_features = input.shape[1]
        self.num_features = num_features
        if self.affine:
            self.register_parameter("weight", Parameter(_Tensor.ones([num_features])))
            self.register_parameter("bias", Parameter(_Tensor.zeros([num_features])))

    def forward(self, x):
        if self.num_features is None:
            self.initialize_parameters(x)
        return super().forward(x)


class LocalResponseNorm(Module):
    """Applies local response normalization over an input signal."""
    def __init__(self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        from . import functional as F
        return F.local_response_norm(input, self.size, alpha=self.alpha, beta=self.beta, k=self.k)


class _ConvNd(Module):
    """Shared implementation of Conv1d / Conv2d / Conv3d: tuple kernel sizes, stride, dilation, groups, 'same' / 'valid' padding,
    `padding_mode` ('zeros', 'reflect', 'replicate', 'circular'), PyTorch's default initialisation."""
    _n = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", device=None, dtype=None):
        super().__init__()
        n = self._n
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        if padding_mode not in ("zeros", "reflect", "replicate", "circular"):
            raise ValueError(f"padding_mode must be 'zeros', 'reflect', 'replicate' or 'circular', got {padding_mode!r}")
        if isinstance(padding, str) and padding not in ("same", "valid"):
            raise ValueError(f"padding must be 'same', 'valid', an int or a tuple, got {padding!r}")
        tup = lambda v: (int(v),) * n if isinstance(v, (int, np.integer)) else tuple(int(x) for x in v)
        elems = (out_channels * in_channels // max(groups, 1)) * int(np.prod([int(k) for k in tup(kernel_size)], dtype=object))
        if in_channels <= 0 or out_channels <= 0 or elems > 2 ** 31:
            raise RuntimeError(f"Conv{n}d: the weight size overflows the supported tensor size ({in_channels}, {out_channels}, {kernel_size})")
        self.in_channels, self.out_channels, self.groups, self.padding_mode = in_channels, out_channels, groups, padding_mode
        self.kernel_size, self.stride, self.dilation = tup(kernel_size), tup(stride), tup(dilation)
        self.padding = padding if isinstance(padding, str) else tup(padding)
        fan_in = (in_channels // groups) * int(np.prod(self.kernel_size))
        k = 1.0 / math.sqrt(fan_in)
        w = np.random.uniform(-k, k, (out_channels, in_channels // groups) + self.kernel_size).astype(np.float32)
        self.weight = Parameter(_from_numpy(w))
        self.bias = Parameter(_from_numpy(np.random.uniform(-k, k, (out_channels,)).astype(np.float32))) if bias else None

    def forward(self, input):
        from . import functional as F
        fn = (F.conv1d, F.conv2d, F.conv3d)[self._n - 1]
        pad = self.padding
        if self.padding_mode != "zeros" and not isinstance(pad, str):
            spec = []
            for p_ in reversed(pad):
                spec += [p_, p_]
            input = F.pad(input, spec, mode=self.padding_mode)
            pad = (0,) * self._n
        one = self._n == 1
        pick = (lambda v: v[0]) if one else (lambda v: v)
        return fn(input, self.weight, self.bias, stride=pick(self.stride), padding=pad if isinstance(pad, str) else pick(pad),
                  dilation=pick(self.dilation), groups=self.groups)

    def extra_repr(self):
        return f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}"


class Conv1d(_ConvNd):
    """Applies a 1D convolution over an input signal composed of several input planes."""
    _n = 1


class Conv2d(_ConvNd):
    """Applies a 2D convolution over an input image composed of several input planes."""
    _n = 2


class Conv3d(_ConvNd):
    """Applies a 3D convolution over an input volume composed of several input planes."""
    _n = 3

class ConvTranspose1d(Module):
    """Applies a 1D transposed convolution operator over an input signal."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

        k = 1.0 / math.sqrt(in_channels * kernel_size)
        w = np.random.uniform(-k, k, (in_channels, out_channels // groups, kernel_size)).astype(np.float32)
        self.weight = Parameter(_from_numpy(w))
        if bias:
            b = np.random.uniform(-k, k, (out_channels,)).astype(np.float32)
            self.bias = Parameter(_from_numpy(b))
        else:
            self.bias = None

    def forward(self, input):
        from . import functional as F
        return F.conv_transpose1d(
            input, self.weight, self.bias, stride=self.stride, padding=self.padding,
            output_padding=self.output_padding, groups=self.groups, dilation=self.dilation
        )


class ConvTranspose2d(Module):
    """Applies a 2D transposed convolution operator over an input image."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
    ):
        super().__init__()
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

        scale = 1.0 / math.sqrt(in_channels * k[0] * k[1])
        w = np.random.uniform(-scale, scale, (in_channels, out_channels // groups, k[0], k[1])).astype(np.float32)
        self.weight = Parameter(_from_numpy(w))
        if bias:
            b = np.random.uniform(-scale, scale, (out_channels,)).astype(np.float32)
            self.bias = Parameter(_from_numpy(b))
        else:
            self.bias = None

    def forward(self, input):
        from . import functional as F
        return F.conv_transpose2d(
            input, self.weight, self.bias, stride=self.stride, padding=self.padding,
            output_padding=self.output_padding, groups=self.groups, dilation=self.dilation
        )


class ConvTranspose3d(Module):
    """Applies a 3D transposed convolution operator over an input volume."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
    ):
        super().__init__()
        k = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

        scale = 1.0 / math.sqrt(in_channels * k[0] * k[1] * k[2])
        w = np.random.uniform(-scale, scale, (in_channels, out_channels // groups, k[0], k[1], k[2])).astype(np.float32)
        self.weight = Parameter(_from_numpy(w))
        if bias:
            b = np.random.uniform(-scale, scale, (out_channels,)).astype(np.float32)
            self.bias = Parameter(_from_numpy(b))
        else:
            self.bias = None

    def forward(self, input):
        from . import functional as F
        return F.conv_transpose3d(
            input, self.weight, self.bias, stride=self.stride, padding=self.padding,
            output_padding=self.output_padding, groups=self.groups, dilation=self.dilation
        )


class MaxPool1d(Module):
    """Applies a 1D max pooling over an input signal."""
    def __init__(self, kernel_size: int, stride: Optional[int] = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        from . import functional as F
        return F.max_pool1d(input, self.kernel_size, stride=self.stride, padding=self.padding)


class MaxPool3d(Module):
    """Applies a 3D max pooling over an input volume."""
    def __init__(self, kernel_size: Union[int, Tuple[int, int, int]], stride: Optional[Union[int, Tuple[int, int, int]]] = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        from . import functional as F
        return F.max_pool3d(input, self.kernel_size, stride=self.stride, padding=self.padding)


class AvgPool1d(Module):
    """Applies a 1D average pooling over an input signal."""
    def __init__(self, kernel_size: int, stride: Optional[int] = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        from . import functional as F
        return F.avg_pool1d(input, self.kernel_size, stride=self.stride, padding=self.padding)


class AvgPool3d(Module):
    """Applies a 3D average pooling over an input volume."""
    def __init__(self, kernel_size: Union[int, Tuple[int, int, int]], stride: Optional[Union[int, Tuple[int, int, int]]] = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        from . import functional as F
        return F.avg_pool3d(input, self.kernel_size, stride=self.stride, padding=self.padding)


class AdaptiveAvgPool1d(Module):
    """Applies a 1D adaptive average pooling over an input signal."""
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        from . import functional as F
        return F.adaptive_avg_pool1d(input, self.output_size)


class AdaptiveAvgPool3d(Module):
    """Applies a 3D adaptive average pooling over an input volume."""
    def __init__(self, output_size: Union[int, Tuple[int, int, int]]):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        from . import functional as F
        return F.adaptive_avg_pool3d(input, self.output_size)


class AdaptiveMaxPool1d(Module):
    """Applies a 1D adaptive max pooling over an input signal."""
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        from . import functional as F
        return F.adaptive_max_pool1d(input, self.output_size)


class AdaptiveMaxPool2d(Module):
    """Applies a 2D adaptive max pooling over an input image."""
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        from . import functional as F
        return F.adaptive_max_pool2d(input, self.output_size)


class AdaptiveMaxPool3d(Module):
    """Applies a 3D adaptive max pooling over an input volume."""
    def __init__(self, output_size: Union[int, Tuple[int, int, int]]):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        from . import functional as F
        return F.adaptive_max_pool3d(input, self.output_size)


class LPPool1d(Module):
    """Applies a 1D power-average pooling over an input signal."""
    def __init__(self, norm_type: float, kernel_size: int, stride: Optional[int] = None):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        from . import functional as F
        return F.lp_pool1d(input, self.norm_type, self.kernel_size, stride=self.stride)


class LPPool2d(Module):
    """Applies a 2D power-average pooling over an input image."""
    def __init__(self, norm_type: float, kernel_size: Union[int, Tuple[int, int]], stride: Optional[Union[int, Tuple[int, int]]] = None):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        from . import functional as F
        return F.lp_pool2d(input, self.norm_type, self.kernel_size, stride=self.stride)


class TransformerEncoderLayer(Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network."""
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout) if dropout > 0 else Identity()
        self.dropout1 = Dropout(dropout) if dropout > 0 else Identity()
        self.dropout2 = Dropout(dropout) if dropout > 0 else Identity()
        self.activation = GELU() if activation == "gelu" else ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal: bool = False):
        x = src
        if self.norm_first:
            sa_out, _ = self.self_attn(
                self.norm1(x),
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )
            x = x + self.dropout1(sa_out)
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x)))))
            x = x + self.dropout2(ff_out)
        else:
            sa_out, _ = self.self_attn(
                x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )
            x = self.norm1(x + self.dropout1(sa_out))
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = self.norm2(x + self.dropout2(ff_out))
        return x


class TransformerEncoder(Module):
    """TransformerEncoder is a stack of N encoder layers."""
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int, norm=None):
        super().__init__()
        num_layers = _check_repeats("num_layers", num_layers)
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        output = src
        for mod in self.layers:
            if is_causal is None:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            else:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output



class TransformerDecoderLayer(Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network."""
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout) if dropout > 0 else Identity()
        self.dropout1 = Dropout(dropout) if dropout > 0 else Identity()
        self.dropout2 = Dropout(dropout) if dropout > 0 else Identity()
        self.dropout3 = Dropout(dropout) if dropout > 0 else Identity()
        self.activation = GELU() if activation == "gelu" else ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal: bool = False, memory_is_causal: bool = False):
        x = tgt
        if self.norm_first:
            sa_out, _ = self.self_attn(
                self.norm1(x),
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
                is_causal=tgt_is_causal,
            )
            x = x + self.dropout1(sa_out)
            ma_out, _ = self.multihead_attn(
                self.norm2(x),
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
                is_causal=memory_is_causal,
            )
            x = x + self.dropout2(ma_out)
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(x)))))
            x = x + self.dropout3(ff_out)
        else:
            sa_out, _ = self.self_attn(
                x,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
                is_causal=tgt_is_causal,
            )
            x = self.norm1(x + self.dropout1(sa_out))
            ma_out, _ = self.multihead_attn(
                x,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
                is_causal=memory_is_causal,
            )
            x = self.norm2(x + self.dropout2(ma_out))
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = self.norm3(x + self.dropout3(ff_out))
        return x


class TransformerDecoder(Module):
    """TransformerDecoder is a stack of N decoder layers."""
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int, norm=None):
        super().__init__()
        num_layers = _check_repeats("num_layers", num_layers)
        self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(Module):
    """A full Transformer model."""
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        custom_encoder: Optional[Module] = None,
        custom_decoder: Optional[Module] = None,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first
            )
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first
            )
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output


class RNNCell(Module):
    """An Elman RNN cell with tanh or ReLU non-linearity."""
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        k = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Parameter(_Tensor.from_numpy(np.random.uniform(-k, k, (hidden_size, input_size)).astype(np.float32)))
        self.weight_hh = Parameter(_Tensor.from_numpy(np.random.uniform(-k, k, (hidden_size, hidden_size)).astype(np.float32)))
        if bias:
            self.bias_ih = Parameter(_Tensor.zeros([hidden_size]))
            self.bias_hh = Parameter(_Tensor.zeros([hidden_size]))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x, hx=None):
        if hx is None:
            batch_size = x.shape[0] if len(x.shape) > 1 else 1
            h = _Tensor.zeros([batch_size, self.hidden_size])
        else:
            h = hx

        gates = x @ self.weight_ih.t() + h @ self.weight_hh.t()
        if self.bias:
            gates = gates + self.bias_ih + self.bias_hh

        return gates.relu() if self.nonlinearity == "relu" else gates.tanh()


class RNN(RNNBase):
    """Multi-layer Elman RNN with tanh or ReLU non-linearity."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__("RNN", input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        self.nonlinearity = nonlinearity

        num_layers = _check_repeats("num_layers", num_layers)
        self.cells = ModuleList([
            RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity)
            for i in range(num_layers)
        ])

    def forward(self, x, hx=None):
        x_seq = x if self.batch_first else x.transpose(0, 1)
        batch_size, seq_len, _ = (int(v) for v in x_seq.shape)
        if hx is None:
            h_prev = [_Tensor.zeros([batch_size, self.hidden_size], dtype=x_seq.dtype, device=x_seq.device) for _ in range(self.num_layers)]
        else:
            h_prev = [hx[l] for l in range(self.num_layers)] if isinstance(hx, _Tensor) else list(hx)

        layer_inputs = [x_seq[:, t, :] for t in range(seq_len)]
        for l, cell in enumerate(self.cells):
            next_layer_inputs = []
            h = h_prev[l]
            for t in range(seq_len):
                h = cell(layer_inputs[t], h)
                next_layer_inputs.append(h)
            layer_inputs = next_layer_inputs
            h_prev[l] = h

        out = _Tensor.stack(layer_inputs, dim=1)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, _Tensor.stack(h_prev, dim=0)


class _AdaptiveLogSoftmaxOutput:
    def __init__(self, output: _Tensor, loss: _Tensor):
        self.output = output
        self.loss = loss


class AdaptiveLogSoftmaxWithLoss(Module):
    """Adaptive log softmax with loss for efficient training over large vocabularies."""
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        cutoffs: Sequence[int],
        div_value: float = 4.0,
        head_bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = list(cutoffs) + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        head_size = self.cutoffs[0] + len(self.cutoffs) - 1
        self.head = Linear(in_features, head_size, bias=head_bias)

        self.tail = ModuleList()
        for i in range(len(self.cutoffs) - 1):
            start = self.cutoffs[i]
            end = self.cutoffs[i + 1]
            cluster_size = end - start
            proj_dim = max(1, int(in_features / (div_value ** (i + 1))))
            tail_cluster = Sequential(
                Linear(in_features, proj_dim, bias=False),
                Linear(proj_dim, cluster_size, bias=False),
            )
            self.tail.append(tail_cluster)

    def forward(self, input: _Tensor, target: _Tensor) -> _AdaptiveLogSoftmaxOutput:
        from . import functional as F
        N = input.shape[0]
        head_out = self.head(input)
        log_prob_head = F.log_softmax(head_out, dim=-1)

        # Compute nll loss on targets in head and clusters
        target_np = target.to_numpy().reshape(-1).astype(np.int64)
        losses = []
        outputs = []
        for b in range(N):
            tgt = int(target_np[b])
            if tgt < self.cutoffs[0]:
                loss_b = -log_prob_head.narrow(0, b, 1).squeeze(0).narrow(0, tgt, 1).squeeze(0)
                out_b = log_prob_head.narrow(0, b, 1).squeeze(0).narrow(0, tgt, 1).squeeze(0)
            else:
                # Find cluster
                cluster_idx = 0
                for ci in range(len(self.cutoffs) - 1):
                    if self.cutoffs[ci] <= tgt < self.cutoffs[ci + 1]:
                        cluster_idx = ci
                        break
                cluster_head_tgt = self.cutoffs[0] + cluster_idx
                head_log_p = log_prob_head.narrow(0, b, 1).squeeze(0).narrow(0, cluster_head_tgt, 1).squeeze(0)
                cluster_logits = self.tail[cluster_idx](input.narrow(0, b, 1))
                cluster_log_p = F.log_softmax(cluster_logits, dim=-1).squeeze(0)
                sub_idx = tgt - self.cutoffs[cluster_idx]
                tail_log_p = cluster_log_p.narrow(0, sub_idx, 1).squeeze(0)
                total_log_p = head_log_p + tail_log_p
                loss_b = -total_log_p
                out_b = total_log_p
            losses.append(loss_b)
            outputs.append(out_b)

        total_loss = _native_stack(losses, dim=0).mean()
        all_outputs = _native_stack(outputs, dim=0)
        return _AdaptiveLogSoftmaxOutput(all_outputs, total_loss)


class Bilinear(Module):
    """Applies a bilinear transformation to the incoming data: y = x_1^T A x_2 + b."""
    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias

        k = 1.0 / math.sqrt(in1_features)
        self.weight = Parameter(_Tensor.from_numpy(np.random.uniform(-k, k, (out_features, in1_features, in2_features)).astype(np.float32)))
        if bias:
            self.bias_param = Parameter(_Tensor.zeros([out_features]))
        else:
            self.bias_param = None

    def forward(self, input1: _Tensor, input2: _Tensor) -> _Tensor:
        from . import functional as F
        return F.bilinear(input1, input2, self.weight, self.bias_param)


class CosineSimilarity(Module):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    def __init__(self, dim: int = 1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: _Tensor, x2: _Tensor) -> _Tensor:
        from . import functional as F
        return F.cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)


class PairwiseDistance(Module):
    """Computes the pairwise distance between input vectors, or between columns of input matrices."""
    def __init__(self, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False):
        super().__init__()
        self.p = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1: _Tensor, x2: _Tensor) -> _Tensor:
        from . import functional as F
        return F.pairwise_distance(x1, x2, p=self.p, eps=self.eps, keepdim=self.keepdim)


class ChannelShuffle(Module):
    """Divide channels into groups and permute to rearrange channels."""
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        return F.channel_shuffle(input, self.groups)


class ZeroPad1d(Module):
    """Pads the input tensor boundaries with zero."""
    def __init__(self, padding: Union[int, Tuple[int, int]]):
        super().__init__()
        self.padding = padding

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="constant", value=0.0)


class ZeroPad2d(Module):
    """Pads the input tensor boundaries with zero."""
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__()
        self.padding = padding

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding, self.padding, self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="constant", value=0.0)


class ZeroPad3d(Module):
    """Pads the input tensor boundaries with zero."""
    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]]):
        super().__init__()
        self.padding = padding

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding,) * 6 if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="constant", value=0.0)


class ConstantPad1d(Module):
    """Pads the input tensor boundaries with a constant value."""
    def __init__(self, padding: Union[int, Tuple[int, int]], value: float):
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="constant", value=self.value)


class ConstantPad2d(Module):
    """Pads the input tensor boundaries with a constant value."""
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]], value: float):
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding, self.padding, self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="constant", value=self.value)


class ConstantPad3d(Module):
    """Pads the input tensor boundaries with a constant value."""
    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]], value: float):
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding,) * 6 if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="constant", value=self.value)


class ReflectionPad1d(Module):
    """Pads the input tensor using the reflection of the input boundary."""
    def __init__(self, padding: Union[int, Tuple[int, int]]):
        super().__init__()
        self.padding = padding

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="reflect")


class ReflectionPad2d(Module):
    """Pads the input tensor using the reflection of the input boundary."""
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__()
        self.padding = padding

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding, self.padding, self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="reflect")


class ReflectionPad3d(Module):
    """Pads the input tensor using the reflection of the input boundary."""
    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]]):
        super().__init__()
        self.padding = padding

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding,) * 6 if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="reflect")


class ReplicationPad1d(Module):
    """Pads the input tensor using replication of the input boundary."""
    def __init__(self, padding: Union[int, Tuple[int, int]]):
        super().__init__()
        self.padding = padding

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="replicate")


class ReplicationPad2d(Module):
    """Pads the input tensor using replication of the input boundary."""
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__()
        self.padding = padding

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding, self.padding, self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="replicate")


class ReplicationPad3d(Module):
    """Pads the input tensor using replication of the input boundary."""
    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]]):
        super().__init__()
        self.padding = padding

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding,) * 6 if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="replicate")


class CircularPad1d(Module):
    """Pads the input tensor using circular padding."""
    def __init__(self, padding: Union[int, Tuple[int, int]]):
        super().__init__()
        self.padding = padding

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="circular")


class CircularPad2d(Module):
    """Pads the input tensor using circular padding."""
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__()
        self.padding = padding

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding, self.padding, self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="circular")


class CircularPad3d(Module):
    """Pads the input tensor using circular padding."""
    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]]):
        super().__init__()
        self.padding = padding

    def forward(self, input: _Tensor) -> _Tensor:
        from . import functional as F
        pad = (self.padding,) * 6 if isinstance(self.padding, int) else self.padding
        return F.pad(input, pad, mode="circular")


class MultiMarginLoss(Module):
    """Creates a criterion that optimizes a multi-class classification hinge loss."""
    def __init__(self, p: int = 1, margin: float = 1.0, weight: Optional[_Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.p = p
        self.margin = margin
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: _Tensor, target: _Tensor) -> _Tensor:
        if self.reduction not in ("mean", "sum", "none"):
            raise ValueError(f"MultiMarginLoss: reduction must be 'mean', 'sum' or 'none', got {self.reduction!r}")
        from . import functional as F
        n = int(input.shape[0])
        c = int(input.shape[1])
        tidx = _from_numpy(target.to_numpy().astype(np.int64)).reshape([n, 1])
        correct = input.gather(1, tidx)
        margins = ((self.margin - correct + input).relu()) ** self.p
        mask = F.one_hot(tidx.reshape([n]), c).to_dtype(str(input.dtype))
        margins = margins * (1.0 - mask)
        if self.weight is not None:
            margins = margins * self.weight
        per = margins.mean(dim=-1)
        if self.reduction == "mean":
            return per.mean()
        if self.reduction == "sum":
            return per.sum()
        return per


class MultiLabelMarginLoss(Module):
    """Creates a criterion that optimizes a multi-class multi-classification hinge loss."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: _Tensor, target: _Tensor) -> _Tensor:
        if self.reduction not in ("mean", "sum", "none"):
            raise ValueError(f"MultiLabelMarginLoss: reduction must be 'mean', 'sum' or 'none', got {self.reduction!r}")
        n = int(input.shape[0])
        c = int(input.shape[1])
        t_np = target.to_numpy().astype(int)
        valid = np.zeros((n, c), dtype=np.float32)
        for b in range(n):
            for y in t_np[b]:
                if y < 0:
                    continue
                if y >= c:
                    raise ValueError(f"MultiLabelMarginLoss: target {y} out of range for {c} classes")
                valid[b, y] = 1.0
        V = _from_numpy(np.ascontiguousarray(valid))
        M = (1.0 - input.unsqueeze(-1) + input.unsqueeze(1)).relu()
        P = V.unsqueeze(-1) * (1.0 - V.unsqueeze(1))
        per = (M * P).reshape([n, -1]).sum(dim=-1) / max(1, c)
        if self.reduction == "mean":
            return per.mean()
        if self.reduction == "sum":
            return per.sum()
        return per


class MultiLabelSoftMarginLoss(Module):
    """Creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy."""
    def __init__(self, weight: Optional[_Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: _Tensor, target: _Tensor) -> _Tensor:
        if self.reduction not in ("mean", "sum", "none"):
            raise ValueError(f"MultiLabelSoftMarginLoss: reduction must be 'mean', 'sum' or 'none', got {self.reduction!r}")
        lse = (1.0 + (input * -1.0).exp()).log()
        loss = lse * target + (input + lse) * (1.0 - target)
        if self.weight is not None:
            loss = loss * self.weight
        per = loss.mean(dim=-1)
        if self.reduction == "mean":
            return per.mean()
        if self.reduction == "sum":
            return per.sum()
        return per


from .losses_ext import TripletMarginWithDistanceLoss


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

from .distributed import DataParallel, DistributedDataParallel, FullyShardedDataParallel


# ---------------------------------------------------------------------------
# Module API for the native layers (Linear, Embedding, Dropout, ReLU, ...): wrap in a one-child Sequential and strip the prefix
# ---------------------------------------------------------------------------

def _native_module_api() -> None:
    def holder(self):
        return Sequential(self)

    def named_parameters(self, prefix="", recurse=True):
        for n, p in holder(self).named_parameters():
            n = n[2:]
            yield (f"{prefix}.{n}" if prefix else n), p

    def parameters_(self, recurse=True):
        return list(self.parameters_native())

    def state_dict(self, *args, **kwargs):
        return OrderedDict((n[2:], v) for n, v in holder(self).state_dict(*args, **kwargs).items())

    def load_state_dict(self, state, strict=True):
        return holder(self).load_state_dict({f"0.{k}": v for k, v in state.items()}, strict=strict)

    def zero_grad(self, set_to_none=False):
        holder(self).zero_grad(set_to_none=set_to_none)

    def requires_grad_(self, requires_grad=True):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def to(self, *args, **kwargs):
        return self  # native layers live on the CPU in float32

    def apply(self, fn):
        fn(self)
        return self

    def named_modules(self, prefix=""):
        yield prefix, self

    def modules(self):
        yield self

    def children(self):
        return iter(())

    def named_children(self):
        return iter(())

    def buffers(self, recurse=True):
        return iter(())

    def named_buffers(self, prefix="", recurse=True):
        return iter(())

    api = dict(named_parameters=named_parameters, state_dict=state_dict, load_state_dict=load_state_dict, zero_grad=zero_grad,
               requires_grad_=requires_grad_, to=to, apply=apply, named_modules=named_modules, modules=modules, children=children,
               named_children=named_children, buffers=buffers, named_buffers=named_buffers)
    for cls in (Linear, SpectralNormLinear, ConvLstmCell, DropPath, Dropout, Embedding, Maxout, Snake, Swish, Upsample2d, ReLU, MLP,
                _NativeConv2d, _NativeConv3d, _NativeLayerNorm):
        for name, fn in api.items():
            if not hasattr(cls, name):
                setattr(cls, name, fn)
        for name in ("train", "eval"):
            orig = getattr(cls, name, None)
            if orig is None:
                setattr(cls, name, lambda self, mode=True: self)
            else:  # the native method returns None; PyTorch returns the module so calls chain
                def chained(self, *a, _o=orig, **k):
                    _o(self, *a, **k)
                    return self
                setattr(cls, name, chained)
        if not hasattr(cls, "training"):
            cls.training = property(lambda self: False)


_native_module_api()

__all__ = [
    # Core Base Classes & Containers
    "Module",
    "Parameter",
    "Sequential",
    "ModuleList",
    "ModuleDict",
    "ParameterList",
    "ParameterDict",
    "RNNBase",

    # Layers & Models
    "Linear",
    "LazyLinear",
    "SpectralNormLinear",
    "Bilinear",
    "Conv1d",
    "Conv2d",
    "LazyConv2d",
    "Conv3d",
    "ConvLstmCell",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "Embedding",
    "EmbeddingBag",
    "LayerNorm",
    "RMSNorm",
    "BatchNorm1d",
    "Threshold",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LocalResponseNorm",
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    "CosineSimilarity",
    "PairwiseDistance",
    "ChannelShuffle",
    "MaxPool1d",
    "MaxPool2d",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "MaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "LPPool1d",
    "LPPool2d",
    "Upsample",
    "Upsample2d",
    "PixelShuffle",
    "PixelUnshuffle",
    "Unfold",
    "Fold",
    "RNNCell",
    "RNN",
    "LSTMCell",
    "GRUCell",
    "LSTM",
    "GRU",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "DropPath",
    "AlphaDropout",
    "Maxout",
    "Snake",
    "Swish",
    "ReLU",
    "ReLU6",
    "GELU",
    "SiLU",
    "Sigmoid",
    "Tanh",
    "LeakyReLU",
    "Softmax",
    "LogSoftmax",
    "Softmin",
    "Softplus",
    "LogSigmoid",
    "Softshrink",
    "Hardshrink",
    "Softsign",
    "Hardswish",
    "Hardsigmoid",
    "Hardtanh",
    "ELU",
    "SELU",
    "CELU",
    "Mish",
    "GLU",
    "PReLU",
    "RReLU",
    "Tanhshrink",
    "MultiheadAttention",
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "TransformerDecoderLayer",
    "TransformerDecoder",
    "Transformer",
    "MLP",
    "Identity",
    "Flatten",
    "Unflatten",

    # Class-Based Loss Modules
    "MSELoss",
    "CrossEntropyLoss",
    "L1Loss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "SmoothL1Loss",
    "HuberLoss",
    "NLLLoss",
    "KLDivLoss",
    "PoissonNLLLoss",
    "GaussianNLLLoss",
    "CTCLoss",
    "FocalLoss",
    "CosineEmbeddingLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
    "MarginRankingLoss",
    "HingeEmbeddingLoss",
    "SoftMarginLoss",
    "MultiMarginLoss",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",

    # Submodules
    "functional",
    "F",
    "utils",
    "init",

    # Functional Losses
    "mse_loss",
    "cross_entropy",
    "l1_loss",
    "binary_cross_entropy_with_logits",
    "huber_loss",
    "nll_loss",
    "kl_div_loss",
    "focal_loss",
    "cosine_embedding_loss",
    "triplet_margin_loss",
    "info_nce_loss",
    "dpo_loss",
    "ppo_clipped_loss",
    "simpo_loss",
    "siglip_loss",
    "barlow_twins_loss",
    "vicreg_loss",
    "grpo_loss",
    "margin_ranking_loss",
    "hinge_embedding_loss",
    "soft_margin_loss",
    "ctc_loss",

    # Serialization
    "save_file",
    "load_file",
    "save",
    "load",

    # Layers & Distributed
    "Softmax2d",
    "SyncBatchNorm",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
    "FeatureAlphaDropout",
    "UpsamplingNearest2d",
    "UpsamplingBilinear2d",
    "AdaptiveLogSoftmaxWithLoss",
    "DataParallel",
    "DistributedDataParallel",
    "FullyShardedDataParallel",
]


# ---------------------------------------------------------------------------------------------------------------------
# Max pooling modules with dilation / ceil_mode / return_indices, LPPool3d, lazy convolutions, CrossMapLRN2d, Buffer,
# uninitialized (lazy) parameters, RNNCellBase, NLLLoss2d, Container, LinearCrossEntropyLoss
# ---------------------------------------------------------------------------------------------------------------------

class _MaxPoolNd(Module):
    _nd = 2

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x):
        f = getattr(functional, f"max_pool{self._nd}d")
        return f(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)

    def extra_repr(self) -> str:
        return (f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, "
                f"ceil_mode={self.ceil_mode}")


class MaxPool1d(_MaxPoolNd):
    """1-D max pooling (kernel_size, stride, padding, dilation, return_indices, ceil_mode as in PyTorch)."""
    _nd = 1


class MaxPool2d(_MaxPoolNd):
    """2-D max pooling (kernel_size, stride, padding, dilation, return_indices, ceil_mode as in PyTorch)."""
    _nd = 2

    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)


class MaxPool3d(_MaxPoolNd):
    """3-D max pooling (kernel_size, stride, padding, dilation, return_indices, ceil_mode as in PyTorch)."""
    _nd = 3


class _AdaptiveMaxPoolNd(Module):
    _nd = 2

    def __init__(self, output_size, return_indices: bool = False):
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, x):
        return getattr(functional, f"adaptive_max_pool{self._nd}d")(x, self.output_size, self.return_indices)


class AdaptiveMaxPool1d(_AdaptiveMaxPoolNd):
    """Adaptive 1-D max pooling to `output_size` (with `return_indices`)."""
    _nd = 1


class AdaptiveMaxPool2d(_AdaptiveMaxPoolNd):
    """Adaptive 2-D max pooling to `output_size` (with `return_indices`)."""
    _nd = 2


class AdaptiveMaxPool3d(_AdaptiveMaxPoolNd):
    """Adaptive 3-D max pooling to `output_size` (with `return_indices`)."""
    _nd = 3


class LPPool3d(Module):
    """3-D power-average pooling: (sum over the window of x ** p) ** (1 / p)."""

    def __init__(self, norm_type: float, kernel_size, stride=None, ceil_mode: bool = False):
        super().__init__()
        self.norm_type, self.kernel_size, self.stride, self.ceil_mode = norm_type, kernel_size, stride, ceil_mode

    def forward(self, x):
        return functional.lp_pool3d(x, self.norm_type, self.kernel_size, self.stride, self.ceil_mode)


class _LazyConvNd(Module):
    """A convolution whose `in_channels` is taken from the first input."""

    _nd = 2
    _transposed = False

    def __init__(self, out_channels: int, kernel_size, stride=1, padding=0, output_padding=0, groups: int = 1,
                 bias: bool = True, dilation=1, padding_mode: str = "zeros", device=None, dtype=None):
        super().__init__()
        if padding_mode != "zeros":
            raise NotImplementedError("lazy convolutions support padding_mode='zeros' only")
        nd = self._nd
        tup = lambda v: tuple(v) if isinstance(v, (list, tuple)) else (v,) * nd
        self.in_channels = None
        self.out_channels = out_channels
        self.kernel_size, self.stride, self.padding = tup(kernel_size), tup(stride), tup(padding)
        self.output_padding, self.dilation = tup(output_padding), tup(dilation)
        self.groups = groups
        self._has_bias = bias
        self.padding_mode = padding_mode
        self.weight = None
        self.bias = None

    def has_uninitialized_params(self) -> bool:
        return self.weight is None

    def initialize_parameters(self, input) -> None:
        c_in = int(input.shape[1]) if len(input.shape) == self._nd + 2 else int(input.shape[0])
        if c_in % self.groups or self.out_channels % self.groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.in_channels = c_in
        k = int(np.prod(self.kernel_size))
        if self._transposed:
            shape = (c_in, self.out_channels // self.groups) + self.kernel_size
            fan_in = (self.out_channels // self.groups) * k
        else:
            shape = (self.out_channels, c_in // self.groups) + self.kernel_size
            fan_in = (c_in // self.groups) * k
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        self.register_parameter("weight", _from_numpy(np.random.uniform(-bound, bound, size=shape).astype(np.float32)).requires_grad_(True))
        if self._has_bias:
            self.register_parameter("bias", _from_numpy(np.random.uniform(-bound, bound, size=(self.out_channels,)).astype(np.float32)).requires_grad_(True))

    def forward(self, x):
        if self.weight is None:
            self.initialize_parameters(x)
        # the functional convolutions take an int where all dimensions agree (some accept nothing else)
        arg = lambda t: t[0] if len(set(t)) == 1 else t
        kw = dict(stride=arg(self.stride), padding=arg(self.padding), dilation=arg(self.dilation), groups=self.groups)
        if self._transposed:
            return getattr(functional, f"conv_transpose{self._nd}d")(x, self.weight, self.bias, output_padding=arg(self.output_padding), **kw)
        return getattr(functional, f"conv{self._nd}d")(x, self.weight, self.bias, **kw)


class LazyConv1d(_LazyConvNd):
    """`Conv1d` whose in_channels is inferred from the first input."""
    _nd = 1

    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None):
        super().__init__(out_channels, kernel_size, stride, padding, 0, groups, bias, dilation, padding_mode)


class LazyConv3d(_LazyConvNd):
    """`Conv3d` whose in_channels is inferred from the first input."""
    _nd = 3

    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None):
        super().__init__(out_channels, kernel_size, stride, padding, 0, groups, bias, dilation, padding_mode)


class LazyConvTranspose1d(_LazyConvNd):
    """`ConvTranspose1d` whose in_channels is inferred from the first input."""
    _nd, _transposed = 1, True


class LazyConvTranspose2d(_LazyConvNd):
    """`ConvTranspose2d` whose in_channels is inferred from the first input."""
    _nd, _transposed = 2, True


class LazyConvTranspose3d(_LazyConvNd):
    """`ConvTranspose3d` whose in_channels is inferred from the first input."""
    _nd, _transposed = 3, True


def _cross_map_window(channels: int, size: int) -> np.ndarray:
    """W[c, j] = 1 when channel j enters channel c's sum in PyTorch's legacy CrossMapLRN2d (whose window is slightly
    asymmetric), found by running its running-sum recurrence on indicator vectors."""
    pre_pad = int((size - 1) / 2 + 1)
    w = np.zeros((channels, channels), dtype=np.float32)
    w[0, :min(pre_pad, channels)] = 1.0
    for c in range(1, channels):
        w[c] = w[c - 1]
        if c < channels - pre_pad + 1:
            w[c, c + pre_pad - 1] += 1.0
        if c > pre_pad:
            w[c, c - pre_pad] -= 1.0
    return w


class CrossMapLRN2d(Module):
    """PyTorch's legacy cross-channel local response normalization for 4-D input:
    y = x * (k + alpha / size * sum of x^2 over neighbouring channels) ** -beta."""

    def __init__(self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0):
        super().__init__()
        self.size, self.alpha, self.beta, self.k = size, alpha, beta, k

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError(f"CrossMapLRN2d: Expected input to be 4D, got {len(x.shape)}D instead.")
        w = _from_numpy(np.ascontiguousarray(_cross_map_window(int(x.shape[1]), self.size).T))
        sq = (x * x).permute([0, 2, 3, 1])
        summed = sq.matmul(w).permute([0, 3, 1, 2])
        return x * (summed * (self.alpha / self.size) + self.k).pow(-self.beta)


class Buffer:
    """Marks a tensor as a buffer: `self.running = nn.Buffer(t)` inside a Module registers it with `register_buffer`
    (`persistent=False` keeps it out of the state dict)."""

    def __init__(self, data: Any = None, *, persistent: bool = True):
        self.data = data if isinstance(data, _Tensor) or data is None else _from_numpy(np.asarray(data))
        self.persistent = persistent


_module_setattr = Module.__setattr__


def _setattr_with_buffers(self, name, value):
    if isinstance(value, Buffer) and hasattr(self, "_modules"):
        self.register_buffer(name, value.data, persistent=value.persistent)
        return
    _module_setattr(self, name, value)


Module.__setattr__ = _setattr_with_buffers


class UninitializedParameter:
    """A parameter whose shape is not known yet (for lazy modules). `materialize(shape)` returns the real parameter,
    zero-filled and requiring grad, which the owner assigns in place of this placeholder."""

    cls_to_become = "Parameter"

    def __init__(self, requires_grad: bool = True, device=None, dtype=None):
        self.requires_grad = requires_grad
        self.dtype = dtype

    def materialize(self, shape, device=None, dtype=None) -> _Tensor:
        t = _from_numpy(np.zeros(tuple(shape), dtype=np.float32))
        return t.requires_grad_(True) if self.requires_grad else t


class UninitializedBuffer(UninitializedParameter):
    """A buffer whose shape is not known yet; `materialize(shape)` returns the zero-filled tensor."""

    cls_to_become = "Tensor"

    def __init__(self, requires_grad: bool = False, device=None, dtype=None, persistent: bool = True):
        super().__init__(requires_grad, device, dtype)
        self.persistent = persistent


def is_lazy(param: Any) -> bool:
    """Whether `param` is an uninitialized (lazy) parameter or buffer."""
    return isinstance(param, UninitializedParameter)


class RNNCellBase(Module):
    """Base class of RNN cells: `weight_ih` (num_chunks * hidden, input), `weight_hh` (num_chunks * hidden, hidden) and
    optional biases, all initialized uniformly in +-1/sqrt(hidden_size). Subclasses implement `forward`."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int):
        super().__init__()
        self.input_size, self.hidden_size, self.bias = input_size, hidden_size, bias
        bound = 1.0 / math.sqrt(hidden_size) if hidden_size > 0 else 0.0
        mk = lambda *s: _from_numpy(np.random.uniform(-bound, bound, size=s).astype(np.float32)).requires_grad_(True)
        self.weight_ih = mk(num_chunks * hidden_size, input_size)
        self.weight_hh = mk(num_chunks * hidden_size, hidden_size)
        if bias:
            self.bias_ih = mk(num_chunks * hidden_size)
            self.bias_hh = mk(num_chunks * hidden_size)
        else:
            self.bias_ih = self.bias_hh = None

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}" + ("" if self.bias else ", bias=False")


class NLLLoss2d(NLLLoss):
    """Deprecated alias of `NLLLoss` (kept for old code)."""


class Container(Module):
    """Deprecated: a Module that adds the keyword arguments as child modules."""

    def __init__(self, **kwargs: Any):
        super().__init__()
        for name, module in kwargs.items():
            self.add_module(name, module)


class LinearCrossEntropyLoss(Module):
    """A classification head and its loss in one: holds the (num_classes, in_features) weight (and an optional bias)
    and returns cross_entropy(input @ weight.T + bias, target)."""

    def __init__(self, in_features: int, num_classes: int, *, out_features=(), bias: bool = False, device=None, dtype=None,
                 reduction: str = "mean", weight: Optional[_Tensor] = None, ignore_index: Optional[int] = None,
                 label_smoothing: float = 0.0, options: Any = None):
        super().__init__()
        if tuple(out_features):
            raise NotImplementedError("LinearCrossEntropyLoss: out_features is not supported")
        bound = 1.0 / math.sqrt(in_features)
        self.linear_weight = _from_numpy(np.random.uniform(-bound, bound, size=(num_classes, in_features)).astype(np.float32)).requires_grad_(True)
        self.linear_bias = _from_numpy(np.random.uniform(-bound, bound, size=(num_classes,)).astype(np.float32)).requires_grad_(True) if bias else None
        self.reduction, self.class_weight = reduction, weight
        self.ignore_index = -100 if ignore_index is None else ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        return functional.linear_cross_entropy(input, self.linear_weight, target, linear_bias=self.linear_bias,
                                               weight=self.class_weight, reduction=self.reduction,
                                               ignore_index=self.ignore_index, label_smoothing=self.label_smoothing)


__all__ += [
    "MaxPool1d", "MaxPool2d", "MaxPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d", "LPPool3d",
    "LazyConv1d", "LazyConv3d", "LazyConvTranspose1d", "LazyConvTranspose2d", "LazyConvTranspose3d", "CrossMapLRN2d",
    "Buffer", "UninitializedParameter", "UninitializedBuffer", "is_lazy", "RNNCellBase", "NLLLoss2d", "Container",
    "LinearCrossEntropyLoss",
]
__all__ = list(dict.fromkeys(__all__))
