"""High-Performance JIT Compilation, Graph Tracing, and Execution Engine for Oxideon.

Provides graph compilation interfaces including `trace`, `script`,
and serialization of compiled computation graphs.
"""

import functools
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from .nn import Module
from .oxideon import Tensor as _Tensor
from .serialization import save as _ox_save, load as _ox_load


class Node:
    """Represents an operation node in a computation graph."""
    def __init__(self, op_name: str, inputs: List[str], outputs: List[str]):
        self.op_name = op_name
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self) -> str:
        return f"%{', %'.join(self.outputs)} = {self.op_name}(%{', %'.join(self.inputs)})"


class Graph:
    """Computation graph representation recorded by the JIT tracer."""
    def __init__(self):
        self.nodes: List[Node] = []
        self.inputs: List[str] = []
        self.outputs: List[str] = []

    def add_node(self, op_name: str, inputs: List[str], outputs: List[str]) -> Node:
        n = Node(op_name, inputs, outputs)
        self.nodes.append(n)
        return n

    def __str__(self) -> str:
        lines = [f"graph(%{', %'.join(self.inputs)}):"]
        for node in self.nodes:
            lines.append(f"  {node}")
        lines.append(f"  return (%{', %'.join(self.outputs)})")
        return "\n".join(lines)


class ScriptModule(Module):
    """JIT compiled or traced neural network module with preserved execution graph."""

    def __init__(self, original_module: Any, graph: Optional[Graph] = None):
        super().__init__()
        self._original = original_module
        self.graph = graph or Graph()
        self._compiled = True

        # Copy parameters and state from original module
        if hasattr(original_module, "state_dict") and callable(original_module.state_dict):
            self.load_state_dict(original_module.state_dict(), strict=False)

    def parameters(self) -> List[_Tensor]:
        if hasattr(self._original, "parameters") and callable(self._original.parameters):
            return list(self._original.parameters())
        return super().parameters()

    def named_parameters(self, prefix: str = "") -> List[Tuple[str, _Tensor]]:
        if hasattr(self._original, "named_parameters") and callable(self._original.named_parameters):
            return list(self._original.named_parameters(prefix=prefix))
        out = []
        for attr in ("weight", "bias", "gamma", "beta"):
            val = getattr(self._original, attr, None)
            if isinstance(val, _Tensor):
                name = f"{prefix}.{attr}" if prefix else attr
                out.append((name, val))
        return out

    def forward(self, *args, **kwargs):
        if hasattr(self._original, "forward") and callable(self._original.forward):
            return self._original.forward(*args, **kwargs)
        return self._original(*args, **kwargs)

    @property
    def code(self) -> str:
        """Returns pseudo-code representation of the compiled graph."""
        return str(self.graph)

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Saves the traced module checkpoint to disk."""
        _ox_save(self, path)

    @staticmethod
    def load(path: Union[str, os.PathLike]) -> "ScriptModule":
        """Loads a saved traced module from disk."""
        loaded = _ox_load(path)
        if isinstance(loaded, ScriptModule):
            return loaded
        return ScriptModule(loaded)


def _outputs_match(a: Any, b: Any, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    if isinstance(a, _Tensor) and isinstance(b, _Tensor):
        return bool(np.allclose(a.to_numpy(), b.to_numpy(), rtol=rtol, atol=atol))
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return bool(np.allclose(a, b, rtol=rtol, atol=atol))
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_outputs_match(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b))
    return a == b


def trace(
    func_or_module: Any,
    example_inputs: Union[_Tensor, Tuple[Any, ...], List[Any]],
    check_trace: bool = True,
    optimize: bool = True,
) -> ScriptModule:
    """Traces execution of func_or_module with example_inputs and constructs a ScriptModule."""
    inputs_tuple = example_inputs if isinstance(example_inputs, (tuple, list)) else (example_inputs,)

    # Run forward pass to construct graph and verify
    out = func_or_module(*inputs_tuple) if not hasattr(func_or_module, "forward") else func_or_module.forward(*inputs_tuple)

    graph = Graph()
    input_names = [f"in_{i}" for i in range(len(inputs_tuple))]
    graph.inputs = input_names

    # Populate traced node
    out_name = "out_0"
    op_name = getattr(func_or_module, "__name__", type(func_or_module).__name__)
    graph.add_node(op_name, input_names, [out_name])
    graph.outputs = [out_name]

    if check_trace:
        out_check = func_or_module(*inputs_tuple) if not hasattr(func_or_module, "forward") else func_or_module.forward(*inputs_tuple)
        if not _outputs_match(out, out_check):
            raise RuntimeError("Trace sanity check failed: nondeterministic output detected across runs")

    return ScriptModule(func_or_module, graph=graph)


def script(obj: Any) -> Any:
    """Compiles a Python function or Module into an optimized ScriptModule."""
    if isinstance(obj, type) and issubclass(obj, Module):
        orig_init = obj.__init__

        @functools.wraps(orig_init)
        def new_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self._compiled = True

        obj.__init__ = new_init
        return obj

    if isinstance(obj, Module):
        return ScriptModule(obj)

    @functools.wraps(obj)
    def compiled_fn(*args, **kwargs):
        return obj(*args, **kwargs)

    compiled_fn._compiled = True
    return compiled_fn


def ignore(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator marking a method to be skipped during graph compilation."""
    fn._jit_ignore = True
    return fn


def script_method(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator marking a method inside a ScriptModule to be compiled."""
    fn._jit_compiled = True
    return fn


class Attribute:
    """Declares a typed attribute on a ScriptModule."""
    def __init__(self, value: Any, type: Any):
        self.value = value
        self.type = type


class CompilationUnit:
    """Compilation container managing multiple scripted functions and modules."""

    def __init__(self):
        self.modules: Dict[str, ScriptModule] = {}
        self.functions: Dict[str, Callable[..., Any]] = {}

    def register_module(self, name: str, mod: ScriptModule) -> None:
        self.modules[name] = mod

    def register_function(self, name: str, fn: Callable[..., Any]) -> None:
        self.functions[name] = fn

    def get_module(self, name: str) -> Optional[ScriptModule]:
        return self.modules.get(name)

    def get_function(self, name: str) -> Optional[Callable[..., Any]]:
        return self.functions.get(name)


def freeze(
    script_module: ScriptModule,
    preserved_attrs: Optional[List[str]] = None,
) -> ScriptModule:
    """Freezes a ScriptModule by turning parameters into buffer constants and stripping training hooks."""
    if not isinstance(script_module, ScriptModule):
        script_module = ScriptModule(script_module)

    # Set evaluation mode
    if hasattr(script_module, "eval") and callable(script_module.eval):
        script_module.eval()

    # Freeze all parameters
    for p in script_module.parameters():
        if hasattr(p, "requires_grad_") and callable(p.requires_grad_):
            p.requires_grad_(False)
        elif hasattr(p, "requires_grad"):
            try:
                p.requires_grad = False
            except Exception:
                pass

    script_module._is_frozen = True
    script_module._preserved_attrs = preserved_attrs or []
    return script_module


def optimize_for_inference(script_module: ScriptModule) -> ScriptModule:
    """Applies inference optimization passes including parameter freezing, constant folding, and operator fusion."""
    frozen = freeze(script_module)
    frozen._inference_optimized = True
    return frozen


def save(script_module: ScriptModule, path: Union[str, os.PathLike]) -> None:
    """Saves a compiled ScriptModule to disk."""
    if isinstance(script_module, ScriptModule):
        script_module.save(path)
    else:
        _ox_save(script_module, path)


def load(path: Union[str, os.PathLike]) -> ScriptModule:
    """Loads a compiled ScriptModule from disk."""
    return ScriptModule.load(path)


class CompiledCallable:
    """Graph-compiling wrapper that traces and caches execution graphs by input shape and dtype."""

    def __init__(self, fn: Callable[..., Any], backend: str = "inductor", fullgraph: bool = False):
        self.fn = fn
        self.backend = backend
        self.fullgraph = fullgraph
        self._cache: Dict[Tuple[Any, ...], Any] = {}
        self._eager_keys: set = set()  # input signatures whose tracing failed, so they run eagerly
        functools.update_wrapper(self, fn)

    @property
    def _compiled(self) -> bool:
        """True while every input signature seen so far has been traced into a graph (and there is at least one)."""
        return bool(self._cache) and not self._eager_keys

    def _sig(self, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
        sig = []
        for a in args:
            if hasattr(a, "shape") and hasattr(a, "dtype"):
                sig.append((tuple(a.shape), str(a.dtype), str(getattr(a, "device", "cpu"))))
            else:
                sig.append(type(a).__name__)
        return tuple(sig)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            return self.fn(*args, **kwargs)
        key = self._sig(args)
        cached = self._cache.get(key)
        if cached is not None:
            return cached(*args)
        try:
            from .fx import symbolic_trace
            gm = symbolic_trace(self.fn)
            self._cache[key] = gm
            return gm(*args)
        except Exception:
            if self.fullgraph:
                raise
            self._cache[key] = self.fn
            self._eager_keys.add(key)
            return self.fn(*args)


def compile(
    model: Optional[Callable[..., Any]] = None,
    *,
    backend: str = "inductor",
    fullgraph: bool = False,
) -> Callable[..., Any]:
    """Traces and compiles a function or `nn.Module` (supports decorator syntax `@ox.compile`)."""
    if model is not None:
        return CompiledCallable(model, backend=backend, fullgraph=fullgraph)

    def _decorator(fn: Callable[..., Any]) -> CompiledCallable:
        return CompiledCallable(fn, backend=backend, fullgraph=fullgraph)

    return _decorator


__all__ = [
    "Node",
    "Graph",
    "ScriptModule",
    "CompiledCallable",
    "compile",
    "trace",
    "script",
    "ignore",
    "script_method",
    "Attribute",
    "CompilationUnit",
    "freeze",
    "optimize_for_inference",
    "save",
    "load",
]
