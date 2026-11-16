"""Tracer mechanism for capturing Module computation graphs symbolically."""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .graph import Graph
from .node import Node
from .proxy import Proxy
from ..nn import Module


class _SubmoduleProxyWrapper:
    def __init__(self, mod: Any, path: str, tracer: "Tracer"):
        self._orig_mod = mod
        self._path = path
        self._tracer = tracer

    def __call__(self, *args, **kwargs):
        unwrapped_args = tuple(self._tracer.create_arg(a) for a in args)
        unwrapped_kwargs = {k: self._tracer.create_arg(v) for k, v in kwargs.items()}
        node = self._tracer.graph.call_module(self._path, unwrapped_args, unwrapped_kwargs)
        return Proxy(node, self._tracer)

    def __getattr__(self, item: str):
        return getattr(self._orig_mod, item)


class Tracer:
    """Symbolic tracer that creates proxy objects for inputs and records operations executed by a Module."""

    def __init__(self):
        self.graph = Graph()
        self.root: Optional[Module] = None
        self.submodule_paths: Dict[int, str] = {}

    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        """Determines if a module should be traced through or preserved as a single atomic call_module node."""
        leaf_names = (
            "Linear", "Conv1d", "Conv2d", "Conv3d", "LayerNorm", "BatchNorm1d", "BatchNorm2d",
            "Dropout", "Embedding", "ReLU", "GELU", "SiLU", "RMSNorm", "MaxPool2d", "AvgPool2d",
        )
        return m.__class__.__name__ in leaf_names or not hasattr(m, "_modules") or len(m._modules) == 0

    def create_arg(self, a: Any) -> Any:
        return Proxy._unwrap(a)

    def trace(self, root: Module, concrete_args: Optional[Dict[str, Any]] = None) -> Graph:
        """Traces the `forward()` method of `root` and constructs a computational Graph."""
        self.root = root
        self.graph = Graph(owning_module=root)
        self.submodule_paths = {id(root): ""}

        def _index_submodules(mod: Module, path: str):
            for name, sub in getattr(mod, "_modules", {}).items():
                full_path = f"{path}.{name}" if path else name
                self.submodule_paths[id(sub)] = full_path
                _index_submodules(sub, full_path)

        _index_submodules(root, "")

        sig = inspect.signature(root.forward)
        args_to_pass = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self",):
                continue
            if concrete_args and param_name in concrete_args:
                args_to_pass.append(concrete_args[param_name])
            else:
                placeholder_node = self.graph.placeholder(param_name)
                args_to_pass.append(Proxy(placeholder_node, self))

        orig_modules = {}
        for path in list(self.submodule_paths.values()):
            if path:
                mod = self._get_submodule(path)
                if mod is not None and self.is_leaf_module(mod, path):
                    parts = path.split(".")
                    parent = self.root if len(parts) == 1 else self._get_submodule(".".join(parts[:-1]))
                    last_name = parts[-1]
                    if parent is not None:
                        orig_modules[path] = (parent, last_name, mod)
                        wrapped = _SubmoduleProxyWrapper(mod, path, self)
                        if hasattr(parent, "_modules") and last_name in parent._modules:
                            parent._modules[last_name] = wrapped
                        if hasattr(parent, "__dict__") and last_name in parent.__dict__:
                            parent.__dict__[last_name] = wrapped

        try:
            output = root.forward(*args_to_pass)
        finally:
            for path, (parent, last_name, orig_mod) in orig_modules.items():
                if hasattr(parent, "_modules") and last_name in parent._modules:
                    parent._modules[last_name] = orig_mod
                if hasattr(parent, "__dict__") and last_name in parent.__dict__:
                    parent.__dict__[last_name] = orig_mod

        unwrapped_output = self.create_arg(output)
        self.graph.output(unwrapped_output)
        return self.graph

    def _get_submodule(self, path: str) -> Optional[Module]:
        curr = self.root
        for part in path.split("."):
            if not hasattr(curr, part) and hasattr(curr, "_modules") and part in curr._modules:
                curr = curr._modules[part]
            else:
                curr = getattr(curr, part, None)
            if curr is None:
                return None
        return curr


__all__ = ["Tracer"]
