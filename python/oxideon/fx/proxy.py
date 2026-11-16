"""Symbolic proxy tracing wrapper for intercepting operations in Oxideon FX."""

from typing import Any, Tuple, Union

from .node import Node


class Proxy:
    """Wraps a Node and intercepts Python expressions during module tracing."""

    def __init__(self, node: Node, tracer: "Tracer"):
        self.node = node
        self.tracer = tracer

    @property
    def name(self) -> str:
        return self.node.name

    def __repr__(self) -> str:
        return f"Proxy({self.node.name})"

    @classmethod
    def _unwrap(cls, arg: Any) -> Any:
        if isinstance(arg, Proxy):
            return arg.node
        elif isinstance(arg, tuple):
            return tuple(cls._unwrap(x) for x in arg)
        elif isinstance(arg, list):
            return [cls._unwrap(x) for x in arg]
        elif isinstance(arg, dict):
            return {k: cls._unwrap(v) for k, v in arg.items()}
        return arg

    def _create_bin_op(self, method_name: str, other: Any) -> "Proxy":
        unwrapped_args = (self.node, self._unwrap(other))
        n = self.node.graph.create_node("call_method", method_name, args=unwrapped_args)
        return Proxy(n, self.tracer)

    def _create_rbin_op(self, method_name: str, other: Any) -> "Proxy":
        # Reflected op: the dunder runs on self with the other operand as its
        # argument (e.g. `2 - p` records `p.__rsub__(2)`), preserving runtime
        # reflection semantics for non-tensor `other`.
        unwrapped_args = (self.node, self._unwrap(other))
        n = self.node.graph.create_node("call_method", method_name, args=unwrapped_args)
        return Proxy(n, self.tracer)

    def _create_unary_op(self, method_name: str) -> "Proxy":
        n = self.node.graph.create_node("call_method", method_name, args=(self.node,))
        return Proxy(n, self.tracer)

    def __add__(self, other): return self._create_bin_op("__add__", other)
    def __radd__(self, other): return self._create_rbin_op("__radd__", other)
    def __sub__(self, other): return self._create_bin_op("__sub__", other)
    def __rsub__(self, other): return self._create_rbin_op("__rsub__", other)
    def __mul__(self, other): return self._create_bin_op("__mul__", other)
    def __rmul__(self, other): return self._create_rbin_op("__rmul__", other)
    def __matmul__(self, other): return self._create_bin_op("__matmul__", other)
    def __rmatmul__(self, other): return self._create_rbin_op("__rmatmul__", other)
    def __truediv__(self, other): return self._create_bin_op("__truediv__", other)
    def __rtruediv__(self, other): return self._create_rbin_op("__rtruediv__", other)
    def __floordiv__(self, other): return self._create_bin_op("__floordiv__", other)
    def __rfloordiv__(self, other): return self._create_rbin_op("__rfloordiv__", other)
    def __pow__(self, other): return self._create_bin_op("__pow__", other)
    def __rpow__(self, other): return self._create_rbin_op("__rpow__", other)
    def __neg__(self): return self._create_unary_op("__neg__")
    def __abs__(self): return self._create_unary_op("__abs__")
    def __invert__(self): return self._create_unary_op("__invert__")

    def __getitem__(self, idx):
        n = self.node.graph.create_node("call_method", "__getitem__", args=(self.node, self._unwrap(idx)))
        return Proxy(n, self.tracer)

    def __getattr__(self, name: str) -> "Proxy":
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        # Record method call when invoked
        def _method_call(*args, **kwargs):
            unwrapped_args = (self.node,) + tuple(self._unwrap(a) for a in args)
            unwrapped_kwargs = {k: self._unwrap(v) for k, v in kwargs.items()}
            n = self.node.graph.create_node("call_method", name, args=unwrapped_args, kwargs=unwrapped_kwargs)
            return Proxy(n, self.tracer)
        return _method_call

    def __call__(self, *args, **kwargs):
        unwrapped_args = (self.node,) + tuple(self._unwrap(a) for a in args)
        unwrapped_kwargs = {k: self._unwrap(v) for k, v in kwargs.items()}
        n = self.node.graph.create_node("call_method", "__call__", args=unwrapped_args, kwargs=unwrapped_kwargs)
        return Proxy(n, self.tracer)

    def __bool__(self) -> bool:
        raise RuntimeError(
            "bool() is not supported on Proxy objects; "
            "rewrite the traced code to avoid branching on traced values"
        )


__all__ = ["Proxy"]
