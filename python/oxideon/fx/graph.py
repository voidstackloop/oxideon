"""Directed Acyclic Graph (DAG) representation for symbolic computation in Oxideon FX."""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from .node import Node


def _format_code_value(obj: Any) -> str:
    """Renders a node arg for generated code, recursing into containers."""
    if isinstance(obj, Node):
        return obj.name
    if isinstance(obj, tuple):
        inner = ", ".join(_format_code_value(v) for v in obj)
        return f"({inner}{',' if len(obj) == 1 else ''})"
    if isinstance(obj, list):
        return "[" + ", ".join(_format_code_value(v) for v in obj) + "]"
    if isinstance(obj, dict):
        return "{" + ", ".join(f"{k!r}: {_format_code_value(v)}" for k, v in obj.items()) + "}"
    return repr(obj)


def _format_function_target(target: Any) -> str:
    """Renders a call_function target, qualifying exported oxideon callables."""
    name = getattr(target, "__name__", None)
    module = getattr(target, "__module__", "") or ""
    if name and module.startswith("oxideon"):
        import oxideon as _ox  # deferred: graph.py must stay import-light

        if getattr(_ox, name, None) is target:
            return f"ox.{name}"
    return getattr(target, "__qualname__", getattr(target, "__name__", str(target)))


class Graph:
    """Contains an ordered execution sequence of Node objects representing a computation DAG."""

    def __init__(self, owning_module: Optional[Any] = None):
        self.owning_module = owning_module
        self._root = Node(self, "", "", None, (), {})
        self._root.prev = self._root
        self._root.next = self._root
        self._insert_point = self._root
        self._name_counter: Dict[str, int] = {}

    @property
    def nodes(self) -> Iterator[Node]:
        """Iterates over all nodes in execution order."""
        curr = self._root.next
        while curr is not self._root:
            yield curr
            curr = curr.next

    def _unique_name(self, prefix: str) -> str:
        # Sanitize prefix
        safe_prefix = "".join(c if c.isalnum() else "_" for c in prefix).strip("_") or "node"
        count = self._name_counter.get(safe_prefix, 0)
        self._name_counter[safe_prefix] = count + 1
        return safe_prefix if count == 0 else f"{safe_prefix}_{count}"

    def create_node(
        self,
        op: str,
        target: Any,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        """Appends a new node into the graph at current insertion point."""
        args = args or ()
        kwargs = kwargs or {}

        if name is None:
            if op == "call_function":
                name = getattr(target, "__name__", "call_fn")
            elif op == "call_method":
                name = str(target)
            elif op == "call_module":
                name = str(target)
            elif op == "get_attr":
                name = str(target)
            elif op == "output":
                name = "output"
            else:
                name = op
        unique_name = self._unique_name(name)

        new_node = Node(self, unique_name, op, target, args, kwargs, type=type_expr)

        # Splice node into doubly linked list after current insert point
        new_node.prev = self._insert_point
        new_node.next = self._insert_point.next
        self._insert_point.next.prev = new_node
        self._insert_point.next = new_node
        self._insert_point = new_node

        return new_node

    def placeholder(self, name: str, type_expr: Optional[Any] = None) -> Node:
        return self.create_node("placeholder", name, name=name, type_expr=type_expr)

    def get_attr(self, target: str) -> Node:
        return self.create_node("get_attr", target)

    def call_module(self, module_name: str, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None) -> Node:
        return self.create_node("call_module", module_name, args=args, kwargs=kwargs)

    def call_function(self, the_function: Callable[..., Any], args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None) -> Node:
        return self.create_node("call_function", the_function, args=args, kwargs=kwargs)

    def call_method(self, method_name: str, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None) -> Node:
        return self.create_node("call_method", method_name, args=args, kwargs=kwargs)

    def output(self, result: Any) -> Node:
        return self.create_node("output", "output", args=(result,))

    @contextmanager
    def inserting_before(self, n: Optional[Node] = None):
        """Context manager setting insertion point immediately before node n."""
        prev_point = self._insert_point
        self._insert_point = n.prev if n is not None else self._root.prev
        try:
            yield
        finally:
            self._insert_point = prev_point

    @contextmanager
    def inserting_after(self, n: Optional[Node] = None):
        """Context manager setting insertion point immediately after node n."""
        prev_point = self._insert_point
        self._insert_point = n if n is not None else self._root
        try:
            yield
        finally:
            self._insert_point = prev_point

    def erase_node(self, to_erase: Node) -> None:
        """Removes a node from the graph."""
        if to_erase.users:
            raise RuntimeError(f"Cannot erase node {to_erase.name} because it is still used by {list(to_erase.users.keys())}")
        for dep in Node._all_nodes(to_erase.args, to_erase.kwargs):
            dep.users.pop(to_erase, None)
        to_erase.prev.next = to_erase.next
        to_erase.next.prev = to_erase.prev

    def eliminate_dead_code(self) -> bool:
        """Removes nodes whose outputs are never used (excluding side effects and output node)."""
        changed = False
        nodes = list(self.nodes)
        for n in reversed(nodes):
            if n.op != "output" and len(n.users) == 0:
                self.erase_node(n)
                changed = True
        return changed

    def lint(self) -> None:
        """Validates that the graph is topologically sorted and has valid connections."""
        seen = set()
        for node in self.nodes:
            for dep in Node._all_nodes(node.args, node.kwargs):
                if dep not in seen:
                    raise RuntimeError(f"Node {node.name} uses {dep.name} before it was defined in topological order.")
            seen.add(node)

    def python_code(self, root_module_name: str = "self") -> str:
        """Generates valid, executable Python source code reproducing the graph."""
        lines = []
        placeholders = [n.name for n in self.nodes if n.op == "placeholder"]
        args_str = ", ".join(placeholders)
        lines.append(f"def forward({root_module_name}, {args_str}):")

        for n in self.nodes:
            if n.op == "placeholder":
                continue
            elif n.op == "output":
                first = n.args[0] if n.args else None
                lines.append(f"    return {_format_code_value(first)}")
            elif n.op == "get_attr":
                lines.append(f"    {n.name} = {root_module_name}.{n.target}")
            elif n.op == "call_module":
                arg_strs = [_format_code_value(a) for a in n.args]
                kwarg_strs = [f"{k}={_format_code_value(v)}" for k, v in n.kwargs.items()]
                all_args = ", ".join(arg_strs + kwarg_strs)
                lines.append(f"    {n.name} = {root_module_name}.{n.target}({all_args})")
            elif n.op == "call_function":
                target_str = _format_function_target(n.target)
                arg_strs = [_format_code_value(a) for a in n.args]
                kwarg_strs = [f"{k}={_format_code_value(v)}" for k, v in n.kwargs.items()]
                all_args = ", ".join(arg_strs + kwarg_strs)
                lines.append(f"    {n.name} = {target_str}({all_args})")
            elif n.op == "call_method":
                first = n.args[0]
                first_str = _format_code_value(first)
                arg_strs = [_format_code_value(a) for a in n.args[1:]]
                kwarg_strs = [f"{k}={_format_code_value(v)}" for k, v in n.kwargs.items()]
                all_args = ", ".join(arg_strs + kwarg_strs)
                lines.append(f"    {n.name} = {first_str}.{n.target}({all_args})")
            else:
                raise ValueError(f"unsupported op {n.op!r} in python_code (node {n.name!r})")

        return "\n".join(lines)

    def print_tabular(self) -> None:
        """Prints a human-readable tabular view of all graph nodes."""
        print(f"{'opcode':<15} {'name':<15} {'target':<25} {'args':<25} {'kwargs':<10}")
        print("-" * 90)
        for n in self.nodes:
            target_str = getattr(n.target, "__name__", str(n.target))
            print(f"{n.op:<15} {n.name:<15} {target_str:<25} {str(n.args):<25} {str(n.kwargs):<10}")

    def __str__(self) -> str:
        lines = ["graph():"]
        for n in self.nodes:
            lines.append(f"    {n.format_node()}")
        return "\n".join(lines)


__all__ = ["Graph"]
