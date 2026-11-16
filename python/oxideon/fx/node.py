"""Symbolic computational graph node representation for Oxideon FX."""

from typing import Any, Dict, List, Optional, Set, Tuple, Union


class Node:
    """Represents an atomic operation in an Oxideon FX computational graph."""

    def __init__(
        self,
        graph: "Graph",
        name: str,
        op: str,
        target: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        type: Optional[Any] = None,
    ):
        self.graph = graph
        self.name = name
        self.op = op  # 'placeholder', 'call_function', 'call_method', 'call_module', 'get_attr', 'output'
        self.target = target
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        self.type = type
        self.users: Dict["Node", None] = {}
        self.prev: "Node" = self
        self.next: "Node" = self

        # Register users for dependency tracking
        for a in self._all_nodes(self.args, self.kwargs):
            a.users[self] = None

    @staticmethod
    def _all_nodes(args: Any, kwargs: Any) -> List["Node"]:
        nodes = []
        def _extract(obj):
            if isinstance(obj, Node):
                nodes.append(obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _extract(item)
            elif isinstance(obj, dict):
                for val in obj.values():
                    _extract(val)
        _extract(args)
        _extract(kwargs)
        return nodes

    def replace_all_uses_with(self, new_node: "Node") -> None:
        """Replaces all uses of this node in other nodes with `new_node`."""
        if new_node is self:
            return
        for user in list(self.users.keys()):
            def _replace(obj):
                if obj is self:
                    return new_node
                elif isinstance(obj, tuple):
                    return tuple(_replace(x) for x in obj)
                elif isinstance(obj, list):
                    return [_replace(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: _replace(v) for k, v in obj.items()}
                return obj

            user.args = _replace(user.args)
            user.kwargs = _replace(user.kwargs)
            new_node.users[user] = None
        self.users.clear()

    def format_node(self) -> str:
        """Returns a string representation of this node in the graph."""
        if self.op == "placeholder":
            return f"%{self.name} : placeholder"
        elif self.op == "get_attr":
            return f"%{self.name} = self.{self.target}"
        elif self.op == "call_module":
            arg_str = ", ".join(f"%{a.name}" if isinstance(a, Node) else repr(a) for a in self.args)
            return f"%{self.name} = self.{self.target}({arg_str})"
        elif self.op == "call_function":
            target_name = getattr(self.target, "__name__", str(self.target))
            arg_str = ", ".join(f"%{a.name}" if isinstance(a, Node) else repr(a) for a in self.args)
            return f"%{self.name} = {target_name}({arg_str})"
        elif self.op == "call_method":
            first = self.args[0]
            first_name = f"%{first.name}" if isinstance(first, Node) else repr(first)
            rest_args = ", ".join(f"%{a.name}" if isinstance(a, Node) else repr(a) for a in self.args[1:])
            return f"%{self.name} = {first_name}.{self.target}({rest_args})"
        elif self.op == "output":
            first = self.args[0] if self.args else "None"
            out_str = f"%{first.name}" if isinstance(first, Node) else repr(first)
            return f"return {out_str}"
        return f"%{self.name} : {self.op}"

    def __repr__(self) -> str:
        return f"%{self.name}"


__all__ = ["Node"]
