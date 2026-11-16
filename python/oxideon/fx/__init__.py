"""Oxideon FX: Symbolic Computation Graph Tracing and Program Transformation Toolkit.

Enables full programmatic introspection, pattern matching, graph transformation,
and Python-to-Python code generation for neural network models.
"""

from typing import Any, Callable, Dict, Optional

from .node import Node
from .graph import Graph
from .proxy import Proxy
from .tracer import Tracer
from .graph_module import GraphModule
from ..nn import Module


def symbolic_trace(root: Module, concrete_args: Optional[Dict[str, Any]] = None) -> GraphModule:
    """Symbolically traces an `ox.nn.Module` and returns an executable, serializable `GraphModule`.

    Args:
        root: The module instance to trace.
        concrete_args: Optional inputs that should not be treated as symbolic proxies.

    Example:
    ```python
    import oxideon as ox
    import oxideon.fx as fx

    class MyNet(ox.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = ox.nn.Linear(4, 4)

        def forward(self, x):
            return ox.relu(self.linear(x))

    model = MyNet()
    traced = fx.symbolic_trace(model)
    print(traced.code)
    y = traced(ox.randn([2, 4]))
    ```
    """
    tracer = Tracer()
    graph = tracer.trace(root, concrete_args=concrete_args)
    return GraphModule(root, graph)


__all__ = [
    "symbolic_trace",
    "Graph",
    "Node",
    "Proxy",
    "Tracer",
    "GraphModule",
]
