"""Executable compiled Module backed by an Oxideon FX Graph."""

from typing import Any, Dict, Optional

from .graph import Graph
from ..nn import Module


class GraphModule(Module):
    """An `ox.nn.Module` generated from a symbolic `Graph`.

    Executes operations according to the graph structure and supports code generation,
    graph transformations, and runtime debugging.
    """

    def __init__(self, root: Module, graph: Graph):
        super().__init__()
        self.graph = graph
        self._code = ""

        # Copy over submodules, parameters, and buffers from root module
        for name, sub in getattr(root, "_modules", {}).items():
            setattr(self, name, sub)
        for name, param in getattr(root, "_parameters", {}).items():
            setattr(self, name, param)
        for name, buf in getattr(root, "_buffers", {}).items():
            setattr(self, name, buf)

        # Recompile graph into forward() method
        self.recompile()

    @property
    def code(self) -> str:
        """Returns the Python source code generated from the internal computational graph."""
        return self._code

    def recompile(self) -> None:
        """Recompiles the Python forward implementation from the current graph."""
        source_code = self.graph.python_code(root_module_name="self")
        self._code = source_code

        # Compile and bind forward function
        glob_env: Dict[str, Any] = {"self": self}
        # Add oxideon functions into global namespace for call_function targets
        import oxideon as ox
        glob_env.update({
            "ox": ox,
            "relu": ox.relu,
            "gelu": getattr(ox, "gelu", None),
            "silu": getattr(ox, "silu", None),
        })

        loc_env: Dict[str, Any] = {}
        exec(source_code, glob_env, loc_env)
        compiled_forward = loc_env["forward"]
        setattr(self, "forward", compiled_forward.__get__(self, GraphModule))

    def print_readable(self) -> None:
        """Prints the graph and generated python code in a readable format."""
        print("=== Oxideon FX Graph ===")
        print(self.graph)
        print("\n=== Compiled Python Code ===")
        print(self._code)

    def __repr__(self) -> str:
        return f"GraphModule(\n  {self.graph}\n)"


__all__ = ["GraphModule"]
