"""Model graph tracing, Graphviz visualization, and state dictionary export for Oxideon.

Enables debugging computation graphs, visual graph exports, and state management:
  - export_state_dict: Exports model parameters into structured state dictionary
  - load_state_dict: Loads weights into model with dimension verification
  - ComputationalGraph: DAG representation of operations and autograd dependencies
  - to_dot: Generates Graphviz DOT diagram for visualization
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter


def _iter_named_params_and_buffers(model: Any) -> List[Tuple[str, _Tensor]]:
    if hasattr(model, "named_parameters") and callable(model.named_parameters):
        items = list(model.named_parameters())
        if hasattr(model, "named_buffers") and callable(model.named_buffers):
            items.extend(list(model.named_buffers()))
        return items
    out: List[Tuple[str, _Tensor]] = []
    for attr in ("weight", "bias", "gamma", "beta", "running_mean", "running_var"):
        val = getattr(model, attr, None)
        if isinstance(val, _Tensor):
            out.append((attr, val))
    if out:
        return out
    if hasattr(model, "parameters") and callable(model.parameters):
        return [(f"param_{i}", p) for i, p in enumerate(model.parameters())]
    return []


def export_state_dict(model: Any) -> Dict[str, np.ndarray]:
    """Serializes all parameters in a module hierarchy into a flattened state dictionary."""
    state = {}
    for name, param in _iter_named_params_and_buffers(model):
        state[name] = param.to_numpy().copy()
    return state


def load_state_dict(model: Any, state_dict: Dict[str, Any], strict: bool = True) -> List[str]:
    """Loads weights from a state dictionary into model parameters."""
    missing_keys = []
    unexpected_keys = list(state_dict.keys())

    for name, param in _iter_named_params_and_buffers(model):
        if name in state_dict:
            unexpected_keys.remove(name)
            val = state_dict[name]
            arr = val.to_numpy() if isinstance(val, _Tensor) else np.asarray(val)
            if list(arr.shape) != list(param.shape):
                raise ValueError(
                    f"Shape mismatch for parameter '{name}': expected {param.shape}, got {arr.shape}"
                )
            parts = name.split(".")
            curr = model
            for p in parts[:-1]:
                curr = getattr(curr, p)
            t = _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
            if isinstance(curr, Module):
                setattr(curr, parts[-1], Parameter(t))
            else:
                t.requires_grad_(True)
                setattr(curr, parts[-1], t)
        else:
            missing_keys.append(name)

    if strict and (missing_keys or unexpected_keys):
        raise KeyError(
            f"Error loading state_dict: missing={missing_keys}, unexpected={unexpected_keys}"
        )

    return missing_keys


class ComputationalGraph:
    """Directed Acyclic Graph (DAG) representation of a tensor computation."""

    def __init__(self, root: Any):
        self.nodes = []
        self.edges = []
        self._build(root)

    def _build(self, root: Any):
        visited: Set[int] = set()

        def visit(node: Any, name: str):
            node_id = id(node)
            if node_id in visited:
                return str(node_id)
            visited.add(node_id)

            shape_str = str(list(node.shape)) if hasattr(node, "shape") else "scalar"
            label = f"{name}\\n{shape_str}"
            self.nodes.append((str(node_id), label))
            return str(node_id)

        visit(root, root.__class__.__name__)

    def to_dot(self) -> str:
        """Serializes graph to Graphviz DOT language."""
        lines = ["digraph ComputationGraph {", "  rankdir=LR;", "  node [shape=box, style=rounded, fontname=Helvetica];"]
        for node_id, label in self.nodes:
            lines.append(f'  "{node_id}" [label="{label}"];')
        for src, dst in self.edges:
            lines.append(f'  "{src}" -> "{dst}";')
        lines.append("}")
        return "\n".join(lines)


def to_dot(tensor_or_model: Any) -> str:
    """Exports tensor computation or model architecture as Graphviz DOT text."""
    if isinstance(tensor_or_model, Module):
        lines = [
            f"digraph {tensor_or_model.__class__.__name__} {{",
            "  rankdir=TB;",
            "  node [shape=record, fontname=Helvetica];",
        ]
        prev_name = None
        for name, mod in getattr(tensor_or_model, "_modules", {}).items():
            lines.append(f'  "{name}" [label="{name}: {mod.__class__.__name__}"];')
            if prev_name is not None:
                lines.append(f'  "{prev_name}" -> "{name}";')
            prev_name = name
        lines.append("}")
        return "\n".join(lines)
    else:
        graph = ComputationalGraph(tensor_or_model)
        return graph.to_dot()


__all__ = [
    "export_state_dict",
    "load_state_dict",
    "ComputationalGraph",
    "to_dot",
]
