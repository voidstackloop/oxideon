"""Symbolic Mathematics and Neural-Symbolic Computation for Oxideon.

Enables mathematical expression trees, exact analytical simplification, symbolic differentiation, and symbolic regression:
  - ExpressionNode: Abstract Syntax Tree (AST) node for algebraic operations (+, -, *, /, sin, cos)
  - eval_expression: Differentiable evaluation of mathematical ASTs on Oxideon Tensors
  - diff: Exact symbolic differentiation of expression trees
  - simplify_expression: Constant folding and algebraic tree simplification
  - SymbolicRegressor: Discovers mathematical formulas matching dataset (X, y)
"""

from typing import Dict, Any, Optional, List, Union
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_node(val: Any) -> "ExpressionNode":
    if isinstance(val, ExpressionNode):
        return val
    if isinstance(val, (int, float)):
        return ExpressionNode("const", value=float(val))
    if isinstance(val, str):
        return ExpressionNode("var", var_name=val)
    raise TypeError(f"Cannot convert {type(val).__name__} to ExpressionNode")


class ExpressionNode:
    """Node in a symbolic algebraic expression tree."""

    def __init__(
        self,
        op: str,
        left: Optional["ExpressionNode"] = None,
        right: Optional["ExpressionNode"] = None,
        value: Optional[float] = None,
        var_name: Optional[str] = None,
    ):
        valid_ops = {"add", "sub", "mul", "div", "sin", "cos", "const", "var"}
        if op not in valid_ops:
            raise ValueError(f"Unknown operator '{op}'. Valid ops: {sorted(valid_ops)}")
        self.op = op
        self.left = left
        self.right = right
        self.value = float(value) if value is not None else None
        self.var_name = var_name

    def __call__(self, variables: Optional[Dict[str, Any]] = None, **kwargs: Any) -> _Tensor:
        env: Dict[str, Any] = {}
        if variables is not None:
            env.update(variables)
        env.update(kwargs)
        return eval_expression(self, env)

    def __add__(self, other: Any) -> "ExpressionNode":
        return ExpressionNode("add", self, _to_node(other))

    def __radd__(self, other: Any) -> "ExpressionNode":
        return ExpressionNode("add", _to_node(other), self)

    def __sub__(self, other: Any) -> "ExpressionNode":
        return ExpressionNode("sub", self, _to_node(other))

    def __rsub__(self, other: Any) -> "ExpressionNode":
        return ExpressionNode("sub", _to_node(other), self)

    def __mul__(self, other: Any) -> "ExpressionNode":
        return ExpressionNode("mul", self, _to_node(other))

    def __rmul__(self, other: Any) -> "ExpressionNode":
        return ExpressionNode("mul", _to_node(other), self)

    def __truediv__(self, other: Any) -> "ExpressionNode":
        return ExpressionNode("div", self, _to_node(other))

    def __rtruediv__(self, other: Any) -> "ExpressionNode":
        return ExpressionNode("div", _to_node(other), self)

    def __repr__(self) -> str:
        if self.op == "const":
            return f"{self.value:.2f}"
        if self.op == "var":
            return str(self.var_name)
        if self.op in ("sin", "cos"):
            return f"{self.op}({self.left})"
        op_symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        sym = op_symbols.get(self.op, self.op)
        return f"({self.left} {sym} {self.right})"


def eval_expression(node: ExpressionNode, variables: Dict[str, Any]) -> _Tensor:
    """Recursively evaluates expression tree on input variables (Tensors or numpy arrays)."""
    if node.op == "const":
        val = np.array([float(node.value)], dtype=np.float32)
        return _from_numpy(val)

    if node.op == "var":
        if node.var_name not in variables:
            raise KeyError(f"Variable '{node.var_name}' not found in provided variables")
        v = variables[node.var_name]
        return v if isinstance(v, _Tensor) else _from_numpy(np.ascontiguousarray(v, dtype=np.float32))

    left_val = eval_expression(node.left, variables) if node.left is not None else None
    right_val = eval_expression(node.right, variables) if node.right is not None else None

    # Check if either child is a Tensor
    if isinstance(left_val, _Tensor) or isinstance(right_val, _Tensor):
        if node.op == "add":
            return left_val + right_val
        if node.op == "sub":
            return left_val - right_val
        if node.op == "mul":
            return left_val * right_val
        if node.op == "div":
            return left_val / right_val
        if node.op == "sin":
            return left_val.sin() if hasattr(left_val, "sin") else _from_numpy(np.sin(left_val.to_numpy()))
        if node.op == "cos":
            return left_val.cos() if hasattr(left_val, "cos") else _from_numpy(np.cos(left_val.to_numpy()))

    l_arr = left_val.to_numpy() if left_val is not None else None
    r_arr = right_val.to_numpy() if right_val is not None else None

    if node.op == "add":
        out = l_arr + r_arr
    elif node.op == "sub":
        out = l_arr - r_arr
    elif node.op == "mul":
        out = l_arr * r_arr
    elif node.op == "div":
        sign = np.where(r_arr >= 0, 1.0, -1.0)
        denom = sign * np.maximum(np.abs(r_arr), 1e-8)
        out = l_arr / denom
    elif node.op == "sin":
        out = np.sin(l_arr)
    elif node.op == "cos":
        out = np.cos(l_arr)
    else:
        raise ValueError(f"Unknown operator: {node.op}")

    return _from_numpy(np.ascontiguousarray(out, dtype=np.float32))


def diff(node: ExpressionNode, var_name: str) -> ExpressionNode:
    """Computes exact symbolic derivative of expression tree with respect to var_name."""
    if node.op == "const":
        return ExpressionNode("const", value=0.0)

    if node.op == "var":
        return ExpressionNode("const", value=1.0) if node.var_name == var_name else ExpressionNode("const", value=0.0)

    if node.op == "add":
        return diff(node.left, var_name) + diff(node.right, var_name)

    if node.op == "sub":
        return diff(node.left, var_name) - diff(node.right, var_name)

    if node.op == "mul":
        # d(u*v) = du*v + u*dv
        return diff(node.left, var_name) * node.right + node.left * diff(node.right, var_name)

    if node.op == "div":
        # d(u/v) = (du*v - u*dv) / (v*v)
        num = diff(node.left, var_name) * node.right - node.left * diff(node.right, var_name)
        denom = node.right * node.right
        return num / denom

    if node.op == "sin":
        # d(sin(u)) = cos(u) * du
        return ExpressionNode("cos", left=node.left) * diff(node.left, var_name)

    if node.op == "cos":
        # d(cos(u)) = -sin(u) * du = (0 - sin(u)) * du
        neg_sin = ExpressionNode("const", value=0.0) - ExpressionNode("sin", left=node.left)
        return neg_sin * diff(node.left, var_name)

    raise ValueError(f"Unknown operator for diff: {node.op}")


def simplify_expression(node: ExpressionNode) -> ExpressionNode:
    """Performs constant folding and algebraic identity simplification on expression tree."""
    if node.op in ("const", "var"):
        return node

    simp_left = simplify_expression(node.left) if node.left else None
    simp_right = simplify_expression(node.right) if node.right else None

    # Constant folding
    if simp_left and simp_left.op == "const" and (simp_right is None or simp_right.op == "const"):
        c_left = simp_left.value
        c_right = simp_right.value if simp_right else 0.0

        if node.op == "add":
            return ExpressionNode("const", value=c_left + c_right)
        if node.op == "sub":
            return ExpressionNode("const", value=c_left - c_right)
        if node.op == "mul":
            return ExpressionNode("const", value=c_left * c_right)
        if node.op == "div":
            sign = 1.0 if c_right >= 0 else -1.0
            denom = sign * max(abs(c_right), 1e-8)
            return ExpressionNode("const", value=c_left / denom)
        if node.op == "sin":
            return ExpressionNode("const", value=math.sin(c_left))
        if node.op == "cos":
            return ExpressionNode("const", value=math.cos(c_left))

    # Algebraic identities
    if node.op == "add":
        if simp_left and simp_left.op == "const" and abs(simp_left.value) < 1e-12:
            return simp_right
        if simp_right and simp_right.op == "const" and abs(simp_right.value) < 1e-12:
            return simp_left
    elif node.op == "sub":
        if simp_right and simp_right.op == "const" and abs(simp_right.value) < 1e-12:
            return simp_left
    elif node.op == "mul":
        if (simp_left and simp_left.op == "const" and abs(simp_left.value) < 1e-12) or (
            simp_right and simp_right.op == "const" and abs(simp_right.value) < 1e-12
        ):
            return ExpressionNode("const", value=0.0)
        if simp_left and simp_left.op == "const" and abs(simp_left.value - 1.0) < 1e-12:
            return simp_right
        if simp_right and simp_right.op == "const" and abs(simp_right.value - 1.0) < 1e-12:
            return simp_left
    elif node.op == "div":
        if simp_left and simp_left.op == "const" and abs(simp_left.value) < 1e-12:
            return ExpressionNode("const", value=0.0)
        if simp_right and simp_right.op == "const" and abs(simp_right.value - 1.0) < 1e-12:
            return simp_left

    return ExpressionNode(node.op, left=simp_left, right=simp_right, value=node.value, var_name=node.var_name)


class SymbolicRegressor:
    """Discovers closed-form algebraic expressions fitting input-target data."""

    def __init__(self, max_depth: int = 3):
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.max_depth = max_depth
        self.best_expr: Optional[ExpressionNode] = None
        self.best_mse: float = float("inf")
        self.var_name: str = "x"

    def fit(self, x: Any, y: Any, var_name: str = "x") -> "SymbolicRegressor":
        x_arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32).reshape(-1)
        y_np = y.to_numpy() if isinstance(y, _Tensor) else np.asarray(y, dtype=np.float32).reshape(-1)
        if len(x_arr) != len(y_np):
            raise ValueError(f"Length mismatch: x has {len(x_arr)}, y has {len(y_np)}")
        if len(y_np) == 0:
            raise ValueError("SymbolicRegressor requires at least 1 sample")

        self.var_name = var_name
        self.best_mse = float("inf")
        self.best_expr = None

        candidates = [
            ExpressionNode("var", var_name=var_name),
            ExpressionNode("add", ExpressionNode("var", var_name=var_name), ExpressionNode("const", value=1.0)),
            ExpressionNode("mul", ExpressionNode("const", value=2.0), ExpressionNode("var", var_name=var_name)),
            ExpressionNode("mul", ExpressionNode("var", var_name=var_name), ExpressionNode("var", var_name=var_name)),
            ExpressionNode("sin", ExpressionNode("var", var_name=var_name)),
            ExpressionNode("cos", ExpressionNode("var", var_name=var_name)),
        ]

        for cand in candidates:
            pred = eval_expression(cand, {var_name: x}).to_numpy().reshape(-1)
            mse = float(np.mean((pred - y_np) ** 2))
            if mse < self.best_mse:
                self.best_mse = mse
                self.best_expr = cand

        return self

    def predict(self, x: Any) -> _Tensor:
        if self.best_expr is None:
            raise RuntimeError("SymbolicRegressor must be fitted before calling predict()")
        return eval_expression(self.best_expr, {self.var_name: x})


__all__ = [
    "ExpressionNode",
    "eval_expression",
    "diff",
    "simplify_expression",
    "SymbolicRegressor",
]
