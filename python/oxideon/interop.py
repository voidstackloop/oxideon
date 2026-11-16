"""Multi-Framework Interoperability Bridge for Oxideon.

Enables seamless bidirectional zero-copy and zero-overhead conversion between
Oxideon Tensors/Modules and:
- **NetworkX**: Graphs, DiGraphs, adjacency matrices, and node features
- **SymPy**: Analytical mathematical expressions, grids, PDEs, and matrices
- **SciPy**: Sparse matrices (CSR, CSC, COO) and special functions
- **PyTorch**: Model weights, state_dict bidirectional loading/exporting, and torch.Tensors
- **JAX**: Array interchange via DLPack / NumPy
- **DLPack**: Zero-copy tensor exchange via `to_dlpack` / `from_dlpack`
- **Apache Arrow / Hugging Face**: Table columns and record batches to/from Tensor mappings
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter
from .sparse_matrix import CSRMatrix


class _DLPackBuffer:
    """DLPack-compatible buffer wrapper supporting both `from_dlpack` and `np.from_dlpack`."""

    def __init__(self, arr: np.ndarray, device: str = "cpu"):
        self._arr = np.ascontiguousarray(arr)
        self.device = device

    def __dlpack__(self, stream: Any = None) -> Any:
        return self._arr.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> Tuple[int, int]:
        return self._arr.__dlpack_device__()

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return self._arr if dtype is None else self._arr.astype(dtype)


class _ArrowTableCompat:
    """Lightweight Arrow-compatible table representation when pyarrow is not installed."""

    def __init__(self, data: Dict[str, np.ndarray]):
        self._data = {k: np.ascontiguousarray(v) for k, v in data.items()}
        self.column_names = list(self._data.keys())
        self.columns = list(self._data.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def to_pydict(self) -> Dict[str, List[Any]]:
        return {k: v.tolist() for k, v in self._data.items()}


def to_dlpack(tensor: Union[_Tensor, np.ndarray]) -> Any:
    """Exports an Oxideon Tensor as a zero-copy DLPack PyCapsule (or wrapper for NumPy arrays)."""
    if isinstance(tensor, _Tensor):
        return tensor.__dlpack__()
    arr = np.asarray(tensor)
    return _DLPackBuffer(arr, device="cpu")


def from_dlpack(ext_tensor: Any, device: str = "cpu") -> _Tensor:
    """Imports an external tensor or DLPack capsule into an Oxideon Tensor."""
    if isinstance(ext_tensor, _DLPackBuffer):
        return _from_numpy(ext_tensor._arr, device=device)
    if isinstance(ext_tensor, _Tensor):
        return ext_tensor.to(device)
    if type(ext_tensor).__name__ == "PyCapsule" and hasattr(_Tensor, "from_dlpack_capsule"):
        t = _Tensor.from_dlpack_capsule(ext_tensor)
        return t if str(device) == "cpu" else t.to(device)
    if hasattr(ext_tensor, "__dlpack__"):
        try:
            cap = ext_tensor.__dlpack__()
            if type(cap).__name__ == "PyCapsule" and hasattr(_Tensor, "from_dlpack_capsule"):
                t = _Tensor.from_dlpack_capsule(cap)
                return t if str(device) == "cpu" else t.to(device)
        except Exception:
            pass
        arr = np.from_dlpack(ext_tensor)
        return _from_numpy(np.ascontiguousarray(arr), device=device)
    arr = np.asarray(ext_tensor)
    return _from_numpy(np.ascontiguousarray(arr), device=device)


def from_networkx(
    graph: Any,
    weight: Optional[str] = "weight",
    dtype: str = "f32",
    device: str = "cpu",
) -> _Tensor:
    """Converts a NetworkX Graph or DiGraph into a dense adjacency Oxideon Tensor [N, N]."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("from_networkx requires NetworkX: pip install networkx")

    np_dtypes = {
        "f32": np.float32, "f64": np.float64,
        "i32": np.int32, "i64": np.int64,
        "u8": np.uint8, "bool": np.bool_,
    }
    if dtype not in np_dtypes:
        raise ValueError(f"from_networkx: unsupported dtype {dtype!r}, expected one of {sorted(np_dtypes)}")
    adj_arr = nx.to_numpy_array(graph, weight=weight, dtype=np_dtypes[dtype])
    return _from_numpy(np.ascontiguousarray(adj_arr), device=device)


def to_networkx(
    tensor: Union[_Tensor, np.ndarray, CSRMatrix],
    create_using: Any = None,
    threshold: float = 0.0,
) -> Any:
    """Converts a 2D adjacency Tensor or CSRMatrix into a NetworkX Graph or DiGraph."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("to_networkx requires NetworkX: pip install networkx")

    if isinstance(tensor, CSRMatrix):
        arr = tensor.to_dense()
    elif isinstance(tensor, _Tensor):
        arr = tensor.to_numpy()
    else:
        arr = np.asarray(tensor)

    if arr.ndim != 2:
        raise ValueError(f"Adjacency tensor must be 2D, got shape {arr.shape}")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {arr.shape}")

    if create_using is None:
        is_symmetric = np.allclose(arr, arr.T, atol=1e-5)
        create_using = nx.Graph if is_symmetric else nx.DiGraph

    if threshold > 0.0:
        arr_filtered = np.where(np.abs(arr) >= threshold, arr, 0.0)
        return nx.from_numpy_array(arr_filtered, create_using=create_using)
    return nx.from_numpy_array(arr, create_using=create_using)


def from_sympy(
    expr: Any,
    symbols: Sequence[Any],
    ranges: Sequence[Tuple[float, float]],
    steps: Union[int, Sequence[int]] = 50,
    dtype: str = "f32",
    device: str = "cpu",
) -> _Tensor:
    """Evaluates a symbolic SymPy mathematical expression across an N-dimensional coordinate grid."""
    try:
        import sympy as sp
    except ImportError:
        raise ImportError("from_sympy requires SymPy: pip install sympy")

    if len(symbols) != len(ranges):
        raise ValueError(
            f"Number of symbols ({len(symbols)}) must match ranges ({len(ranges)})"
        )

    if dtype not in ("f32", "f64"):
        raise ValueError(f"from_sympy dtype must be 'f32' or 'f64', got {dtype!r}")
    step_list = [steps] * len(symbols) if isinstance(steps, int) else list(steps)
    grid_dtype = np.float32 if dtype == "f32" else np.float64
    grids = [
        np.linspace(r[0], r[1], s, dtype=grid_dtype)
        for r, s in zip(ranges, step_list)
    ]
    mesh = np.meshgrid(*grids, indexing="ij")
    f_lambdified = sp.lambdify(symbols, expr, modules=["numpy"])
    vals = f_lambdified(*mesh)
    arr = np.asarray(vals, dtype=np.float32 if dtype == "f32" else np.float64)
    return _from_numpy(np.ascontiguousarray(arr), device=device)


def to_sympy(
    tensor: Union[_Tensor, np.ndarray],
) -> Any:
    """Converts a 1D vector or 2D matrix Oxideon Tensor into a SymPy Matrix."""
    try:
        import sympy as sp
    except ImportError:
        raise ImportError("to_sympy requires SymPy: pip install sympy")

    arr = tensor.to_numpy() if isinstance(tensor, _Tensor) else np.asarray(tensor)
    if arr.ndim == 1 or arr.ndim == 2:
        return sp.Matrix(arr.tolist())
    else:
        raise ValueError(f"to_sympy requires 1D or 2D tensor, got shape {arr.shape}")


def from_scipy_sparse(
    sp_mat: Any,
    as_csr_matrix: bool = True,
    device: str = "cpu",
) -> Union[CSRMatrix, _Tensor]:
    """Converts a SciPy sparse matrix (csr_matrix, csc_matrix, coo_matrix) into an Oxideon CSRMatrix or Tensor."""
    try:
        import scipy.sparse as sp  # noqa: F401
    except ImportError:
        raise ImportError("from_scipy_sparse requires SciPy: pip install scipy")

    csr = sp_mat.tocsr()
    if as_csr_matrix:
        return CSRMatrix(
            values=np.ascontiguousarray(csr.data, dtype=np.float32),
            col_indices=np.ascontiguousarray(csr.indices, dtype=np.int64),
            row_ptrs=np.ascontiguousarray(csr.indptr, dtype=np.int64),
            shape=csr.shape,
        )
    else:
        dense_arr = np.ascontiguousarray(csr.toarray())
        return _from_numpy(dense_arr, device=device)


def to_scipy_sparse(
    matrix_or_tensor: Union[CSRMatrix, _Tensor, np.ndarray],
    format: str = "csr",
) -> Any:
    """Converts an Oxideon CSRMatrix or 2D Tensor into a SciPy sparse matrix."""
    try:
        import scipy.sparse as sp
    except ImportError:
        raise ImportError("to_scipy_sparse requires SciPy: pip install scipy")

    if isinstance(matrix_or_tensor, CSRMatrix):
        csr = sp.csr_matrix(
            (matrix_or_tensor.values, matrix_or_tensor.col_indices, matrix_or_tensor.row_ptrs),
            shape=matrix_or_tensor.shape,
        )
    elif isinstance(matrix_or_tensor, _Tensor):
        arr = matrix_or_tensor.to_numpy()
        csr = sp.csr_matrix(arr)
    else:
        csr = sp.csr_matrix(np.asarray(matrix_or_tensor))

    fmt = format.lower()
    if fmt == "csr":
        return csr
    elif fmt == "csc":
        return csr.tocsc()
    elif fmt == "coo":
        return csr.tocoo()
    else:
        return csr.asformat(fmt)


def load_torch_state_dict(
    model: Any,
    torch_state: Union[Dict[str, Any], str],
    strict: bool = True,
) -> List[str]:
    """Loads a PyTorch state_dict (or .pt/.pth filepath) into an Oxideon Module hierarchy."""
    try:
        import torch
    except ImportError:
        raise ImportError("load_torch_state_dict requires PyTorch: pip install torch")

    if isinstance(torch_state, str):
        state = torch.load(torch_state, map_location="cpu", weights_only=True)
    else:
        state = torch_state

    np_state = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            np_state[k] = v.detach().cpu().numpy()
        else:
            np_state[k] = np.asarray(v)

    from .export import load_state_dict
    return load_state_dict(model, np_state, strict=strict)


def export_torch_state_dict(model: Any) -> Dict[str, Any]:
    """Exports all parameters in an Oxideon Module as a dictionary of PyTorch Tensors."""
    try:
        import torch
    except ImportError:
        raise ImportError("export_torch_state_dict requires PyTorch: pip install torch")

    from .export import export_state_dict
    raw_sd = export_state_dict(model)
    return {name: torch.from_numpy(arr.copy()) for name, arr in raw_sd.items()}


def from_jax(jax_array: Any, device: str = "cpu") -> _Tensor:
    """Converts a JAX Array into an Oxideon Tensor via NumPy array sharing."""
    try:
        import jax.numpy as jnp  # noqa: F401
    except ImportError:
        raise ImportError("from_jax requires JAX: pip install jax")

    arr = np.asarray(jax_array)
    return _from_numpy(np.ascontiguousarray(arr), device=device)


def to_jax(tensor: _Tensor) -> Any:
    """Converts an Oxideon Tensor into a JAX Array."""
    try:
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("to_jax requires JAX: pip install jax")

    arr = tensor.to_numpy()
    return jnp.array(arr)


def from_arrow(
    table_or_dataset: Any,
    columns: Optional[Sequence[str]] = None,
    device: str = "cpu",
) -> Dict[str, _Tensor]:
    """Converts an Apache Arrow Table or Hugging Face Dataset into a dictionary of Oxideon Tensors."""
    cols_to_extract = (
        columns
        if columns is not None
        else getattr(table_or_dataset, "column_names", [])
    )
    if not cols_to_extract and hasattr(table_or_dataset, "columns"):
        cols_to_extract = list(table_or_dataset.columns)
    if not cols_to_extract and isinstance(table_or_dataset, dict):
        cols_to_extract = list(table_or_dataset.keys())

    tensors = {}
    for col_name in cols_to_extract:
        col = table_or_dataset[col_name]
        if hasattr(col, "to_numpy"):
            try:
                arr = col.to_numpy()
            except Exception:
                arr = np.asarray(col.to_pylist())
        else:
            arr = np.asarray(col)
        if arr.dtype == object:
            arr = np.stack([np.asarray(x) for x in arr], axis=0)
        if np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
            tensors[col_name] = _from_numpy(np.ascontiguousarray(arr), device=device)
    return tensors


def to_arrow(tensors: Dict[str, Union[_Tensor, np.ndarray]]) -> Any:
    """Converts a dictionary of Oxideon Tensors or NumPy arrays into an Apache Arrow Table (or compatible table)."""
    np_dict = {
        k: (v.to_numpy() if isinstance(v, _Tensor) else np.asarray(v))
        for k, v in tensors.items()
    }
    try:
        import pyarrow as pa
        py_dict = {k: v.tolist() for k, v in np_dict.items()}
        return pa.Table.from_pydict(py_dict)
    except ImportError:
        return _ArrowTableCompat(np_dict)


__all__ = [
    "from_networkx",
    "to_networkx",
    "from_sympy",
    "to_sympy",
    "from_scipy_sparse",
    "to_scipy_sparse",
    "load_torch_state_dict",
    "export_torch_state_dict",
    "from_jax",
    "to_jax",
    "to_dlpack",
    "from_dlpack",
    "from_arrow",
    "to_arrow",
]
