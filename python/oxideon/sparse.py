"""Sparse tensor representations and sparse-dense linear algebra for Oxideon.

Implements memory-efficient representations for high-dimensional sparse data:
  - COOTensor: Coordinate format sparse matrix (indices, values, shape)
  - CSRTensor: Compressed Sparse Row format (indptr, indices, values, shape)
  - coo_matrix_from_dense: Converts dense Tensor into COOTensor
  - to_dense: Reconstructs dense Oxideon Tensor
  - sparse_transpose: Transposes sparse coordinate representation
  - spmm: Sparse-Dense Matrix Multiplication (A_sparse @ B_dense)
  - spspmm: Sparse-Sparse Matrix Multiplication (C = A @ B)
  - sparse_add: Adds two sparse coordinate matrices
  - sparse_sub: Subtracts two sparse coordinate matrices
  - sparse_norm: Computes p-norm of sparse matrix
"""

from typing import Tuple, Union, Any, List, Optional
import numpy as np

from .oxideon import (
    Tensor as _Tensor,
    from_numpy as _from_numpy,
    SparseCoo as _SparseCoo,
    SparseCsr as _SparseCsr,
)


class COOTensor:
    """Coordinate format (COO) sparse matrix."""

    def __init__(
        self,
        indices: Any,
        values: Any,
        shape: Tuple[int, int],
    ):
        if isinstance(indices, _Tensor):
            indices = indices.to_numpy().astype(np.int64)
        else:
            indices = np.asarray(indices, dtype=np.int64)

        if isinstance(values, _Tensor):
            values = values.to_numpy().astype(np.float32)
        else:
            values = np.asarray(values, dtype=np.float32)

        if indices.ndim != 2 or indices.shape[0] != 2:
            raise ValueError(f"indices must have shape [2, nnz], got {indices.shape}")
        if values.ndim != 1 or values.shape[0] != indices.shape[1]:
            raise ValueError(f"values must have shape [{indices.shape[1]}], got {values.shape}")

        self.indices = indices
        self.values = values
        self.shape = (int(shape[0]), int(shape[1]))

    @property
    def nnz(self) -> int:
        return self.values.shape[0]

    @property
    def density(self) -> float:
        total = self.shape[0] * self.shape[1]
        return float(self.nnz / max(total, 1))

    @property
    def sparsity(self) -> float:
        return 1.0 - self.density

    def norm(self, p: float = 2) -> float:
        """Computes p-norm across non-zero values."""
        return float(np.linalg.norm(self.values, ord=p))

    def transpose(self) -> "COOTensor":
        return sparse_transpose(self)

    def to_csr(self) -> "CSRTensor":
        """Converts COO matrix into CSR representation."""
        nrows, ncols = self.shape
        row_idx = self.indices[0]
        col_idx = self.indices[1]
        vals = self.values

        order = np.lexsort((col_idx, row_idx))
        sorted_rows = row_idx[order]
        sorted_cols = col_idx[order]
        sorted_vals = vals[order]

        indptr = np.zeros(nrows + 1, dtype=np.int64)
        np.add.at(indptr[1:], sorted_rows, 1)
        np.cumsum(indptr, out=indptr)

        return CSRTensor(indptr, sorted_cols, sorted_vals, self.shape)

    def to_dense(self) -> _Tensor:
        """Converts COOTensor to a dense Oxideon Tensor."""
        dense = np.zeros(self.shape, dtype=np.float32)
        np.add.at(dense, (self.indices[0], self.indices[1]), self.values)
        return _from_numpy(np.ascontiguousarray(dense, dtype=np.float32))

    def to_oxideon_coo(self) -> _SparseCoo:
        """Converts to native Rust SparseCoo."""
        pairs = [(int(r), int(c)) for r, c in zip(self.indices[0], self.indices[1])]
        vals = [float(v) for v in self.values]
        return _SparseCoo(pairs, vals, self.shape)

    @classmethod
    def from_oxideon_coo(cls, coo: _SparseCoo) -> "COOTensor":
        """Constructs COOTensor from native Rust SparseCoo."""
        idxs = coo.indices()
        if len(idxs) == 0:
            indices = np.zeros((2, 0), dtype=np.int64)
            values = np.zeros(0, dtype=np.float32)
        else:
            rows = [rc[0] for rc in idxs]
            cols = [rc[1] for rc in idxs]
            indices = np.array([rows, cols], dtype=np.int64)
            values = np.array(coo.values(), dtype=np.float32)
        return cls(indices, values, coo.shape)

    def __add__(self, other: Any) -> "COOTensor":
        if isinstance(other, COOTensor):
            return sparse_add(self, other)
        raise TypeError(f"Unsupported operand type for +: COOTensor and {type(other)}")

    def __sub__(self, other: Any) -> "COOTensor":
        if isinstance(other, COOTensor):
            return sparse_sub(self, other)
        raise TypeError(f"Unsupported operand type for -: COOTensor and {type(other)}")

    def __mul__(self, scalar: float) -> "COOTensor":
        scalar = float(scalar)
        return COOTensor(self.indices.copy(), self.values * scalar, self.shape)

    def __rmul__(self, scalar: float) -> "COOTensor":
        return self.__mul__(scalar)

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, (COOTensor, CSRTensor)):
            return spspmm(self, other)
        return spmm(self, other)

    def __repr__(self) -> str:
        return (
            f"COOTensor(shape={self.shape}, nnz={self.nnz}, "
            f"density={self.density * 100:.2f}%)"
        )


class CSRTensor:
    """Compressed Sparse Row (CSR) format sparse matrix."""

    def __init__(
        self,
        indptr: Any,
        indices: Any,
        values: Any,
        shape: Tuple[int, int],
    ):
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = (int(shape[0]), int(shape[1]))

    @property
    def nnz(self) -> int:
        return self.values.shape[0]

    @property
    def density(self) -> float:
        total = self.shape[0] * self.shape[1]
        return float(self.nnz / max(total, 1))

    @property
    def sparsity(self) -> float:
        return 1.0 - self.density

    def norm(self, p: float = 2) -> float:
        """Computes p-norm across non-zero values."""
        return float(np.linalg.norm(self.values, ord=p))

    def to_coo(self) -> COOTensor:
        """Converts CSR matrix into COO representation."""
        nrows, ncols = self.shape
        row_indices = []
        col_indices = []
        vals = []
        for i in range(nrows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            if start < end:
                for idx in range(start, end):
                    row_indices.append(i)
                    col_indices.append(self.indices[idx])
                    vals.append(self.values[idx])
        if len(row_indices) == 0:
            idx = np.zeros((2, 0), dtype=np.int64)
            v = np.zeros(0, dtype=np.float32)
        else:
            idx = np.array([row_indices, col_indices], dtype=np.int64)
            v = np.array(vals, dtype=np.float32)
        return COOTensor(idx, v, self.shape)

    def transpose(self) -> "CSRTensor":
        """Transposes CSRTensor."""
        return self.to_coo().transpose().to_csr()

    def to_dense(self) -> _Tensor:
        nrows, ncols = self.shape
        dense = np.zeros((nrows, ncols), dtype=np.float32)
        for i in range(nrows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            cols = self.indices[start:end]
            vals = self.values[start:end]
            dense[i, cols] = vals
        return _from_numpy(np.ascontiguousarray(dense, dtype=np.float32))

    def to_oxideon_csr(self) -> _SparseCsr:
        """Converts to native Rust SparseCsr."""
        row_ptr = [int(p) for p in self.indptr]
        col_idx = [int(c) for c in self.indices]
        vals = [float(v) for v in self.values]
        return _SparseCsr(row_ptr, col_idx, vals, self.shape)

    @classmethod
    def from_oxideon_csr(cls, csr: _SparseCsr) -> "CSRTensor":
        """Builds CSRTensor from native Rust SparseCsr."""
        return cls(
            indptr=np.array(csr.row_ptr(), dtype=np.int64),
            indices=np.array(csr.col_idx(), dtype=np.int64),
            values=np.array(csr.values(), dtype=np.float32),
            shape=csr.shape,
        )

    def __add__(self, other: Any) -> "CSRTensor":
        if isinstance(other, CSRTensor):
            return sparse_add(self.to_coo(), other.to_coo()).to_csr()
        if isinstance(other, COOTensor):
            return sparse_add(self.to_coo(), other).to_csr()
        raise TypeError(f"Unsupported operand type for +: CSRTensor and {type(other)}")

    def __sub__(self, other: Any) -> "CSRTensor":
        if isinstance(other, CSRTensor):
            return sparse_sub(self.to_coo(), other.to_coo()).to_csr()
        if isinstance(other, COOTensor):
            return sparse_sub(self.to_coo(), other).to_csr()
        raise TypeError(f"Unsupported operand type for -: CSRTensor and {type(other)}")

    def __mul__(self, scalar: float) -> "CSRTensor":
        scalar = float(scalar)
        return CSRTensor(self.indptr.copy(), self.indices.copy(), self.values * scalar, self.shape)

    def __rmul__(self, scalar: float) -> "CSRTensor":
        return self.__mul__(scalar)

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, (COOTensor, CSRTensor)):
            return spspmm(self, other)
        return spmm(self, other)

    def __repr__(self) -> str:
        return f"CSRTensor(shape={self.shape}, nnz={self.nnz}, sparsity={self.sparsity * 100:.2f}%)"


def coo_matrix_from_dense(dense_tensor: Any, tol: float = 1e-8) -> COOTensor:
    """Constructs a COOTensor from a dense 2D Tensor."""
    arr = dense_tensor.to_numpy() if isinstance(dense_tensor, _Tensor) else np.asarray(dense_tensor)
    if arr.ndim != 2:
        raise ValueError(f"Dense tensor must be 2D, got shape {arr.shape}")

    rows, cols = np.nonzero(np.abs(arr) > tol)
    vals = arr[rows, cols]
    indices = np.stack([rows, cols], axis=0)
    return COOTensor(indices, vals, arr.shape)


def to_dense(sparse_tensor: Union[COOTensor, CSRTensor]) -> _Tensor:
    """Converts a sparse tensor (COO or CSR) into a dense Oxideon Tensor."""
    return sparse_tensor.to_dense()


def sparse_transpose(coo_mat: COOTensor) -> COOTensor:
    """Transposes a COOTensor."""
    new_indices = np.stack([coo_mat.indices[1], coo_mat.indices[0]], axis=0)
    new_shape = (coo_mat.shape[1], coo_mat.shape[0])
    return COOTensor(new_indices, coo_mat.values.copy(), new_shape)


def sparse_add(a: COOTensor, b: COOTensor) -> COOTensor:
    """Adds two coordinate format sparse matrices."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for sparse_add: {a.shape} vs {b.shape}")

    new_indices = np.concatenate([a.indices, b.indices], axis=1)
    new_values = np.concatenate([a.values, b.values], axis=0)
    return COOTensor(new_indices, new_values, a.shape)


def sparse_sub(a: COOTensor, b: COOTensor) -> COOTensor:
    """Subtracts two coordinate format sparse matrices (a - b)."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for sparse_sub: {a.shape} vs {b.shape}")

    neg_b_values = -b.values
    new_indices = np.concatenate([a.indices, b.indices], axis=1)
    new_values = np.concatenate([a.values, neg_b_values], axis=0)
    return COOTensor(new_indices, new_values, a.shape)


def sparse_norm(sparse_mat: Union[COOTensor, CSRTensor], p: float = 2) -> float:
    """Calculates matrix norm over non-zero values of sparse matrix."""
    return float(np.linalg.norm(sparse_mat.values, ord=p))


def spmm(sparse_mat: Union[COOTensor, CSRTensor], dense_mat: Any) -> _Tensor:
    """Sparse-dense matrix multiplication: out = sparse_mat @ dense_mat.

    Leverages high-performance native O(E) sparse-dense linear algebra kernels
    with full autograd gradient propagation.
    """
    if isinstance(dense_mat, _Tensor):
        if len(dense_mat.shape) != 2:
            raise ValueError(f"Dense matrix must be 2D, got shape {dense_mat.shape}")
        if sparse_mat.shape[1] != dense_mat.shape[0]:
            raise ValueError(
                f"Shape mismatch for spmm: sparse {sparse_mat.shape} vs dense {dense_mat.shape}"
            )
        if isinstance(sparse_mat, COOTensor):
            return sparse_mat.to_oxideon_coo() @ dense_mat
        elif isinstance(sparse_mat, CSRTensor):
            return sparse_mat.to_oxideon_csr() @ dense_mat

    b = dense_mat.to_numpy() if isinstance(dense_mat, _Tensor) else np.asarray(dense_mat, dtype=np.float32)
    if b.ndim != 2:
        raise ValueError(f"Dense matrix must be 2D, got shape {b.shape}")

    if sparse_mat.shape[1] != b.shape[0]:
        raise ValueError(
            f"Shape mismatch for spmm: sparse {sparse_mat.shape} vs dense {b.shape}"
        )

    out_rows = sparse_mat.shape[0]
    out_cols = b.shape[1]
    out = np.zeros((out_rows, out_cols), dtype=np.float32)

    if isinstance(sparse_mat, COOTensor):
        row_idx = sparse_mat.indices[0]
        col_idx = sparse_mat.indices[1]
        vals = sparse_mat.values[:, None]  # [nnz, 1]
        # Multiply and accumulate
        weighted_b = vals * b[col_idx]  # [nnz, out_cols]
        np.add.at(out, row_idx, weighted_b)
    elif isinstance(sparse_mat, CSRTensor):
        for i in range(out_rows):
            start = sparse_mat.indptr[i]
            end = sparse_mat.indptr[i + 1]
            if start < end:
                cols = sparse_mat.indices[start:end]
                vals = sparse_mat.values[start:end, None]
                out[i] = np.sum(vals * b[cols], axis=0)

    return _from_numpy(np.ascontiguousarray(out, dtype=np.float32))


def spspmm(
    a: Union[COOTensor, CSRTensor],
    b: Union[COOTensor, CSRTensor],
) -> Union[COOTensor, CSRTensor]:
    """Sparse-Sparse matrix multiplication: out = a @ b using native Gustavson engine."""
    csr_a = a.to_oxideon_csr() if isinstance(a, CSRTensor) else a.to_csr().to_oxideon_csr()
    csr_b = b.to_oxideon_csr() if isinstance(b, CSRTensor) else b.to_csr().to_oxideon_csr()
    csr_c = csr_a.spspmm(csr_b)
    out_csr = CSRTensor.from_oxideon_csr(csr_c)
    if isinstance(a, COOTensor) and isinstance(b, COOTensor):
        return out_csr.to_coo()
    return out_csr


def sparse_coo_tensor(
    indices: Any,
    values: Any,
    size: Optional[Tuple[int, ...]] = None,
    dtype: Optional[Any] = None,
    device: Optional[Any] = None,
    requires_grad: bool = False,
) -> COOTensor:
    """Constructs a sparse COO matrix from coordinate indices and non-zero values."""
    if size is None:
        idx_np = indices.to_numpy() if isinstance(indices, _Tensor) else np.asarray(indices)
        size = (int(idx_np[0].max() + 1), int(idx_np[1].max() + 1))
    return COOTensor(indices=indices, values=values, shape=size)


__all__ = [
    "COOTensor",
    "CSRTensor",
    "coo_matrix_from_dense",
    "to_dense",
    "sparse_transpose",
    "sparse_add",
    "sparse_sub",
    "sparse_norm",
    "spmm",
    "spspmm",
    "sparse_coo_tensor",
]
