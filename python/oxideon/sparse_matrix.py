"""High-performance Compressed Sparse Row (CSR) matrix representation and sparse linear algebra.

Implements:
  - CSRMatrix: Standard compressed sparse row matrix format with transpose() and autograd-aware spmm().
  - spmm: Sparse-Dense Matrix Multiplication (Y = A @ X) with optimized BLAS kernel paths.
  - spspmm: Sparse-Sparse Matrix Multiplication (C = A @ B).
"""

from typing import Tuple, Optional, Any, Union
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


class CSRMatrix:
    """Compressed Sparse Row (CSR) matrix for memory-efficient sparse computing."""

    def __init__(
        self,
        values: np.ndarray,
        col_indices: np.ndarray,
        row_ptrs: np.ndarray,
        shape: Tuple[int, int],
    ):
        if shape[0] < 0 or shape[1] < 0:
            raise ValueError(f"CSRMatrix shape must be non-negative, got {shape}")
        if len(row_ptrs) != shape[0] + 1:
            raise ValueError(f"CSR invariant broken: row_ptrs length ({len(row_ptrs)}) must equal nrows + 1 ({shape[0] + 1})")
        if len(row_ptrs) > 0 and row_ptrs[0] != 0:
            raise ValueError(f"CSR invariant broken: row_ptrs[0] must be 0, got {row_ptrs[0]}")
        if len(values) != len(col_indices):
            raise ValueError(f"CSR invariant broken: values ({len(values)}) and col_indices ({len(col_indices)}) must have matching length")
        self.values = values.astype(np.float32)
        self.col_indices = col_indices.astype(np.int64)
        self.row_ptrs = row_ptrs.astype(np.int64)
        self.shape = shape

    @property
    def nnz(self) -> int:
        return len(self.values)

    @property
    def sparsity(self) -> float:
        total = self.shape[0] * self.shape[1]
        if total == 0:
            return 1.0
        return 1.0 - (float(self.nnz) / float(total))

    @classmethod
    def from_dense(cls, matrix: Union[_Tensor, np.ndarray], eps: float = 1e-7) -> "CSRMatrix":
        """Constructs a CSRMatrix from a 2D dense tensor or array."""
        if hasattr(matrix, "to_numpy"):
            mat = matrix.to_numpy()
        else:
            mat = np.asarray(matrix)

        if mat.ndim != 2:
            raise ValueError(f"Dense matrix must be 2D, got shape {mat.shape}")

        nrows, ncols = mat.shape
        row_ptrs = [0]
        values = []
        col_indices = []

        for r in range(nrows):
            row = mat[r]
            non_zeros = np.where(np.abs(row) > eps)[0]
            for c in non_zeros:
                values.append(row[c])
                col_indices.append(c)
            row_ptrs.append(len(values))

        return cls(
            values=np.array(values, dtype=np.float32),
            col_indices=np.array(col_indices, dtype=np.int64),
            row_ptrs=np.array(row_ptrs, dtype=np.int64),
            shape=(nrows, ncols),
        )

    def to_dense(self) -> _Tensor:
        """Converts back to a 2D dense Oxideon Tensor."""
        nrows, ncols = self.shape
        dense = np.zeros((nrows, ncols), dtype=np.float32)
        for r in range(nrows):
            start = self.row_ptrs[r]
            end = self.row_ptrs[r + 1]
            cols = self.col_indices[start:end]
            vals = self.values[start:end]
            np.add.at(dense[r], cols, vals)

        return _from_numpy(np.ascontiguousarray(dense))

    def transpose(self) -> "CSRMatrix":
        """Returns the transposed CSRMatrix $A^T$ of shape `(ncols, nrows)`."""
        nrows, ncols = self.shape
        if self.nnz == 0:
            return CSRMatrix(
                values=np.zeros(0, dtype=np.float32),
                col_indices=np.zeros(0, dtype=np.int64),
                row_ptrs=np.zeros(ncols + 1, dtype=np.int64),
                shape=(ncols, nrows),
            )
        row_counts = np.diff(self.row_ptrs)
        row_indices = np.repeat(np.arange(nrows, dtype=np.int64), row_counts)
        order = np.argsort(self.col_indices, kind="stable")
        new_vals = self.values[order]
        new_cols = row_indices[order]
        col_counts = np.bincount(self.col_indices, minlength=ncols)
        new_row_ptrs = np.concatenate([[0], np.cumsum(col_counts)]).astype(np.int64)
        return CSRMatrix(
            values=new_vals,
            col_indices=new_cols,
            row_ptrs=new_row_ptrs,
            shape=(ncols, nrows),
        )

    def spmm(self, x: Union[_Tensor, np.ndarray]) -> _Tensor:
        """Sparse-Dense matrix multiplication: $Y = A X$.

        Preserves Tensor autograd when `x` is a Tensor with `requires_grad=True`.
        """
        is_tensor = isinstance(x, _Tensor)
        if hasattr(x, "to_numpy"):
            x_np = x.to_numpy()
        else:
            x_np = np.asarray(x)

        if x_np.ndim not in (1, 2):
            raise ValueError(f"Dense tensor must be 1D or 2D, got shape {x_np.shape}")
        nrows, ncols = self.shape
        if x_np.shape[0] != ncols:
            raise ValueError(f"Shape mismatch: CSR {self.shape} vs Dense {x_np.shape}")

        if is_tensor and getattr(x, "requires_grad", False):
            a_dense = self.to_dense()
            if x_np.ndim == 1:
                return a_dense.matmul(x.reshape([ncols, 1])).reshape([nrows])
            return a_dense.matmul(x)

        if x_np.ndim == 1:
            row_counts = np.diff(self.row_ptrs)
            row_idx = np.repeat(np.arange(nrows, dtype=np.int64), row_counts)
            scaled = self.values * x_np[self.col_indices]
            out = np.bincount(row_idx, weights=scaled, minlength=nrows).astype(np.float32)
            return _from_numpy(np.ascontiguousarray(out))

        k_dim = x_np.shape[1]
        out = np.zeros((nrows, k_dim), dtype=np.float32)

        for r in range(nrows):
            start = self.row_ptrs[r]
            end = self.row_ptrs[r + 1]
            if start < end:
                out[r] = self.values[start:end] @ x_np[self.col_indices[start:end]]

        return _from_numpy(np.ascontiguousarray(out.astype(np.float32)))

    def spspmm(self, other: "CSRMatrix") -> "CSRMatrix":
        """Sparse-Sparse matrix multiplication: $C = A B$ returning a new CSRMatrix."""
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        nrows, ncols = self.shape[0], other.shape[1]
        out_row_ptrs = [0]
        out_col_indices = []
        out_values = []

        for r in range(nrows):
            a_start = self.row_ptrs[r]
            a_end = self.row_ptrs[r + 1]
            if a_start == a_end:
                out_row_ptrs.append(len(out_values))
                continue

            row_acc: dict = {}
            for idx in range(a_start, a_end):
                a_val = self.values[idx]
                k = self.col_indices[idx]

                b_start = other.row_ptrs[k]
                b_end = other.row_ptrs[k + 1]
                for b_idx in range(b_start, b_end):
                    c = other.col_indices[b_idx]
                    b_val = other.values[b_idx]
                    row_acc[c] = row_acc.get(c, 0.0) + (a_val * b_val)

            for col in sorted(row_acc.keys()):
                val = row_acc[col]
                if abs(val) > 1e-9:
                    out_col_indices.append(col)
                    out_values.append(val)

            out_row_ptrs.append(len(out_values))

        return CSRMatrix(
            values=np.array(out_values, dtype=np.float32),
            col_indices=np.array(out_col_indices, dtype=np.int64),
            row_ptrs=np.array(out_row_ptrs, dtype=np.int64),
            shape=(nrows, ncols),
        )


__all__ = [
    "CSRMatrix",
]
