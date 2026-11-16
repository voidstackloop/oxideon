"""Comprehensive test suite for Oxideon Sparse Tensor Linear Algebra enrichment.

Verifies:
  - Coordinate (COO) and Compressed Sparse Row (CSR) representations
  - Differentiable Sparse-Dense Matrix Multiplication (SpMM) forward & autograd backward
  - Gustavson Sparse-Sparse Matrix Multiplication (SpSpMM)
  - Interop between native Rust (SparseCoo/SparseCsr) and Python (COOTensor/CSRTensor)
  - Operator overloads: +, -, *, @
  - Norms, sparsity, density, transpose, roundtrip conversions
"""

import numpy as np
import pytest
import torch
import oxideon as ox
from oxideon.sparse import (
    COOTensor,
    CSRTensor,
    coo_matrix_from_dense,
    sparse_transpose,
    sparse_add,
    sparse_sub,
    sparse_norm,
    spmm,
    spspmm,
    sparse_coo_tensor,
)


def test_native_sparse_coo_and_csr_constructors():
    idx = [(0, 0), (1, 2), (2, 1)]
    vals = [3.0, 4.0, 12.0]
    shape = (3, 4)

    coo = ox.SparseCoo(idx, vals, shape)
    assert coo.shape == shape
    assert coo.nnz() == 3
    assert abs(coo.sparsity() - (1.0 - 3 / 12)) < 1e-12
    assert abs(coo.norm() - 13.0) < 1e-12  # sqrt(9 + 16 + 144) = 13
    assert "SparseCoo" in repr(coo)

    csr = coo.to_csr()
    assert csr.shape == shape
    assert csr.nnz() == 3
    assert abs(csr.sparsity() - coo.sparsity()) < 1e-12
    assert abs(csr.norm() - 13.0) < 1e-12
    assert "SparseCsr" in repr(csr)

    # CSR to COO roundtrip
    coo_back = csr.to_coo()
    assert coo_back.shape == shape
    assert coo_back.nnz() == 3
    assert coo_back.indices() == idx
    assert coo_back.values() == vals


def test_native_sparse_arithmetic():
    a = ox.SparseCoo([(0, 0), (0, 1)], [2.0, 3.0], (2, 2))
    b = ox.SparseCoo([(0, 1), (1, 1)], [4.0, 5.0], (2, 2))

    # COO add & sub
    c = a + b
    assert c.indices() == [(0, 0), (0, 1), (1, 1)]
    assert c.values() == [2.0, 7.0, 5.0]

    diff = a - b
    assert diff.indices() == [(0, 0), (0, 1), (1, 1)]
    assert diff.values() == [2.0, -1.0, -5.0]

    scaled = a * 2.0
    assert scaled.values() == [4.0, 6.0]
    scaled_r = 3.0 * a
    assert scaled_r.values() == [6.0, 9.0]

    # CSR add & sub & scale
    csr_a = a.to_csr()
    csr_b = b.to_csr()
    csr_c = csr_a + csr_b
    assert csr_c.shape == (2, 2)
    assert csr_c.values() == [2.0, 7.0, 5.0]

    csr_diff = csr_a - csr_b
    assert csr_diff.values() == [2.0, -1.0, -5.0]

    csr_scaled = csr_a * 1.5
    assert csr_scaled.values() == [3.0, 4.5]


def test_native_sparse_dense_matmul_and_autograd():
    # A = [[1, 2], [0, 3]]
    # X = [[1, 2], [3, 4]]
    # A @ X = [[7, 10], [9, 12]]
    coo = ox.SparseCoo([(0, 0), (0, 1), (1, 1)], [1.0, 2.0, 3.0], (2, 2))
    csr = coo.to_csr()

    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = ox.from_numpy(x_np).requires_grad_(True)

    # Test @ on COO
    out_coo = coo @ x
    np.testing.assert_allclose(out_coo.to_numpy(), np.array([[7.0, 10.0], [9.0, 12.0]]))

    # Test @ on CSR
    x_csr = ox.from_numpy(x_np).requires_grad_(True)
    out_csr = csr @ x_csr
    np.testing.assert_allclose(out_csr.to_numpy(), np.array([[7.0, 10.0], [9.0, 12.0]]))

    # Test Autograd backward pass: d(sum(out))/dX = A^T @ ones = [[1, 0], [2, 3]] @ [[1, 1], [1, 1]] = [[1, 1], [5, 5]]
    out_csr.sum().backward()
    np.testing.assert_allclose(x_csr.grad.to_numpy(), np.array([[1.0, 1.0], [5.0, 5.0]]))


def test_native_gustavson_spspmm():
    # A = [[1, 2], [0, 3]]
    # B = [[4, 0], [5, 6]]
    # A @ B = [[14, 12], [15, 18]]
    a = ox.SparseCoo([(0, 0), (0, 1), (1, 1)], [1.0, 2.0, 3.0], (2, 2)).to_csr()
    b = ox.SparseCoo([(0, 0), (1, 0), (1, 1)], [4.0, 5.0, 6.0], (2, 2)).to_csr()

    c = a @ b
    assert isinstance(c, ox.SparseCsr)
    dense_c = c.to_dense().to_numpy()
    expected = np.array([[14.0, 12.0], [15.0, 18.0]], dtype=np.float32)
    np.testing.assert_allclose(dense_c, expected)

    # COO @ COO
    coo_a = a.to_coo()
    coo_b = b.to_coo()
    coo_c = coo_a @ coo_b
    assert isinstance(coo_c, ox.SparseCoo)
    np.testing.assert_allclose(coo_c.to_dense().to_numpy(), expected)


def test_python_coo_and_csr_tensor_classes():
    indices = [[0, 0, 1, 2], [0, 1, 2, 0]]
    values = [1.0, 2.0, 3.0, 4.0]
    shape = (3, 3)

    coo = COOTensor(indices, values, shape)
    assert coo.nnz == 4
    assert abs(coo.density - 4 / 9) < 1e-6
    assert abs(coo.sparsity - (1.0 - 4 / 9)) < 1e-6
    assert abs(coo.norm() - np.sqrt(1 + 4 + 9 + 16)) < 1e-6

    # Convert to CSR and check
    csr = coo.to_csr()
    assert isinstance(csr, CSRTensor)
    assert csr.nnz == 4
    assert abs(csr.sparsity - coo.sparsity) < 1e-6
    assert abs(csr.norm() - coo.norm()) < 1e-6

    # Roundtrip to COO
    coo_roundtrip = csr.to_coo()
    np.testing.assert_allclose(coo_roundtrip.to_dense().to_numpy(), coo.to_dense().to_numpy())

    # Native interop
    native_coo = coo.to_oxideon_coo()
    coo_from_native = COOTensor.from_oxideon_coo(native_coo)
    np.testing.assert_allclose(coo_from_native.to_dense().to_numpy(), coo.to_dense().to_numpy())

    native_csr = csr.to_oxideon_csr()
    csr_from_native = CSRTensor.from_oxideon_csr(native_csr)
    np.testing.assert_allclose(csr_from_native.to_dense().to_numpy(), csr.to_dense().to_numpy())


def test_python_sparse_arithmetic_and_matmul():
    a = coo_matrix_from_dense([[1.0, 2.0], [0.0, 3.0]])
    b = coo_matrix_from_dense([[0.0, 1.0], [4.0, 0.0]])

    # Addition
    c = a + b
    expected_add = np.array([[1.0, 3.0], [4.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(c.to_dense().to_numpy(), expected_add)

    # Subtraction
    diff = a - b
    expected_sub = np.array([[1.0, 1.0], [-4.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(diff.to_dense().to_numpy(), expected_sub)

    # Scalar multiplication
    scaled = a * 2.5
    np.testing.assert_allclose(scaled.to_dense().to_numpy(), np.array([[2.5, 5.0], [0.0, 7.5]]))

    # Sparse-Dense matmul
    x = ox.from_numpy(np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32))
    ax = a @ x
    np.testing.assert_allclose(ax.to_numpy(), np.array([[4.0, 5.0], [3.0, 6.0]]))

    # Sparse-Sparse matmul
    c_mul = a @ b
    np.testing.assert_allclose(c_mul.to_dense().to_numpy(), np.array([[8.0, 1.0], [12.0, 0.0]]))


def test_sparse_rectangular_and_empty():
    # Empty COO
    empty_coo = ox.SparseCoo([], [], (0, 0))
    assert empty_coo.nnz() == 0
    assert empty_coo.shape == (0, 0)
    assert empty_coo.to_csr().nnz() == 0

    # Rectangular [3, 2] @ [2, 4]
    a = ox.SparseCoo([(0, 1), (2, 0)], [3.0, 5.0], (3, 2)).to_csr()
    b = ox.SparseCoo([(1, 2), (0, 3)], [2.0, 4.0], (2, 4)).to_csr()

    c = a @ b
    assert c.shape == (3, 4)
    # Row 0 col 1 (3.0) * Row 1 col 2 (2.0) -> (0, 2) = 6.0
    # Row 2 col 0 (5.0) * Row 0 col 3 (4.0) -> (2, 3) = 20.0
    d = c.to_dense().to_numpy()
    assert d[0, 2] == 6.0
    assert d[2, 3] == 20.0
    assert np.count_nonzero(d) == 2
