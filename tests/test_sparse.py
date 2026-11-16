"""Z11: sparse COO/CSR (`ox.SparseCoo`/`ox.SparseCsr`) vs torch.sparse ground truth.

Pins construction/coalescing (incl. the deliberate explicit-zero divergence),
dense round-trips, transpose, CSR layout, differentiable spmm values + backward,
and fail-closed validation errors.
"""
import numpy as np
import pytest
import torch

import oxideon as ox

# torch-internal notice fired by the reference sparse constructors below.
pytestmark = pytest.mark.filterwarnings("ignore:Sparse invariant checks.*")

T = ox.from_numpy


def torch_coo(indices, values, shape):
    idx = torch.tensor(indices, dtype=torch.long).t().contiguous()
    val = torch.tensor(values, dtype=torch.float64)
    return torch.sparse_coo_tensor(idx, val, size=shape).coalesce()


def test_coo_coalesce_matches_torch_values():
    idx = [(0, 0), (0, 0), (1, 2), (2, 1)]
    val = [1.0, 2.0, 5.0, -3.0]
    coo = ox.SparseCoo(idx, val, (3, 4))
    assert coo.nnz() == 3
    assert coo.indices() == [(0, 0), (1, 2), (2, 1)]
    assert coo.values() == [3.0, 5.0, -3.0]
    th = torch_coo(idx, val, (3, 4))
    np.testing.assert_allclose(coo.to_dense(dtype="float64").to_numpy(), th.to_dense().numpy())
    # Deliberate divergence: oxideon drops exact zeros, torch keeps them.
    zd = ox.SparseCoo([(0, 0), (1, 1)], [0.0, 4.0], (2, 2))
    assert zd.nnz() == 1
    thz = torch_coo([(0, 0), (1, 1)], [0.0, 4.0], (2, 2))
    assert thz._nnz() == 2
    np.testing.assert_allclose(zd.to_dense(dtype="float64").to_numpy(), thz.to_dense().numpy())


def test_coo_from_dense_transpose_and_errors():
    dense = np.array([[0.0, 2.0, 0.0], [-1.0, 0.0, 0.5]])
    coo = ox.SparseCoo.from_dense(T(dense), 0.25)
    assert coo.nnz() == 3
    tr = coo.transpose()
    assert tr.shape == (3, 2)
    th = torch.from_numpy(dense).to_sparse().coalesce()
    np.testing.assert_allclose(tr.to_dense(dtype="float64").to_numpy(),
                               th.to_dense().numpy().T)
    with pytest.raises(Exception, match="len"):
        ox.SparseCoo([(0, 0)], [1.0, 2.0], (1, 1))
    with pytest.raises(Exception, match="bounds"):
        ox.SparseCoo([(0, 5)], [1.0], (2, 2))
    with pytest.raises(Exception, match="2D"):
        ox.SparseCoo.from_dense(T(np.ones(4)), 0.0)


def test_csr_layout_and_spmm_match_torch():
    idx = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
    val = [1.0, 2.0, 3.0, 4.0, 5.0]
    csr = ox.SparseCoo(idx, val, (3, 3)).to_csr()
    assert csr.row_ptr() == [0, 2, 3, 5]
    assert csr.col_idx() == [0, 2, 1, 0, 2]
    assert csr.values() == val
    assert abs(csr.sparsity() - 4 / 9) < 1e-12
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3, 4))
    y = csr.spmm(T(x)).to_numpy()
    th = torch_coo(idx, val, (3, 3)).to_dense().numpy()
    np.testing.assert_allclose(y, th @ x, rtol=1e-5, atol=1e-6)
    # Rectangular: [2, 4] sparse @ [4, 3] dense.
    csr2 = ox.SparseCoo([(0, 1), (1, 3)], [2.0, -1.0], (2, 4)).to_csr()
    x2 = rng.standard_normal((4, 3))
    th2 = torch_coo([(0, 1), (1, 3)], [2.0, -1.0], (2, 4)).to_dense().numpy()
    np.testing.assert_allclose(csr2.spmm(T(x2)).to_numpy(), th2 @ x2, rtol=1e-5, atol=1e-6)
    with pytest.raises(Exception, match="spmm|ShapeMismatch|shape"):
        csr.spmm(T(rng.standard_normal((2, 2))))


def test_spmm_backward_matches_dense_matmul_grad():
    idx = [(0, 0), (0, 1), (1, 1)]
    val = [1.0, 2.0, 3.0]
    csr = ox.SparseCoo(idx, val, (2, 2)).to_csr()
    rng = np.random.default_rng(6)
    x = rng.standard_normal((2, 3)).astype(np.float32)
    t = T(x).requires_grad_(True)
    csr.spmm(t).sum().backward()
    th = torch_coo(idx, val, (2, 2)).to_dense().numpy()
    np.testing.assert_allclose(t.grad.to_numpy(), th.T @ np.ones((2, 3)), rtol=1e-5, atol=1e-6)
    # CSR transpose values match torch.
    tr = csr.transpose()
    assert tr.shape == (2, 2)
    np.testing.assert_allclose(tr.to_dense(dtype="float64").to_numpy(), th.T)
