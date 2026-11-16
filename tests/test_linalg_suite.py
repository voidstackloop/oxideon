"""Comprehensive tests for Oxideon Linear Algebra (ox.linalg) module.

Validates vector & matrix norms, matrix inversions, pseudoinverses, determinants,
signed log-determinants, Cholesky, QR, SVD, eigenvalue decompositions, linear solves,
least squares, matrix exponential/powers, vector cross products, condition numbers,
and rank calculations.
"""

import math
import numpy as np
import pytest

import oxideon as ox
import oxideon.linalg as la


def test_matrix_and_vector_norms():
    x = ox.tensor([3.0, 4.0])
    assert math.isclose(la.vector_norm(x, ord=2).item(), 5.0, rel_tol=1e-5)
    assert math.isclose(la.vector_norm(x, ord=1).item(), 7.0, rel_tol=1e-5)
    assert math.isclose(la.vector_norm(x, ord=float("inf")).item(), 4.0, rel_tol=1e-5)

    m = ox.tensor([[1.0, 2.0], [3.0, 4.0]])
    fro = math.sqrt(1 + 4 + 9 + 16)
    assert math.isclose(la.matrix_norm(m, ord="fro").item(), fro, rel_tol=1e-5)
    assert math.isclose(la.norm(m).item(), fro, rel_tol=1e-5)


def test_inv_and_pinv():
    m = ox.tensor([[1.0, 2.0], [3.0, 4.0]])
    inv_m = la.inv(m)
    prod = ox.mm(m, inv_m)
    assert np.allclose(prod.to_numpy(), np.eye(2), atol=1e-4)

    # Rectangular matrix pseudo-inverse
    rect = ox.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pinv_rect = la.pinv(rect)
    assert pinv_rect.shape == (2, 3)
    rec = ox.mm(pinv_rect, rect)
    assert np.allclose(rec.to_numpy(), np.eye(2), atol=1e-4)


def test_det_and_slogdet():
    m = ox.tensor([[2.0, 1.0], [1.0, 2.0]])
    d = la.det(m)
    assert math.isclose(d.item(), 3.0, rel_tol=1e-5)

    sign, logabsdet = la.slogdet(m)
    assert math.isclose(sign.item(), 1.0, rel_tol=1e-5)
    assert math.isclose(logabsdet.item(), math.log(3.0), rel_tol=1e-5)


def test_cholesky_suite():
    # Symmetric positive-definite matrix
    A = ox.tensor([[4.0, 2.0], [2.0, 5.0]])
    L = la.cholesky(A)
    rec = ox.mm(L, L.t())
    assert np.allclose(rec.to_numpy(), A.to_numpy(), atol=1e-4)

    # Cholesky solve A X = B
    B = ox.tensor([[2.0], [1.0]])
    X = la.cholesky_solve(B, L)
    assert np.allclose(ox.mm(A, X).to_numpy(), B.to_numpy(), atol=1e-4)

    # Cholesky inverse
    inv_A = la.cholesky_inverse(L)
    assert np.allclose(ox.mm(A, inv_A).to_numpy(), np.eye(2), atol=1e-4)


def test_qr_decomposition():
    A = ox.tensor([[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]])
    res = la.qr(A)
    Q, R = res.Q, res.R
    assert Q.shape == (3, 3)
    assert R.shape == (3, 3)

    # Q is orthogonal
    QtQ = ox.mm(Q.t(), Q)
    assert np.allclose(QtQ.to_numpy(), np.eye(3), atol=1e-4)

    # Reconstruction A = Q @ R
    rec = ox.mm(Q, R)
    assert np.allclose(rec.to_numpy(), A.to_numpy(), atol=1e-3)


def test_svd_and_svdvals():
    A = ox.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    U, S, Vh = la.svd(A, full_matrices=False)
    assert U.shape == (3, 2)
    assert S.shape == (2,)
    assert Vh.shape == (2, 2)

    s_vals = la.svdvals(A)
    assert np.allclose(s_vals.to_numpy(), S.to_numpy(), atol=1e-4)

    # Reconstruct A = U @ diag(S) @ Vh
    diag_S = ox.diag(S) if hasattr(ox, "diag") else ox.from_numpy(np.diag(S.to_numpy()))
    rec = ox.mm(ox.mm(U, diag_S), Vh)
    assert np.allclose(rec.to_numpy(), A.to_numpy(), atol=1e-4)


def test_eigensystems():
    # Symmetric matrix for eigh
    A = ox.tensor([[2.0, -1.0], [-1.0, 2.0]])
    vals, vecs = la.eigh(A)
    assert np.allclose(np.sort(vals.to_numpy()), [1.0, 3.0], atol=1e-4)

    vals_only = la.eigvalsh(A)
    assert np.allclose(np.sort(vals_only.to_numpy()), [1.0, 3.0], atol=1e-4)


def test_solve_and_lstsq():
    A = ox.tensor([[3.0, 1.0], [1.0, 2.0]])
    B = ox.tensor([[9.0], [8.0]])
    X = la.solve(A, B)
    # Solution is [2.0, 3.0]
    assert np.allclose(X.to_numpy().reshape(-1), [2.0, 3.0], atol=1e-4)

    # Overdetermined system with lstsq
    A_over = ox.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    B_over = ox.tensor([[6.0], [5.0], [7.0]])
    res = la.lstsq(A_over, B_over)
    assert res.solution.shape == (2, 1)
    assert res.rank == 2


def test_matrix_power_and_matrix_exp():
    A = ox.tensor([[1.0, 2.0], [0.0, 1.0]])
    A3 = la.matrix_power(A, 3)
    assert np.allclose(A3.to_numpy(), [[1.0, 6.0], [0.0, 1.0]], atol=1e-4)

    zero_mat = ox.zeros(2, 2)
    exp_zero = la.matrix_exp(zero_mat)
    assert np.allclose(exp_zero.to_numpy(), np.eye(2), atol=1e-4)


def test_cross_cond_rank_multidot():
    v1 = ox.tensor([1.0, 0.0, 0.0])
    v2 = ox.tensor([0.0, 1.0, 0.0])
    c = la.cross(v1, v2)
    assert np.allclose(c.to_numpy(), [0.0, 0.0, 1.0], atol=1e-5)

    ident = ox.eye(4)
    assert math.isclose(la.cond(ident).item(), 1.0, rel_tol=1e-4)
    assert la.matrix_rank(ident).item() == 4

    # Rank-deficient matrix
    rank_def = ox.tensor([[1.0, 2.0], [2.0, 4.0]])
    assert la.matrix_rank(rank_def).item() == 1

    # Multi-dot
    A = ox.randn(2, 3)
    B = ox.randn(3, 4)
    C = ox.randn(4, 2)
    md = la.multi_dot([A, B, C])
    expected = ox.mm(ox.mm(A, B), C)
    assert np.allclose(md.to_numpy(), expected.to_numpy(), atol=1e-4)
