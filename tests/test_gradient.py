"""Z42: gradient/trapezoid/trapz vs torch ground truth.

The native engine kernels (slicing composition in the promoted width,
general 3-point schemes for coordinate spacing, differentiable) serve the
module functions via pure delegation. Values, shapes, dtypes, edges, and
gradients are pinned against torch.

Deliberate divergences from torch (all pinned below):
- Bad dims (incl. `dim` on 0-d) raise `RuntimeError` (torch: `IndexError`).
- Bool input raises `RuntimeError` (torch: `NotImplementedError`).
"""
import numpy as np
import pytest
import torch

import oxideon as ox

T = ox.from_numpy
th = torch.from_numpy

X = np.array([1.0, 2.0, 4.0, 7.0, 11.0], dtype=np.float32)
A = np.arange(12, dtype=np.float32).reshape(3, 4)


def test_gradient_uniform_matches_torch():
    for e in (1, 2):
        o = ox.gradient(T(X), edge_order=e)
        t = torch.gradient(th(X), edge_order=e)
        assert isinstance(o, tuple) and len(o) == len(t) == 1
        np.testing.assert_allclose(o[0].to_numpy(), t[0].numpy(), rtol=1e-6)
    o = ox.gradient(T(A))
    t = torch.gradient(th(A))
    assert len(o) == len(t) == 2
    for oo, tt in zip(o, t):
        assert oo.to_numpy().shape == tt.numpy().shape
        np.testing.assert_allclose(oo.to_numpy(), tt.numpy(), rtol=1e-6)
    for dim in (1, (0, 1), [0]):
        o = ox.gradient(T(A), dim=dim, edge_order=2)
        t = torch.gradient(th(A), dim=dim, edge_order=2)
        assert len(o) == len(t)
        for oo, tt in zip(o, t):
            np.testing.assert_allclose(oo.to_numpy(), tt.numpy(), rtol=1e-6)


def test_gradient_scalar_and_list_spacing():
    o = ox.gradient(T(X), spacing=2.0)[0].to_numpy()
    t = torch.gradient(th(X), spacing=2.0)[0].numpy()
    np.testing.assert_allclose(o, t, rtol=1e-6)
    o = ox.gradient(T(X), spacing=2)[0].to_numpy()
    np.testing.assert_allclose(o, t, rtol=1e-6)
    o = ox.gradient(T(A), spacing=[1.0, 2.0])
    t = torch.gradient(th(A), spacing=[1.0, 2.0])
    for oo, tt in zip(o, t):
        np.testing.assert_allclose(oo.to_numpy(), tt.numpy(), rtol=1e-6)
    o = ox.gradient(T(A), spacing=[2.0], dim=(1,))
    t = torch.gradient(th(A), spacing=[2.0], dim=(1,))
    np.testing.assert_allclose(o[0].to_numpy(), t[0].numpy(), rtol=1e-6)


def test_gradient_coord_spacing_matches_torch():
    c = np.array([0.0, 1.0, 3.0, 6.0, 10.0], dtype=np.float32)
    for e in (1, 2):
        o = ox.gradient(T(X), spacing=(T(c),), edge_order=e)[0].to_numpy()
        t = torch.gradient(th(X), spacing=(th(c),), edge_order=e)[0].numpy()
        np.testing.assert_allclose(o, t, rtol=1e-5)
    # Quadratic data: second-order schemes are near-exact (true slopes 2t).
    t = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)
    y = (t ** 2).astype(np.float32)
    o = ox.gradient(T(y), spacing=(T(t),), edge_order=2)[0].to_numpy()
    np.testing.assert_allclose(o, 2 * t, rtol=1e-4, atol=1e-4)
    tt = torch.gradient(th(y), spacing=(th(t),), edge_order=2)[0].numpy()
    np.testing.assert_allclose(o, tt, rtol=1e-5)


def test_gradient_tuple_shapes_and_0d():
    assert ox.gradient(T(X)) and len(ox.gradient(T(X), dim=0)) == 1
    assert ox.gradient(T(np.array(5.0, dtype=np.float32))) == ()
    assert torch.gradient(th(np.array(5.0, dtype=np.float32))) == ()


def test_gradient_validation_matches_torch():
    with pytest.raises(RuntimeError):
        ox.gradient(T(X), edge_order=3)
    with pytest.raises(RuntimeError):
        torch.gradient(th(X), edge_order=3)
    with pytest.raises(RuntimeError):
        ox.gradient(T(np.array([5.0], dtype=np.float32)))
    with pytest.raises(RuntimeError):
        torch.gradient(th(np.array([5.0], dtype=np.float32)))
    with pytest.raises(RuntimeError):
        ox.gradient(T(np.array([5.0, 7.0], dtype=np.float32)), edge_order=2)
    with pytest.raises(RuntimeError):
        ox.gradient(T(A), dim=5)
    with pytest.raises(IndexError):
        torch.gradient(th(A), dim=5)
    with pytest.raises(RuntimeError):
        ox.gradient(T(A), dim=(1, 1))
    with pytest.raises(RuntimeError):
        torch.gradient(th(A), dim=(1, 1))
    with pytest.raises(TypeError):
        ox.gradient(T(X), spacing=T(np.array([0.0, 1.0, 3.0, 6.0, 10.0], dtype=np.float32)))
    with pytest.raises(TypeError):
        torch.gradient(th(X), spacing=th(np.array([0.0, 1.0, 3.0, 6.0, 10.0], dtype=np.float32)))
    with pytest.raises(RuntimeError):
        ox.gradient(T(X), spacing=(T(np.array(2.0, dtype=np.float32)).sum(),))
    with pytest.raises(RuntimeError):
        ox.gradient(T(A), spacing=[1.0])
    with pytest.raises(RuntimeError):
        torch.gradient(th(A), spacing=[1.0])
    with pytest.raises(RuntimeError):
        ox.gradient(T(A), spacing=[1.0, 2.0], dim=(1,))
    with pytest.raises(RuntimeError):
        ox.gradient(T(X), spacing=(T(np.array([0.0, 1.0], dtype=np.float32)),))


def test_gradient_dtypes_and_grad():
    g = ox.gradient(T(np.array([1, 2, 4])))[0]
    assert g.to_numpy().dtype == np.float32
    np.testing.assert_allclose(g.to_numpy(), [1.0, 1.5, 2.0], rtol=1e-6)
    assert torch.gradient(th(np.array([1, 2, 4])))[0].dtype == torch.float32
    with pytest.raises(RuntimeError):
        ox.gradient(T(np.array([True, False, True])))
    with pytest.raises(NotImplementedError):
        torch.gradient(th(np.array([True, False, True])))
    h = ox.gradient(T(np.array([1.0, 2.0, 4.0], dtype=np.float16)))[0]
    assert h.dtype == "f16" 
    m = ox.gradient(T(np.array([1.0, 2.0, 4.0], dtype=np.float16)),
                    spacing=(T(np.array([0.0, 1.0, 3.0], dtype=np.float64)),))[0]
    assert m.to_numpy().dtype == np.float64
    o = T(X).requires_grad_(True)
    ox.gradient(o)[0].sum().backward()
    t = th(X).requires_grad_(True)
    torch.gradient(t)[0].sum().backward()
    np.testing.assert_allclose(o.grad.to_numpy(), t.grad.numpy(), rtol=1e-6)


def test_trapezoid_matches_torch():
    assert ox.trapezoid(T(X)).to_numpy() == 19.0
    assert ox.trapz(T(X)).to_numpy() == 19.0
    c = np.array([0.0, 1.0, 3.0, 6.0, 10.0], dtype=np.float32)
    assert ox.trapezoid(T(X), x=T(c)).to_numpy() == 60.0
    assert ox.trapezoid(T(X), dx=2.0).to_numpy() == 38.0
    o = ox.trapezoid(T(A), dim=0).to_numpy()
    np.testing.assert_array_equal(o, [8.0, 10.0, 12.0, 14.0])
    z = ox.trapezoid(T(A), x=T(np.ones((3, 4), dtype=np.float32))).to_numpy()
    np.testing.assert_array_equal(z, [0.0, 0.0, 0.0])
    assert ox.trapezoid(T(np.array([5.0], dtype=np.float32))).to_numpy() == 0.0
    assert ox.trapezoid(T(np.empty((0,), dtype=np.float32))).to_numpy() == 0.0
    r = ox.trapezoid(T(X), x=T(c.astype(np.float64)))
    assert r.to_numpy().dtype == np.float64
    assert torch.trapezoid(th(X), th(c.astype(np.float64))).dtype == torch.float64


def test_trapezoid_validation_matches_torch():
    c = np.array([0.0, 1.0, 3.0, 6.0, 10.0], dtype=np.float32)
    with pytest.raises(TypeError):
        ox.trapezoid(T(X), x=T(c), dx=2.0)
    with pytest.raises(TypeError):
        torch.trapezoid(th(X), th(c), dx=2.0)
    with pytest.raises(RuntimeError):
        ox.trapezoid(T(X), x=T(np.array([0.0, 1.0], dtype=np.float32)))
    with pytest.raises(RuntimeError):
        torch.trapezoid(th(X), th(np.array([0.0, 1.0], dtype=np.float32)))
    with pytest.raises(TypeError):
        ox.trapezoid(T(A), dim=(0, 1))
    with pytest.raises(TypeError):
        torch.trapezoid(th(A), dim=(0, 1))


def test_trapezoid_dtypes_and_grad():
    r = ox.trapezoid(T(np.array([1, 2, 4])))
    assert r.to_numpy().dtype == np.float32
    assert r.to_numpy() == 4.5
    with pytest.raises(RuntimeError):
        ox.trapezoid(T(np.array([True, False, True])))
    with pytest.raises(NotImplementedError):
        torch.trapezoid(th(np.array([True, False, True])))
    h = ox.trapezoid(T(np.array([1.0, 2.0, 4.0], dtype=np.float16)))
    assert h.dtype == "f16" 
    o = T(X).requires_grad_(True)
    ox.trapezoid(o).backward()
    t = th(X).requires_grad_(True)
    torch.trapezoid(t).backward()
    np.testing.assert_allclose(o.grad.to_numpy(), t.grad.numpy(), rtol=1e-6)
