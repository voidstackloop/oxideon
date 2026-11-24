"""Regression tests for bugs found by systematic comparison against PyTorch (each test names the symptom it pins)."""
import numpy as np
import pytest

import oxideon as ox

torch = pytest.importorskip("torch")


def T(a):
    return ox.from_numpy(np.ascontiguousarray(a))


def A(t):
    return t.to_numpy()


@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "atan2", "hypot", "fmod", "remainder", "maximum", "minimum", "copysign",
                                  "xlogy", "fmax", "fmin", "logaddexp", "logaddexp2"])
def test_binary_methods_accept_python_scalars(name):
    a = np.array([[0.5, 1.5], [2.5, 3.5]])
    got = A(getattr(T(a), name)(2.0))
    want = getattr(torch, name)(torch.from_numpy(a), torch.tensor(2.0, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_float_scalar_on_integer_tensor_promotes_to_float():
    got = T(np.array([1, 2, 3], dtype=np.int64)).mul(0.5)
    assert got.dtype == "f32"
    np.testing.assert_allclose(A(got), [0.5, 1.0, 1.5])


def test_integer_true_division_gives_floats():
    i = np.array([7, -7, 6, -6, 0], dtype=np.int64)
    j = np.array([2, 2, -4, -4, 3], dtype=np.int64)
    for got, want in [(T(i) / T(j), torch.from_numpy(i) / torch.from_numpy(j)), (T(i) / 2, torch.from_numpy(i) / 2),
                      (T(i).div(T(j)), torch.div(torch.from_numpy(i), torch.from_numpy(j))), (7 / T(j + 5), 7 / torch.from_numpy(j + 5))]:
        assert got.dtype == "f32"
        np.testing.assert_allclose(A(got), want.numpy(), rtol=1e-6)
    np.testing.assert_array_equal(A(T(i).div(T(j), rounding_mode="floor")), torch.div(torch.from_numpy(i), torch.from_numpy(j), rounding_mode="floor").numpy())


def test_floor_division_modulo_and_shift_operators():
    i = np.array([7, -7, 6, -6, 0], dtype=np.int64)
    j = np.array([2, 2, -4, -4, 3], dtype=np.int64)
    ti, tj = torch.from_numpy(i), torch.from_numpy(j)
    np.testing.assert_array_equal(A(T(i) // T(j)), (ti // tj).numpy())
    np.testing.assert_array_equal(A(T(i) % T(j)), (ti % tj).numpy())
    np.testing.assert_array_equal(A(T(i) // 2), (ti // 2).numpy())
    np.testing.assert_array_equal(A(T(i) << 2), (ti << 2).numpy())
    np.testing.assert_array_equal(A(T(i) >> 1), (ti >> 1).numpy())
    f = np.array([5.5, -5.5, 0.25])
    np.testing.assert_allclose(A(T(f) % 2.0), (torch.from_numpy(f) % 2.0).numpy())
    np.testing.assert_allclose(A(10.0 % T(np.array([3.0, 4.0]))), [1.0, 2.0])
    np.testing.assert_allclose(A(T(f) // 2.0), (torch.from_numpy(f) // 2.0).numpy())
    assert A(+T(f)).tolist() == f.tolist()


def test_integer_floor_division_by_zero_raises():
    with pytest.raises(RuntimeError):
        T(np.array([1], dtype=np.int64)) // T(np.array([0], dtype=np.int64))


def test_in_place_operators_write_into_the_tensor():
    a = T(np.array([1.0, 2.0], dtype=np.float32))
    alias = a
    a += 1
    a *= 2
    a -= 1
    a /= 2
    assert a is alias
    np.testing.assert_allclose(A(alias), [1.5, 2.5])
    w = T(np.array([1.0, 2.0], dtype=np.float32)).requires_grad_(True)
    y = w * 2
    y += 1  # part of a graph: the out-of-place result keeps the gradient correct
    y.sum().backward()
    np.testing.assert_allclose(A(w.grad), [2.0, 2.0])


def test_nan_to_num_keeps_shape_dtype_and_gradient():
    assert tuple(ox.nan_to_num(ox.from_numpy(np.array(np.nan))).shape) == ()
    x = np.array([1.0, np.nan, np.inf, -np.inf, 2.0], dtype=np.float32)
    xo = T(x).requires_grad_(True)
    out = ox.nan_to_num(xo, nan=-1.0, posinf=9.0)
    want = torch.nan_to_num(torch.from_numpy(x), nan=-1.0, posinf=9.0).numpy()
    np.testing.assert_array_equal(A(out), want)
    out.sum().backward()
    np.testing.assert_array_equal(A(xo.grad), [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(A(ox.nan_to_num(T(np.array([np.inf, -np.inf], dtype=np.float32)))),
                                  torch.nan_to_num(torch.tensor([np.inf, -np.inf])).numpy())
    ints = T(np.array([1, 2], dtype=np.int64))
    assert A(ox.nan_to_num(ints)).tolist() == [1, 2]


def test_angle_of_real_input_matches_torch():
    x = np.array([-0.0, 0.0, -1.0, 2.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    np.testing.assert_array_equal(A(T(x).angle()), torch.from_numpy(x).angle().numpy())
    c = np.array([1 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex64)
    np.testing.assert_allclose(A(T(c).angle()), torch.from_numpy(c).angle().numpy(), rtol=1e-6)


# ---------------------------------------------------------------------------------------- hunt 2: reductions / shape / indexing

X3 = np.random.default_rng(5).standard_normal((3, 4, 5))


def both(fn, grad=True):
    """Runs `fn(x)` on an oxideon and a torch tensor of the same data; returns both results and the two input gradients."""
    xo, xt = T(X3.copy()), torch.from_numpy(X3.copy())
    if grad:
        xo.requires_grad_(True)
        xt.requires_grad_(True)
    ro, rt = fn(xo, ox), fn(xt, torch)
    return xo, xt, ro, rt


def same(fn, grad=True, tol=1e-9):
    xo, xt, ro, rt = both(fn, grad)
    assert tuple(ro.shape) == tuple(rt.shape)
    np.testing.assert_allclose(A(ro), rt.detach().numpy(), rtol=tol, atol=tol)
    if grad:
        w = np.random.default_rng(6).standard_normal(tuple(rt.shape))
        (ro * T(w)).sum().backward()
        (rt * torch.from_numpy(w)).sum().backward()
        np.testing.assert_allclose(A(xo.grad), xt.grad.numpy(), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("name", ["sum", "mean", "amax", "amin", "var", "std", "nansum", "logsumexp"])
@pytest.mark.parametrize("dims", [(0, 1), (1, 2), (0, 2), (-1, -2), (0, 1, 2)])
@pytest.mark.parametrize("keep", [False, True])
def test_reductions_over_several_dims(name, dims, keep):
    same(lambda x, M: getattr(x, name)(dims, keepdim=keep))


@pytest.mark.parametrize("name", ["amax", "amin", "prod", "all", "any", "count_nonzero", "sum", "var"])
def test_full_reductions_are_zero_dimensional(name):
    assert tuple(getattr(T(X3), name)().shape) == () == tuple(getattr(torch.from_numpy(X3), name)().shape)


def test_amax_amin_share_gradient_between_ties_like_torch():
    a = np.array([[1.0, 3.0, 3.0], [2.0, 2.0, 0.0]])
    for name in ("amax", "amin"):
        xo, xt = T(a).requires_grad_(True), torch.from_numpy(a).requires_grad_(True)
        getattr(xo, name)(1).sum().backward()
        getattr(xt, name)(1).sum().backward()
        np.testing.assert_allclose(A(xo.grad), xt.grad.numpy())


def test_var_std_correction_keyword():
    for c in (0, 1, 2):
        same(lambda x, M, c=c: x.var(1, correction=c))
        same(lambda x, M, c=c: x.std((0, 2), correction=c))


def test_max_min_with_dim_return_values_and_indices_and_binary_form():
    for dim, keep in [(0, False), (1, True), (-1, False)]:
        ro = T(X3).max(dim, keepdim=keep)
        rt = torch.from_numpy(X3).max(dim, keepdim=keep)
        np.testing.assert_array_equal(A(ro.values), rt.values.numpy())
        np.testing.assert_array_equal(A(ro.indices), rt.indices.numpy())
        v, i = T(X3).min(dim=dim, keepdim=keep)
        np.testing.assert_array_equal(A(i), torch.from_numpy(X3).min(dim, keepdim=keep).indices.numpy())
    np.testing.assert_array_equal(A(T(X3).max(T(X3 * 0.5))), np.maximum(X3, X3 * 0.5))
    assert float(T(X3).max()) == X3.max()
    s = T(X3).sort(dim=1, descending=True)
    np.testing.assert_array_equal(A(s.values), torch.from_numpy(X3).sort(dim=1, descending=True).values.numpy())
    assert T(X3).topk(2, dim=2).indices.shape == [3, 4, 2]


def test_varargs_dims_and_transpose_properties():
    same(lambda x, M: x.permute(2, 0, 1))
    same(lambda x, M: x.permute((2, 0, 1)))
    same(lambda x, M: x[:, :1].expand(3, 4, 5))
    same(lambda x, M: x[:, :1].expand(-1, 4, -1))
    same(lambda x, M: x.flip(0, 2))
    same(lambda x, M: x.flip((0, 2)))
    same(lambda x, M: x.mT)
    same(lambda x, M: x[0].T)


@pytest.mark.parametrize("fn", [
    lambda x, M: x.repeat(2, 1, 2), lambda x, M: x.repeat(2, 1, 1, 2), lambda x, M: x.tile((2, 1)), lambda x, M: x.repeat_interleave(2, 1),
    lambda x, M: x.repeat_interleave(2), lambda x, M: M.hstack([x, x]), lambda x, M: M.vstack([x[0], x[1]]), lambda x, M: M.dstack([x, x]),
    lambda x, M: x.diag_embed(), lambda x, M: x.diag_embed(1), lambda x, M: x.diag_embed(-1, 0, 2), lambda x, M: M.meshgrid(x[0, 0], x[1, 1])[1],
    lambda x, M: M.meshgrid(x[0, 0], x[1, 1], x[2, 2], indexing="xy")[0], lambda x, M: M.cumulative_trapezoid(x, dim=1),
    lambda x, M: M.einsum("ijk,ijl->kl", x, x), lambda x, M: M.einsum("bij,bkj->bik", x, x), lambda x, M: M.where(x > 0, x, 0.0),
], ids=lambda f: "")
def test_ops_now_differentiable_and_matching_torch(fn):
    same(fn)


def test_take_index_add_index_copy_put_are_differentiable():
    idx = np.array([0, 5, 59, -1])
    same(lambda x, M: x.take(M.from_numpy(idx)))
    same(lambda x, M: x.index_add(1, M.from_numpy(np.array([0, 0, 2])), x[:, :3]))
    same(lambda x, M: x.index_copy(1, M.from_numpy(np.array([2, 0])), x[:, :2]))
    same(lambda x, M: x.index_add(1, M.from_numpy(np.array([1, 3])), x[:, :2], alpha=2.0))
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    got = A(T(a).put(ox.from_numpy(np.array([0, 4, -1])), ox.from_numpy(np.array([10.0, 20.0, 30.0]))))
    np.testing.assert_array_equal(got, torch.from_numpy(a).put(torch.tensor([0, 4, -1]), torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)).numpy())
    np.testing.assert_array_equal(A(T(a).scatter(1, ox.from_numpy(np.array([[0], [2]])), 9.0)),
                                  torch.from_numpy(a).scatter(1, torch.tensor([[0], [2]]), 9.0).numpy())


@pytest.mark.parametrize("key", ["[[0, 2]]", "[[0, 1], [1, 2]]", "[:, [2, 0]]", "[:, [1, 2], [0, 3]]", "[[0, 2], :, [1, 3]]", "[..., [0, 1]]",
                                 "[1, :, [0, 1]]", "[None, [0, 1]]", "[-1, [0, 1]]", "[[0, 2], 1]", "[[-1, 0], 1:3]"])
def test_advanced_indexing_matches_torch(key):
    same(eval("lambda x, M: x" + key))


def test_advanced_indexing_with_tensors_and_masks():
    same(lambda x, M: x[M.from_numpy(np.array([2, 0]))])
    same(lambda x, M: x[:, M.from_numpy(np.array([2, 0]))])
    same(lambda x, M: x[x > 0])
    same(lambda x, M: x[M.from_numpy(np.array([True, False, True]))])
    same(lambda x, M: x[0][M.from_numpy(np.array([[True, False, True, False, False]] * 4))])
    with pytest.raises(IndexError):
        T(X3)[[0, 7]]


def test_to_accepts_dtype_tags_strings_and_numpy_types():
    x = T(np.ones(2, dtype=np.float32))
    assert x.to(T(np.ones(1)).dtype).dtype == "f64"
    assert x.to("float64").dtype == "f64" and x.to(np.float64).dtype == "f64" and x.to(np.dtype("int32")).dtype == "i32"
    assert x.to("cpu").dtype == "f32"


def test_triangle_indices_are_one_2xn_tensor():
    for f in ("triu_indices", "tril_indices"):
        np.testing.assert_array_equal(A(getattr(ox, f)(4, 5, 1)), getattr(torch, f)(4, 5, 1).numpy())


def test_zero_d_results_stay_zero_d_through_numpy_backed_functions():
    assert tuple(ox.from_numpy(np.array(3.0)).sum().shape) == ()
    assert tuple(ox.nanmean(T(X3)).shape) == ()
    assert tuple(ox.count_nonzero(T(X3)).shape) == ()
