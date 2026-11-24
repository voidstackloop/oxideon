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


def test_softmax_last_axis_second_order_grad_matches_pytorch():
    rng = np.random.RandomState(7)
    x_np = rng.randn(3, 5).astype(np.float64)
    v_np = rng.randn(3, 5).astype(np.float64)
    x = T(x_np).requires_grad_(True)
    v = T(v_np)
    y = x.softmax(-1)
    g1 = ox.autograd.grad((y * y).sum(), x, create_graph=True)[0]
    g2 = ox.autograd.grad((g1 * v).sum(), x)[0]

    px = torch.from_numpy(x_np.copy()).requires_grad_(True)
    pv = torch.from_numpy(v_np.copy())
    py = torch.softmax(px, dim=-1)
    pg1 = torch.autograd.grad((py * py).sum(), px, create_graph=True)[0]
    pg2 = torch.autograd.grad((pg1 * pv).sum(), px)[0]
    np.testing.assert_allclose(A(g2), pg2.numpy(), atol=1e-10, rtol=1e-10)


def test_matmul_mismatch_error_reports_distinct_shapes():
    a = ox.zeros(5, 4)
    b = ox.zeros(3, 2)
    with pytest.raises(Exception, match=r"inner dims 4 != 3"):
        _ = a @ b


def test_optimizers_sgd_adam_adamw_flags_match_pytorch():
    rng = np.random.RandomState(11)
    w0 = rng.randn(4, 3).astype(np.float32)
    g_seq = [rng.randn(4, 3).astype(np.float32) for _ in range(4)]

    # SGD with nesterov / maximize
    for kwargs in [
        dict(lr=0.1, momentum=0.9, dampening=0.0, nesterov=True),
        dict(lr=0.1, momentum=0.5, dampening=0.25, nesterov=False, maximize=True),
    ]:
        ow = T(w0.copy()).requires_grad_(True)
        pw = torch.from_numpy(w0.copy()).requires_grad_(True)
        oopt = ox.optim.SGD([ow], **kwargs)
        popt = torch.optim.SGD([pw], **kwargs)
        for g in g_seq:
            oopt.zero_grad()
            popt.zero_grad()
            (ow * T(g)).sum().backward()
            (pw * torch.from_numpy(g)).sum().backward()
            oopt.step()
            popt.step()
        np.testing.assert_allclose(A(ow), pw.detach().numpy(), atol=1e-5)

    # Adam / AdamW with amsgrad, maximize, betas, eps
    for ocls, pcls in [(ox.optim.Adam, torch.optim.Adam), (ox.optim.AdamW, torch.optim.AdamW)]:
        ow = T(w0.copy()).requires_grad_(True)
        pw = torch.from_numpy(w0.copy()).requires_grad_(True)
        kwargs = dict(lr=1e-2, betas=(0.85, 0.99), eps=1e-6, amsgrad=True, maximize=True)
        oopt = ocls([ow], **kwargs)
        popt = pcls([pw], **kwargs)
        for g in g_seq:
            oopt.zero_grad()
            popt.zero_grad()
            (ow * T(g)).sum().backward()
            (pw * torch.from_numpy(g)).sum().backward()
            oopt.step()
            popt.step()
        np.testing.assert_allclose(A(ow), pw.detach().numpy(), atol=1e-5)


def test_addmm_baddbmm_addmv_addr_differentiable_and_beta_zero():
    rng = np.random.RandomState(13)
    m_np = rng.randn(3, 4)
    a_np = rng.randn(3, 5)
    b_np = rng.randn(5, 4)
    for beta in (0.0, 0.7):
        om = T(m_np.copy()).requires_grad_(True)
        oa = T(a_np.copy()).requires_grad_(True)
        ob = T(b_np.copy()).requires_grad_(True)
        pm = torch.from_numpy(m_np.copy()).requires_grad_(True)
        pa = torch.from_numpy(a_np.copy()).requires_grad_(True)
        pb = torch.from_numpy(b_np.copy()).requires_grad_(True)

        oy = ox.addmm(om, oa, ob, beta=beta, alpha=1.5)
        py = torch.addmm(pm, pa, pb, beta=beta, alpha=1.5)
        np.testing.assert_allclose(A(oy), py.detach().numpy(), atol=1e-12)
        (oy * oy).sum().backward()
        (py * py).sum().backward()
        np.testing.assert_allclose(A(oa.grad), pa.grad.numpy(), atol=1e-11)
        np.testing.assert_allclose(A(ob.grad), pb.grad.numpy(), atol=1e-11)
        if beta == 0.0:
            assert om.grad is None or np.allclose(A(om.grad), 0.0)
        else:
            np.testing.assert_allclose(A(om.grad), pm.grad.numpy(), atol=1e-11)


def test_losses_and_rms_norm_match_pytorch():
    import oxideon.nn.functional as F
    import torch.nn.functional as tF

    rng = np.random.RandomState(19)
    # rms_norm default eps matches finfo(dtype).eps
    for dtype in (np.float32, np.float64):
        x_np = rng.randn(2, 4).astype(dtype)
        w_np = rng.randn(4).astype(dtype)
        ox_x = T(x_np.copy()).requires_grad_(True)
        pt_x = torch.from_numpy(x_np.copy()).requires_grad_(True)
        oy = F.rms_norm(ox_x, (4,), weight=T(w_np.copy()))
        py = tF.rms_norm(pt_x, (4,), weight=torch.from_numpy(w_np.copy()))
        np.testing.assert_allclose(A(oy), py.detach().numpy(), atol=1e-6)
        (oy * oy).sum().backward()
        (py * py).sum().backward()
        np.testing.assert_allclose(A(ox_x.grad), pt_x.grad.numpy(), atol=1e-5)

    # huber_loss, kl_div, nll_loss reductions & options
    inp_np = rng.randn(3, 5).astype(np.float32)
    tgt_np = rng.randn(3, 5).astype(np.float32)
    for red in ("none", "mean", "sum"):
        ox_i = T(inp_np.copy()).requires_grad_(True)
        pt_i = torch.from_numpy(inp_np.copy()).requires_grad_(True)
        oy = F.huber_loss(ox_i, T(tgt_np.copy()), reduction=red, delta=1.2)
        py = tF.huber_loss(pt_i, torch.from_numpy(tgt_np.copy()), reduction=red, delta=1.2)
        np.testing.assert_allclose(A(oy), py.detach().numpy(), atol=1e-5)
        oy.sum().backward()
        py.sum().backward()
        np.testing.assert_allclose(A(ox_i.grad), pt_i.grad.numpy(), atol=1e-5)

    log_p = tF.log_softmax(torch.from_numpy(inp_np), dim=-1).numpy()
    log_q = tF.log_softmax(torch.from_numpy(tgt_np), dim=-1).numpy()
    q_prob = np.exp(log_q)
    for red in ("none", "mean", "sum", "batchmean"):
        oy = F.kl_div(T(log_p.copy()), T(log_q.copy()), reduction=red, log_target=True)
        py = tF.kl_div(torch.from_numpy(log_p.copy()), torch.from_numpy(log_q.copy()), reduction=red, log_target=True)
        np.testing.assert_allclose(A(oy), py.detach().numpy(), atol=1e-5)
        oy2 = F.kl_div_loss(T(log_p.copy()), T(q_prob.copy()), reduction=red)
        py2 = tF.kl_div(torch.from_numpy(log_p.copy()), torch.from_numpy(q_prob.copy()), reduction=red)
        np.testing.assert_allclose(A(oy2), py2.detach().numpy(), atol=1e-5)

    targets = np.array([1, -100, 4], dtype=np.int64)
    wt = np.array([1.0, 2.0, 0.5, 1.5, 3.0], dtype=np.float32)
    for red in ("none", "mean", "sum"):
        ox_i = T(log_p.copy()).requires_grad_(True)
        pt_i = torch.from_numpy(log_p.copy()).requires_grad_(True)
        oy = F.nll_loss(ox_i, T(targets.copy()), weight=T(wt.copy()), ignore_index=-100, reduction=red)
        py = tF.nll_loss(pt_i, torch.from_numpy(targets.copy()), weight=torch.from_numpy(wt.copy()), ignore_index=-100, reduction=red)
        np.testing.assert_allclose(A(oy), py.detach().numpy(), atol=1e-5)
        oy.sum().backward()
        py.sum().backward()
        np.testing.assert_allclose(A(ox_i.grad), pt_i.grad.numpy(), atol=1e-5)


def test_fft_accepts_native_complex_tensors():
    rng = np.random.RandomState(23)
    c = (rng.randn(4, 8) + 1j * rng.randn(4, 8)).astype(np.complex64)
    oy = A(ox.fft.fft(T(c)))
    py = torch.fft.fft(torch.from_numpy(c))
    np.testing.assert_allclose(oy[..., 0] + 1j * oy[..., 1], py.numpy(), atol=1e-4)


def test_py_layers_state_dict_conv2d_norms_pools_and_rnns():
    import oxideon.nn as nn
    import oxideon.nn.functional as F
    import torch.nn as tnn
    import torch.nn.functional as tF

    for layer in [
        nn.Linear(4, 3),
        nn.Embedding(6, 4),
        nn.LayerNorm(4),
        nn.Conv2d(4, 6, (3, 2), stride=(1, 2), padding=(1, 0), dilation=(1, 1), groups=2, bias=False),
        nn.Conv3d(2, 3, 2),
    ]:
        sd = layer.state_dict()
        assert set(sd.keys()) == set(dict(layer.named_parameters()).keys())
        layer.load_state_dict(sd)
        layer.train()
        layer.eval()
        layer.zero_grad()

    with pytest.raises(Exception):
        nn.Linear(4, 3).load_state_dict({"weight": ox.zeros(2, 4), "bias": ox.zeros(3)})

    rng = np.random.RandomState(29)
    x_np = rng.randn(2, 4, 5, 5).astype(np.float32)
    bn = nn.BatchNorm2d(4)
    assert tuple(bn.weight.shape) == (4,) and tuple(bn.bias.shape) == (4,)
    assert tuple(bn.running_mean.shape) == (4,) and tuple(bn.running_var.shape) == (4,)
    gn = nn.GroupNorm(2, 4)
    assert tuple(gn.weight.shape) == (4,) and tuple(gn.bias.shape) == (4,)

    assert np.allclose(
        A(F.max_pool2d(T(x_np), 2, stride=2, ceil_mode=True)),
        tF.max_pool2d(torch.from_numpy(x_np), 2, stride=2, ceil_mode=True).numpy(),
        atol=1e-6,
    )
    assert np.allclose(
        A(F.avg_pool2d(T(x_np), 3, stride=2, padding=1, ceil_mode=True, count_include_pad=False)),
        tF.avg_pool2d(torch.from_numpy(x_np), 3, stride=2, padding=1, ceil_mode=True, count_include_pad=False).numpy(),
        atol=1e-6,
    )

    inp_np = rng.randn(2, 4, 3).astype(np.float32)
    for cls, tcls, extra in [
        (nn.LSTM, tnn.LSTM, {}),
        (nn.GRU, tnn.GRU, {}),
        (nn.RNN, tnn.RNN, {"nonlinearity": "tanh"}),
    ]:
        om = cls(3, 5, batch_first=True, **extra)
        tm = tcls(3, 5, batch_first=True, **extra)
        sd = {}
        for pk, pv in tm.state_dict().items():
            base, layer = pk.rsplit("_l", 1)
            sd[f"cells.{layer}.{base}"] = T(pv.detach().numpy().copy())
        om.load_state_dict(sd)
        ox_i = T(inp_np.copy()).requires_grad_(True)
        pt_i = torch.from_numpy(inp_np.copy()).requires_grad_(True)
        oo, _ = om(ox_i)
        to, _ = tm(pt_i)
        np.testing.assert_allclose(A(oo), to.detach().numpy(), atol=1e-5)
        (oo * oo).sum().backward()
        (to * to).sum().backward()
        np.testing.assert_allclose(A(ox_i.grad), pt_i.grad.numpy(), atol=1e-5)
        for pk, tp in tm.named_parameters():
            base, layer = pk.rsplit("_l", 1)
            ok = f"cells.{layer}.{base}"
            np.testing.assert_allclose(A(dict(om.named_parameters())[ok].grad), tp.grad.numpy(), atol=1e-5)


def test_softmax_and_log_softmax_empty_and_zero_d():
    for fn in ("softmax", "log_softmax"):
        oe = T(np.empty((3, 0), dtype=np.float32)).requires_grad_(True)
        oy = getattr(oe, fn)(-1)
        assert tuple(oy.shape) == (3, 0)
        oy.sum().backward()
        assert tuple(oe.grad.shape) == (3, 0)

        for ax in (-1, 0):
            o0 = T(np.array(2.0, dtype=np.float32)).requires_grad_(True)
            p0 = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
            oy0 = getattr(o0, fn)(ax)
            py0 = getattr(p0, fn)(ax)
            np.testing.assert_allclose(A(oy0), py0.detach().numpy())
            oy0.backward()
            py0.backward()
            np.testing.assert_allclose(A(o0.grad), p0.grad.numpy())


def test_scatter_reduce_gradients_all_modes():
    for red in ("sum", "mean", "prod", "amax", "amin"):
        for inc in (True, False):
            x_np = np.array([2.0 if inc else -1.0, 1.0, 3.0, 5.0], dtype=np.float64)
            s_np = np.array([2.0, 2.0, 0.0, 4.0], dtype=np.float64)
            idx_np = np.array([0, 0, 1, 2], dtype=np.int64)
            w_np = np.array([1.2, 0.7, 2.1, -0.5], dtype=np.float64)

            ox_x = T(x_np.copy()).requires_grad_(True)
            ox_s = T(s_np.copy()).requires_grad_(True)
            pt_x = torch.from_numpy(x_np.copy()).requires_grad_(True)
            pt_s = torch.from_numpy(s_np.copy()).requires_grad_(True)

            oy = ox_x.scatter_reduce(0, T(idx_np), ox_s, reduce=red, include_self=inc)
            py = pt_x.scatter_reduce(0, torch.from_numpy(idx_np), pt_s, reduce=red, include_self=inc)
            np.testing.assert_allclose(A(oy), py.detach().numpy(), atol=1e-12)
            (oy * T(w_np)).sum().backward()
            (py * torch.from_numpy(w_np)).sum().backward()
            np.testing.assert_allclose(A(ox_x.grad), pt_x.grad.numpy(), atol=1e-12)
            np.testing.assert_allclose(A(ox_s.grad), pt_s.grad.numpy(), atol=1e-12)


def test_einsum_ellipsis_and_gradients():
    rng = np.random.RandomState(31)
    a_np = rng.randn(1, 2, 3, 4)
    b_np = rng.randn(5, 2, 4, 3)
    m_np = rng.randn(2, 4, 4)
    for eq, arrs in [
        ("...ij,...jk->...ik", [a_np, b_np]),
        ("...ij,...jk", [a_np, b_np]),
        ("...ii->...i", [m_np]),
        ("...ii", [m_np]),
    ]:
        ox_ts = [T(x.copy()).requires_grad_(True) for x in arrs]
        pt_ts = [torch.from_numpy(x.copy()).requires_grad_(True) for x in arrs]
        oy = ox.einsum(eq, *ox_ts)
        py = torch.einsum(eq, *pt_ts)
        np.testing.assert_allclose(A(oy), py.detach().numpy(), atol=1e-12)
        (oy * oy).sum().backward()
        (py * py).sum().backward()
        for ot, pt in zip(ox_ts, pt_ts):
            np.testing.assert_allclose(A(ot.grad), pt.grad.numpy(), atol=1e-11)


def test_cross_entropy_and_five_losses_full_parity():
    import oxideon.nn.functional as F
    import torch.nn.functional as tF

    rng = np.random.RandomState(37)
    logits = rng.randn(4, 5).astype(np.float32)
    targets = np.array([0, 2, -100, 4], dtype=np.int64)
    weight = np.array([1.0, 2.0, 0.5, 1.5, 0.8], dtype=np.float32)
    for red in ("none", "mean", "sum"):
        ox_l = T(logits.copy()).requires_grad_(True)
        pt_l = torch.from_numpy(logits.copy()).requires_grad_(True)
        oy = F.cross_entropy(ox_l, T(targets), weight=T(weight), ignore_index=-100, label_smoothing=0.15, reduction=red)
        py = tF.cross_entropy(pt_l, torch.from_numpy(targets), weight=torch.from_numpy(weight), ignore_index=-100, label_smoothing=0.15, reduction=red)
        np.testing.assert_allclose(A(oy), py.detach().numpy(), atol=1e-5)
        oy.sum().backward()
        py.sum().backward()
        np.testing.assert_allclose(A(ox_l.grad), pt_l.grad.numpy(), atol=1e-5)

    logits3d = rng.randn(2, 5, 4).astype(np.float32)
    targets2d = np.array([[0, 1, -100, 4], [2, 3, 1, 0]], dtype=np.int64)
    np.testing.assert_allclose(
        A(F.cross_entropy(T(logits3d), T(targets2d))),
        tF.cross_entropy(torch.from_numpy(logits3d), torch.from_numpy(targets2d)).numpy(),
        atol=1e-5,
    )
    probs3d = tF.softmax(torch.from_numpy(rng.randn(2, 5, 4).astype(np.float32)), dim=1).numpy()
    np.testing.assert_allclose(
        A(F.cross_entropy(T(logits3d), T(probs3d), weight=T(weight), label_smoothing=0.1)),
        tF.cross_entropy(torch.from_numpy(logits3d), torch.from_numpy(probs3d), weight=torch.from_numpy(weight), label_smoothing=0.1).numpy(),
        atol=1e-5,
    )

    x1 = rng.randn(4, 6).astype(np.float32)
    x2 = rng.randn(4, 6).astype(np.float32)
    x3 = rng.randn(4, 6).astype(np.float32)
    y_pm = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    for red in ("none", "mean", "sum"):
        np.testing.assert_allclose(A(F.cosine_embedding_loss(T(x1), T(x2), T(y_pm), reduction=red)), tF.cosine_embedding_loss(torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(y_pm), reduction=red).numpy(), atol=1e-5)
        np.testing.assert_allclose(A(F.triplet_margin_loss(T(x1), T(x2), T(x3), swap=True, reduction=red)), tF.triplet_margin_loss(torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), swap=True, reduction=red).numpy(), atol=1e-5)
        np.testing.assert_allclose(A(F.margin_ranking_loss(T(x1[:, 0]), T(x2[:, 0]), T(y_pm), reduction=red)), tF.margin_ranking_loss(torch.from_numpy(x1[:, 0]), torch.from_numpy(x2[:, 0]), torch.from_numpy(y_pm), reduction=red).numpy(), atol=1e-5)
        np.testing.assert_allclose(A(F.hinge_embedding_loss(T(x1[:, 0]), T(y_pm), reduction=red)), tF.hinge_embedding_loss(torch.from_numpy(x1[:, 0]), torch.from_numpy(y_pm), reduction=red).numpy(), atol=1e-5)
        np.testing.assert_allclose(A(F.soft_margin_loss(T(x1), T(np.sign(x2)), reduction=red)), tF.soft_margin_loss(torch.from_numpy(x1), torch.from_numpy(np.sign(x2)), reduction=red).numpy(), atol=1e-5)


def test_rmsnorm_mha_transformer_and_pylayer_module_interface():
    import oxideon.nn as nn
    import torch.nn as tnn

    rng = np.random.RandomState(41)
    x_np = rng.randn(2, 5, 8).astype(np.float32)
    om_rms = nn.RMSNorm(8)
    tm_rms = tnn.RMSNorm(8)
    ox_i = T(x_np.copy()).requires_grad_(True)
    pt_i = torch.from_numpy(x_np.copy()).requires_grad_(True)
    oy = om_rms(ox_i)
    py = tm_rms(pt_i)
    np.testing.assert_allclose(A(oy), py.detach().numpy(), atol=1e-5)
    oy.sum().backward()
    py.sum().backward()
    np.testing.assert_allclose(A(ox_i.grad), pt_i.grad.numpy(), atol=1e-5)

    mask = np.triu(np.full((5, 5), -np.inf, dtype=np.float32), k=1)
    kpm = np.array([[False, False, False, True, True], [False, False, True, True, True]])
    om = nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    tm = tnn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    tsd = tm.state_dict()
    osd = {}
    for k, v in tsd.items():
        arr = v.detach().numpy().copy()
        if k == "self_attn.in_proj_weight":
            osd["self_attn.q_proj.weight"] = T(arr[:8].copy())
            osd["self_attn.k_proj.weight"] = T(arr[8:16].copy())
            osd["self_attn.v_proj.weight"] = T(arr[16:].copy())
        elif k == "self_attn.in_proj_bias":
            osd["self_attn.q_proj.bias"] = T(arr[:8].copy())
            osd["self_attn.k_proj.bias"] = T(arr[8:16].copy())
            osd["self_attn.v_proj.bias"] = T(arr[16:].copy())
        else:
            osd[k] = T(arr)
    om.load_state_dict(osd)
    oo = om(T(x_np.copy()), src_mask=T(mask), src_key_padding_mask=T(kpm))
    po = tm(torch.from_numpy(x_np.copy()), src_mask=torch.from_numpy(mask), src_key_padding_mask=torch.from_numpy(kpm))
    np.testing.assert_allclose(A(oo), po.detach().numpy(), atol=1e-5)

    enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True), num_layers=2)
    assert enc.layers[0] is not enc.layers[1]

    for cls_name, args, sample in [
        ("Dropout", (0.0,), ox.ones(2, 4)),
        ("DropPath", (0.0,), ox.ones(2, 4)),
        ("Maxout", (2,), ox.ones(2, 4)),
        ("Snake", (4,), ox.ones(2, 4)),
        ("Swish", (), ox.ones(2, 4)),
        ("Upsample2d", ((4, 4),), ox.ones(1, 2, 2, 2)),
        ("ReLU", (), ox.ones(2, 4)),
        ("ReLU6", (), ox.ones(2, 4)),
        ("Hardtanh", (), ox.ones(2, 4)),
        ("Softshrink", (), ox.ones(2, 4)),
        ("Hardshrink", (), ox.ones(2, 4)),
        ("LogSigmoid", (), ox.ones(2, 4)),
        ("MLP", ([4, 3, 2],), ox.ones(2, 4)),
        ("SpectralNormLinear", (4, 3), ox.ones(2, 4)),
        ("WeightNormLinear", (4, 3), ox.ones(2, 4)),
        ("MaxPool1d", (2,), ox.ones(1, 2, 4)),
        ("AvgPool1d", (2,), ox.ones(1, 2, 4)),
        ("ConvTranspose3d", (2, 3, 2), ox.ones(1, 2, 3, 3, 3)),
        ("EmbeddingBag", (6, 4), T(np.array([[0, 1], [2, 3]], dtype=np.int64))),
    ]:
        cls = getattr(nn, cls_name, None) or getattr(ox, cls_name)
        layer = cls(*args)
        _ = layer(sample)
        sd = layer.state_dict()
        assert set(sd.keys()) == set(dict(layer.named_parameters()).keys()), cls_name
        layer.load_state_dict(sd)
        layer.train()
        layer.eval()
        layer.zero_grad()


def test_gelu_approximate_and_padding_same_valid():
    import oxideon.nn as nn
    import oxideon.nn.functional as F
    import torch.nn as tnn
    import torch.nn.functional as tF

    rng = np.random.RandomState(43)
    x_np = rng.randn(3, 8).astype(np.float64)
    for approx in ("none", "tanh"):
        ox_x = T(x_np.copy()).requires_grad_(True)
        pt_x = torch.from_numpy(x_np.copy()).requires_grad_(True)
        oy = F.gelu(ox_x, approximate=approx)
        py = tF.gelu(pt_x, approximate=approx)
        np.testing.assert_allclose(A(oy), py.detach().numpy(), atol=1e-6)
        (oy * oy).sum().backward()
        (py * py).sum().backward()
        np.testing.assert_allclose(A(ox_x.grad), pt_x.grad.numpy(), atol=1e-5)
        assert tuple(nn.GELU(approximate=approx)(ox_x).shape) == (3, 8)

    x4 = rng.randn(2, 4, 7, 7).astype(np.float32)
    x3 = x4[:, :, 0, :]
    for ks, pad in [(3, "same"), (2, "same"), (3, "valid")]:
        om2 = nn.Conv2d(4, 5, ks, padding=pad)
        tm2 = tnn.Conv2d(4, 5, ks, padding=pad)
        om2.load_state_dict({"weight": T(tm2.weight.detach().numpy().copy()), "bias": T(tm2.bias.detach().numpy().copy())})
        ox_i = T(x4.copy()).requires_grad_(True)
        pt_i = torch.from_numpy(x4.copy()).requires_grad_(True)
        oy = om2(ox_i)
        py = tm2(pt_i)
        np.testing.assert_allclose(A(oy), py.detach().numpy(), atol=1e-5)
        oy.sum().backward()
        py.sum().backward()
        np.testing.assert_allclose(A(ox_i.grad), pt_i.grad.numpy(), atol=1e-5)

        om1 = nn.Conv1d(4, 5, ks, padding=pad)
        tm1 = tnn.Conv1d(4, 5, ks, padding=pad)
        om1.load_state_dict({"weight": T(tm1.weight.detach().numpy().copy()), "bias": T(tm1.bias.detach().numpy().copy())})
        ox_1 = T(x3.copy()).requires_grad_(True)
        pt_1 = torch.from_numpy(x3.copy()).requires_grad_(True)
        oy1 = om1(ox_1)
        py1 = tm1(pt_1)
        np.testing.assert_allclose(A(oy1), py1.detach().numpy(), atol=1e-5)


def test_conv_transpose_groups_batchnorm_track_lr_plateau_and_embedding_padding_idx():
    import oxideon.nn as nn
    import torch.nn as tnn

    rng = np.random.RandomState(47)
    x4 = rng.randn(2, 4, 5, 5).astype(np.float32)
    om_ct2 = nn.ConvTranspose2d(4, 6, 3, dilation=2, groups=2, bias=False)
    tm_ct2 = tnn.ConvTranspose2d(4, 6, 3, dilation=2, groups=2, bias=False)
    om_ct2.load_state_dict({"weight": T(tm_ct2.weight.detach().numpy().copy())})
    ox_i = T(x4.copy()).requires_grad_(True)
    pt_i = torch.from_numpy(x4.copy()).requires_grad_(True)
    oy = om_ct2(ox_i)
    py = tm_ct2(pt_i)
    np.testing.assert_allclose(A(oy), py.detach().numpy(), atol=1e-5)
    oy.sum().backward()
    py.sum().backward()
    np.testing.assert_allclose(A(ox_i.grad), pt_i.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(A(om_ct2.weight.grad), tm_ct2.weight.grad.numpy(), atol=1e-5)

    for cls, inp in [
        (nn.BatchNorm1d, x4[:, :, 0, :]),
        (nn.BatchNorm2d, x4),
        (nn.BatchNorm3d, rng.randn(2, 4, 3, 3, 3).astype(np.float32)),
    ]:
        m = cls(4, affine=False, track_running_stats=False)
        assert m.running_mean is None and m.running_var is None
        m.eval()
        assert tuple(m(T(inp)).shape) == inp.shape

    ow = ox.ones(2, requires_grad=True)
    pw = torch.ones(2, requires_grad=True)
    oo = ox.optim.SGD([ow], lr=0.05)
    po = torch.optim.SGD([pw], lr=0.05)
    os_ = ox.optim.lr_scheduler.ReduceLROnPlateau(oo, factor=0.5, patience=2)
    ps_ = torch.optim.lr_scheduler.ReduceLROnPlateau(po, factor=0.5, patience=2)
    for _ in range(8):
        oo.step()
        po.step()
        os_.step(1.0)
        ps_.step(1.0)
        np.testing.assert_allclose(os_.get_last_lr()[0], ps_.get_last_lr()[0], rtol=1e-5)
        np.testing.assert_allclose(oo.param_groups[0]["lr"], po.param_groups[0]["lr"], rtol=1e-5)

    res = ox.zeros(4).scatter_reduce(0, T(np.array([0, 1], dtype=np.int64)), T(np.array([3.0, 4.0], dtype=np.float32)), "sum", True)
    np.testing.assert_allclose(A(res), [3.0, 4.0, 0.0, 0.0])

    oe = nn.Embedding(6, 4, padding_idx=0)
    np.testing.assert_allclose(A(oe.weight)[0], np.zeros(4))
    pe = tnn.Embedding(6, 4, padding_idx=0)
    oe.load_state_dict({"weight": T(pe.weight.detach().numpy().copy())})
    idx = np.array([[0, 2, 0, 5], [1, 0, 3, 4]], dtype=np.int64)
    oy_e = oe(T(idx))
    py_e = pe(torch.from_numpy(idx))
    np.testing.assert_allclose(A(oy_e), py_e.detach().numpy(), atol=1e-5)
    (oy_e * oy_e).sum().backward()
    (py_e * py_e).sum().backward()
    np.testing.assert_allclose(A(oe.weight.grad), pe.weight.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(A(oe.weight.grad)[0], np.zeros(4))


def test_squeeze_tuple_clamp_tensor_layernorm_affine_bias_utils_and_opt_state_dict():
    import os
    import tempfile
    import oxideon.nn as nn
    import torch.nn as tnn

    t = ox.ones(1, 3, 1, 4, requires_grad=True)
    s = t.squeeze((0, 2))
    assert tuple(s.shape) == (3, 4)
    (s * 2.0).sum().backward()
    np.testing.assert_allclose(A(t.grad), np.full((1, 3, 1, 4), 2.0))

    rng = np.random.RandomState(53)
    x_np = rng.randn(3, 4)
    lo_np = rng.uniform(-1.0, 0.0, (1, 4))
    hi_np = rng.uniform(0.1, 1.0, (3, 4))
    ox_x = T(x_np.copy()).requires_grad_(True)
    pt_x = torch.from_numpy(x_np.copy()).requires_grad_(True)
    oy = ox.clamp(ox_x, min=T(lo_np), max=T(hi_np))
    py = torch.clamp(pt_x, min=torch.from_numpy(lo_np), max=torch.from_numpy(hi_np))
    np.testing.assert_allclose(A(oy), py.detach().numpy(), atol=1e-12)
    (oy * oy).sum().backward()
    (py * py).sum().backward()
    np.testing.assert_allclose(A(ox_x.grad), pt_x.grad.numpy(), atol=1e-12)

    x2 = rng.randn(2, 6).astype(np.float32)
    ln_nb = nn.LayerNorm(6, bias=False)
    tln_nb = tnn.LayerNorm(6, bias=False)
    assert ln_nb.bias is None and list(ln_nb.state_dict().keys()) == ["weight"]
    ox_x2 = T(x2.copy()).requires_grad_(True)
    pt_x2 = torch.from_numpy(x2.copy()).requires_grad_(True)
    oy2 = ln_nb(ox_x2)
    py2 = tln_nb(pt_x2)
    np.testing.assert_allclose(A(oy2), py2.detach().numpy(), atol=1e-5)
    (oy2 * oy2).sum().backward()
    (py2 * py2).sum().backward()
    np.testing.assert_allclose(A(ox_x2.grad), pt_x2.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(A(ln_nb.weight.grad), tln_nb.weight.grad.numpy(), atol=1e-5)

    ln_na = nn.LayerNorm(6, elementwise_affine=False)
    tln_na = tnn.LayerNorm(6, elementwise_affine=False)
    assert ln_na.weight is None and ln_na.bias is None and ln_na.state_dict() == {}
    np.testing.assert_allclose(A(ln_na(T(x2.copy()))), tln_na(torch.from_numpy(x2.copy())).detach().numpy(), atol=1e-5)

    seq = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    vec = nn.utils.parameters_to_vector(seq.parameters())
    assert tuple(vec.shape) == (4 * 3 + 3 + 3 * 2 + 2,)
    nn.utils.vector_to_parameters(vec * 0.0 + 0.5, seq.parameters())
    for p in seq.parameters():
        np.testing.assert_allclose(A(p), 0.5)

    for ocls in (ox.optim.SGD, ox.optim.Adam, ox.optim.AdamW):
        kwargs = dict(lr=0.05, momentum=0.9) if ocls is ox.optim.SGD else dict(lr=0.01, amsgrad=True)
        w_a = T(rng.randn(4, 3).astype(np.float32)).requires_grad_(True)
        w_b = T(A(w_a).copy()).requires_grad_(True)
        g1 = T(rng.randn(4, 3).astype(np.float32))
        g2 = T(rng.randn(4, 3).astype(np.float32))
        opt_a = ocls([w_a], **kwargs)
        (w_a * g1).sum().backward()
        opt_a.step()

        with tempfile.TemporaryDirectory() as td:
            ckpt = os.path.join(td, "ckpt.ox")
            ox.save({"w": w_a, "opt": opt_a.state_dict()}, ckpt)
            loaded = ox.load(ckpt)
            with ox.no_grad():
                w_b.copy_(loaded["w"])
            opt_b = ocls([w_b], **kwargs)
            opt_b.load_state_dict(loaded["opt"])

        opt_a.zero_grad()
        opt_b.zero_grad()
        (w_a * g2).sum().backward()
        (w_b * g2).sum().backward()
        opt_a.step()
        opt_b.step()
        np.testing.assert_allclose(A(w_a), A(w_b), atol=1e-7)
        assert len(opt_b.param_groups) == 1

    for cls, inp in [
        (nn.InstanceNorm1d, rng.randn(2, 4, 6).astype(np.float32)),
        (nn.InstanceNorm2d, rng.randn(2, 4, 5, 5).astype(np.float32)),
        (nn.InstanceNorm3d, rng.randn(2, 4, 3, 3, 3).astype(np.float32)),
    ]:
        m = cls(4, affine=True, track_running_stats=True)
        assert m.running_mean is not None and tuple(m(T(inp)).shape) == inp.shape


def test_exact_jvp_hessian_and_persistent_buffers():
    from collections import OrderedDict
    import oxideon.nn as nn

    x_np = np.array([1.2, -0.7, 0.4], dtype=np.float64)
    v_np = np.array([0.5, 1.0, -0.3], dtype=np.float64)
    f_ox = lambda z: (z.sin() + z * z * 0.5).sum()
    f_pt = lambda z: (z.sin() + z * z * 0.5).sum()

    o1, j1 = ox.autograd.functional.jvp(f_ox, (T(x_np),), (T(v_np),))
    p1, pj1 = torch.autograd.functional.jvp(f_pt, (torch.from_numpy(x_np),), (torch.from_numpy(v_np),))
    np.testing.assert_allclose(A(o1), p1.numpy(), atol=1e-12)
    np.testing.assert_allclose(A(j1), pj1.numpy(), atol=1e-12)

    oh = ox.autograd.functional.hessian(f_ox, T(x_np))
    ph = torch.autograd.functional.hessian(f_pt, torch.from_numpy(x_np))
    assert oh.dtype == "f64"
    np.testing.assert_allclose(A(oh), ph.numpy(), atol=1e-12)
    np.testing.assert_allclose(A(ox.func.hessian(f_ox)(T(x_np))), ph.numpy(), atol=1e-12)

    f2_ox = lambda a, b: (a * b).sum() + (a ** 2).sum() * 0.5
    f2_pt = lambda a, b: (a * b).sum() + (a ** 2).sum() * 0.5
    oa = T(np.array([1.0, 2.0]))
    ob = T(np.array([3.0, 4.0]))
    pa = torch.tensor([1.0, 2.0], dtype=torch.float64)
    pb = torch.tensor([3.0, 4.0], dtype=torch.float64)
    h2o = ox.autograd.functional.hessian(f2_ox, (oa, ob))
    h2p = torch.autograd.functional.hessian(f2_pt, (pa, pb))
    for r in range(2):
        for c in range(2):
            np.testing.assert_allclose(A(h2o[r][c]), h2p[r][c].numpy(), atol=1e-12)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.wn = ox.WeightNormLinear(4, 3)
            self.register_buffer("perm", ox.ones(3), persistent=True)
            self.register_buffer("scratch", ox.zeros(3), persistent=False)

    m = Net()
    sd = m.state_dict()
    assert set(sd.keys()) == {"wn.weight_g", "wn.weight_v", "wn.bias", "perm"}
    assert "scratch" in dict(m.named_buffers())
    m.load_state_dict(sd)

    s = nn.Sequential(OrderedDict([("fc1", nn.Linear(4, 3)), ("act", nn.ReLU()), ("fc2", nn.Linear(3, 2))]))
    assert tuple(s(ox.ones(2, 4)).shape) == (2, 2)
    assert s[-1] is s.fc2 and len(s[:2]) == 2
