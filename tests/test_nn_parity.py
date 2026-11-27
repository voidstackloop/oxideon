"""nn / functional parity with PyTorch: convolutions (padding='same', groups, dilation, transposed), Module versions of Conv / LayerNorm /
Upsample / activations, and the Module API on the native layers."""
import numpy as np
import pytest

import oxideon as ox

torch = pytest.importorskip("torch")
F, TF = ox.functional, torch.nn.functional
rng = np.random.default_rng(11)


def O(a):
    return ox.from_numpy(np.ascontiguousarray(a))


def T(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def f32(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def close(got, want, tol=1e-4):
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_allclose(got.to_numpy(), want.detach().numpy(), rtol=tol, atol=tol)


@pytest.mark.parametrize("kw", [dict(padding="same"), dict(padding="valid"), dict(padding="same", dilation=2), dict(stride=(2, 1), padding=(1, 0))])
def test_conv2d_padding_modes(kw):
    x, w, b = f32(2, 4, 7, 7), f32(6, 4, 3, 3), f32(6)
    close(F.conv2d(O(x), O(w), O(b), **kw), TF.conv2d(T(x), T(w), T(b), **kw))


def test_conv_same_with_even_kernel_and_1d():
    x, w = f32(2, 4, 7, 7), f32(6, 4, 4, 2)
    close(F.conv2d(O(x), O(w), padding="same"), TF.conv2d(T(x), T(w), padding="same"))
    x1, w1 = f32(2, 4, 9), f32(3, 4, 3)
    for kw in (dict(padding="same"), dict(padding="valid"), dict(padding="same", dilation=2)):
        close(F.conv1d(O(x1), O(w1), **kw), TF.conv1d(T(x1), T(w1), **kw))


@pytest.mark.parametrize("kw", [dict(groups=2), dict(groups=2, dilation=2, padding=2), dict(padding="same", groups=2),
                                dict(stride=2, groups=2, dilation=(1, 2, 1), padding=1)])
def test_conv3d_groups_dilation_and_gradients(kw):
    x, w, b = f32(1, 4, 5, 5, 5), f32(6, 2, 3, 3, 3), f32(6)
    close(F.conv3d(O(x), O(w), O(b), **kw), TF.conv3d(T(x), T(w), T(b), **kw))
    xo, wo, xt, wt = O(x).requires_grad_(True), O(w).requires_grad_(True), T(x).requires_grad_(True), T(w).requires_grad_(True)
    F.conv3d(xo, wo, **kw).sum().backward()
    TF.conv3d(xt, wt, **kw).sum().backward()
    close(xo.grad, xt.grad, 1e-3)
    close(wo.grad, wt.grad, 1e-3)


@pytest.mark.parametrize("n,xs,ws,kw", [
    (1, (2, 4, 6), (4, 3, 3), dict(stride=2, padding=1, output_padding=1)), (1, (2, 4, 6), (4, 3, 3), dict(groups=2, dilation=2)),
    (2, (2, 4, 5, 5), (4, 3, 3, 3), dict(stride=2, padding=1, output_padding=1)), (2, (2, 4, 5, 5), (4, 3, 3, 3), dict(dilation=2, groups=2, stride=(2, 1))),
    (3, (1, 4, 3, 3, 3), (4, 3, 2, 2, 2), dict(groups=2, padding=1)),
])
def test_transposed_convolutions_use_native_kernels(n, xs, ws, kw):
    x, w, b = f32(*xs), f32(*ws), f32(ws[1] * kw.get("groups", 1))
    fo, ft = getattr(F, f"conv_transpose{n}d"), getattr(TF, f"conv_transpose{n}d")
    close(fo(O(x), O(w), O(b), **kw), ft(T(x), T(w), T(b), **kw))
    xo, wo, xt, wt = O(x).requires_grad_(True), O(w).requires_grad_(True), T(x).requires_grad_(True), T(w).requires_grad_(True)
    fo(xo, wo, **kw).sum().backward()
    ft(xt, wt, **kw).sum().backward()
    close(xo.grad, xt.grad, 1e-3)
    close(wo.grad, wt.grad, 1e-3)


def copy_weights(src, dst):
    for (_, a), (_, b) in zip(src.named_parameters(), dst.named_parameters()):
        b.assign(O(a.detach().numpy()))


@pytest.mark.parametrize("ctor,shape", [
    ("Conv2d(4, 6, 3, groups=2)", (2, 4, 6, 6)), ("Conv2d(2, 3, 3, dilation=2, padding=2, bias=False)", (1, 2, 8, 8)),
    ("Conv2d(3, 4, (3, 2), stride=(2, 1), padding=(1, 0))", (2, 3, 7, 6)), ("Conv2d(2, 3, 3, padding='same')", (1, 2, 5, 5)),
    ("Conv2d(2, 3, 3, padding=1, padding_mode='reflect')", (1, 2, 5, 5)), ("Conv2d(2, 3, 3, padding=1, padding_mode='circular')", (1, 2, 5, 5)),
    ("Conv1d(3, 4, 3, padding=1, padding_mode='replicate')", (2, 3, 7)), ("Conv3d(2, 3, 3, padding=1, groups=1)", (1, 2, 4, 4, 4)),
    ("LayerNorm((4, 5))", (3, 4, 5)), ("LayerNorm(5, elementwise_affine=False)", (3, 5)), ("LayerNorm(5, bias=False)", (3, 5)),
])
def test_modules_match_torch_forward_and_backward(ctor, shape):
    tl, ol = eval("torch.nn." + ctor), eval("ox.nn." + ctor)
    assert [tuple(p.shape) for _, p in tl.named_parameters()] == [tuple(p.shape) for _, p in ol.named_parameters()]
    copy_weights(tl, ol)
    x = f32(*shape)
    xo, xt = O(x).requires_grad_(True), T(x).requires_grad_(True)
    yo, yt = ol(xo), tl(xt)
    close(yo, yt)
    w = f32(*yt.shape)
    (yo * O(w)).sum().backward()
    (yt * T(w)).sum().backward()
    close(xo.grad, xt.grad, 1e-3)
    for (_, a), (_, b) in zip(tl.named_parameters(), ol.named_parameters()):
        close(b.grad, a.grad, 1e-3)


@pytest.mark.parametrize("ctor,shape", [
    ("Upsample(scale_factor=2)", (1, 2, 3, 3)), ("Upsample(scale_factor=2, mode='bilinear', align_corners=False)", (1, 2, 3, 3)),
    ("Upsample(size=(5, 6), mode='bilinear', align_corners=True)", (1, 2, 3, 3)), ("Upsample(scale_factor=2, mode='bicubic')", (1, 1, 4, 4)),
    ("Upsample(scale_factor=2, mode='linear')", (1, 2, 4)), ("Upsample(scale_factor=2, mode='trilinear')", (1, 1, 3, 3, 3)),
    ("Softplus(beta=2, threshold=5)", (3, 5)), ("Hardshrink(0.3)", (3, 5)), ("Softshrink(0.3)", (3, 5)), ("LogSigmoid()", (3, 5)),
])
def test_resampling_and_activation_modules(ctor, shape):
    x = f32(*shape)
    xo, xt = O(x).requires_grad_(True), T(x).requires_grad_(True)
    yo, yt = eval("ox.nn." + ctor)(xo), eval("torch.nn." + ctor)(xt)
    close(yo, yt)
    yo.sum().backward()
    yt.sum().backward()
    close(xo.grad, xt.grad)


def test_native_layers_have_the_module_api():
    lin = ox.nn.Linear(4, 3)
    assert [n for n, _ in lin.named_parameters()] == ["weight", "bias"] and list(lin.state_dict()) == ["weight", "bias"]
    other = ox.nn.Linear(4, 3)
    other.load_state_dict(lin.state_dict())
    np.testing.assert_array_equal(other.weight.to_numpy(), lin.weight.to_numpy())
    lin.requires_grad_(False)
    assert not lin.weight.requires_grad
    assert lin.eval() is lin and lin.zero_grad() is None
    assert list(ox.nn.Embedding(5, 3).state_dict()) == ["weight"]
    assert [tuple(p.shape) for p in ox.nn.InstanceNorm2d(3, affine=True).parameters()] == [(3,), (3,)]


def test_module_double_casts_python_module_parameters():
    c = ox.nn.Conv2d(2, 3, 3).double()
    assert c.weight.dtype == "f64" and c(ox.randn([1, 2, 5, 5]).double()).dtype == "f64"
    assert c.get_parameter("bias") is c.bias and c.get_submodule("") is c


@pytest.mark.parametrize("ctor,shape", [("BatchNorm1d(4)", (6, 4)), ("BatchNorm1d(4)", (3, 4, 5)), ("BatchNorm2d(3)", (4, 3, 5, 5)),
                                        ("BatchNorm2d(3, affine=False)", (4, 3, 4, 4)), ("BatchNorm3d(2)", (2, 2, 3, 3, 3)),
                                        ("BatchNorm2d(3, momentum=None)", (4, 3, 4, 4)), ("BatchNorm1d(4, track_running_stats=False)", (6, 4))])
def test_batchnorm_buffers_and_state_dict_match_torch(ctor, shape):
    t, o = eval("torch.nn." + ctor), eval("ox.nn." + ctor)
    for _ in range(3):
        x = f32(*shape)
        close(o(O(x)), t(T(x)))
    assert list(o.state_dict()) == list(t.state_dict())
    if t.track_running_stats:
        close(o.running_mean, t.running_mean, 1e-5)
        close(o.running_var, t.running_var, 1e-5)
        assert int(o.num_batches_tracked.item()) == int(t.num_batches_tracked)
    t.eval()
    o.eval()
    x = f32(*shape)
    close(o(O(x)), t(T(x)))


def test_lazy_batchnorm_initialises_buffers_on_first_call():
    layer = ox.nn.LazyBatchNorm2d()
    assert layer(ox.randn([2, 3, 4, 4])).shape == [2, 3, 4, 4]
    assert list(layer.state_dict()) == ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]
