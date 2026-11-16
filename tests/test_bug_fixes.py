"""
Unit tests confirming bug fixes and edge-case handling across Oxideon:
- Empty array safety in precision_recall_f1 and ToTensor
- Exact tie-handling in roc_auc_score (Wilcoxon-Mann-Whitney formulation)
- Per-sample row-wise L2 normalization in ArcFaceLoss
- Probability clamping in functional binary_cross_entropy
- 4D batched tensor support and zero-std protection in Normalize
- Exact root endpoint returns in bisection and 1D gradient extraction in newton_raphson
- Dataset size vs cluster validation in KMeans
- Terminal step alpha_bar handling in DDIMScheduler
- Zero-size array safety in amplitude_to_db
- Non-positive timestep rejection in diffusion schedulers
- Inference-steps-above-training rejection in diffusion set_timesteps
- Negative-k rejection in topk
- Clear error for grad() without inputs
- Torch-parity degenerate window sizes
- topk range errors and partition parity
"""
import numpy as np
import pytest
import warnings
import oxideon as ox
from oxideon import functional as F
from oxideon import metrics, transforms, losses, root, clustering, diffusion, audio
import oxideon.nn as nn
import oxideon.optim as optim
import oxideon.testing as testing
import oxideon.utils as utils
from oxideon.data import ConcatDataset

def test_precision_recall_f1_empty_inputs():
    res = metrics.precision_recall_f1([], [], average="macro")
    assert res == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    res_bin = metrics.precision_recall_f1([], [], average="binary")
    assert res_bin["precision"] == 0.0
    assert res_bin["recall"] == 0.0

def test_roc_auc_score_tie_handling():
    # When all predictions are tied, AUC should be exactly 0.5 (random guess)
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    targets = np.array([0, 0, 1, 1])
    auc = metrics.roc_auc_score(scores, targets)
    assert auc == 0.5

    # Empty inputs
    assert metrics.roc_auc_score([], []) == 0.5

    # Single class
    assert metrics.roc_auc_score([0.1, 0.9], [1, 1]) == 0.5

def test_arcface_per_sample_normalization():
    arcface = losses.ArcFaceLoss(in_features=8, out_features=4, s=30.0, m=0.5)
    embeddings = ox.randn([5, 8])
    labels = ox.from_numpy(np.array([0, 1, 2, 3, 0], dtype=np.int64))

    loss = arcface(embeddings, labels)
    assert float(loss.to_numpy().flat[0]) > 0.0

def test_binary_cross_entropy_clamping():
    # Input slightly beyond [0, 1] due to numerical float drift
    inp = ox.from_numpy(np.array([1.0001, -0.0001], dtype=np.float32))
    target = ox.from_numpy(np.array([1.0, 0.0], dtype=np.float32))

    # Should not produce NaN
    loss = F.binary_cross_entropy(inp, target)
    loss_val = float(loss.to_numpy().flat[0])
    assert not np.isnan(loss_val) and not np.isinf(loss_val)
    assert loss_val >= 0.0

def test_normalize_batched_4d_and_zero_std():
    # Batched 4D tensor: [B, C, H, W]
    img_4d = ox.randn([2, 3, 16, 16])
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    out = normalizer(img_4d)
    assert out.shape == [2, 3, 16, 16]

    # Zero std should not cause division by zero crash
    zero_std_normalizer = transforms.Normalize(mean=[0.5], std=[0.0])
    img_1d = ox.from_numpy(np.array([0.5, 1.0], dtype=np.float32))
    out_safe = zero_std_normalizer(img_1d)
    assert not np.isnan(out_safe.to_numpy()).any()

def test_totensor_empty_array():
    totensor = transforms.ToTensor()
    empty = np.array([], dtype=np.float32)
    t = totensor(empty)
    assert t.to_numpy().size == 0

def test_root_finding_exact_endpoints():
    # Root exactly at a
    res = root.bisection(lambda x: x - 2.0, a=2.0, b=5.0)
    assert res.converged is True
    assert res.root == 2.0
    assert res.iterations == 0

    # Newton raphson with 1D tensor
    res_nr = root.newton_raphson(lambda x: x * x - 4.0, x0=3.0)
    assert res_nr.converged is True
    assert np.isclose(res_nr.root, 2.0, atol=1e-5)

def test_kmeans_validation():
    km = clustering.KMeans(n_clusters=10)
    x_small = ox.randn([3, 4])  # 3 samples < 10 clusters
    with pytest.raises(ValueError, match="n_samples .* must be >= n_clusters"):
        km.fit(x_small)

def test_ddim_terminal_step():
    ddim = diffusion.DDIMScheduler(num_train_timesteps=100)
    ddim.set_timesteps(num_inference_steps=10)

    model_output = ox.randn([1, 4])
    sample = ox.randn([1, 4])

    # Reverse step at final step t=0
    out = ddim.step(model_output, timestep=0, sample=sample, eta=0.0)
    assert out.shape == [1, 4]
    assert not np.isnan(out.to_numpy()).any()

def test_amplitude_to_db_empty():
    empty_spec = ox.from_numpy(np.array([], dtype=np.float32))
    db = audio.amplitude_to_db(empty_spec)
    assert db.to_numpy().size == 0

def test_diffusion_schedulers_reject_nonpositive_timesteps():
    ctor_cases = (
        diffusion.DDPMScheduler,
        diffusion.DDIMScheduler,
        ox.EulerDiscreteScheduler,
        ox.DPMSolverMultistepScheduler,
        ox.DDIMScheduler,
    )
    for cls in ctor_cases:
        for bad in (0, -1, -100):
            with pytest.raises(ValueError, match="at least 1"):
                cls(num_train_timesteps=bad)

    schedulers = [
        diffusion.DDIMScheduler(num_train_timesteps=100),
        ox.EulerDiscreteScheduler(num_train_timesteps=100),
        ox.DPMSolverMultistepScheduler(num_train_timesteps=100),
        ox.DDIMScheduler(num_train_timesteps=100),
    ]
    for sched in schedulers:
        for bad in (0, -1, -100):
            with pytest.raises(ValueError, match="at least 1"):
                sched.set_timesteps(bad)

    for sched in schedulers:
        sched.set_timesteps(10)
        assert len(sched.timesteps) == 10

def test_diffusion_set_timesteps_rejects_more_than_training():
    schedulers = [
        diffusion.DDIMScheduler(num_train_timesteps=100),
        ox.EulerDiscreteScheduler(num_train_timesteps=100),
        ox.DPMSolverMultistepScheduler(num_train_timesteps=100),
        ox.DDIMScheduler(num_train_timesteps=100),
    ]
    for sched in schedulers:
        with pytest.raises(ValueError, match="exceeds"):
            sched.set_timesteps(101)
        with pytest.raises(ValueError, match="exceeds"):
            sched.set_timesteps(200)
    for sched in schedulers:
        sched.set_timesteps(100)
        assert len(sched.timesteps) == 100
        sched.set_timesteps(1)
        assert len(sched.timesteps) == 1

def test_topk_rejects_negative_k():
    x = ox.randn([4])
    with pytest.raises(ValueError, match="non-negative"):
        ox.topk(x, -1)
    with pytest.raises(ValueError, match="non-negative"):
        ox.topk(x, -100)
    vals, idx = ox.topk(x, 0)
    assert vals.to_numpy().size == 0
    assert idx.to_numpy().size == 0
    vals, idx = ox.topk(x, 2)
    assert vals.shape == [2]
    assert idx.shape == [2]

def test_grad_missing_inputs_is_a_clear_error():
    x = ox.randn([3]).requires_grad_(True)
    y = (x * 2).sum()
    with pytest.raises(TypeError, match="inputs"):
        ox.grad(y)
    grads = ox.grad(y, x)
    assert grads[0].shape == [3]

def test_grad_mode_contexts_match_torch():
    # torch-verified context behavior (nested no_grad/enable_grad/set_guards).
    # Deliberate divergence (not asserted here): a bare set_grad_enabled(mode)
    # call applies eagerly in torch but is pure here (pinned by
    # test_set_grad_enabled_construction_pure_and_nesting).
    assert ox.is_grad_enabled() is True
    with ox.no_grad():
        assert ox.is_grad_enabled() is False
        with ox.enable_grad():
            assert ox.is_grad_enabled() is True
        assert ox.is_grad_enabled() is False
    assert ox.is_grad_enabled() is True
    with ox.set_grad_enabled(False):
        assert ox.is_grad_enabled() is False
        with ox.set_grad_enabled(True):
            assert ox.is_grad_enabled() is True
        assert ox.is_grad_enabled() is False
    assert ox.is_grad_enabled() is True

def test_bitwise_logical_and_predicates_match_torch():
    # torch-verified values; dunders + methods share kernels
    a = ox.Tensor.from_slice([12, 10, 0], [3], dtype="u8")
    b = ox.Tensor.from_slice([10, 6, 255], [3], dtype="u8")
    assert (a & b).tolist() == [8, 2, 0]
    assert (a | b).tolist() == [14, 14, 255]
    assert (a ^ b).tolist() == [6, 12, 255]
    assert (~a).tolist() == [243, 245, 255]
    assert a.bitwise_and(b).tolist() == [8, 2, 0]
    i = ox.Tensor.from_slice([6], [1], dtype="i32")
    assert (i & 3).tolist() == [2]
    f = ox.Tensor.from_slice([0.0, 2.0, float("nan")], [3])
    assert f.isnan().tolist() == [False, False, True]
    assert f.isfinite().tolist() == [True, True, False]
    t = ox.Tensor.from_slice([True, True, False], [3], dtype="bool")
    assert f.logical_and(t).tolist() == [False, True, False]
    assert f.logical_or(t).tolist() == [True, True, True]
    with pytest.raises(RuntimeError):
        a.bitwise_and(f)
    with pytest.raises(RuntimeError):
        f & f

def test_index_select_add_gather_scatter_add():
    # torch-verified values incl. duplicates, wrap, and i64 exactness.
    # index_select/gather/scatter_add run the Rust kernels; index_add runs the
    # numpy shim (which additionally supports alpha like torch).
    v = ox.Tensor.from_slice([1.0, 2.0, 3.0], [3])
    idx = ox.Tensor.from_slice([2, 0, 2], [3], dtype="i64")
    assert v.index_select(0, idx).tolist() == [3.0, 1.0, 3.0]
    neg = ox.Tensor.from_slice([-1], [1], dtype="i64")
    assert v.index_select(0, neg).tolist() == [3.0]
    d = ox.Tensor.from_slice([0.0, 0.0], [2])
    dup = ox.Tensor.from_slice([1, 1, 0], [3], dtype="i64")
    s3 = ox.Tensor.from_slice([1.0, 2.0, 4.0], [3])
    assert d.index_add(0, dup, s3).tolist() == [4.0, 3.0]
    assert d.index_add(0, dup, s3, alpha=2.0).tolist() == [8.0, 6.0]
    g = ox.Tensor.from_slice([1.0, 2.0, 3.0, 4.0], [2, 2])
    gi = ox.Tensor.from_slice([1, 0, 0, 1], [2, 2], dtype="i64")
    assert g.gather(1, gi).tolist() == [[2.0, 1.0], [3.0, 4.0]]
    z = ox.Tensor.from_slice([0.0, 0.0], [2])
    si = ox.Tensor.from_slice([1, 1], [2], dtype="i64")
    sv = ox.Tensor.from_slice([1.0, 2.0], [2])
    assert z.scatter_add(0, si, sv).tolist() == [0.0, 3.0]
    # i64 path end-to-end (beyond-2^53 exactness is proven Rust-side; the
    # from_slice f64 funnel cannot express such literals from Python)
    bi = ox.Tensor.from_slice([7, 5], [2], dtype="i64")
    one = ox.Tensor.from_slice([0], [1], dtype="i64")
    assert bi.index_select(0, one).tolist() == [7]
    bi1 = ox.Tensor.from_slice([7], [1], dtype="i64")
    with pytest.raises(RuntimeError):
        d.index_add(0, one, bi1)

def test_window_functions_match_torch_at_degenerate_sizes():
    fns = [
        ox.hann_window,
        ox.hamming_window,
        ox.blackman_window,
        ox.bartlett_window,
        ox.kaiser_window,
        ox.gaussian_window,
        ox.exponential_window,
        ox.chebyshev_window,
    ]
    for fn in fns:
        assert fn(1).to_numpy().tolist() == [1.0]
        assert fn(0).to_numpy().size == 0
        with pytest.raises(ValueError, match="non-negative"):
            fn(-1)
        with pytest.raises(ValueError, match="non-negative"):
            fn(-100)

def test_topk_range_errors_and_partition_parity():
    x = ox.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    with pytest.raises(ValueError, match="out of range"):
        ox.topk(x, 9)
    with pytest.raises(ValueError, match="out of range"):
        ox.topk(x, 2, dim=5)
    with pytest.raises(ValueError, match="out of range"):
        ox.topk(x, 2, dim=-2)

    rng = np.random.default_rng(0)
    for shape, dim, k, largest, sorted in [
        ((1000,), -1, 10, True, True),
        ((1000,), 0, 10, False, True),
        ((50, 60), 1, 7, True, True),
        ((50, 60), 0, 7, False, False),
        ((30,), -1, 30, True, True),
    ]:
        af = rng.standard_normal(shape).astype(np.float32)
        t = ox.from_numpy(af)
        vals, idx = ox.topk(t, k, dim=dim, largest=largest, sorted=sorted)
        ax = dim % len(shape)
        expected_shape = list(shape)
        expected_shape[ax] = k
        assert list(vals.shape) == expected_shape
        assert list(idx.shape) == expected_shape
        v = vals.to_numpy()
        sel = idx.to_numpy()
        assert np.allclose(np.take_along_axis(af, sel, axis=ax), v, atol=0)
        ref = np.sort(af, axis=ax)
        ref_top = ref.take(np.arange(shape[ax] - k, shape[ax]) if largest else np.arange(k), axis=ax)
        assert np.allclose(np.sort(v, axis=ax), np.sort(ref_top, axis=ax), atol=1e-6)
        if sorted:
            d = np.diff(v, axis=ax)
            assert (d <= 1e-6).all() if largest else (d >= -1e-6).all()

# ---------------------------------------------------------------------------
# Bug-hunt 2026-11-04 regressions: normalization autograd, state-dict identity,
# safe checkpoint loading, mode(), and advanced indexing.
# ---------------------------------------------------------------------------

def test_batch_norm_updates_running_stats_and_flows_grads():
    import oxideon.nn as nn

    x = ox.randn([4, 3, 5, 5]).requires_grad_(True)
    rm = ox.zeros([3])
    rv = ox.ones([3])
    w = ox.ones([3]).requires_grad_(True)
    y = F.batch_norm(x, rm, rv, weight=w, training=True)
    assert list(y.shape) == [4, 3, 5, 5]
    # Running stats used to be written to a detached numpy copy (never updated).
    assert float(rm.to_numpy().sum()) != 0.0
    assert not np.allclose(rv.to_numpy(), 1.0)
    y.sum().backward()
    assert x.grad is not None
    assert w.grad is not None


def test_group_norm_instance_norm_values_and_grads():
    xg = ox.randn([2, 4, 6]).requires_grad_(True)
    yg = F.group_norm(xg, 2)
    ref = xg.to_numpy().reshape(2, 2, -1)
    ref = (ref - ref.mean(-1, keepdims=True)) / np.sqrt(ref.var(-1, keepdims=True) + 1e-5)
    assert np.allclose(yg.to_numpy(), ref.reshape(2, 4, 6), atol=1e-5)
    yg.sum().backward()
    assert xg.grad is not None
    with pytest.raises(ValueError):
        F.group_norm(xg, 0)

    xi = ox.randn([2, 3, 4, 4]).requires_grad_(True)
    yi = F.instance_norm(xi)
    r = xi.to_numpy()
    m = r.mean(axis=(2, 3), keepdims=True)
    v = r.var(axis=(2, 3), keepdims=True)
    assert np.allclose(yi.to_numpy(), (r - m) / np.sqrt(v + 1e-5), atol=1e-5)
    yi.sum().backward()
    assert xi.grad is not None
    # use_input_stats=False must honor (not ignore) the running stats.
    rm = ox.zeros([3])
    rv = ox.ones([3])
    ye = F.instance_norm(xi.detach(), rm, rv, use_input_stats=False)
    m2 = rm.to_numpy().reshape(1, 3, 1, 1)
    v2 = rv.to_numpy().reshape(1, 3, 1, 1)
    assert np.allclose(ye.to_numpy(), (r - m2) / np.sqrt(v2 + 1e-5), atol=1e-5)


def test_bilinear_values_and_grads():
    x1 = ox.randn([5, 4]).requires_grad_(True)
    x2 = ox.randn([5, 6]).requires_grad_(True)
    w = ox.randn([7, 4, 6]).requires_grad_(True)
    y = F.bilinear(x1, x2, w)
    ref = np.einsum("bi,oij,bj->bo", x1.to_numpy(), w.to_numpy(), x2.to_numpy())
    assert np.allclose(y.to_numpy(), ref, atol=1e-4)
    y.sum().backward()
    assert x1.grad is not None and x2.grad is not None and w.grad is not None


def test_dropout_nd_zero_at_p1_and_flows_grads():
    z = F.dropout2d(ox.ones([2, 3, 4, 4]), 1.0)
    assert np.all(z.to_numpy() == 0.0)  # used to produce NaN
    x = ox.ones([2, 3, 4, 4]).requires_grad_(True)
    F.dropout2d(x, 0.5).sum().backward()
    assert x.grad is not None


def test_load_state_dict_preserves_identity_and_buffers():
    import oxideon.nn as nn

    class P(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("w", ox.randn([3, 3]))

    pm = P()
    w0 = pm.w
    sd = pm.state_dict()
    sd["w"] = ox.ones([3, 3])
    pm.load_state_dict(sd)
    assert pm.w is w0  # live optimizers keep working
    assert float(pm.w.to_numpy().sum()) == 9.0

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)
            self.register_buffer("rm", ox.zeros([3]))

        def forward(self, x):
            return self.fc(x)

    mm = M()
    opt = ox.optim.Adam(mm.parameters(), lr=0.01)
    sd = mm.state_dict()
    sd["fc.weight"] = ox.ones_like(sd["fc.weight"])
    sd["rm"] = ox.ones([3]) * 5.0
    mm.load_state_dict(sd)
    assert float(mm._buffers["rm"].to_numpy().sum()) == 15.0  # buffers were skipped
    before = mm.fc.weight.to_numpy().copy()
    mm.zero_grad()
    mm(ox.randn([2, 4])).sum().backward()
    opt.step()
    assert not np.array_equal(before, mm.fc.weight.to_numpy())  # optimizer drives live params


def test_module_to_preserves_frozen_and_identity():
    import oxideon.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

    m = M()
    m.freeze()
    m.to(device="cpu")
    assert all(not p.requires_grad for p in m.parameters())  # to() used to unfreeze

    m2 = nn.Module()
    m2.register_parameter("w", ox.randn([2, 2]))
    w0 = m2.w
    m2.to(device="cpu")
    assert m2.w is w0

    # An actual dtype cast must land (assign()-based moves silently no-op'd).
    m3 = nn.Module()
    m3.register_parameter("w", ox.randn([2, 2]))
    w_before = m3.w
    m3.to(dtype="f64")
    assert m3.w is w_before
    assert str(m3.w.dtype) == "f64"
    assert m3.w.shape == (2, 2)


def test_load_refuses_unsafe_pickle_fallback(tmp_path):
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not a safetensors file at all...." * 10)
    # A failing .safetensors load must surface its real error, not unpickle.
    with pytest.raises(Exception):
        ox.load(str(bad))
    pt = tmp_path / "c.pt"
    ox.save({"epoch": 3}, str(pt))
    assert ox.load(str(pt))["epoch"] == 3  # trusted .pt roundtrip still works
    with pytest.raises(ValueError, match="weights_only"):
        ox.load(str(pt), weights_only=True)


def test_mode_without_scipy_and_correct_indices():
    import builtins
    import oxideon.primitives as P

    x = ox.from_numpy(np.array([[1.0, 2.0, 2.0, 3.0], [5.0, 5.0, 1.0, 1.0]], dtype=np.float32))
    real_import = builtins.__import__

    def no_scipy(name, *a, **k):
        if name.startswith("scipy"):
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    builtins.__import__ = no_scipy
    try:
        vals, idx = P.mode(x, dim=-1)
    finally:
        builtins.__import__ = real_import
    assert vals.to_numpy().tolist() == [2.0, 1.0]  # used to return the first element
    assert idx.to_numpy().tolist() == [1, 2]
    _, idx2 = P.mode(x, dim=-1)
    assert idx2.to_numpy().tolist() == [1, 2]  # used to return counts, not indices


def test_advanced_index_getitem_fancy_and_grads():
    x = ox.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4)).requires_grad_(True)
    idx = ox.from_numpy(np.array([[0, 1], [1, 0]], dtype=np.int64))
    sel = x[idx]  # 0/1 int indices used to be misread as a boolean mask
    assert list(sel.shape) == [2, 2, 4]
    assert np.array_equal(sel.to_numpy(), x.to_numpy()[[[0, 1], [1, 0]]])
    sel.sum().backward()
    assert x.grad is not None

    y = ox.from_numpy(np.arange(6, dtype=np.float32).reshape(3, 2)).requires_grad_(True)
    m = ox.from_numpy(np.array([True, False, True]))
    selm = y[m]
    assert np.array_equal(selm.to_numpy(), np.array([[0, 1], [4, 5]], dtype=np.float32))
    selm.sum().backward()
    assert y.grad is not None


def _tiny_trainer_setup(lr=0.01):
    from oxideon import data, nn, optim
    X = ox.randn([8, 2])
    y = ox.randn([8, 1])
    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=4)
    model = nn.Sequential(nn.Linear(2, 1))
    return loader, model, optim.SGD(model.parameters(), lr=lr), nn.MSELoss()


def test_trainer_scheduler_routing_and_error_propagation():
    loader, model, opt, crit = _tiny_trainer_setup()

    calls = []
    class Plateau:
        def step(self, metric):
            calls.append(metric)
    tr = ox.Trainer(model, opt, crit, lr_scheduler=Plateau())
    tr.fit(loader, val_loader=loader, epochs=1, verbose=False)
    assert len(calls) == 1 and isinstance(calls[0], float)

    plain_calls = []
    class Plain:
        def step(self):
            plain_calls.append(1)
    tr2 = ox.Trainer(model, opt, crit, lr_scheduler=Plain())
    tr2.fit(loader, val_loader=loader, epochs=1, verbose=False)
    assert plain_calls == [1]

    class Broken:
        def step(self, metric):
            raise TypeError("inner boom")
    tr3 = ox.Trainer(model, opt, crit, lr_scheduler=Broken())
    with pytest.raises(TypeError, match="inner boom"):
        tr3.fit(loader, val_loader=loader, epochs=1, verbose=False)


def test_trainer_predict_empty_loader_raises():
    from oxideon import data
    loader, model, opt, crit = _tiny_trainer_setup()
    tr = ox.Trainer(model, opt, crit)
    empty = data.DataLoader(
        data.TensorDataset(ox.from_numpy(np.zeros((0, 2), dtype=np.float32))),
        batch_size=4,
    )
    with pytest.raises(ValueError, match="empty"):
        tr.predict(empty)


def test_trainer_epoch_accuracy_exact_fraction():
    from oxideon import data, nn, optim
    X = ox.from_numpy(np.array([[0.9], [0.1], [0.9], [0.9]], dtype=np.float32))
    y = ox.from_numpy(np.ones((4, 1), dtype=np.float32))
    loader = data.DataLoader(data.TensorDataset(X, y), batch_size=2, shuffle=False)
    model = nn.Linear(1, 1)
    model.weight.assign(ox.ones(list(model.weight.shape)))
    model.bias.assign(ox.zeros(list(model.bias.shape)))
    opt = optim.SGD(model.parameters(), lr=0.0)
    tr = ox.Trainer(model, opt, nn.MSELoss())
    h = tr.fit(loader, epochs=1, verbose=False)
    assert h["train_acc"][0] == 0.75


def test_default_collate_preserves_bool():
    from oxideon import data
    t = data.default_collate([True, False])
    assert str(t.dtype) == "bool"
    assert t.to_numpy().tolist() == [True, False]


def test_size_hashable():
    from oxideon.primitives import Size
    assert hash(Size([2, 3])) == hash((2, 3))
    assert len({Size([2, 3]), Size([2, 3])}) == 1


def test_set_grad_enabled_construction_pure_and_nesting():
    from oxideon.primitives import set_grad_enabled, is_grad_enabled
    before = is_grad_enabled()
    g = set_grad_enabled(False)
    assert is_grad_enabled() == before
    with set_grad_enabled(False):
        assert is_grad_enabled() is False
        with set_grad_enabled(True):
            assert is_grad_enabled() is True
        assert is_grad_enabled() is False
    assert is_grad_enabled() == before


def test_grad_setter_preserves_target_dtype():
    t = ox.from_numpy(np.ones((2,), dtype=np.float64)).requires_grad_(True)
    t.grad = np.array([1.0, 2.0])
    assert str(t.grad.dtype) == "f64"


def test_squeeze_invalid_dim_raises():
    with pytest.raises(IndexError):
        ox.ones([2, 3]).squeeze(5)
    with pytest.raises(IndexError):
        ox.ones([2, 3]).squeeze(-3)
    assert list(ox.ones([2, 3]).squeeze(0).shape) == [2, 3]


def test_split_sections_mismatch_raises():
    with pytest.raises(ValueError, match="sum exactly"):
        ox.ones([10]).split([3, 3])
    with pytest.raises(ValueError):
        ox.ones([10]).split(0)
    parts = ox.ones([10]).split([4, 6])
    assert [list(p.shape) for p in parts] == [[4], [6]]


def test_tensor_type_names():
    assert ox.ones([1]).type() == "oxideon.FloatTensor"
    assert ox.from_numpy(np.ones(1, dtype=np.int64)).type() == "oxideon.LongTensor"


def test_normalize_device_keeps_index_and_to_native():
    from oxideon.primitives import _normalize_device, device
    assert _normalize_device(device("cuda:1")) == "wgpu:1"
    t = ox.ones([1])
    assert _normalize_device(t.device) == "cpu"
    assert t.to(t.device) is t
    with pytest.raises(TypeError):
        t.to(123)


def test_jvp_central_difference_documented_and_accurate():
    from oxideon.primitives import _FunctionalAutograd as FA
    assert "finite" in FA.jvp.__doc__.lower()
    x = ox.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    v = ox.from_numpy(np.array([1.0, 1.0], dtype=np.float32))
    _, j = FA.jvp(lambda t: (t * t).sum(), x, v)
    assert abs(float(j.to_numpy().reshape(-1)[0]) - 6.0) < 1e-2


def test_hessian_unused_input_gives_zeros():
    from oxideon.primitives import _FunctionalAutograd as FA
    x = ox.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    const_out = ox.from_numpy(np.array(3.0, dtype=np.float32)).requires_grad_(True)
    H = FA.hessian(lambda t: const_out, x)
    assert np.abs(H.to_numpy()).max() == 0.0


def test_autograd_grad_honors_retain_graph():
    from oxideon.primitives import _AutogradModule as AM
    a = ox.from_numpy(np.array([2.0], dtype=np.float32)).requires_grad_(True)
    b = a * a
    AM.grad(b, a, retain_graph=True)
    b.backward()  # must not raise: graph was retained


def test_dropout_nd_p1_zero_no_nan():
    x = ox.from_numpy(np.ones((2, 3, 4, 4), dtype=np.float32))
    for fn, y in [
        (F.dropout1d, ox.from_numpy(np.ones((2, 3, 8), dtype=np.float32))),
        (F.dropout2d, x),
        (F.dropout3d, ox.from_numpy(np.ones((2, 3, 4, 4, 4), dtype=np.float32))),
    ]:
        out = fn(y, 1.0).to_numpy()
        assert not np.isnan(out).any()
        assert (out == 0.0).all()


def test_instance_norm_honors_running_stats():
    x = ox.from_numpy(np.random.randn(2, 3, 8).astype(np.float32))
    rm = ox.from_numpy(np.zeros(3, dtype=np.float32))
    rv = ox.from_numpy(np.ones(3, dtype=np.float32))
    y_eval = F.instance_norm(x, rm, rv, use_input_stats=False)
    assert list(y_eval.shape) == [2, 3, 8]
    rm2 = ox.from_numpy(np.zeros(3, dtype=np.float32))
    rv2 = ox.from_numpy(np.ones(3, dtype=np.float32))
    F.instance_norm(x, rm2, rv2, use_input_stats=True)
    assert abs(float(rm2.to_numpy().sum())) > 0


def test_group_norm_validation():
    x = ox.from_numpy(np.random.randn(2, 3, 8).astype(np.float32))
    with pytest.raises(ValueError):
        F.group_norm(x, 0)
    with pytest.raises(ValueError):
        F.group_norm(ox.from_numpy(np.ones(5, dtype=np.float32)), 1)
    assert list(F.group_norm(x, 3).shape) == [2, 3, 8]


def test_sdpa_f16_and_full_mask_finite():
    q = ox.from_numpy(np.random.randn(1, 2, 4, 8).astype(np.float16)).to_dtype("f16")
    k = ox.from_numpy(np.random.randn(1, 2, 4, 8).astype(np.float16)).to_dtype("f16")
    v = ox.from_numpy(np.random.randn(1, 2, 4, 8).astype(np.float16)).to_dtype("f16")
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert bool(np.isfinite(out.to_numpy().astype(np.float32)).all())
    q32 = ox.from_numpy(np.random.randn(1, 1, 3, 4).astype(np.float32))
    k32 = ox.from_numpy(np.random.randn(1, 1, 3, 4).astype(np.float32))
    v32 = ox.from_numpy(np.random.randn(1, 1, 3, 4).astype(np.float32))
    m = ox.from_numpy(np.zeros((3, 3), dtype=bool))
    o2 = F.scaled_dot_product_attention(q32, k32, v32, attn_mask=m)
    assert bool(np.isfinite(o2.to_numpy()).all())


def test_from_networkx_dtypes():
    nx = pytest.importorskip("networkx")
    from oxideon import interop
    g = nx.Graph()
    g.add_edge(0, 1, weight=2.5)
    assert str(interop.from_networkx(g, dtype="i32").dtype) == "i32"
    assert str(interop.from_networkx(g, dtype="i64").dtype) == "i64"
    assert str(interop.from_networkx(g, dtype="bool", weight=None).dtype) == "bool"
    with pytest.raises(ValueError):
        interop.from_networkx(g, dtype="c64")


def test_to_networkx_rejects_rectangular():
    pytest.importorskip("networkx")
    from oxideon import interop
    with pytest.raises(ValueError, match="square"):
        interop.to_networkx(np.ones((2, 3)))


def test_from_scipy_sparse_preserves_dtype():
    sp = pytest.importorskip("scipy.sparse")
    from oxideon import interop
    m = sp.csr_matrix(np.array([[1.5, 0.0], [0.0, 2.5]]))
    d = interop.from_scipy_sparse(m, as_csr_matrix=False)
    assert str(d.dtype) == "f64"
    c = interop.from_scipy_sparse(m, as_csr_matrix=True)
    assert c.shape == (2, 2)


def test_from_numpy_int_widening_and_complex_rejection():
    t = ox.from_numpy(np.array([-300, 30000], dtype=np.int16))
    assert str(t.dtype) == "i32"
    assert t.to_numpy().tolist() == [-300, 30000]
    t = ox.from_numpy(np.array([0, 65535], dtype=np.uint16))
    assert str(t.dtype) == "i32"
    c = ox.from_numpy(np.array([1 + 2j], dtype=np.complex64))
    assert str(c.dtype) == "c64"
    assert np.allclose(c.to_numpy(), np.array([1 + 2j], dtype=np.complex64))
    with pytest.raises(OverflowError):
        ox.from_numpy(np.array([2**63], dtype=np.uint64))


def test_frozen_tensor_attr_in_state_dict():
    from oxideon import nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = ox.from_numpy(np.ones((2, 2), dtype=np.float32))
            self.w.requires_grad_(False)

    m = M()
    assert "w" in dict(m.named_parameters())
    assert "w" in m.state_dict()
    m2 = M()
    m2.load_state_dict({"w": ox.from_numpy(np.full((2, 2), 7.0, dtype=np.float32))})
    assert float(m2.state_dict()["w"].to_numpy().sum()) == 28.0
    assert m2.w.requires_grad is False


def test_weighted_sampler_rejects_bad_weights():
    from oxideon import data
    with pytest.raises(ValueError):
        data.WeightedRandomSampler([-1.0, 1.0], 2)
    with pytest.raises(ValueError):
        data.WeightedRandomSampler([float("nan"), 1.0], 2)
    with pytest.raises(ValueError):
        data.WeightedRandomSampler([], 2)
    s = data.WeightedRandomSampler([0.0, 1.0], 2, replacement=False)
    assert sorted(s) == [0, 1]


# ---------------------------------------------------------------------------
# LOW bug-hunt regressions (2026-11-04)
# ---------------------------------------------------------------------------


class _NativeLikeParam:
    """Mimics a native tensor: has .grad and a .zero_grad() that ignores set_to_none."""

    def __init__(self):
        self.grad = [1.0, 2.0]
        self.zero_grad_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1


class _ParamHolder:
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return iter(self._params)


def test_low_zero_grad_set_to_none_wins_over_native_zero_grad():
    p = _NativeLikeParam()
    nn.Module.zero_grad(_ParamHolder([p]), set_to_none=True)
    assert p.grad is None
    assert p.zero_grad_calls == 0


def test_low_zero_grad_default_still_delegates():
    p = _NativeLikeParam()
    nn.Module.zero_grad(_ParamHolder([p]))
    assert p.zero_grad_calls == 1
    assert p.grad == [1.0, 2.0]


def test_low_weighted_average_exact_for_perfect_single_class():
    res = metrics.precision_recall_f1([0, 0, 0], [0, 0, 0], average="weighted")
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0


def test_low_weighted_average_matches_hand_computation():
    res = metrics.precision_recall_f1([0, 0, 1, 1], [0, 0, 0, 1], average="weighted")
    assert res["precision"] == pytest.approx(0.875)
    assert res["recall"] == pytest.approx(0.75)
    assert res["f1"] == pytest.approx((3 * 0.8 + 2 / 3) / 4)


def test_low_confusion_matrix_oor_pred_raises():
    with pytest.raises(ValueError, match="num_classes"):
        metrics.confusion_matrix([0, 5], [0, 1], num_classes=2)


def test_low_confusion_matrix_negative_label_raises():
    with pytest.raises(ValueError, match="num_classes"):
        metrics.confusion_matrix([0, -1], [0, 1], num_classes=2)


def test_low_summary_writer_preserves_zero_walltime(tmp_path):
    w = utils.SummaryWriter(log_dir=str(tmp_path))
    w.add_scalar("loss", 1.0, walltime=0.0)
    assert w.events[-1]["walltime"] == 0.0
    assert w.scalars["loss"][-1][2] == 0.0


def test_low_summary_writer_default_walltime_is_now(tmp_path):
    import time

    w = utils.SummaryWriter(log_dir=str(tmp_path))
    before = time.time()
    w.add_scalar("loss", 1.0)
    after = time.time()
    assert before <= w.events[-1]["walltime"] <= after


def _stub_param(numel, dtype):
    import types

    return types.SimpleNamespace(
        shape=(numel,), numel=lambda: numel, dtype=dtype, requires_grad=True
    )


def _stub_model(params):
    import types

    return types.SimpleNamespace(parameters=lambda: list(params))


def test_low_model_summary_uses_dtype_byte_widths():
    s = utils.model_summary(_stub_model([_stub_param(10, "f64")]))
    # 10 params * 8 bytes = 80 bytes; the old 4-bytes-always math gave 0.04 KB.
    assert "0.08 KB" in s


def test_low_model_summary_unknown_dtype_falls_back_to_4_bytes():
    s = utils.model_summary(_stub_model([_stub_param(10, "q8_0")]))
    assert "0.04 KB" in s


def test_low_assert_close_f32_default_atol_matches_doc():
    a = np.array([0.0], dtype=np.float32)
    testing.assert_close(a, np.array([5e-9], dtype=np.float32))  # within atol=1e-8
    with pytest.raises(AssertionError):
        testing.assert_close(a, np.array([5e-7], dtype=np.float32))  # above atol=1e-8


def test_low_make_tensor_deterministic_per_seed():
    a = testing.make_tensor((2, 3), seed=7).to_numpy()
    b = testing.make_tensor((2, 3), seed=7).to_numpy()
    c = testing.make_tensor((2, 3), seed=8).to_numpy()
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_low_make_tensor_does_not_perturb_global_rng():
    np.random.seed(1234)
    before = np.random.rand(4)
    np.random.seed(1234)
    testing.make_tensor((4,), seed=0)
    after = np.random.rand(4)
    np.testing.assert_array_equal(before, after)


def test_low_optimizer_state_dict_returns_copies():
    p = ox.randn(3)
    opt = optim.Optimizer([p], {"lr": 0.1})
    opt.state[id(p)]["buf"] = np.ones(3)
    sd = opt.state_dict()
    sd["state"][id(p)]["buf"][0] = 99.0
    sd["param_groups"][0]["lr"] = 999.0
    assert opt.state[id(p)]["buf"][0] == 1.0
    assert opt.param_groups[0]["lr"] == 0.1


def test_low_optimizer_load_state_dict_copies():
    p = ox.randn(3)
    opt = optim.Optimizer([p], {"lr": 0.1})
    src = {
        "state": {id(p): {"buf": np.ones(2)}},
        "param_groups": [{"params": [p], "lr": 0.5}],
    }
    opt.load_state_dict(src)
    src["state"][id(p)]["buf"][0] = 99.0
    src["param_groups"][0]["lr"] = 999.0
    assert opt.state[id(p)]["buf"][0] == 1.0
    assert opt.param_groups[0]["lr"] == 0.5


def test_low_optimizer_prunes_dead_id_keyed_state():
    p = ox.randn(3)
    opt = optim.Optimizer([p], {"lr": 0.1})
    opt.state[999999937]["stale"] = 1.0
    opt.load_state_dict(
        {"state": dict(opt.state), "param_groups": [{"params": [p], "lr": 0.1}]}
    )
    assert 999999937 not in opt.state


class _Seq:
    def __init__(self, vals):
        self._vals = list(vals)

    def __len__(self):
        return len(self._vals)

    def __getitem__(self, i):
        return self._vals[i]


def test_low_concat_dataset_positive_oob_raises():
    ds = ConcatDataset([_Seq([10, 11]), _Seq([20])])
    assert len(ds) == 3
    assert ds[2] == 20
    with pytest.raises(IndexError, match="out of range"):
        ds[3]
    with pytest.raises(IndexError, match="out of range"):
        ds[99]


def test_low_concat_dataset_negative_indexing():
    ds = ConcatDataset([_Seq([10, 11]), _Seq([20])])
    assert ds[-1] == 20
    assert ds[-3] == 10
    with pytest.raises(IndexError):
        ds[-4]


def _repo_root():
    from pathlib import Path

    import oxideon as _ox

    here = Path(_ox.__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / "tools" / "edge_fuzz.py").exists():
            return cand
    raise AssertionError("repo root with tools/ not found")


def test_low_edge_fuzz_classifies_unsolicited_sigkill_as_oom():
    import ast

    src = (_repo_root() / "tools" / "edge_fuzz.py").read_text()
    kinds = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and n.value == "oom"]
    assert kinds, "edge_fuzz must classify unsolicited SIGKILL as oom, not hang"
    assert "killed_for_stall" in src


def test_r2_steplr_rejects_non_positive_step_size():
    from oxideon.optim.lr_scheduler import StepLR

    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="step_size"):
            StepLR(object(), bad)
    s = StepLR(object(), 2)
    s.step()
    assert s.get_lr() == [1e-3]


def test_r2_accept_longest_path_cycle_raises():
    with pytest.raises(RuntimeError, match="cycle"):
        ox.TreeSpeculativeDecoder.accept_longest_path([7, 7], [7, 7], [1, 0])
    path, n = ox.TreeSpeculativeDecoder.accept_longest_path([10, 20, 30], [10, 20, 99], [0, 0, 1])
    assert (list(path), n) == ([10, 20], 2)


def test_r2_trainer_feeds_metric_only_to_plateau_schedulers():
    from oxideon.optim.lr_scheduler import ReduceLROnPlateau, StepLR
    from oxideon.training import _step_takes_metric

    # Epoch-style schedulers declare optional `epoch`: never feed them val_loss.
    assert _step_takes_metric(StepLR(object(), 2).step) is False
    # Plateau schedulers require a metric positional.
    assert _step_takes_metric(ReduceLROnPlateau(object()).step) is True
    # And a StepLR stepped by a Trainer keeps integer epochs.
    sched = StepLR(object(), 2)
    sched.step()
    assert isinstance(sched.last_epoch, int) and not isinstance(sched.last_epoch, bool)


def test_r2_functional_preserves_f64_precision():
    from oxideon import functional as F

    x = ox.from_numpy(np.array([2.0**40, 1.0], dtype=np.float64))
    # exp() of huge positive inputs overflows, but those lanes are discarded
    # by the where/minimum select, so no warning may escape.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        y = F.selu(x)
        F.elu(x)
        F.celu(x)
    assert str(y.dtype) == "f64"
    # selu(x) = scale*x for x > 0; exact in f64, rounded in f32.
    expected = 1.0507009873554804934193349852946 * np.array([2.0**40, 1.0])
    np.testing.assert_allclose(y.to_numpy(), expected, rtol=0, atol=0)


def test_r2_trainer_predict_preserves_output_dtype():
    from oxideon.training import Trainer

    class F64Model:
        def eval(self):
            pass

        def __call__(self, x):
            return ox.from_numpy(np.ones((2, 3), dtype=np.float64))

    trainer = Trainer(F64Model(), optimizer=None, criterion=None)
    loader = [[ox.zeros([2, 3])], [ox.zeros([2, 3])]]
    preds = trainer.predict(loader)
    assert str(preds.dtype) == "f64"
    assert preds.shape == (4, 3)


def test_r2_train_classification_rejects_fractional_labels():
    from oxideon.oxideon import Sequential as NativeSequential

    m = NativeSequential()
    m.add_linear(4, 3)
    x = ox.randn([8, 4])
    y_bad = ox.from_numpy(np.array([0.0, 1.5, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float32))
    with pytest.raises(RuntimeError, match="not a valid class index"):
        m.train_classification_in_rust(x, y_bad, 1, batch_size=8)
    y_ok = ox.from_numpy(np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64))
    hist = m.train_classification_in_rust(x, y_ok, 1, batch_size=8)
    assert len(hist) == 1


def test_r2_micro_average_exact_for_perfect_predictions():
    res = metrics.precision_recall_f1([0, 1, 1, 0], [0, 1, 1, 0], average="micro")
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0


def test_r2_weighted_sampler_rejects_oversampling_without_replacement():
    from oxideon.data import WeightedRandomSampler

    with pytest.raises(ValueError, match="without replacement"):
        WeightedRandomSampler([0.5, 0.5], 3, replacement=False)
    s = WeightedRandomSampler([0.5, 0.5], 2, replacement=False)
    assert len(list(s)) == 2 == len(s)


def test_r2_from_sympy_f64_grid_keeps_precision():
    sp = pytest.importorskip("sympy")
    x = sp.Symbol("x")
    # Midpoint 2^40 + 100000 is exact in f64; an f32-built grid rounds it.
    t = ox.from_sympy(
        x, symbols=[x], ranges=[(2.0**40, 2.0**40 + 200_000.0)], steps=3, dtype="f64"
    )
    assert str(t.dtype) == "f64"
    assert t.to_numpy().reshape(-1)[1] == 2.0**40 + 100_000.0
    with pytest.raises(ValueError, match="dtype"):
        ox.from_sympy(x, symbols=[x], ranges=[(0.0, 1.0)], steps=3, dtype="f16")


# ---------------------------------------------------------------------------
# Round 3: torch-parity config validation across optimizers, schedulers,
# losses, dropout, one_hot, interpolate.
# ---------------------------------------------------------------------------


def _r3_param():
    return ox.from_numpy(np.ones(4, dtype=np.float32))


def test_r3_python_optimizers_reject_invalid_hparams():
    p = _r3_param()
    with pytest.raises(ValueError, match="learning rate"):
        optim.RMSprop([p], lr=-1.0)
    with pytest.raises(ValueError, match="lr_decay"):
        optim.Adagrad([p], lr_decay=-1.0)
    with pytest.raises(ValueError, match="rho"):
        optim.Adadelta([p], rho=2.0)
    with pytest.raises(ValueError, match="beta parameter at index 1"):
        optim.Adamax([p], betas=(0.9, 1.0))
    with pytest.raises(ValueError, match="momentum_decay"):
        optim.NAdam([p], momentum_decay=-0.1)
    with pytest.raises(ValueError, match="epsilon"):
        optim.RAdam([p], eps=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        optim.ASGD([p], alpha=-1.0)
    with pytest.raises(ValueError, match="parameter groups"):
        optim.LBFGS([{"params": [p]}, {"params": [p]}])
    # Valid configs still construct.
    optim.RMSprop([p], lr=0.01)
    optim.LBFGS([p])


def test_r3_native_optimizers_validate():
    p = _r3_param()
    with pytest.raises(ValueError):
        optim.AdamW([p], lr=-1.0)
    with pytest.raises(ValueError):
        optim.AdamW([p], weight_decay=-1.0)
    with pytest.raises(ValueError):
        optim.Muon([], lr=0.01)
    with pytest.raises(ValueError):
        optim.Muon([p], lr=float("inf"))
    with pytest.raises(ValueError):
        optim.ScheduleFreeAdamW([p], weight_decay=-1.0)
    opt = optim.AdamW([p], lr=0.01)
    with pytest.raises(ValueError):
        opt.set_learning_rate(-1.0)
    # LR accessors exist on every native optimizer so schedulers can drive them
    # (AdamW/Muon/ScheduleFree silently ignored schedulers before).
    from oxideon.optim import lr_scheduler as lrs

    for make in (
        lambda: optim.AdamW([_r3_param()], lr=0.1),
        lambda: optim.Muon([_r3_param()], lr=0.1),
        lambda: optim.ScheduleFreeAdamW([_r3_param()], lr=0.1),
    ):
        o = make()
        assert o.lr == 0.1
        o.set_learning_rate(0.05)
        assert o.lr == 0.05
        sched = lrs.StepLR(o, step_size=1, gamma=0.5)
        sched.step()
        assert o.lr == pytest.approx(0.025)


def _r3_sgd():
    return optim.SGD([_r3_param()], lr=0.1)


def test_r3_schedulers_reject_invalid_configs():
    from oxideon.optim import lr_scheduler as lrs

    with pytest.raises(ValueError, match="T_max"):
        lrs.CosineAnnealingLR(_r3_sgd(), T_max=0)
    with pytest.raises(ValueError, match="T_0"):
        lrs.CosineAnnealingWarmRestarts(_r3_sgd(), T_0=0)
    with pytest.raises(ValueError, match="factor"):
        lrs.LinearLR(_r3_sgd(), start_factor=2.0)
    with pytest.raises(ValueError, match="factor"):
        lrs.LinearLR(_r3_sgd(), end_factor=-0.5)
    with pytest.raises(ValueError, match="total_iters"):
        lrs.LinearLR(_r3_sgd(), total_iters=0)
    with pytest.raises(ValueError, match="total_steps"):
        lrs.OneCycleLR(_r3_sgd(), max_lr=0.1, total_steps=0)
    with pytest.raises(ValueError, match="pct_start"):
        lrs.OneCycleLR(_r3_sgd(), max_lr=0.1, total_steps=10, pct_start=1.5)
    with pytest.raises(ValueError, match="div_factor"):
        lrs.OneCycleLR(_r3_sgd(), max_lr=0.1, total_steps=10, div_factor=0.0)
    with pytest.raises(ValueError, match="warmup_steps"):
        lrs.WarmupCosineScheduler(_r3_sgd(), warmup_steps=5, total_steps=5)
    with pytest.raises(ValueError, match="total_iters"):
        lrs.PolynomialLR(_r3_sgd(), total_iters=0)
    with pytest.raises(ValueError, match="step_size_up"):
        lrs.CyclicLR(_r3_sgd(), base_lr=0.01, max_lr=0.1, step_size_up=0)
    with pytest.raises(ValueError, match="at least one scheduler"):
        lrs.ChainedScheduler([])
    with pytest.raises(ValueError, match="between 0 and 1"):
        lrs.ConstantLR(_r3_sgd(), factor=2.0)
    s1, s2 = lrs.StepLR(_r3_sgd(), step_size=1), lrs.StepLR(_r3_sgd(), step_size=1)
    with pytest.raises(ValueError, match="milestone"):
        lrs.SequentialLR(_r3_sgd(), [s1, s2], [])
    # Valid configs still construct and step.
    lrs.CosineAnnealingLR(_r3_sgd(), T_max=10).step()
    lrs.WarmupCosineScheduler(_r3_sgd(), warmup_steps=0, total_steps=10).step()
    lrs.SequentialLR(_r3_sgd(), [s1, s2], [5]).step()


def test_r3_functional_losses_reject_bad_reduction():
    x = ox.from_numpy(np.array([0.5, -1.5], dtype=np.float32))
    y = ox.from_numpy(np.array([0.0, 1.0], dtype=np.float32))
    for loss_fn in (
        F.smooth_l1_loss,
        F.binary_cross_entropy,
        F.poisson_nll_loss,
    ):
        with pytest.raises(ValueError, match="reduction"):
            loss_fn(x, y, reduction="bogus")
    v = ox.from_numpy(np.array([1.0, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="reduction"):
        F.gaussian_nll_loss(x, y, v, reduction="bogus")
    with pytest.raises(ValueError, match="beta"):
        F.smooth_l1_loss(x, y, beta=0.0)
    # Accepted modes still work.
    assert F.smooth_l1_loss(x, y, reduction="none").to_numpy().shape == (2,)
    assert F.smooth_l1_loss(x, y, reduction="mean").to_numpy().size == 1


def test_r3_one_hot_validates_indices():
    good = ox.from_numpy(np.array([0, 2, 1], dtype=np.int64))
    assert F.one_hot(good, 3).to_numpy().shape == (3, 3)
    bad_neg = ox.from_numpy(np.array([0, -1], dtype=np.int64))
    with pytest.raises(ValueError, match="out of range"):
        F.one_hot(bad_neg, 3)
    bad_hi = ox.from_numpy(np.array([0, 3], dtype=np.int64))
    with pytest.raises(ValueError, match="out of range"):
        F.one_hot(bad_hi, 3)
    empty = ox.from_numpy(np.array([], dtype=np.int64))
    assert F.one_hot(empty, 4).to_numpy().shape == (0, 4)
    with pytest.raises(ValueError, match="empty"):
        F.one_hot(empty)


def test_r3_interpolate_validates_sizes():
    x = ox.from_numpy(np.zeros((1, 1, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        F.interpolate(x, size=0)
    with pytest.raises(ValueError, match="spatial size"):
        F.interpolate(x, size=(2, 2, 2))
    with pytest.raises(ValueError, match="positive"):
        F.interpolate(x, scale_factor=0.0)
    out = F.interpolate(x, size=(2, 2))
    assert out.to_numpy().shape == (1, 1, 2, 2)


def test_r3_dropout_rejects_out_of_range_p():
    x = ox.from_numpy(np.ones((2, 2), dtype=np.float32))
    for fn in (F.dropout, F.dropout1d, F.dropout2d, F.dropout3d, F.alpha_dropout):
        with pytest.raises(ValueError, match="between 0 and 1"):
            fn(x, 2.0)
        with pytest.raises(ValueError, match="between 0 and 1"):
            fn(x, -0.5)
    # Boundaries still behave.
    assert F.dropout(x, 0.0).to_numpy().shape == (2, 2)
    assert float(F.dropout(x, 1.0).to_numpy().sum()) == 0.0


def test_r3_fold_validates_geometry():
    x = ox.from_numpy(np.zeros((1, 4, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        F.fold(x, (2, 2), (2, 2), stride=0)
    with pytest.raises(ValueError, match="positive"):
        F.fold(x, (0, 2), (1, 1))


def test_r3_nn_modules_validate_configs():
    with pytest.raises(ValueError, match="num_heads"):
        nn.MultiheadAttention(8, 0)
    with pytest.raises(ValueError, match="num_heads"):
        nn.MultiheadAttention(7, 2)
    with pytest.raises(ValueError, match="num_groups"):
        nn.GroupNorm(0, 8)
    with pytest.raises(ValueError, match="num_groups"):
        nn.GroupNorm(3, 8)
    shuffle = nn.ChannelShuffle(3)
    bad = ox.from_numpy(np.zeros((1, 8, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="groups"):
        shuffle(bad)
    nn.MultiheadAttention(8, 2)
    nn.GroupNorm(2, 8)
