"""Everyday tensor usage from Python: scalar operators, indexing, reductions, error behaviour.

Run standalone (`python tests/test_python_ergonomics.py`) against an installed wheel, or with pytest.
"""

import math

import oxideon as ox


def vals(t):
    return [round(v, 4) for v in t.to_list()]


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except BaseException as e:  # a wrong exception type is a failure too
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


def sq():
    return ox.Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])


def test_scalar_arithmetic_both_sides():
    t = sq()
    assert vals(t + 1.0) == [2, 3, 4, 5] and vals(1.0 + t) == [2, 3, 4, 5]
    assert vals(t - 1) == [0, 1, 2, 3] and vals(1.0 - t) == [0, -1, -2, -3]
    assert vals(t * 2.0) == [2, 4, 6, 8] and vals(2 * t) == [2, 4, 6, 8]
    assert vals(t / 2.0) == [0.5, 1, 1.5, 2] and vals(2.0 / t) == [2, 1, 0.6667, 0.5]
    assert vals(t**2) == [1, 4, 9, 16] and vals(t**t) == [1, 4, 27, 256]
    assert vals(-t) == [-1, -2, -3, -4] and vals(abs(-t)) == [1, 2, 3, 4]
    assert vals(t + t) == [2, 4, 6, 8] and vals(t @ t) == [7, 10, 15, 22]
    raises(TypeError, lambda: t + "x")
    raises(TypeError, lambda: "x" + t)
    raises(TypeError, lambda: t * None)


def test_comparisons_are_elementwise():
    t = sq()
    assert vals(t > 2.0) == [0, 0, 1, 1] and vals(t <= 2.0) == [1, 1, 0, 0]
    assert vals(t >= t) == [1, 1, 1, 1] and vals(t < t) == [0, 0, 0, 0]
    assert vals(t == t) == [1, 1, 1, 1] and vals(t != 2.0) == [1, 0, 1, 1]
    assert (t == None) is False  # noqa: E711  (falls back to identity)
    assert {t: 1}[t] == 1  # still hashable


def test_indexing_and_iteration():
    t = sq()
    assert vals(t[0]) == [1, 2] and vals(t[-1]) == [3, 4]
    assert t[1, 0].item() == 3.0 and vals(t[:, 1]) == [2, 4] and vals(t[..., 0]) == [1, 3]
    assert t[None].shape == [1, 2, 2] and t[:, None].shape == [2, 1, 2]
    v = ox.Tensor.arange(0.0, 6.0, 1.0)
    assert vals(v[1:5:2]) == [1, 3] and v[4:2].shape == [0] and vals(v[-2:]) == [4, 5]
    assert [vals(r) for r in t] == [[1, 2], [3, 4]]
    assert len(t) == 2
    raises(IndexError, lambda: t[2])
    raises(IndexError, lambda: t[0, 0, 0])
    raises(IndexError, lambda: t[..., ..., 0])
    assert vals(v[::-1]) == [5, 4, 3, 2, 1, 0]
    raises(TypeError, lambda: t["a"])
    raises(TypeError, lambda: len(ox.Tensor([5.0], shape=[])))


def test_number_conversions():
    one = ox.Tensor([3.5], shape=[1])
    assert float(one) == 3.5 and int(one) == 3 and bool(one) is True
    assert bool(ox.Tensor([0.0], shape=[1])) is False
    raises(ValueError, lambda: float(sq()))
    raises(ValueError, lambda: bool(sq()))


def test_reshape_infers_minus_one():
    t = sq()
    assert t.reshape([-1]).shape == [4] and t.reshape([4, -1]).shape == [4, 1] and t.view([1, -1]).shape == [1, 4]
    raises(ValueError, lambda: t.reshape([-1, -1]))
    raises(ValueError, lambda: t.reshape([3, -1]))
    raises(ValueError, lambda: t.reshape([-2, 2]))
    raises(RuntimeError, lambda: t.reshape([3, 3]))


def test_reductions_with_dim():
    t = sq()
    assert vals(t.sum()) == [10] and vals(t.sum(0)) == [4, 6] and t.sum(1, keepdim=True).shape == [2, 1]
    assert vals(t.sum(dim=1)) == [3, 7] and vals(t.mean(0)) == [2, 3] and vals(t.mean()) == [2.5]
    assert vals(t.max()) == [4] and vals(t.max(0)) == [3, 4] and vals(t.min(1)) == [1, 3] and vals(t.min()) == [1]
    assert t.argmax().item() == 3 and vals(t.argmax(1)) == [1, 1] and vals(t.argmin(0)) == [0, 0]
    assert abs(t.var().item() - 5 / 3) < 1e-4 and abs(t.var(unbiased=False).item() - 1.25) < 1e-4
    assert vals(t.std(0)) == [round(math.sqrt(2), 4)] * 2
    assert abs(t.norm().item() - math.sqrt(30)) < 1e-4
    assert vals(t.cumsum(0)) == [1, 2, 4, 6]
    ls = t.log_softmax(1)
    assert all(abs(sum(math.exp(x) for x in row.to_list()) - 1.0) < 1e-5 for row in ls)


def test_reductions_over_nothing():
    empty = ox.Tensor.zeros([0])
    assert vals(empty.sum()) == [0]
    for fn in (empty.max, empty.min, empty.argmax, empty.argmin):
        raises(RuntimeError, fn)  # used to be a Rust panic (PanicException, which is a BaseException)


def test_elementwise_helpers():
    x = ox.Tensor([-1.5, 0.4, 2.5], shape=[3])
    assert vals(x.abs()) == [1.5, 0.4, 2.5] and vals(x.neg()) == [1.5, -0.4, -2.5]
    assert vals(x.sign()) == [-1, 1, 1] and vals(x.floor()) == [-2, 0, 2] and vals(x.ceil()) == [-1, 1, 3]
    assert vals(x.clamp(-1.0, 1.0)) == [-1, 0.4, 1] and vals(x.pow(2.0)) == [2.25, 0.16, 6.25]
    assert vals(ox.Tensor([1.0, 4.0], shape=[2]).recip()) == [1, 0.25]
    assert vals(ox.Tensor([1.0, math.e], shape=[2]).log()) == [0, 1]
    assert vals(x.maximum(x.neg())) == [1.5, 0.4, 2.5] and vals(x.minimum(x.neg())) == [-1.5, -0.4, -2.5]
    assert vals(x.masked_fill(x > 0.0, 9.0)) == [-1.5, 9, 9]
    assert abs(sq().det().item() - (-2.0)) < 1e-4


def test_shape_helpers_and_constructors():
    t = sq()
    assert t.flatten().shape == [4] and t.flatten(1).shape == [2, 2] and t.expand([3, 2, 2]).shape == [3, 2, 2]
    assert ox.Tensor.zeros([2, 3, 4]).flatten(1, 2).shape == [2, 12]
    assert t.size() == [2, 2] and t.size(0) == 2 and t.size(-1) == 2
    assert vals(t.zeros_like()) == [0] * 4 and vals(t.ones_like()) == [1] * 4 and vals(t.full_like(7.0)) == [7] * 4
    assert vals(ox.Tensor.full([2, 2], 7.0)) == [7] * 4 and vals(ox.Tensor.from_slice([1.0, 2.0], [2])) == [1, 2]
    r = ox.Tensor.rand([200]).to_list()
    assert 0.0 <= min(r) and max(r) < 1.0 and len(set(r)) > 100


def test_module_functions():
    t = sq()
    assert ox.cat([t, t], 0).shape == [4, 2] and ox.cat([t, t], 1).shape == [2, 4]
    assert ox.stack([t, t], 0).shape == [2, 2, 2]
    assert vals(ox.where(t > 2.0, t, t.zeros_like())) == [0, 0, 3, 4]
    assert vals(ox.einsum("ij,jk->ik", t, t)) == [7, 10, 15, 22]
    raises(RuntimeError, lambda: ox.stack([ox.Tensor.zeros([2]), ox.Tensor.zeros([3])], 0))
    raises(TypeError, lambda: ox.einsum("i->i", 1.0))


def test_stack_stays_differentiable():
    a = ox.Tensor([1.0, 2.0], shape=[2], requires_grad=True)
    b = ox.Tensor([3.0, 4.0], shape=[2], requires_grad=True)
    s = ox.stack([a, b], dim=0)
    assert s.shape == [2, 2]
    s.sum().backward()
    assert vals(a.grad()) == [1.0, 1.0]
    assert vals(b.grad()) == [1.0, 1.0]


def test_cat_preserves_i64_exactness_and_gradients():
    big = ox.Tensor.from_i64([2**60 + 1, 5], [2])
    small = ox.Tensor.from_i64([7, 8], [2])
    assert ox.cat([big, small], dim=0).to_i64_list() == [2**60 + 1, 5, 7, 8]
    # Float path stays differentiable through the native kernel.
    a = ox.Tensor([1.0, 2.0], shape=[2], requires_grad=True)
    b = ox.Tensor([3.0], shape=[1], requires_grad=True)
    ox.cat([a, b], dim=0).sum().backward()
    assert vals(a.grad()) == [1.0, 1.0]
    assert vals(b.grad()) == [1.0]
    # Mixed dtypes keep NumPy promotion semantics.
    i = ox.Tensor([1, 2], shape=[2], dtype="i32")
    f = ox.Tensor([0.5], shape=[1], dtype="f32")
    m = ox.cat([i, f], dim=0)
    assert m.dtype == "f64" and vals(m) == [1.0, 2.0, 0.5]


def test_autograd_through_new_operators():
    x = ox.Tensor([2.0, 3.0], shape=[2], requires_grad=True)
    loss = ((x * 3.0 + 1.0) ** 2).sum()  # d/dx (3x+1)^2 = 6 (3x+1)
    loss.backward()
    assert vals(x.grad()) == [42, 60]
    y = ox.Tensor([2.0], shape=[1], requires_grad=True)
    (1.0 / y).sum().backward()
    assert vals(y.grad()) == [-0.25]
    z = ox.Tensor.from_slice([3.0], [1], requires_grad=True)
    dz = ox.grad([z**3], [z], create_graph=True)[0]
    d2z = ox.grad([dz], [z], create_graph=False)[0]
    assert dz.item() == 27.0 and d2z.item() == 18.0


def test_dtype_and_device_errors_are_clean():
    raises(RuntimeError, lambda: ox.Tensor.zeros([2], dtype="float128"))
    raises(RuntimeError, lambda: ox.Tensor.zeros([2], device="mars:0"))
    raises(RuntimeError, lambda: ox.Tensor([1.0, 2.0, 3.0], shape=[2]))


def test_absurd_sizes_raise_instead_of_killing_the_process():
    raises(RuntimeError, lambda: ox.Tensor.zeros([10**12]))
    raises(RuntimeError, lambda: ox.Tensor.zeros([2**31 + 1]))  # above the element limit: refused whatever the OS would allow
    raises(RuntimeError, lambda: ox.Tensor.zeros([2**40, 2**40]))
    raises(RuntimeError, lambda: ox.Tensor.randn([10**12]))
    raises(RuntimeError, lambda: ox.Tensor.arange(0.0, 1e15, 1.0))
    raises(RuntimeError, lambda: ox.Tensor.from_i64([1], [1]) / ox.Tensor.from_i64([0], [1]))


def test_engines_with_absurd_arguments_return_promptly():
    raises(RuntimeError, lambda: ox.CkksContext(2**30))
    assert ox.RothBourgainCapSetBounds(10**9).capset_growth_base_and_upper_bound()[1] > 0
    assert ox.AllardVarifoldMonotonicityEngine(2**60, 0.05).density_ratio_and_regularity_check(1.0, 1.0)
    assert math.isfinite(ox.MeanCurvatureFlowEngine(2**60, 1.0).colding_minicozzi_sphere_entropy())


def test_bad_layer_configuration_raises_ordinary_exceptions():
    # These used to be Rust `assert!`s: a PanicException (a BaseException that `except Exception` does not catch).
    raises(RuntimeError, lambda: ox.CausalLM(100, 14, 2, 2, 32, 16))  # head_dim 7 is odd (RoPE needs an even one)
    raises(RuntimeError, lambda: ox.CausalLM(100, 16, 3, 2, 32, 16))  # 16 is not divisible by 3 heads
    raises(RuntimeError, lambda: ox.CausalLM(100, 16, 0, 2, 32, 16))  # zero heads
    raises(ValueError, lambda: ox.Sequential().add_dropout(1.0))
    raises(ValueError, lambda: ox.TransfusionModel(dropout=1.0))
    raises(RuntimeError, lambda: ox.GatedCrossAttention(8, 0, 8))
    raises(ValueError, lambda: ox.RingAttention(0, 0))
    raises(ValueError, lambda: ox.RingAttention(2, 5))
    raises(ValueError, lambda: ox.PrefillDecodeInterleaver(0))
    raises(ValueError, lambda: ox.DiffusionForcingScheduler(0))
    raises(RuntimeError, lambda: ox.ConformalRiskController("fnr", float("nan"), 1.0))


def test_basic_layers_are_callable():
    x = ox.Tensor([1.0, -2.0, 3.0, 4.0], shape=[2, 2])
    assert vals(ox.ReLU()(x)) == [1, 0, 3, 4] and ox.ReLU().parameters() == []
    ln = ox.LayerNorm(2)
    assert len(ln.parameters()) == 2 and ln.weight.shape == [2] and ln.bias.shape == [2]
    assert all(abs(row.mean().item()) < 1e-5 for row in ln(x)) and ox.LayerNorm([2]).weight.shape == [2]
    emb = ox.Embedding(5, 3)
    out = emb(ox.Tensor.from_i64([0, 4, 4], [3]))
    assert out.shape == [3, 3] and vals(out[1]) == vals(out[2]) and emb.weight.shape == [5, 3]
    assert (emb.num_embeddings, emb.embedding_dim) == (5, 3) and len(emb.parameters()) == 1
    assert ox.Linear(2, 3)(x).shape == [2, 3]
    seq = ox.Sequential()
    seq.add_linear(2, 4)
    seq.add_relu()
    assert seq(x).shape == [2, 4] and ox.MLP([2, 4, 1])(x).shape == [2, 1]


def test_dropout_modes():
    ones = ox.Tensor.ones([2000])
    drop = ox.Dropout(0.5)
    kept = drop(ones).to_list()
    assert 0.3 < kept.count(0.0) / len(kept) < 0.7 and set(kept) <= {0.0, 2.0}  # survivors are scaled by 1 / (1 - p)
    drop.eval()
    assert drop(ones).to_list() == ones.to_list()
    drop.train()
    assert 0.0 in drop(ones).to_list()
    assert ox.Dropout(0.0)(ones).to_list() == ones.to_list()
    for bad in (1.0, -0.1, float("nan")):
        raises(ValueError, lambda: ox.Dropout(bad))


def fit_line(make_optimizer, steps):
    """Fits y = 2x + 1 with one Linear layer and returns the final mean squared error."""
    ox.manual_seed(0)
    x = ox.Tensor([1.0, 2.0, 3.0, 4.0], shape=[4, 1])
    y = ox.Tensor([3.0, 5.0, 7.0, 9.0], shape=[4, 1])
    model = ox.Linear(1, 1)
    opt = make_optimizer(model.parameters())
    first = ox.mse_loss(model(x), y).item()
    for _ in range(steps):
        opt.zero_grad()
        loss = ox.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    return first, ox.mse_loss(model(x), y).item()


def trained(first, last):
    """Training worked if the loss fell by two orders of magnitude to near zero (Adam hovers around 1e-3, it does not settle)."""
    return last < 0.05 and last * 100 < first


def test_sgd_and_adam_train_a_model():
    assert trained(*fit_line(lambda p: ox.SGD(p, lr=0.02, momentum=0.9), 300))
    assert trained(*fit_line(lambda p: ox.Adam(p, lr=0.05), 600))
    assert trained(*fit_line(lambda p: ox.Adam(p, lr=0.05, betas=(0.8, 0.9), weight_decay=1e-4), 600))
    opt = ox.SGD([ox.Tensor([1.0], shape=[1], requires_grad=True)], lr=0.1)
    assert opt.lr == 0.1
    opt.set_learning_rate(0.5)
    assert opt.lr == 0.5


def test_optimizer_misuse_raises_ordinary_errors():
    p = [ox.Tensor([1.0], shape=[1], requires_grad=True)]
    raises(ValueError, lambda: ox.SGD([], lr=0.1))
    raises(ValueError, lambda: ox.Adam([]))
    raises(ValueError, lambda: ox.SGD(p, lr=-1.0))
    raises(ValueError, lambda: ox.SGD(p, lr=float("nan")))
    raises(ValueError, lambda: ox.SGD(p, momentum=-0.5))
    raises(ValueError, lambda: ox.Adam(p, betas=(1.0, 0.999)))
    raises(ValueError, lambda: ox.Adam(p, eps=-1e-8))
    raises(ValueError, lambda: ox.Adam(p).set_learning_rate(-1.0))


def test_loss_functions_match_hand_computed_values():
    pred, target = ox.Tensor([1.0, 2.0], shape=[2]), ox.Tensor([1.0, 4.0], shape=[2])
    assert vals(ox.mse_loss(pred, target)) == [2.0] and vals(ox.l1_loss(pred, target)) == [1.0]
    assert vals(ox.huber_loss(pred, target)) == [0.75] and vals(ox.huber_loss(pred, target, delta=2.0)) == [1.0]
    flat = ox.Tensor.zeros([2, 2])
    ln2 = round(math.log(2), 4)
    assert vals(ox.cross_entropy(flat, [0, 1])) == [ln2]
    assert vals(ox.cross_entropy(flat, ox.Tensor.from_i64([1, 0], [2]))) == [ln2]
    assert vals(ox.nll_loss(flat.log_softmax(1), [0, 1])) == [ln2]
    smooth = ox.cross_entropy(ox.Tensor([2.0, 0.0, 0.0, 2.0], shape=[2, 2]), [0, 1], label_smoothing=0.1).item()
    hard = ox.cross_entropy(ox.Tensor([2.0, 0.0, 0.0, 2.0], shape=[2, 2]), [0, 1]).item()
    assert smooth > hard > 0.0
    assert vals(ox.binary_cross_entropy_with_logits(ox.Tensor.zeros([4]), ox.Tensor.ones([4]))) == [ln2]
    assert ox.focal_loss(ox.Tensor([2.0, 0.0, 0.0, 2.0], shape=[2, 2]), [0, 1]).item() >= 0.0
    # every loss supports autograd
    logits = ox.Tensor([0.5, -0.5, 0.1, 0.9], shape=[2, 2], requires_grad=True)
    ox.cross_entropy(logits, [0, 1]).backward()
    g = logits.grad().to_list()
    assert abs(sum(g)) < 1e-5 and any(abs(v) > 1e-3 for v in g)


def test_losses_reject_bad_input_without_panicking():
    flat = ox.Tensor.zeros([2, 2])
    raises(RuntimeError, lambda: ox.cross_entropy(flat, [0, 5]))  # class index out of range
    raises(RuntimeError, lambda: ox.cross_entropy(flat, [0]))  # wrong number of targets
    raises(RuntimeError, lambda: ox.cross_entropy(ox.Tensor.zeros([4]), [0, 1]))  # not [batch, classes]
    raises(ValueError, lambda: ox.cross_entropy(flat, [0, -1]))
    raises(ValueError, lambda: ox.cross_entropy(flat, [0, 1], label_smoothing=1.0))
    raises(TypeError, lambda: ox.cross_entropy(flat, "ab"))
    raises(TypeError, lambda: ox.cross_entropy(flat, None))
    raises(RuntimeError, lambda: ox.mse_loss(ox.Tensor.zeros([2]), ox.Tensor.zeros([3])))
    scalar, vec = ox.Tensor([1.0], shape=[]), ox.Tensor.zeros([3])
    raises(RuntimeError, lambda: ox.focal_loss(scalar, [0]))
    raises(RuntimeError, lambda: ox.focal_loss(vec, [0]))
    for fn in (ox.info_nce_loss, ox.vicreg_loss, ox.barlow_twins_loss, ox.siglip_loss):
        raises(RuntimeError, lambda: fn(scalar, scalar))
        raises(RuntimeError, lambda: fn(vec, vec))
    raises(RuntimeError, lambda: ox.Embedding(5, 3)(ox.Tensor.from_i64([7], [1])))  # index out of range
    raises(RuntimeError, lambda: ox.Embedding(2**62, 2**62))
    raises(ValueError, lambda: ox.LayerNorm([2, 3]))


def test_absurd_arguments_are_ordinary_errors_not_panics_hangs_or_aborts():
    # Each of these used to panic (a `PanicException`, which `except Exception` misses), hang or abort the process.
    m = 2**63
    cases = {
        "iteration count": lambda: ox.ChirikovStandardMapEngine(0.5).iterate_map(0.5, 0.5, m),
        "pipeline stage": lambda: ox.PipelineSchedule1F1B(2, 2, 2),
        "empty photonic mesh": lambda: ox.PhotonicMesh(0),
        "empty photonic layer": lambda: ox.PhotonicLinear(0, 2),
        "tiny 3D RoPE": lambda: ox.RotaryEmbedding3D(2, 10000.0),
        "plucker index": lambda: ox.AmplituhedronGrassmannianEngine([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]).plucker_minor(7, 2),
        "kv cache size": lambda: ox.QuantizedPagedKVCache(2**40, 2**40, 2, 2, "fp8"),  # overflows usize on any machine
        "kv cache read": lambda: ox.QuantizedPagedKVCache(2, 2, 2, 2, "fp8").read_token_dequantized(2, 2),
        "dual pipe": lambda: ox.DualPipeSchedule(m, m),
        "clamp bounds": lambda: ox.Tensor([1.0], shape=[1]).clamp(1.0, 0.5),
        "layer count": lambda: ox.ConvNeXt(3, 8, 2**31, 4, 10),
        "tpu pod": lambda: ox.TpuCluster("v5e", 2**31, 2, 1),
        "z limit": lambda: ox.SelbergSieveEngine.selberg_j_sum(2**31),
        "non-finite t": lambda: ox.RiemannSiegelZEngine.hardy_z_main_sum(float("inf")),
        "relativistic steps": lambda: ox.RelativisticLbm1D([1.0] * 4, [0.0] * 4, 1.0).step(m),
        "engine integer overflow": lambda: ox.AbcSzpiroArakelovEngine.abc_quality(2**32, 2),
        "vlm zero query heads": lambda: ox.MultimodalVLM(8, 4, 3, 16, 1, 4, 24, 16, 2, 0, 2, 32, 32),
        "vlm layer count": lambda: ox.MultimodalVLM(8, 4, 3, 16, 1, 4, 24, 16, 2**40, 4, 2, 32, 32),
        "vlm sequence length": lambda: ox.MultimodalVLM(8, 4, 3, 16, 1, 4, 24, 16, 2, 4, 2, 32, 2**64 - 1),
    }
    for name, call in cases.items():
        try:
            call()
        except RuntimeError:
            continue
        except BaseException as e:  # noqa: BLE001
            raise AssertionError(f"{name}: expected RuntimeError, got {type(e).__name__}: {e}")
        raise AssertionError(f"{name}: nothing raised")
    # a long closed-form series is summed, not looped
    assert ox.MotivicHodgeEngine.projective_space_motive(10**12, 0.5, 1.0) > 1.99


def test_a_rust_panic_is_an_ordinary_runtime_error():
    # pyo3's PanicException derives from BaseException; the extension re-parents it so `except Exception` works.
    try:
        ox._raise_test_panic()
    except Exception as e:  # noqa: BLE001
        assert isinstance(e, RuntimeError) and "deliberate test panic" in str(e), (type(e), e)
    else:
        raise AssertionError("no exception")


def test_nn_and_optim_namespaces_alias_the_top_level_names():
    for ns in (ox.nn, ox.optim):
        assert ns.__all__ and all(getattr(ns, n) is getattr(ox, n) for n in ns.__all__)
    assert ox.nn.cross_entropy is ox.cross_entropy and ox.optim.SGD is ox.SGD and ox.nn.Embedding is ox.Embedding


def test_clone_is_an_independent_copy():
    a = sq()
    b = a.clone()
    assert vals(b) == vals(a) and b.shape == a.shape and not a.shares_storage(b)
    i = ox.Tensor.from_i64([2**60 + 1, 5], [2]).clone()
    assert i.dtype == "i64" and i.to_i64_list() == [2**60 + 1, 5]  # no round trip through f64
    x = ox.Tensor([1.0, 2.0], shape=[2], requires_grad=True)
    (x.clone() * 3.0).sum().backward()  # the copy stays in the autograd graph
    assert vals(x.grad()) == [3, 3]


def test_numpy_export():
    try:
        import numpy as np
    except ImportError:
        raises(ImportError, lambda: sq().numpy())
        return
    arr = sq().numpy()
    assert isinstance(arr, np.ndarray) and arr.dtype == np.float32 and arr.shape == (2, 2)
    assert arr.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    arr[0, 0] = 99.0  # a copy: writing to it must not touch the tensor
    assert sq().numpy()[0, 0] == 1.0 and vals(sq()) == [1, 2, 3, 4]
    assert ox.Tensor.from_i64([1, 2**40], [2]).numpy().tolist() == [1, 2**40]
    assert ox.Tensor.zeros([2], dtype="f64").numpy().dtype == np.float64
    assert ox.Tensor.zeros([2], dtype="bool").numpy().dtype == np.bool_
    assert ox.Tensor.zeros([2], dtype="u8").numpy().dtype == np.uint8
    assert ox.Tensor.zeros([2], dtype="bf16").numpy().dtype == np.float32
    assert ox.Tensor.zeros([0, 3]).numpy().shape == (0, 3) and ox.Tensor([2.5], shape=[]).numpy().shape == ()
    assert sq().t().numpy().tolist() == [[1.0, 3.0], [2.0, 4.0]]  # non-contiguous view


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"All {len(tests)} Python ergonomics tests passed for oxideon v{ox.__version__}!")
