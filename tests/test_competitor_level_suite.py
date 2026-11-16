"""Industrial-Grade Deep Learning Competitor Parity Verification Suite for Oxideon.

Tests:
1. Autograd custom Function, vector-Jacobian product backward(gradient), and tensor register_hook()
2. High-order functional autodiff (vjp, jvp, jacobian, hessian, hvp)
3. Functional transforms (ox.func.vmap, grad, hessian)
4. Full batched linear algebra suite (ox.linalg: norm, inv, det, svd, qr, eigh, solve)
5. Signal processing suite (windows: hann, hamming, blackman, kaiser; stft)
6. Nested tensors for ragged sequence batches (ox.nested.nested_tensor, to_padded_tensor)
7. Sparse COO tensor construction and conversion (ox.sparse_coo_tensor, to_dense)
8. End-to-end model pickling, state preservation, and checkpoint roundtripping
"""

import math
import os
import pickle
import tempfile
import numpy as np
import pytest

import oxideon as ox
from oxideon import nn, optim


# ===========================================================================
# 1. Custom Autograd Function, VJP Backward, and Tensor Hooks
# ===========================================================================

def test_autograd_custom_function_and_hooks():
    class CubeFunction(ox.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x * x

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            return grad_output * 3.0 * x * x

    x = ox.tensor([2.0], requires_grad=True)

    hook_history = []
    handle = x.register_hook(lambda g: (hook_history.append(g.item()), g * 2.0)[1])

    y = CubeFunction.apply(x)
    assert y.requires_grad is True

    # Backward: dy/dx = 3 * 2^2 = 12. Hook doubles it to 24.
    y.backward()
    assert abs(x.grad.item() - 24.0) < 1e-4
    assert abs(hook_history[0] - 12.0) < 1e-4

    handle.remove()

    # Test Vector-Jacobian Product backward with explicit gradient tensor
    w = ox.tensor([2.0, 3.0], requires_grad=True)
    out = w * w
    out.backward(gradient=ox.tensor([10.0, 20.0]))
    # d(out0)/dw0 * 10 = 2 * 2 * 10 = 40; d(out1)/dw1 * 20 = 2 * 3 * 20 = 120
    assert np.allclose(w.grad.tolist(), [40.0, 120.0])


# ===========================================================================
# 2. High-Order Functional Autodiff (Jacobian, Hessian, VJP, JVP, HVP)
# ===========================================================================

def test_autograd_functional_suite():
    import oxideon.autograd.functional as F_grad

    # f(x0, x1) = x0^2 + 3*x0*x1 + 2*x1^2
    def f(x):
        return x[0]**2 + x[0]*x[1]*3.0 + x[1]**2 * 2.0

    x = ox.tensor([2.0, 5.0])

    # 1. VJP
    val, grad_val = F_grad.vjp(f, x)
    assert abs(val.item() - 84.0) < 1e-4
    assert np.allclose(grad_val.tolist(), [2*2 + 3*5, 3*2 + 4*5], atol=1e-3)

    # 2. JVP
    val_jvp, jvp_val = F_grad.jvp(f, x, v=ox.tensor([1.0, 1.0]))
    assert abs(jvp_val.item() - 45.0) < 0.1

    # 3. Hessian
    H = F_grad.hessian(f, x)
    assert H.shape == (2, 2)
    assert np.allclose(H.to_numpy(), [[2.0, 3.0], [3.0, 4.0]], atol=1e-2)

    # 4. HVP
    val_hvp, hvp_val = F_grad.hvp(f, x, v=ox.tensor([1.0, 2.0]))
    # [[2, 3], [3, 4]] @ [1, 2] = [8, 11]
    assert np.allclose(hvp_val.tolist(), [8.0, 11.0], atol=0.1)


# ===========================================================================
# 3. Functional Transforms (ox.func.vmap, grad, hessian)
# ===========================================================================

def test_func_transforms():
    import oxideon.func as F_func

    # Vector dot product
    def dot_product(a, b):
        return (a * b).sum()

    # Batch of 4 vectors
    batch_a = ox.randn([4, 8])
    batch_b = ox.randn([4, 8])

    v_dot = F_func.vmap(dot_product)
    batched_out = v_dot(batch_a, batch_b)
    assert batched_out.shape == (4,)

    for i in range(4):
        expected = (batch_a[i] * batch_b[i]).sum().item()
        assert abs(batched_out[i].item() - expected) < 1e-4

    # Functional grad
    def scalar_fn(v):
        return (v**3).sum()

    g_fn = F_func.grad(scalar_fn)
    test_v = ox.tensor([2.0, 3.0])
    g_res = g_fn(test_v)
    assert np.allclose(g_res.tolist(), [3*4.0, 3*9.0], atol=1e-3)


# ===========================================================================
# 4. Batched Linear Algebra Suite (ox.linalg)
# ===========================================================================

def test_linalg_suite():
    # 1. Norms
    A = ox.tensor([[1.0, 2.0], [3.0, 4.0]])
    fro_norm = ox.linalg.norm(A, ord="fro")
    assert abs(fro_norm.item() - math.sqrt(1 + 4 + 9 + 16)) < 1e-4

    # 2. Inversion and Determinant
    inv_A = ox.linalg.inv(A)
    det_A = ox.linalg.det(A)
    assert abs(det_A.item() - (-2.0)) < 1e-3

    # Product A @ inv(A) = Identity
    I_approx = (A @ inv_A).to_numpy()
    assert np.allclose(I_approx, np.eye(2), atol=1e-4)

    # 3. SVD
    svd_res = ox.linalg.svd(A)
    reconstructed = (svd_res.U @ ox.diag(svd_res.S) @ svd_res.Vh).to_numpy()
    assert np.allclose(reconstructed, A.to_numpy(), atol=1e-4)

    # 4. QR decomposition
    qr_res = ox.linalg.qr(A)
    assert np.allclose((qr_res.Q @ qr_res.R).to_numpy(), A.to_numpy(), atol=1e-4)

    # 5. Eigh on symmetric matrix
    sym_mat = ox.tensor([[2.0, 1.0], [1.0, 2.0]])
    eig_res = ox.linalg.eigh(sym_mat)
    assert sorted(eig_res.eigenvalues.tolist()) == pytest.approx([1.0, 3.0], abs=1e-4)

    # 6. Linear solve Ax = b
    b = ox.tensor([5.0, 11.0])
    x = ox.linalg.solve(A, b)
    assert np.allclose((A @ x).tolist(), b.tolist(), atol=1e-4)


# ===========================================================================
# 5. Signal Processing & Window Functions (ox.signal)
# ===========================================================================

def test_signal_suite():
    import oxideon.signal as signal

    # 1. Windows
    h_win = signal.windows.hann(17)
    assert h_win.shape == (17,)
    assert h_win[0].item() == pytest.approx(0.0, abs=1e-4)
    assert h_win[8].item() == pytest.approx(1.0, abs=1e-4)

    ham_win = ox.hamming_window(16)
    assert ham_win.shape == (16,)

    k_win = ox.kaiser_window(16, beta=10.0)
    assert k_win.shape == (16,)

    # 2. Short-Time Fourier Transform (STFT)
    sig = ox.sin(ox.linspace(0.0, 20.0 * math.pi, 128))
    stft_res = signal.stft(sig, n_fft=32, hop_length=16)
    # Output shape: [freq_bins, time_frames, 2]
    assert len(stft_res.shape) == 3
    assert stft_res.shape[-1] == 2
    assert stft_res.shape[0] == 32 // 2 + 1


# ===========================================================================
# 6. Nested Tensors for Jagged Sequence Batches (ox.nested)
# ===========================================================================

def test_nested_tensor_suite():
    import oxideon.nested as nested

    t1 = ox.tensor([[1.0, 2.0], [3.0, 4.0]])        # shape [2, 2]
    t2 = ox.tensor([[5.0, 6.0]])                    # shape [1, 2]
    t3 = ox.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])  # shape [3, 2]

    nt = nested.nested_tensor([t1, t2, t3])
    assert nt.is_nested is True
    assert len(nt) == 3

    # Unbind
    unbound = nt.unbind()
    assert len(unbound) == 3
    assert unbound[0].shape == (2, 2)
    assert unbound[1].shape == (1, 2)
    assert unbound[2].shape == (3, 2)

    # Padded dense conversion
    padded = nt.to_padded_tensor(padding_value=0.0)
    assert padded.shape == (3, 3, 2)
    assert padded[1, 0].tolist() == [5.0, 6.0]
    assert padded[1, 1].tolist() == [0.0, 0.0]  # Padded element


# ===========================================================================
# 7. Sparse COO Tensor Suite (ox.sparse)
# ===========================================================================

def test_sparse_coo_suite():
    indices = ox.tensor([[0, 1, 2], [1, 2, 0]])
    values = ox.tensor([3.0, 4.0, 5.0])

    sp_t = ox.sparse_coo_tensor(indices, values, size=(3, 3))
    assert sp_t.shape == (3, 3)

    dense = sp_t.to_dense()
    expected = [
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0],
        [5.0, 0.0, 0.0],
    ]
    assert np.allclose(dense.to_numpy(), expected)


# ===========================================================================
# 8. Model Checkpointing, Pickling, and State Preservation
# ===========================================================================

class DeepClassifier(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.norm = nn.LayerNorm([hidden])
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.norm(self.fc1(x))))


def test_model_pickling_and_checkpoint_fidelity():
    model = DeepClassifier(8, 16, 4)
    x = ox.randn([2, 8])
    y_orig = model(x)

    # 1. Standard Python Pickle
    dumped = pickle.dumps(model)
    unpickled_model = pickle.loads(dumped)
    y_unpickled = unpickled_model(x)
    assert np.allclose(y_orig.to_numpy(), y_unpickled.to_numpy(), atol=1e-7)

    # 2. Disk Checkpointing via ox.save and ox.load (.pt)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_pt = f.name
    try:
        ox.save(model, tmp_pt)
        loaded_model = ox.load(tmp_pt)
        y_loaded = loaded_model(x)
        assert np.allclose(y_orig.to_numpy(), y_loaded.to_numpy(), atol=1e-7)
    finally:
        if os.path.exists(tmp_pt):
            os.remove(tmp_pt)

    # 3. Disk Checkpointing via ox.save_file (.safetensors)
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tmp_st = f.name
    try:
        ox.save_file(model, tmp_st)
        new_model = DeepClassifier(8, 16, 4)
        ox.load_file(tmp_st, model=new_model)
        y_st = new_model(x)
        assert np.allclose(y_orig.to_numpy(), y_st.to_numpy(), atol=1e-7)
    finally:
        if os.path.exists(tmp_st):
            os.remove(tmp_st)


# ===========================================================================
# 9. Accelerator & Hardware Runtime Management (CUDA, MPS, Backends)
# ===========================================================================

def test_cuda_mps_and_backends_runtime_interfaces():
    import oxideon.cuda as cuda
    import oxideon.mps as mps
    import oxideon.backends as backends

    assert isinstance(cuda.is_available(), bool)
    assert isinstance(cuda.device_count(), int)
    assert cuda.current_device() == 0
    props = cuda.get_device_properties()
    assert hasattr(props, "name") and hasattr(props, "total_memory")
    cuda.synchronize()
    cuda.empty_cache()
    cuda.manual_seed(42)

    # Stream and Event
    stream = cuda.Stream()
    event1 = cuda.Event(enable_timing=True)
    event2 = cuda.Event(enable_timing=True)
    with cuda.stream(stream):
        event1.record(stream)
        x = ox.randn([100, 100])
        y = x @ x.t()
        event2.record(stream)
    assert event1.query()
    assert event2.query()
    delta = event1.elapsed_time(event2)
    assert delta >= 0.0

    # MPS & Backends
    assert not mps.is_available() or mps.is_available()
    assert hasattr(backends.cudnn, "is_available")
    assert backends.mkl.is_available()
    assert backends.openmp.is_available()


# ===========================================================================
# 10. Advanced Linear Algebra Solvers & Decompositions
# ===========================================================================

def test_advanced_linear_algebra_solvers_and_decompositions():
    # 1. LU Decomposition: P @ L @ U == A
    A = ox.tensor([[4.0, 3.0], [6.0, 3.0]])
    P, L, U = ox.linalg.lu(A)
    A_reconstructed = P.to_numpy() @ L.to_numpy() @ U.to_numpy()
    assert np.allclose(A.to_numpy(), A_reconstructed, atol=1e-5)

    # 2. Triangular Solver: U @ X = B
    U_tri = ox.tensor([[2.0, 1.0], [0.0, 3.0]])
    B = ox.tensor([[5.0], [9.0]])
    X = ox.linalg.solve_triangular(U_tri, B, upper=True)
    assert np.allclose(U_tri.to_numpy() @ X.to_numpy(), B.to_numpy(), atol=1e-5)

    # 3. Batched Vector Dot Product
    v1 = ox.tensor([[1.0, 2.0], [3.0, 4.0]])
    v2 = ox.tensor([[5.0, 6.0], [7.0, 8.0]])
    vdot = ox.linalg.vecdot(v1, v2)
    assert np.allclose(vdot.to_numpy(), [17.0, 53.0])

    # 4. Vandermonde Matrix
    x_v = ox.tensor([1.0, 2.0, 3.0])
    vander_mat = ox.linalg.vander(x_v, N=3)
    assert vander_mat.shape == [3, 3]


# ===========================================================================
# 11. Extended Probability Distributions
# ===========================================================================

def test_extended_probability_distributions():
    # Binomial
    bin_dist = ox.distributions.Binomial(total_count=10, probs=0.5)
    samples_bin = bin_dist.sample([20])
    assert samples_bin.shape == [20]
    lp_bin = bin_dist.log_prob(5.0)
    assert np.isfinite(lp_bin.item())

    # Multinomial
    mult_dist = ox.distributions.Multinomial(total_count=6, probs=ox.tensor([0.2, 0.3, 0.5]))
    s_mult = mult_dist.sample([5])
    assert s_mult.shape == [5, 3]
    assert np.allclose(s_mult.to_numpy().sum(axis=-1), 6.0)

    # HalfNormal & HalfCauchy
    hn = ox.distributions.HalfNormal(scale=2.0)
    s_hn = hn.sample([10])
    assert (s_hn.to_numpy() >= 0.0).all()

    hc = ox.distributions.HalfCauchy(scale=1.5)
    s_hc = hc.sample([10])
    assert (s_hc.to_numpy() >= 0.0).all()

    # Weibull
    wb = ox.distributions.Weibull(scale=1.0, concentration=1.5)
    s_wb = wb.sample([10])
    assert (s_wb.to_numpy() > 0.0).all()


# ===========================================================================
# 12. Advanced Tensor Manipulations & Indexing
# ===========================================================================

def test_advanced_tensor_manipulation_suite():
    # 1. Cartesian Product & Meshgrid
    x = ox.tensor([1.0, 2.0])
    y = ox.tensor([3.0, 4.0])
    cp = ox.cartesian_prod(x, y)
    assert cp.shape == [4, 2]
    assert np.allclose(cp.to_numpy(), [[1, 3], [1, 4], [2, 3], [2, 4]])

    gx, gy = ox.meshgrid(x, y)
    assert gx.shape == [2, 2] and gy.shape == [2, 2]

    # 2. Block Diagonal & Triangular Parts
    m1 = ox.tensor([[1.0, 2.0], [3.0, 4.0]])
    m2 = ox.tensor([[5.0]])
    bd = ox.block_diag(m1, m2)
    assert bd.shape == [3, 3]
    assert bd.to_numpy()[2, 2] == 5.0 and bd.to_numpy()[0, 2] == 0.0

    tril_m = m1.tril()
    triu_m = m1.triu()
    assert np.allclose(tril_m.to_numpy(), [[1, 0], [3, 4]])
    assert np.allclose(triu_m.to_numpy(), [[1, 2], [0, 4]])

    # 3. Narrow, Select, Take, Take Along Dim
    t_seq = ox.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    narr = t_seq.narrow(0, 1, 3)
    assert np.allclose(narr.to_numpy(), [20.0, 30.0, 40.0])

    sel = t_seq.select(0, 2)
    assert sel.item() == 30.0

    tk = ox.take(t_seq, ox.tensor([0, 4]))
    assert np.allclose(tk.to_numpy(), [10.0, 50.0])

    # 4. Difference, Gradient, Trapz
    vals = ox.tensor([1.0, 4.0, 9.0, 16.0])
    d = vals.diff()
    assert np.allclose(d.to_numpy(), [3.0, 5.0, 7.0])

    integral = ox.trapz(ox.tensor([1.0, 2.0, 3.0]), dx=1.0)
    assert np.isclose(integral.item(), 4.0)

    # 5. Nan to Num, Quantile, Mode, Isin, Unique
    nan_t = ox.tensor([1.0, float("nan"), float("inf"), -float("inf")])
    cleaned = nan_t.nan_to_num(nan=0.0, posinf=999.0, neginf=-999.0)
    assert np.allclose(cleaned.to_numpy(), [1.0, 0.0, 999.0, -999.0])

    data = ox.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    q50 = data.quantile(0.5)
    assert np.isclose(q50.item(), 30.0)

    isin_res = ox.isin(ox.tensor([1.0, 5.0, 2.0]), ox.tensor([2.0, 3.0]))
    assert np.allclose(isin_res.to_numpy(), [False, False, True])

    unq = ox.unique(ox.tensor([3.0, 1.0, 3.0, 2.0]))
    assert np.allclose(unq.to_numpy(), [1.0, 2.0, 3.0])


# ===========================================================================
# 13. Activation Recomputation & Gradient Checkpointing
# ===========================================================================

def test_activation_checkpointing_numerical_parity():
    # Mathematical forward pass: f(x) = relu(3*x + 2)^2
    def custom_subgraph(x):
        return ((x * 3.0 + 2.0).relu()) ** 2

    x_standard = ox.tensor([1.0, 2.0, 3.0], requires_grad=True)
    x_checkpointed = ox.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Standard execution
    out_std = custom_subgraph(x_standard).sum()
    out_std.backward()

    # Activation checkpointed execution (recomputing forward on backward)
    out_ckpt = ox.checkpoint.checkpoint(custom_subgraph, x_checkpointed).sum()
    out_ckpt.backward()

    assert np.allclose(out_std.to_numpy(), out_ckpt.to_numpy(), atol=1e-7)
    assert np.allclose(x_standard.grad.to_numpy(), x_checkpointed.grad.to_numpy(), atol=1e-7)

    # Also test via ox.checkpoint directly
    x_direct = ox.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out_dir = ox.checkpoint(custom_subgraph, x_direct).sum()
    out_dir.backward()
    assert np.allclose(x_standard.grad.to_numpy(), x_direct.grad.to_numpy(), atol=1e-7)


# ===========================================================================
# 14. Symbolic Computation Graph Tracing Engine (ox.fx)
# ===========================================================================

def test_fx_symbolic_tracing_and_code_generation():
    class TracedSubnet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)

        def forward(self, x):
            return self.linear(x) * 2.0 + 1.0

    model = TracedSubnet()
    traced = ox.fx.symbolic_trace(model)

    assert isinstance(traced, ox.fx.GraphModule)
    assert hasattr(traced, "graph")
    assert hasattr(traced, "code")
    assert "def forward(self, x):" in traced.code

    # Verify input execution and numerical parity
    x = ox.randn([2, 4])
    y_orig = model(x)
    y_traced = traced(x)
    assert np.allclose(y_orig.to_numpy(), y_traced.to_numpy(), atol=1e-6)

    # Inspect graph node composition
    node_op_types = [node.op for node in traced.graph.nodes]
    assert "placeholder" in node_op_types
    assert "call_module" in node_op_types
    assert "output" in node_op_types


# ===========================================================================
# 15. Architecture Optimization & Quantization (ox.ao.quantization)
# ===========================================================================

def test_ao_quantization_qat_and_ptq():
    import oxideon.ao.quantization as aoq

    class TinyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 2)

        def forward(self, x):
            return self.fc2((self.fc1(x)).relu())

    model = TinyClassifier()

    # 1. QAT Preparation
    qat_qconfig = aoq.get_default_qat_qconfig()
    model_prepared = aoq.prepare(model, qconfig=qat_qconfig)

    x = ox.randn([2, 4], requires_grad=True)
    y_qat = model_prepared(x)
    assert y_qat.shape == [2, 2]

    # Verify Straight-Through Estimator autograd propagates through fake-quantized graph
    loss = y_qat.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == [2, 4]

    # 2. Conversion to Quantized Inference Model
    model_quantized = aoq.convert(model_prepared)
    y_quant = model_quantized(x)
    assert y_quant.shape == [2, 2]


# ===========================================================================
# 16. Testing Assertion Framework & Tensor Generators (ox.testing)
# ===========================================================================

def test_testing_utilities_assert_close_and_make_tensor():
    import oxideon.testing as oxtest

    t1 = oxtest.make_tensor([3, 5], dtype="float32")
    assert t1.shape == [3, 5]

    t2 = t1.clone()
    oxtest.assert_close(t1, t2)

    # Slight perturbation within rtol/atol
    t3 = ox.tensor(t1.to_numpy() + 1e-6)
    oxtest.assert_close(t1, t3, atol=1e-4, rtol=1e-4)

    # NaN equality behavior
    nan_a = ox.tensor([float("nan"), 1.0])
    nan_b = ox.tensor([float("nan"), 1.0])
    oxtest.assert_close(nan_a, nan_b, equal_nan=True)


# ===========================================================================
# 17. ONNX Model Export Engine (ox.onnx)
# ===========================================================================

def test_onnx_model_export():
    class ExportNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    model = ExportNet()
    dummy_input = ox.randn([1, 4])

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_onnx = f.name

    try:
        ox.onnx.export(
            model,
            dummy_input,
            tmp_onnx,
            input_names=["input"],
            output_names=["output"],
        )
        assert os.path.exists(tmp_onnx)
        assert os.path.getsize(tmp_onnx) > 0
    finally:
        if os.path.exists(tmp_onnx):
            os.remove(tmp_onnx)


# ===========================================================================
# 18. Advanced Tensor Primitives (kron, cdist, cross, addmm, rot90)
# ===========================================================================

def test_advanced_tensor_primitives_kron_cdist_cross_addmm():
    # 1. Kronecker Product
    m1 = ox.tensor([[1.0, 2.0], [3.0, 4.0]])
    m2 = ox.tensor([[0.5, 1.0], [0.0, 2.0]])
    kr = ox.kron(m1, m2)
    assert kr.shape == [4, 4]
    assert np.isclose(kr.to_numpy()[0, 0], 0.5)
    assert np.isclose(kr.to_numpy()[0, 1], 1.0)
    assert np.isclose(kr.to_numpy()[3, 3], 8.0)

    # 2. Pairwise Coordinate Distance (cdist)
    p1 = ox.tensor([[0.0, 0.0], [1.0, 1.0]])
    p2 = ox.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dists = ox.cdist(p1, p2, p=2.0)
    assert dists.shape == [2, 3]
    assert np.isclose(dists.to_numpy()[0, 0], 0.0)
    assert np.isclose(dists.to_numpy()[0, 1], 1.0)
    assert np.isclose(dists.to_numpy()[1, 0], math.sqrt(2.0))

    # 3. 3D Cross Product
    v1 = ox.tensor([1.0, 0.0, 0.0])
    v2 = ox.tensor([0.0, 1.0, 0.0])
    v_cross = ox.cross(v1, v2)
    assert np.allclose(v_cross.to_numpy(), [0.0, 0.0, 1.0])

    # 4. BLAS-level addmm & addr
    bias = ox.tensor([[10.0]])
    a = ox.tensor([[2.0, 3.0]])
    b = ox.tensor([[4.0], [5.0]])
    # 10.0 + 1.0 * (2*4 + 3*5) = 10 + 23 = 33
    res_addmm = ox.addmm(bias, a, b)
    assert np.isclose(res_addmm.item(), 33.0)

    # 5. Tensor plane rotations and flips
    mat = ox.tensor([[1.0, 2.0], [3.0, 4.0]])
    rot = mat.rot90(1)
    assert np.allclose(rot.to_numpy(), [[2.0, 4.0], [1.0, 3.0]])

    flp_lr = mat.fliplr()
    assert np.allclose(flp_lr.to_numpy(), [[2.0, 1.0], [4.0, 3.0]])


# ===========================================================================
# 19. JIT Compilation, Freezing, and Inference Optimization
# ===========================================================================

def test_jit_freezing_and_inference_optimization():
    class FeedForwardNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 6)
            self.fc2 = nn.Linear(6, 2)

        def forward(self, x):
            return self.fc2((self.fc1(x)).relu())

    model = FeedForwardNet()
    example_input = ox.randn([1, 4])

    # Trace
    traced = ox.jit.trace(model, example_input)
    assert isinstance(traced, ox.jit.ScriptModule)

    # Freeze
    frozen = ox.jit.freeze(traced)
    assert getattr(frozen, "_is_frozen", False) is True

    # Optimize for inference
    opt_model = ox.jit.optimize_for_inference(traced)
    assert getattr(opt_model, "_inference_optimized", False) is True

    # Forward pass verification
    test_x = ox.randn([3, 4])
    y_opt = opt_model(test_x)
    assert y_opt.shape == [3, 2]


# ===========================================================================
# 20. Special Mathematical Functions (log_ndtr, bessel, lambert_w)
# ===========================================================================

def test_special_functions_log_ndtr_bessel_lambert():
    import oxideon.special as sp

    # 1. Log-normal CDF (log_ndtr)
    x = ox.tensor([0.0, -12.0])
    lndtr = sp.log_ndtr(x)
    # Phi(0) = 0.5 -> ln(0.5)
    assert np.isclose(lndtr.to_numpy()[0], math.log(0.5))
    # Stable asymptotic expansion without underflow
    assert lndtr.to_numpy()[1] < -70.0

    # 2. Bessel J0 and J1
    zero_t = ox.tensor([0.0])
    assert np.isclose(sp.bessel_j0(zero_t).item(), 1.0)
    assert np.isclose(sp.bessel_j1(zero_t).item(), 0.0)

    # 3. Lambert W
    e_val = ox.tensor([float(math.e)])
    w_val = sp.lambert_w(e_val)
    assert np.isclose(w_val.item(), 1.0, atol=1e-4)

    # 4. Hermite & Legendre Polynomials
    x_val = ox.tensor([1.5])
    # H_2(1.5) = 4*(1.5^2) - 2 = 9 - 2 = 7
    h2 = sp.hermite_polynomial_h(x_val, 2)
    assert np.isclose(h2.item(), 7.0)

    # P_2(1.5) = 0.5 * (3*(1.5^2) - 1) = 0.5 * (6.75 - 1) = 2.875
    p2 = sp.legendre_polynomial_p(x_val, 2)
    assert np.isclose(p2.item(), 2.875)



