"""Comprehensive Test Suite for Oxideon Industrial Deep Learning Ecosystem Parity.

Verifies:
- Advanced tensor indexing (boolean masks, integer arrays, in-place assignment, unbind, view_as)
- Hardware acceleration controls (ox.cuda.Stream, Event, amp)
- Autograd verification (ox.autograd.gradcheck, detect_anomaly)
- Extended learning rate schedulers (MultiStepLR, ConstantLR, LambdaLR, SequentialLR)
- Advanced dataset samplers and pipelines (ConcatDataset, Samplers, DataLoader integration)
- Special mathematical functions library (ox.special)
- Fast Fourier Transform suite (ox.fft)
- Complete probabilistic distributions & KL divergences (ox.distributions)
- Execution and memory profiler (ox.profiler)
- JIT compilation and graph tracing (ox.jit)
- Extended neural network layers (RNN, Bilinear, CosineSimilarity, Padding, Loss functions)
"""

import math
import os
import tempfile
import numpy as np
import pytest

import oxideon as ox
from oxideon import nn, optim


# ===========================================================================
# 1. Advanced Tensor Indexing, Mutation & Views
# ===========================================================================

def test_tensor_advanced_indexing_and_mutation():
    # 1. Boolean masking
    x = ox.tensor([[1.0, -2.0], [-3.0, 4.0]])
    mask = x > 0
    positives = x[mask]
    assert positives.numel() == 2
    assert sorted(positives.tolist()) == [1.0, 4.0]

    # 2. In-place boolean mutation
    x[x < 0] = 0.0
    assert x.tolist() == [[1.0, 0.0], [0.0, 4.0]]

    # 3. Item assignment
    x[0, 1] = 99.0
    assert x[0, 1].item() == 99.0

    # 4. Slice tensor assignment
    x[1] = ox.tensor([5.0, 6.0])
    assert x.tolist() == [[1.0, 99.0], [5.0, 6.0]]

    # 5. unbind
    slices = x.unbind(dim=0)
    assert len(slices) == 2
    assert slices[0].tolist() == [1.0, 99.0]
    assert slices[1].tolist() == [5.0, 6.0]

    # 6. view_as and expand_as
    target = ox.zeros([4, 1])
    reshaped = x.view_as(target)
    assert reshaped.shape == (4, 1)

    y = ox.tensor([[1.0], [2.0]])
    expanded = y.expand_as(x)
    assert expanded.shape == x.shape

    # 7. new_zeros, new_ones, new_tensor
    z = x.new_zeros([3, 2])
    assert z.shape == (3, 2) and z.sum().item() == 0.0
    o = x.new_ones([2, 2])
    assert o.shape == (2, 2) and o.sum().item() == 4.0
    nt = x.new_tensor([10.0, 20.0])
    assert nt.tolist() == [10.0, 20.0]

    # 8. Top-level operations available as methods on Tensor
    t = ox.tensor([3.0, 1.0, 4.0, 2.0])
    vals, idxs = t.topk(2)
    assert vals.tolist() == [4.0, 3.0]
    assert idxs.tolist() == [2, 0]

    sorted_vals, sorted_idxs = t.sort()
    assert sorted_vals.tolist() == [1.0, 2.0, 3.0, 4.0]

    clamped = t.clamp(1.5, 3.5)
    assert clamped.tolist() == [3.0, 1.5, 3.5, 2.0]

    t.clamp_(1.5, 3.5)
    assert t.tolist() == [3.0, 1.5, 3.5, 2.0]


# ===========================================================================
# 2. CUDA & Autograd Verification
# ===========================================================================

def test_cuda_and_autograd_subsystems():
    # 1. CUDA Stream and Event APIs
    stream = ox.cuda.Stream()
    event_start = ox.cuda.Event(enable_timing=True)
    event_end = ox.cuda.Event(enable_timing=True)

    with ox.cuda.stream(stream):
        event_start.record(stream)
        x = ox.randn([10, 10])
        y = x @ x
        event_end.record(stream)

    elapsed_ms = event_start.elapsed_time(event_end)
    assert elapsed_ms >= 0.0

    # 2. Autocast and GradScaler in cuda.amp
    scaler = ox.cuda.amp.GradScaler()
    assert scaler.get_scale() == 65536.0

    with ox.cuda.amp.autocast():
        a = ox.ones([4, 4])
        b = a + 2.0
    assert b.sum().item() == 48.0

    # 3. Autograd gradcheck
    def simple_fn(t):
        return (t ** 2).sum()

    inp = ox.tensor([1.5, -2.5, 3.0], requires_grad=True)
    passed = ox.autograd.gradcheck(simple_fn, [inp], eps=1e-4, atol=1e-3, rtol=1e-2)
    assert passed is True

    # 4. Detect anomaly context
    with ox.autograd.detect_anomaly():
        z = ox.tensor([2.0, 4.0], requires_grad=True)
        loss = (z * 3.0).sum()
        loss.backward()
        assert z.grad.tolist() == [3.0, 3.0]


# ===========================================================================
# 3. Extended Learning Rate Schedulers
# ===========================================================================

def test_extended_lr_schedulers():
    model = nn.Linear(4, 2)
    opt = optim.SGD(model.parameters(), lr=0.1)

    # 1. MultiStepLR
    s1 = optim.lr_scheduler.MultiStepLR(opt, milestones=[2, 4], gamma=0.5)
    assert math.isclose(s1.get_last_lr()[0], 0.1)
    s1.step()  # epoch 0 -> 1
    assert math.isclose(s1.get_last_lr()[0], 0.1)
    s1.step()  # epoch 1 -> 2 (milestone reached)
    assert math.isclose(s1.get_last_lr()[0], 0.05)
    s1.step()  # epoch 2 -> 3
    assert math.isclose(s1.get_last_lr()[0], 0.05)
    s1.step()  # epoch 3 -> 4 (milestone reached)
    assert math.isclose(s1.get_last_lr()[0], 0.025)

    # 2. ConstantLR
    opt2 = optim.SGD(model.parameters(), lr=0.1)
    s2 = optim.lr_scheduler.ConstantLR(opt2, factor=0.5, total_iters=2)
    assert math.isclose(s2.get_last_lr()[0], 0.05)
    s2.step()  # epoch 0 -> 1
    assert math.isclose(s2.get_last_lr()[0], 0.05)
    s2.step()  # epoch 1 -> 2 (end of decay)
    assert math.isclose(s2.get_last_lr()[0], 0.1)

    # 3. LambdaLR
    opt3 = optim.SGD(model.parameters(), lr=0.1)
    s3 = optim.lr_scheduler.LambdaLR(opt3, lr_lambda=lambda ep: 0.5 ** ep)
    assert math.isclose(s3.get_last_lr()[0], 0.1)
    s3.step()  # epoch 0 -> 1
    assert math.isclose(s3.get_last_lr()[0], 0.05)
    s3.step()  # epoch 1 -> 2
    assert math.isclose(s3.get_last_lr()[0], 0.025)


# ===========================================================================
# 4. Data Loading Ecosystem & Samplers
# ===========================================================================

def test_data_samplers_and_concat():
    t1 = ox.tensor([1.0, 2.0, 3.0])
    t2 = ox.tensor([4.0, 5.0, 6.0])
    ds1 = ox.utils.data.TensorDataset(t1)
    ds2 = ox.utils.data.TensorDataset(t2)

    # ConcatDataset
    concat_ds = ox.utils.data.ConcatDataset([ds1, ds2])
    assert len(concat_ds) == 6
    assert concat_ds[0].item() == 1.0
    assert concat_ds[3].item() == 4.0

    # SequentialSampler and BatchSampler
    seq_sampler = ox.utils.data.SequentialSampler(concat_ds)
    batch_sampler = ox.utils.data.BatchSampler(seq_sampler, batch_size=2, drop_last=False)
    batches = list(batch_sampler)
    assert batches == [[0, 1], [2, 3], [4, 5]]

    # WeightedRandomSampler
    w_sampler = ox.utils.data.WeightedRandomSampler(weights=[1.0, 10.0, 1.0], num_samples=10, replacement=True)
    assert len(w_sampler) == 10
    w_indices = list(w_sampler)
    assert len(w_indices) == 10

    # DataLoader with sampler
    loader = ox.utils.data.DataLoader(concat_ds, sampler=seq_sampler, batch_size=2)
    loaded_batches = [b.tolist() for b in loader]
    assert loaded_batches == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


# ===========================================================================
# 5. Special Mathematical Functions Library (ox.special)
# ===========================================================================

def test_special_functions_suite():
    # 1. erf, erfc, erfinv
    x = ox.tensor([0.0, 0.5, -0.5])
    erf_x = ox.special.erf(x)
    assert math.isclose(erf_x[0].item(), 0.0, abs_tol=1e-5)

    rec_x = ox.special.erfinv(erf_x)
    assert np.allclose(rec_x.to_numpy(), x.to_numpy(), atol=1e-4)

    # 2. expit (sigmoid) and logit
    sig_x = ox.special.expit(x)
    assert math.isclose(sig_x[0].item(), 0.5, abs_tol=1e-5)
    rec_logit = ox.special.logit(sig_x)
    assert np.allclose(rec_logit.to_numpy(), x.to_numpy(), atol=1e-4)

    # 3. gammaln and digamma
    pos = ox.tensor([1.0, 2.0, 5.0])
    lgamma_val = ox.special.gammaln(pos)
    assert math.isclose(lgamma_val[0].item(), 0.0, abs_tol=1e-5)
    assert math.isclose(lgamma_val[2].item(), math.log(24.0), abs_tol=1e-4)

    psi_val = ox.special.digamma(pos)
    assert math.isclose(psi_val[0].item(), -0.5772156649, abs_tol=1e-4)

    # 4. ndtr (Normal CDF) and ndtri (Inverse Normal CDF)
    z = ox.tensor([0.0, 1.0, -1.0])
    cdf = ox.special.ndtr(z)
    assert math.isclose(cdf[0].item(), 0.5, abs_tol=1e-5)
    z_rec = ox.special.ndtri(cdf)
    assert np.allclose(z_rec.to_numpy(), z.to_numpy(), atol=1e-4)

    # 5. Bessel i0, i1
    i0_val = ox.special.i0(ox.tensor([0.0, 1.0]))
    assert math.isclose(i0_val[0].item(), 1.0, abs_tol=1e-5)
    i1_val = ox.special.i1(ox.tensor([0.0, 1.0]))
    assert math.isclose(i1_val[0].item(), 0.0, abs_tol=1e-5)
    assert math.isclose(i1_val[1].item(), 0.565159, abs_tol=1e-4)

    # 6. sinc and entr
    s = ox.special.sinc(ox.tensor([0.0]))
    assert math.isclose(s[0].item(), 1.0, abs_tol=1e-5)
    e = ox.special.entr(ox.tensor([1.0, math.e]))
    assert math.isclose(e[0].item(), 0.0, abs_tol=1e-5)
    assert math.isclose(e[1].item(), -math.e, abs_tol=1e-4)


# ===========================================================================
# 6. Fast Fourier Transform (ox.fft)
# ===========================================================================

def test_fft_suite():
    # 1. 1D real FFT and inverse
    signal = ox.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    spectrum = ox.fft.rfft(signal)
    # output has shape [n // 2 + 1, 2]
    assert spectrum.shape == (5, 2)

    recovered = ox.fft.irfft(spectrum, n=8)
    assert np.allclose(recovered.to_numpy(), signal.to_numpy(), atol=1e-5)

    # 2. Complex FFT and IFFT
    c_spectrum = ox.fft.fft(signal)
    assert c_spectrum.shape == (8, 2)
    c_recovered = ox.fft.ifft(c_spectrum, n=8)
    # Real part of complex recovered matches signal
    assert np.allclose(c_recovered.to_numpy()[..., 0], signal.to_numpy(), atol=1e-5)

    # 3. Frequency calculations
    freqs = ox.fft.fftfreq(8, d=0.1)
    assert freqs.shape == (8,)

    r_freqs = ox.fft.rfftfreq(8, d=0.1)
    assert r_freqs.shape == (5,)

    # 4. Spectral shift
    shifted = ox.fft.fftshift(freqs)
    unshifted = ox.fft.ifftshift(shifted)
    assert np.allclose(unshifted.to_numpy(), freqs.to_numpy(), atol=1e-5)


# ===========================================================================
# 7. Probability Distributions & KL Divergence (ox.distributions)
# ===========================================================================

def test_distributions_suite():
    # 1. Normal Distribution
    norm = ox.distributions.Normal(loc=0.0, scale=1.0)
    samples = norm.sample([100])
    assert samples.shape == (100,)
    log_prob_zero = norm.log_prob(ox.tensor([0.0]))
    assert math.isclose(log_prob_zero.item(), -0.5 * math.log(2.0 * math.pi), abs_tol=1e-4)

    # CDF & ICDF
    cdf_zero = norm.cdf(ox.tensor([0.0]))
    assert math.isclose(cdf_zero.item(), 0.5, abs_tol=1e-4)
    icdf_half = norm.icdf(ox.tensor([0.5]))
    assert math.isclose(icdf_half.item(), 0.0, abs_tol=1e-4)

    # 2. Bernoulli
    bern = ox.distributions.Bernoulli(probs=0.7)
    b_samples = bern.sample([50])
    assert set(b_samples.tolist()).issubset({0.0, 1.0})

    # 3. Categorical
    cat = ox.distributions.Categorical(probs=[0.1, 0.7, 0.2])
    cat_samples = cat.sample([20])
    assert set(cat_samples.tolist()).issubset({0.0, 1.0, 2.0})

    # 4. Uniform
    uni = ox.distributions.Uniform(low=1.0, high=5.0)
    u_samples = uni.sample([30])
    assert all(1.0 <= val <= 5.0 for val in u_samples.tolist())

    # 5. MultivariateNormal
    mvn = ox.distributions.MultivariateNormal(loc=[0.0, 0.0], covariance_matrix=[[1.0, 0.5], [0.5, 1.0]])
    mvn_sample = mvn.sample([10])
    assert mvn_sample.shape == (10, 2)
    mvn_lp = mvn.log_prob(ox.tensor([0.0, 0.0]))
    assert mvn_lp.numel() == 1

    # 6. KL Divergence (Normal || Normal)
    p = ox.distributions.Normal(0.0, 1.0)
    q = ox.distributions.Normal(1.0, 2.0)
    kl = ox.distributions.kl_divergence(p, q)
    # Analytical: log(2) + (1 + 1) / (2 * 4) - 0.5 = 0.693147 + 0.25 - 0.5 = 0.443147
    assert math.isclose(kl.item(), 0.443147, abs_tol=1e-4)


# ===========================================================================
# 8. Execution Profiler (ox.profiler)
# ===========================================================================

def test_profiler_suite():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trace_path = os.path.join(tmp_dir, "trace.json")

        with ox.profiler.profile(
            activities=[ox.profiler.ProfilerActivity.CPU],
            record_shapes=True,
        ) as prof:
            with ox.profiler.record_function("matmul_block"):
                x = ox.randn([20, 20])
                y = x @ x

            with ox.profiler.record_function("norm_block"):
                n = y.norm()

        table_str = prof.table()
        assert "matmul_block" in table_str
        assert "norm_block" in table_str

        prof.export_chrome_trace(trace_path)
        assert os.path.exists(trace_path)
        assert os.path.getsize(trace_path) > 50


# ===========================================================================
# 9. JIT Compilation and Graph Tracing (ox.jit)
# ===========================================================================

class ToyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def test_jit_tracing_suite():
    model = ToyModule()
    example = ox.randn([1, 4])

    traced = ox.jit.trace(model, example)
    assert isinstance(traced, ox.jit.ScriptModule)
    out = traced(example)
    assert out.shape == (1, 2)
    assert "return" in traced.code

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_name = f.name
    try:
        traced.save(tmp_name)
        loaded = ox.jit.load(tmp_name)
        assert isinstance(loaded, ox.jit.ScriptModule)
        out_loaded = loaded(example)
        assert out_loaded.shape == (1, 2)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


# ===========================================================================
# 10. Extended Neural Network Modules & Losses
# ===========================================================================

def test_nn_extended_layers():
    # 1. RNNCell & RNN
    rnn_cell = nn.RNNCell(input_size=4, hidden_size=6, nonlinearity="tanh")
    x_step = ox.randn([2, 4])
    h_step = rnn_cell(x_step)
    assert h_step.shape == (2, 6)

    rnn = nn.RNN(input_size=4, hidden_size=6, num_layers=2, batch_first=True)
    x_seq = ox.randn([2, 5, 4])
    out_seq, h_n = rnn(x_seq)
    assert out_seq.shape == (2, 5, 6)
    assert h_n.shape == (2, 2, 6)

    # 2. Bilinear
    bilinear = nn.Bilinear(4, 5, 3)
    x1 = ox.randn([2, 4])
    x2 = ox.randn([2, 5])
    bi_out = bilinear(x1, x2)
    assert bi_out.shape == (2, 3)

    # 3. CosineSimilarity and PairwiseDistance
    cos_sim = nn.CosineSimilarity(dim=-1)
    sim = cos_sim(x1, x1)
    assert np.allclose(sim.to_numpy(), np.ones([2]), atol=1e-4)

    pwd = nn.PairwiseDistance(p=2)
    dist = pwd(x1, x1)
    assert np.allclose(dist.to_numpy(), np.zeros([2]), atol=1e-4)

    # 4. ChannelShuffle
    cs = nn.ChannelShuffle(groups=2)
    feat = ox.randn([2, 4, 8, 8])
    cs_out = cs(feat)
    assert cs_out.shape == (2, 4, 8, 8)

    # 5. Padding modules
    img = ox.randn([1, 1, 4, 4])
    zp = nn.ZeroPad2d(1)
    assert zp(img).shape == (1, 1, 6, 6)

    cp = nn.ConstantPad2d(1, value=5.0)
    assert cp(img).shape == (1, 1, 6, 6)

    rp = nn.ReflectionPad2d(1)
    assert rp(img).shape == (1, 1, 6, 6)

    # 6. Losses
    logits = ox.tensor([[2.0, 1.0, 0.1], [0.1, 3.0, 0.5]])
    targets = ox.tensor([0, 1])

    mm_loss = nn.MultiMarginLoss()
    l1 = mm_loss(logits, targets)
    assert l1.item() >= 0.0

    multi_targets = ox.tensor([[0, 1, -1], [1, -1, -1]])
    mlm_loss = nn.MultiLabelMarginLoss()
    l2 = mlm_loss(logits, multi_targets)
    assert l2.item() >= 0.0

    binary_targets = ox.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mlsm_loss = nn.MultiLabelSoftMarginLoss()
    l3 = mlsm_loss(logits, binary_targets)
    assert l3.item() >= 0.0
