"""Efficiency benchmarks: MXFP4, FP8, INT4, KV eviction, Radix Prefix Cache, FlashAttention-2, Sparse SpMM, Mamba SSM, Evals.

    python tools/efficiency_bench.py

Reports median wall-clock timings for the Oxideon inference-efficiency and
scientific compute stack. Numbers depend on build (debug/release) and CPU;
see docs/BENCHMARKS.md for reference measurements and theoretical scaling models.
"""
import statistics
import time

import numpy as np

import oxideon as ox
from oxideon.flash_attn import flash_attention_forward, TiledAttention
from oxideon.sparse_matrix import CSRMatrix
from oxideon.prefix_cache import RadixPrefixCache
from oxideon.quantization_advanced import (
    quantize_fp8_e4m3,
    dequantize_fp8_e4m3,
    quantize_fp8_e5m2,
    dequantize_fp8_e5m2,
    WeightOnlyInt4Quantizer,
)
from oxideon.state_space import MambaBlock


def timed(fn, reps=5):
    fn()  # warm-up: allocations, thread pools
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return [s * 1e3 for s in samples]


def med(samples):
    return statistics.median(samples)


print(f"oxideon {ox.__version__}")

# ================================================================
# 1. Quantization Suite
# ================================================================
print("=" * 68)
print("Quantization Suite: 2048x2048 f32 (4.19M parameters)")
print("=" * 68)
w = ox.Tensor.randn([2048, 2048])
w_np = w.to_numpy()

# 1.1 OCP MXFP4
tq_mx = timed(lambda: ox.quantize_mxfp4(w), 3)
codes_mx, scales_mx = ox.quantize_mxfp4(w)
td_mx = timed(lambda: ox.dequantize_mxfp4(codes_mx, scales_mx), 3)
back_mx = ox.dequantize_mxfp4(codes_mx, scales_mx).to_numpy()
err_mx = np.abs(back_mx - w_np)

print(f"  {'OCP MXFP4 (E2M1 + E8M0/32)':<32}quant: {med(tq_mx):>6.1f} ms | dequant: {med(td_mx):>6.1f} ms | MAE: {err_mx.mean():.4f} | 4.25 b/p (7.53x)")

# 1.2 FP8 E4M3
tq_e4 = timed(lambda: quantize_fp8_e4m3(w), 3)
q_e4, scale_e4 = quantize_fp8_e4m3(w)
td_e4 = timed(lambda: dequantize_fp8_e4m3(q_e4, scale_e4), 3)
back_e4 = dequantize_fp8_e4m3(q_e4, scale_e4).to_numpy()
err_e4 = np.abs(back_e4 - w_np)

print(f"  {'FP8 E4M3 (1S 4E 3M)':<32}quant: {med(tq_e4):>6.1f} ms | dequant: {med(td_e4):>6.1f} ms | MAE: {err_e4.mean():.4f} | 8.00 b/p (4.00x)")

# 1.3 FP8 E5M2
tq_e5 = timed(lambda: quantize_fp8_e5m2(w), 3)
q_e5, scale_e5 = quantize_fp8_e5m2(w)
td_e5 = timed(lambda: dequantize_fp8_e5m2(q_e5, scale_e5), 3)
back_e5 = dequantize_fp8_e5m2(q_e5, scale_e5).to_numpy()
err_e5 = np.abs(back_e5 - w_np)

print(f"  {'FP8 E5M2 (1S 5E 2M)':<32}quant: {med(tq_e5):>6.1f} ms | dequant: {med(td_e5):>6.1f} ms | MAE: {err_e5.mean():.4f} | 8.00 b/p (4.00x)")

# 1.4 Weight-Only INT4 (group size 32)
tq_i4 = timed(lambda: WeightOnlyInt4Quantizer.quantize(w, group_size=32), 3)
q_i4, s_i4, z_i4 = WeightOnlyInt4Quantizer.quantize(w, group_size=32)
td_i4 = timed(lambda: WeightOnlyInt4Quantizer.dequantize(q_i4, s_i4, z_i4, group_size=32), 3)
back_i4 = WeightOnlyInt4Quantizer.dequantize(q_i4, s_i4, z_i4, group_size=32).to_numpy()
err_i4 = np.abs(back_i4 - w_np)

print(f"  {'Weight-Only INT4 (g=32)':<32}quant: {med(tq_i4):>6.1f} ms | dequant: {med(td_i4):>6.1f} ms | MAE: {err_i4.mean():.4f} | 4.00 b/p (8.00x)")

# ================================================================
# 2. KV Caches & Prefix Sharing
# ================================================================
print("=" * 68)
print("KV caches: prefill 1024 + 64 decode steps (B=1, H=8, D=64, budget=1024)")
print("=" * 68)


def rand_kv(n):
    k = ox.from_numpy(np.random.randn(1, n, 8, 64).astype(np.float32))
    v = ox.from_numpy(np.random.randn(1, n, 8, 64).astype(np.float32))
    return k, v


def bench_cache(name, make, scored=False):
    cache = make()
    k0, v0 = rand_kv(1024)
    steps = [rand_kv(1) for _ in range(64)]
    prefill_scores = np.random.randn(1, 8, 1024, 1024)
    scores = [np.random.randn(1, 8, 1, 1025) for _ in range(64)]

    def prefill():
        cache.reset()
        if scored:
            cache.update(k0, v0, prefill_scores)
        else:
            cache.update(k0, v0)

    def decode():
        if scored:
            for (k, v), s in zip(steps, scores):
                cache.update(k, v, s)
        else:
            for k, v in steps:
                cache.update(k, v)

    prefill()
    decode()  # warm-up
    t0 = time.perf_counter()
    prefill()
    pre_ms = (time.perf_counter() - t0) * 1e3
    per_step = []
    for _ in range(3):
        prefill()
        t0 = time.perf_counter()
        decode()
        per_step.append((time.perf_counter() - t0) * 1e3 / 64)
    print(f"  {name:<36}{med(per_step):>10.3f} ms/decode-step  (prefill {pre_ms:.1f} ms)")


bench_cache("StandardKVCache (sliding 1024)", lambda: ox.StandardKVCache(1, 1024, 8, 64, sliding_window=1024))
bench_cache("StreamingSinkCache (sink 4)", lambda: ox.StreamingSinkCache(1, 1024, 8, 64, n_sink=4))
bench_cache(
    "ScoredEvictionCache (h2o, recent 64)",
    lambda: ox.ScoredEvictionCache(1, 1024, 8, 64, recent_window=64, policy="h2o"),
    scored=True,
)
full_mb = 2 * 8192 * 8 * 64 * 4 / 1e6
budget_mb = 2 * 1024 * 8 * 64 * 4 / 1e6
print(f"  8k ctx: full {full_mb:.1f} MB vs budgeted {budget_mb:.1f} MB (8.0x less)")

# Radix Prefix KV-Cache micro-benchmark
print("-" * 68)
print("Radix Tree Prefix KV-Cache (vLLM/SGLang style, block_size=16)")
print("-" * 68)
prefix_cache = RadixPrefixCache(block_size=16)
sys_tokens = list(range(128))
sys_blocks = list(range(8))
prefix_cache.insert(sys_tokens, sys_blocks)

# Benchmark prefix matching
t_match = timed(lambda: prefix_cache.match_prefix(sys_tokens + list(range(128, 256))), reps=10)
m_toks, m_blocks = prefix_cache.match_prefix(sys_tokens + list(range(128, 256)))
print(f"  Prefix match latency (128 cached tokens):  {med(t_match):>6.3f} ms  (matched {len(m_toks)} tokens, {len(m_blocks)} blocks)")

# Benchmark insertion of new branches
t_insert = timed(lambda: prefix_cache.insert(list(range(200, 300)), list(range(10, 17))), reps=5)
print(f"  Tree insertion latency (100 new tokens):   {med(t_insert):>6.3f} ms")

# Benchmark LRU eviction
t_evict = timed(lambda: prefix_cache.evict_lru(num_blocks=4), reps=5)
print(f"  LRU eviction latency (4 blocks freed):     {med(t_evict):>6.3f} ms")

# ================================================================
# 3. Attention Forward & FlashAttention-2 Tiling
# ================================================================
print("=" * 68)
print("Attention Forward Kernels (hidden=128, heads=4, batch=2)")
print("=" * 68)


def bench_attn(name, make, is_qkv=False, seqs=(256, 1024)):
    for s in seqs:
        mod = make()
        x = ox.from_numpy(np.random.randn(2, s, 128).astype(np.float32))
        call_fn = (lambda: mod(x, x, x)) if is_qkv else (lambda: mod(x))
        samples = timed(call_fn, 5 if s <= 256 else 3)
        print(f"  {name:<30}s={s:<6}{med(samples):>10.1f} ms")


bench_attn("MultiQueryAttention", lambda: ox.MultiQueryAttention(128, 4))
bench_attn(
    "MLAAttention",
    lambda: ox.MLAAttention(128, 4, q_lora_rank=16, kv_lora_rank=16,
                             qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16),
)
bench_attn(
    "NativeSparseAttention",
    lambda: ox.NativeSparseAttention(128, 4, block_size=32, top_blocks=2, window_size=64),
)
bench_attn("FlashAttention-2 (TiledAttention)", lambda: TiledAttention(128, 4, block_size=64, causal=True), is_qkv=True)

# Direct FlashAttention-2 tiling scaling
print("-" * 68)
print("FlashAttention-2 Exact Forward Scaling (heads=4, head_dim=32, batch=2)")
print("-" * 68)
for seq_len in (256, 512, 1024):
    q_t = ox.from_numpy(np.random.randn(2, 4, seq_len, 32).astype(np.float32))
    k_t = ox.from_numpy(np.random.randn(2, 4, seq_len, 32).astype(np.float32))
    v_t = ox.from_numpy(np.random.randn(2, 4, seq_len, 32).astype(np.float32))
    t_fa = timed(lambda: flash_attention_forward(q_t, k_t, v_t, block_size_r=64, block_size_c=64, causal=True), 3)
    print(f"  FlashAttention-2 (causal)   seq_len={seq_len:<6}{med(t_fa):>10.2f} ms")

# ================================================================
# 4. Sparse Linear Algebra & State-Space Models
# ================================================================
print("=" * 68)
print("Sparse Computing: CSR SpMM vs Dense GEMM (1024x1024 @ 1024x64)")
print("=" * 68)
dense_mat = np.random.randn(1024, 1024).astype(np.float32)
x_dense = ox.from_numpy(np.random.randn(1024, 64).astype(np.float32))
t_dense_gemm = timed(lambda: ox.from_numpy(dense_mat) @ x_dense, reps=5)
dense_gemm_ms = med(t_dense_gemm)
print(f"  {'Dense GEMM (baseline)':<34}{dense_gemm_ms:>10.2f} ms")

for sparsity in (0.90, 0.95, 0.99):
    mask = np.random.rand(1024, 1024) > sparsity
    sparse_data = dense_mat * mask
    csr = CSRMatrix.from_dense(sparse_data)
    t_spmm = timed(lambda: csr.spmm(x_dense), reps=5)
    spmm_ms = med(t_spmm)
    speedup = dense_gemm_ms / max(spmm_ms, 1e-6)
    print(f"  CSR SpMM ({int(sparsity*100)}% sparse, nnz={csr.nnz:<6})   {spmm_ms:>10.2f} ms  ({speedup:.2f}x speedup)")

print("-" * 68)
print("Selective State Space (Mamba SSM: D=64, N=16, expand=2, batch=2)")
print("-" * 68)
mamba = MambaBlock(d_model=64, d_state=16, expand=2)
for seq_len in (128, 256, 512):
    x_mamba = ox.from_numpy(np.random.randn(2, seq_len, 64).astype(np.float32))
    t_mamba = timed(lambda: mamba(x_mamba), reps=3)
    print(f"  MambaBlock Forward          seq_len={seq_len:<6}{med(t_mamba):>10.2f} ms")

# ================================================================
# 5. Offline Evals Harness
# ================================================================
print("=" * 68)
print("Evals harness overhead (fake model, no downloads)")
print("=" * 68)


class FakeLoglik:
    def generate(self, prompts):
        return ["A"] * len(prompts)

    def loglikelihoods(self, contexts, continuations):
        return [0.0 if c.strip() == "A" else -10.0 for c in continuations]


mc = ox.evals.MultipleChoiceTask("mc", ox.evals.demo_choices(50))
samples = timed(lambda: mc.evaluate(FakeLoglik()), 5)
print(f"  {'50 choice items (loglik)':<24}{med(samples):>10.1f} ms  ({50 / (med(samples) / 1e3):.0f} items/s)")


class FakeMath:
    def __init__(self, answers):
        self.answers = answers

    def generate(self, prompts):
        return [f"result {a}" for a in self.answers[: len(prompts)]]


items = [ox.evals.MathItem(f"What is {i} plus {i}?", str(2 * i)) for i in range(50)]
math_task = ox.evals.MathTask("math", items)
answers = [it.answer for it in items]
samples = timed(lambda: math_task.evaluate(FakeMath(answers)), 5)
print(f"  {'50 math items (generate)':<24}{med(samples):>10.1f} ms  ({50 / (med(samples) / 1e3):.0f} items/s)")
print("=" * 68)
print("done")
