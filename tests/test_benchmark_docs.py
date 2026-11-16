import numpy as np
import pytest

import oxideon as ox
from oxideon.flash_attn import flash_attention_forward, TiledAttention
from oxideon.prefix_cache import RadixPrefixCache
from oxideon.quantization_advanced import (
    quantize_fp8_e4m3,
    dequantize_fp8_e4m3,
    quantize_fp8_e5m2,
    dequantize_fp8_e5m2,
    WeightOnlyInt4Quantizer,
)
from oxideon.state_space import MambaBlock
from oxideon.sparse_matrix import CSRMatrix
from oxideon.sparse_attn import SparseMultiheadAttention, build_bigbird_mask, build_sliding_window_mask
from oxideon.synthetic_data import GaussianCopulaSynthesizer, CTGANGenerator, DifferentialPrivacySynthesizer
from oxideon.uncertainty_quant import DirichletLoss, decompose_uncertainty, brier_score
from oxideon.interpret import TCAV, IntegratedHessians, difference_from_reference_multipliers
from oxideon.func import vmap, grad, jacrev, hessian
from oxideon.models import ResNet18, VisionTransformer, MiniGPT
from oxideon.signal import windows, correlate, convolve, resample_poly


def test_docs_radix_prefix_cache_example():
    cache = RadixPrefixCache(block_size=16)
    system_tokens = list(range(100, 132))
    system_blocks = [0, 1]
    cache.insert(system_tokens, system_blocks)

    user_prompt = system_tokens + [200, 201, 202]
    matched_tokens, matched_blocks = cache.match_prefix(user_prompt)
    assert len(matched_tokens) == 32
    assert matched_blocks == [0, 1]

    freed = cache.evict_lru(num_blocks=1)
    assert len(freed) >= 1


def test_docs_quantization_suite_example():
    w = ox.Tensor.randn([64, 64])

    # FP8 E4M3
    q_e4, scale_e4 = quantize_fp8_e4m3(w)
    rec_e4 = dequantize_fp8_e4m3(q_e4, scale_e4)
    assert rec_e4.to_numpy().shape == (64, 64)

    # FP8 E5M2
    q_e5, scale_e5 = quantize_fp8_e5m2(w)
    rec_e5 = dequantize_fp8_e5m2(q_e5, scale_e5)
    assert rec_e5.to_numpy().shape == (64, 64)

    # Weight-Only INT4
    q_int4, s, z = WeightOnlyInt4Quantizer.quantize(w, group_size=32)
    rec_int4 = WeightOnlyInt4Quantizer.dequantize(q_int4, s, z, group_size=32)
    assert rec_int4.to_numpy().shape == (64, 64)


def test_docs_mamba_and_sparse_attention_example():
    # MambaBlock
    mamba = MambaBlock(d_model=32, d_state=8, d_conv=4, expand=2)
    x = ox.from_numpy(np.random.randn(2, 64, 32).astype(np.float32))
    out_m = mamba(x)
    assert out_m.to_numpy().shape == (2, 64, 32)

    # Sparse Multi-head Attention
    mask = build_bigbird_mask(seq_len=64, num_global=2, window_radius=4, num_random=2)
    assert mask.shape == (64, 64)

    smha = SparseMultiheadAttention(embed_dim=32, num_heads=2, window_radius=4)
    out_a = smha(x)
    assert out_a.to_numpy().shape == (2, 64, 32)


def test_docs_csr_sparse_matrix_example():
    dense = np.zeros((32, 32), dtype=np.float32)
    dense[0, 5] = 1.0
    dense[10, 20] = 2.5
    csr = CSRMatrix.from_dense(dense)
    assert csr.nnz == 2

    x = ox.from_numpy(np.ones((32, 4), dtype=np.float32))
    spmm_out = csr.spmm(x)
    assert spmm_out.to_numpy().shape == (32, 4)


def test_docs_canonical_models_example():
    # ResNet18
    resnet = ResNet18(in_channels=3, num_classes=5, hidden_dim=16)
    img = ox.from_numpy(np.random.randn(2, 3, 16, 16).astype(np.float32))
    logits = resnet(img)
    assert logits.to_numpy().shape == (2, 5)

    # VisionTransformer
    vit = VisionTransformer(img_size=16, patch_size=4, in_channels=3, num_classes=5, embed_dim=16, num_heads=2)
    out_v = vit(img)
    assert out_v.to_numpy().shape == (2, 5)

    # MiniGPT
    gpt = MiniGPT(vocab_size=20, max_seq_len=16, embed_dim=16, num_layers=1, num_heads=2)
    toks = gpt.generate(prompt_ids=[1, 2], max_new_tokens=4)
    assert len(toks) == 6


def test_docs_functional_and_interpretability():
    # vmap
    f = lambda a, b: a + b
    vf = vmap(f)
    t1 = ox.tensor(np.ones((4, 2), dtype=np.float32))
    t2 = ox.tensor(np.ones((4, 2), dtype=np.float32))
    res = vf(t1, t2)
    assert res.to_numpy().shape == (4, 2)

    # TCAV
    cav = ox.from_numpy(np.array([1.0, 0.0], dtype=np.float32))
    tcav = TCAV(cav)
    grads = ox.from_numpy(np.array([[1.0, 0.5], [-1.0, 0.2]], dtype=np.float32))
    score = tcav.tcav_score(grads)
    assert 0.0 <= score <= 1.0


def test_docs_synthetic_data_and_signal():
    # DP Synthesizer
    dp = DifferentialPrivacySynthesizer(epsilon=1.0, num_bins=8)
    dp.fit(np.random.randn(100))
    samples = dp.sample(n_samples=20)
    assert len(samples.to_numpy()) == 20

    # Signal
    w = windows.hann(16)
    assert len(w.to_numpy()) == 16
    sig = ox.from_numpy(np.random.randn(32).astype(np.float32))
    resampled = resample_poly(sig, up=2, down=1)
    assert len(resampled.to_numpy()) == 64


import importlib.util
from pathlib import Path
import pytest

def test_new_example_scripts_execute_cleanly():
    repo_root = Path(__file__).resolve().parent.parent
    examples_dir = repo_root / "examples"
    example_names = [
        "radix_prefix_serving.py",
        "mamba_selective_ssm.py",
        "sparse_long_context_attention.py",
        "advanced_quantization_and_qat.py",
        "sparse_matrix_spmm.py",
        "uncertainty_and_interpretability.py",
        "synthetic_tabular_generation.py",
        "science_numerical_methods.py",
        "science_statistics.py",
        "science_physics_sims.py",
        "train_with_scheduler_and_checkpoint.py",
    ]

    for name in example_names:
        script_path = examples_dir / name
        assert script_path.exists(), f"Missing example file: {script_path}"
        spec = importlib.util.spec_from_file_location(f"example_{name[:-3]}", str(script_path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main"), f"Example {name} missing main() function"
        # Execute main to ensure zero crashes and clean exit
        mod.main()
