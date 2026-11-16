"""Absurd integer sizes must raise ordinary errors, never panic or hang.

Regression tests for integer-overflow panics found by edge-value fuzzing
(`tools/edge_fuzz.py`): dimension products that overflowed `usize`/`i64`/`u32`
in constructors and math helpers. Infallible helpers saturate, widen, or use
floating-point math instead of panicking.
"""

import math

import numpy as np
import pytest

import oxideon as ox
import oxideon.diffusion_sampler as ox_sampler
import oxideon.models as ox_models
import oxideon.vision_transformer as ox_vt
import oxideon.signal as ox_signal

H = 1 << 62
I64 = (1 << 63) - 1
U64 = (1 << 64) - 1
U32 = (1 << 32) - 1


def test_conv_fan_in_overflow_is_an_error():
    with pytest.raises(RuntimeError, match="overflow"):
        ox.Conv2d(H, 2, 2)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.Conv3d(H, 2, 2)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.ConvNeXt(in_channels=H)
    # Ordinary sizes still work.
    ox.Conv2d(3, 8, 3)
    ox.Conv3d(2, 4, 2)


def test_recurrent_gate_dims_overflow_is_an_error():
    with pytest.raises(RuntimeError, match="overflow"):
        ox.BiLSTM(2, H, 1)
    ox.BiLSTM(4, 8, 1)


def test_patch_embedding_overflow_is_an_error():
    with pytest.raises(RuntimeError, match="overflow"):
        ox.DiT(1 << 40, 2, 2, 2, 2, 2, 2)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.MLPMixer(1 << 40, 2, 2, 2, 2, 2, 2, 2)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.MultimodalVLM(1 << 40, 2, 2, 2, 2, 2, 100, 8, 1, 2, 2, 8, 16)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.VisionLanguageActionModel(1 << 40, 2, 2, 2, 2, 100, 4, 4)


def test_gguf_config_overflow_is_an_error():
    with pytest.raises(RuntimeError, match="overflow"):
        ox.GGUFCausalLM(16, H, 2, 1)
    ox.GGUFCausalLM(16, 8, 2, 1)


def test_robotics_and_world_model_dims_overflow_is_an_error():
    with pytest.raises(RuntimeError, match="overflow"):
        ox.ActionChunkingTransformer(14, H, 16, 64, 0.01)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.FlowMatchingRoboticsPolicy(U64, 32, 64)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.UnifiedWorldModel(4, 7, H, 64, 2)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.TemporalInteractionGNN(2, I64, 16, 8)
    ox.ActionChunkingTransformer(14, 14, 16, 64, 0.01)


def test_video_and_vision_dims_overflow_is_an_error():
    with pytest.raises(RuntimeError, match="overflow"):
        ox.RectifiedFlowVideoDiT(H, 64, 4, 32)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.VideoVAE3d(H, 2, 2)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.MiniImageClassifier(H, 16, 16, 4)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.MiniImageClassifier(1, I64, 16, 4)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.MiniAutoEncoder(H, 16)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.MiniResNet(H, 4)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.TemporalConvNet(I64, [2, 2], 3)
    with pytest.raises(RuntimeError, match="overflow"):
        ox.Whisper(I64, 64, 2, 4, 2, 4, 128, 100)


def test_graph_wavelet_length_overflow_is_an_error():
    adj = [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(RuntimeError):
        ox.MultiscaleGraphWaveletTransform(adj, 1 << 32, [1.0])
    with pytest.raises(RuntimeError):
        ox.SpatioTemporalWaveletLayer(adj, 1 << 32, [1.0], 2, 2, 2)


def test_hafnian_and_autotune_overflow_is_an_error():
    with pytest.raises(RuntimeError, match="overflow"):
        ox.GaussianCvCircuit.hafnian([1.0, 2.0, 3.0, 4.0], 1 << 32)
    with pytest.raises(RuntimeError):
        ox.KernelAutoTuner().tune_gemm(H, 2, 2)


def test_nas_estimate_saturates_and_search_errors():
    genome = ox.ArchitectureGenome([64, 32], 2, 2)
    assert genome.estimate_parameters(16, 4) == 13732
    assert genome.estimate_parameters(U64, 2) == U64
    automl = ox.EvolutionaryAutoML(4, 1, 0.2, 100_000, 42)
    with pytest.raises(RuntimeError):
        automl.search(H, 2)


def test_absurd_math_inputs_do_not_panic():
    assert ox.CausalTriangulationEngine(1.0, 1.0).regge_action(2, U64, 2) == pytest.approx(3.6893488147419103e19)
    assert ox.ChowHirzebruchRiemannRoch().riemann_roch_curve(I64, 2, 2) == -18446744073709551610
    assert ox.ChowHirzebruchRiemannRoch().riemann_roch_curve(2, 1, 5) == 4
    assert math.isfinite(ox.ChowHirzebruchRiemannRoch().riemann_roch_surface(I64, 2, 2, 2))
    assert ox.ChowHirzebruchRiemannRoch().bezout_intersection_number([U64, 2]) == U64
    assert ox.ChowHirzebruchRiemannRoch().bezout_intersection_number([2, 3, 4]) == 24
    assert ox.HeegaardFloerKnotEngine().satisfies_legendrian_tau_bound(I64, 2, 2) is False
    assert ox.HeegaardFloerKnotEngine().satisfies_legendrian_tau_bound(0, 0, 1) is True
    assert ox.SpinNetworkLqgEngine.is_admissible_trivalent_intertwiner(U32, U32, U32) is False
    assert ox.SpinNetworkLqgEngine.is_admissible_trivalent_intertwiner(2, 2, 2) is True


def test_absurd_layer_counts_fail_fast():
    huge = 1 << 31
    with pytest.raises(ValueError, match="above the limit"):
        ox.LSTM(2, 2, huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox.GRU(2, 2, huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox.RNN(2, 2, huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox.TransformerEncoder(ox.TransformerEncoderLayer(8, 2, 16), huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox.TransformerDecoder(ox.TransformerDecoderLayer(8, 2, 16), huge)
    # Ordinary sizes still work.
    assert len(ox.LSTM(4, 8, 2).cells) == 2
    assert len(ox.GRU(4, 8, 2).cells) == 2
    assert len(ox.RNN(4, 8, 2).cells) == 2


def test_absurd_python_layer_counts_fail_fast():
    huge = 1 << 31
    with pytest.raises(ValueError, match="above the limit"):
        ox.AlibiPositionalBias(huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox.AudioResidualVQ(huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox.CapacityConstrainedMoE(8, 8, num_experts=huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox.RealNVP(2, num_layers=huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox.GraphTransformer(2, num_layers=huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox.PagedKVCache(huge, 16, 4, 8)
    with pytest.raises(ValueError, match="above the limit"):
        ox.MoELayer(8, 8, num_experts=huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox.VisionTransformer(depth=huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox_models.VisionTransformer(depth=huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox_vt.VisionTransformer(depth=huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox.MiniGPT(vocab_size=10, num_layers=huge)
    with pytest.raises(ValueError, match="above the limit"):
        ox.CausalTransformerDecoder(10, num_layers=huge)
    # Ordinary sizes still work.
    assert ox.AlibiPositionalBias(4).slopes.shape == (4,)
    assert len(ox.AudioResidualVQ(2, 16, 8).quantizers) == 2
    assert len(ox.GraphTransformer(2, num_layers=1).layers) == 1
    assert len(ox.MiniGPT(vocab_size=10, num_layers=1).layers) == 1
    assert len(ox.RealNVP(2, num_layers=1).layers) == 1


def test_absurd_buffer_sizes_fail_fast():
    huge = 1 << 32
    wrap = (1 << 63) - 1  # numpy arange silently returns empty() here unguarded
    for name in ("hann", "hamming", "blackman", "bartlett", "kaiser", "gaussian", "exponential", "chebyshev"):
        with pytest.raises(ValueError, match="above the limit"):
            getattr(ox, name + "_window")(huge)
        with pytest.raises(ValueError, match="above the limit"):
            getattr(ox, name + "_window")(wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox.SpatiotemporalRoPE(wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox.RotaryEmbedding(wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox.RotaryEmbedding(16, max_seq_len=wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox.MFCC(n_mfcc=wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox.MFCC(n_mels=wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox.DDPMScheduler(num_train_timesteps=wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox.DDIMScheduler(num_train_timesteps=wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox.DDIMScheduler().set_timesteps(wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox_sampler.EulerDiscreteScheduler(num_train_timesteps=wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox_sampler.EulerDiscreteScheduler().set_timesteps(wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox_sampler.DPMSolverMultistepScheduler(num_train_timesteps=wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox_sampler.DPMSolverMultistepScheduler().set_timesteps(wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox_sampler.DDIMScheduler(num_train_timesteps=wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox_sampler.DDIMScheduler().set_timesteps(wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox.add_self_loops([[0, 1], [1, 2]], wrap)
    with pytest.raises(ValueError, match="above the limit"):
        ox.build_sliding_window_mask(wrap, 3)
    with pytest.raises(ValueError, match="above the limit"):
        ox.topk(ox.tensor([3.0, 1.0, 2.0]), wrap)
    # Ordinary sizes still work.
    assert tuple(ox.hann_window(16).shape) == (16,)
    assert tuple(ox.kaiser_window(16, beta=5.0).shape) == (16,)
    assert tuple(ox.chebyshev_window(16).shape) == (16,)
    assert ox.add_self_loops([[0, 1], [1, 2]], 3).shape[1] == 5
    vals, _ = ox.topk(ox.tensor([3.0, 1.0, 2.0]), 2)
    assert [float(x) for x in vals.to_numpy()] == [3.0, 2.0]


def test_repeat_and_element_helper_boundaries():
    from oxideon.nn import _MAX_ELEMENTS, _MAX_REPEATS, _check_elements, _check_repeats

    assert _MAX_REPEATS == 1 << 16
    assert _MAX_ELEMENTS == 1 << 31
    assert _check_repeats("n", 1 << 16) == 1 << 16
    assert _check_elements("n", 1 << 31) == 1 << 31
    with pytest.raises(ValueError, match="above the limit"):
        _check_repeats("n", (1 << 16) + 1)
    with pytest.raises(ValueError, match="above the limit"):
        _check_elements("n", (1 << 31) + 1)
    # Non-count inputs pass through for downstream handling.
    assert _check_repeats("n", True) is True
    assert _check_elements("n", None) is None
    assert _check_repeats("n", 2.0) == 2.0
    # Integer-like scalars (e.g. numpy ints) are checked, not skipped.
    with pytest.raises(ValueError, match="above the limit"):
        _check_repeats("n", np.int64(1 << 40))
    with pytest.raises(ValueError, match="above the limit"):
        _check_elements("n", np.int64(1 << 40))


def test_vectorized_windows_match_analytical_values():
    assert ox.hann_window(8)[4].item() == pytest.approx(1.0, abs=1e-6)
    assert ox.hamming_window(16)[8].item() == pytest.approx(1.0, abs=1e-2)
    kaiser = ox.kaiser_window(16, beta=10.0).to_numpy()
    assert kaiser[8] == pytest.approx(1.0, abs=1e-6)
    sym = ox.kaiser_window(17, periodic=False, beta=10.0).to_numpy()
    assert sym[0] == pytest.approx(sym[-1], abs=1e-7)
    assert sym[8] == pytest.approx(1.0, abs=1e-6)
    assert ox.bartlett_window(9, periodic=False)[4].item() == pytest.approx(1.0, abs=1e-6)
    assert ox.blackman_window(16)[8].item() == pytest.approx(1.0, abs=1e-2)
    assert ox.gaussian_window(17, periodic=False)[8].item() == pytest.approx(1.0, abs=1e-6)
    assert ox.exponential_window(17, periodic=False)[8].item() == pytest.approx(1.0, abs=1e-6)
    assert tuple(ox.chebyshev_window(16).shape) == (16,)
    slopes = ox.AlibiPositionalBias(4).slopes
    assert list(slopes) == pytest.approx([2.0 ** (-8.0 * (i + 1) / 4) for i in range(4)], abs=1e-7)


def test_forward_size_params_fail_fast():
    huge = (1 << 63) - 1  # numpy arange silently returns empty() here unguarded
    alibi = ox.AlibiPositionalBias(4)
    with pytest.raises(ValueError, match="above the limit"):
        alibi(huge)
    rope = ox.SpatiotemporalRoPE(12)
    with pytest.raises(ValueError, match="above the limit"):
        rope(huge, 3, 3)
    with pytest.raises(ValueError, match="above the limit"):
        rope(2, huge, 3)
    with pytest.raises(ValueError, match="above the limit"):
        rope(2, 3, huge)
    # Ordinary sizes still work.
    assert alibi(8).shape == [4, 8, 8]
    (cos_t, _, _), _ = rope(2, 3, 3)
    assert cos_t.shape == (2, 2)


def test_stft_istft_reject_nonpositive_hop():
    x = ox.randn(1, 256)
    with pytest.raises(ValueError, match="hop_length"):
        ox_signal.stft(x, n_fft=32, hop_length=0)
    with pytest.raises(ValueError, match="hop_length"):
        ox_signal.stft(x, n_fft=32, hop_length=-4)
    spec = ox_signal.stft(x, n_fft=32, hop_length=16)
    with pytest.raises(ValueError, match="hop_length"):
        ox_signal.istft(spec, n_fft=32, hop_length=0)
    y = ox_signal.istft(spec, n_fft=32, hop_length=16)
    assert y.shape[0] == 1
