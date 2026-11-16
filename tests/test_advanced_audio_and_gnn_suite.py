"""Comprehensive test suite for Oxideon Audio Domain Engine and Geometric GNN Suite."""

import numpy as np
import pytest

import oxideon as ox
from oxideon.audio import functional as af, transforms as at
from oxideon.gnn import (
    scatter_add,
    scatter_mean,
    scatter_max,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    MessagePassing,
    GINConv,
    EdgeConv,
)


def test_audio_functional_resample_and_deltas():
    # 1. Resample: 16kHz -> 8kHz
    t = np.linspace(0, 1, 16000, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    wave = ox.tensor(sine[None, :])

    downsampled = af.resample(wave, 16000, 8000)
    assert downsampled.shape == [1, 8000]

    # 2. Spectrogram and Deltas
    spec_transform = at.Spectrogram(n_fft=400, hop_length=160)
    spec = spec_transform(wave)
    assert spec.shape[0] == 1
    assert spec.shape[1] == 201

    deltas = af.compute_deltas(spec, win_length=5)
    assert deltas.shape == spec.shape

    # 3. Audio Gain and Dither
    gained = af.gain(wave, gain_db=6.0)
    assert gained.shape == wave.shape

    dithered = af.dither(wave, density_function="TPDF")
    assert dithered.shape == wave.shape


def test_audio_specaugment_frequency_and_time_masking():
    spec = ox.randn([1, 80, 100])

    fm = at.FrequencyMasking(freq_mask_param=15)
    f_masked = fm(spec)
    assert f_masked.shape == [1, 80, 100]

    tm = at.TimeMasking(time_mask_param=20)
    t_masked = tm(spec)
    assert t_masked.shape == [1, 80, 100]


def test_audio_griffin_lim_reconstruction():
    spec = ox.randn([1, 201, 25])
    reconstructed = af.griffin_lim(spec, n_fft=400, hop_length=160, n_iter=3)
    assert reconstructed.shape[0] == 1
    assert reconstructed.shape[1] > 0


def test_gnn_scatter_and_global_pooling():
    src = ox.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ])
    index = ox.tensor([0, 1, 0, 1])

    # 1. Scatter Add
    s_add = scatter_add(src, index, dim=0, dim_size=2)
    assert s_add.shape == [2, 2]
    assert np.allclose(s_add.to_numpy(), [[6.0, 8.0], [10.0, 12.0]])

    # 2. Scatter Mean
    s_mean = scatter_mean(src, index, dim=0, dim_size=2)
    assert s_mean.shape == [2, 2]
    assert np.allclose(s_mean.to_numpy(), [[3.0, 4.0], [5.0, 6.0]])

    # 3. Scatter Max
    s_max, s_argmax = scatter_max(src, index, dim=0, dim_size=2)
    assert s_max.shape == [2, 2]
    assert np.allclose(s_max.to_numpy(), [[5.0, 6.0], [7.0, 8.0]])

    # 4. Global Graph Pooling
    g_add = global_add_pool(src, batch=index)
    assert g_add.shape == [2, 2]
    g_mean = global_mean_pool(src, batch=index)
    assert g_mean.shape == [2, 2]
    g_max = global_max_pool(src, batch=index)
    assert g_max.shape == [2, 2]


def test_gnn_message_passing_gin_and_edgeconv():
    x = ox.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], requires_grad=True)
    edge_index = ox.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ])

    # 1. GINConv with autograd
    gin_mlp = ox.nn.Sequential(
        ox.nn.Linear(2, 4),
        ox.nn.ReLU(),
        ox.nn.Linear(4, 2),
    )
    gin = GINConv(gin_mlp, eps=0.1)
    y_gin = gin(x, edge_index)
    assert y_gin.shape == [3, 2]

    loss = y_gin.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == [3, 2]

    # 2. Dynamic EdgeConv
    edge_mlp = ox.nn.Linear(4, 3)
    edge_conv = EdgeConv(edge_mlp, aggr="max")
    y_edge = edge_conv(x, edge_index)
    assert y_edge.shape == [3, 3]
