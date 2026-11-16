"""Tests for Oxideon text generation, LLM components, and advanced loss functions."""

import numpy as np
import pytest
import oxideon as ox
from oxideon.text import (
    RotaryEmbedding,
    SwiGLU,
    CausalSelfAttention,
    CausalTransformerDecoderLayer,
    CausalTransformerDecoder,
    top_k_top_p_filtering,
    generate,
)


def test_rmsnorm_forward():
    norm = ox.nn.RMSNorm(dim=16, eps=1e-5)
    x = ox.randn([2, 4, 16])
    out = norm(x)
    assert out.shape == [2, 4, 16]
    arr = out.to_numpy()
    # Check that root-mean-square along last dim is approximately 1.0 (for unit weights)
    rms = np.sqrt(np.mean(arr ** 2, axis=-1))
    np.testing.assert_allclose(rms, 1.0, atol=1e-2)


def test_rotary_embedding_and_swiglu():
    dim = 16
    rope = RotaryEmbedding(dim=dim, max_seq_len=64)
    # [B, H, T, D]
    q = ox.randn([2, 2, 8, dim])
    k = ox.randn([2, 2, 8, dim])
    q_rot, k_rot = rope(q, k, seq_len=8)
    assert q_rot.shape == [2, 2, 8, dim]
    assert k_rot.shape == [2, 2, 8, dim]

    swiglu = SwiGLU(in_features=16, hidden_features=32)
    x = ox.randn([2, 8, 16])
    out = swiglu(x)
    assert out.shape == [2, 8, 16]


def test_causal_self_attention_and_decoder_layer():
    dim = 16
    num_heads = 2
    attn = CausalSelfAttention(dim=dim, num_heads=num_heads)
    x = ox.randn([2, 6, dim])
    attn_out = attn(x)
    assert attn_out.shape == [2, 6, dim]

    layer = CausalTransformerDecoderLayer(dim=dim, num_heads=num_heads, ffn_dim=32)
    layer_out = layer(x)
    assert layer_out.shape == [2, 6, dim]


def test_causal_transformer_decoder_and_generation():
    vocab_size = 50
    dim = 16
    model = CausalTransformerDecoder(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=2,
        num_heads=2,
        max_seq_len=64,
    )

    prompt = [1, 5, 12]
    prompt_tensor = ox.Tensor(prompt, shape=[1, len(prompt)])
    logits = model(prompt_tensor)
    assert logits.shape == [1, len(prompt), vocab_size]

    generated = generate(
        model,
        prompt_ids=prompt,
        max_new_tokens=5,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
    )
    assert len(generated) == len(prompt) + 5
    assert generated[:len(prompt)] == prompt


def test_top_k_top_p_filtering():
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    # Top-K = 2 should keep only 4.0 and 5.0
    filtered = top_k_top_p_filtering(logits, top_k=2)
    assert np.isneginf(filtered[0])
    assert np.isneginf(filtered[1])
    assert np.isneginf(filtered[2])
    assert filtered[3] == 4.0
    assert filtered[4] == 5.0


def test_advanced_losses():
    # 1. HuberLoss
    huber = ox.nn.HuberLoss(delta=1.0)
    pred = ox.Tensor([1.0, 2.0, 5.0], shape=[3])
    target = ox.Tensor([1.2, 1.0, 2.0], shape=[3])
    loss = huber(pred, target)
    assert loss.to_numpy().ndim == 0 or loss.shape == [1] or loss.shape == []
    assert float(loss.to_numpy()) > 0.0

    # 2. CosineEmbeddingLoss
    cos_loss = ox.nn.CosineEmbeddingLoss(margin=0.0)
    x1 = ox.Tensor([1.0, 0.0, 0.0, 1.0], shape=[2, 2])
    x2 = ox.Tensor([1.0, 0.0, 1.0, 0.0], shape=[2, 2])
    target = ox.Tensor([1.0, -1.0], shape=[2])
    l_val = cos_loss(x1, x2, target)
    assert float(l_val.to_numpy()) >= 0.0

    # 3. GaussianNLLLoss
    gnll = ox.nn.GaussianNLLLoss()
    pred = ox.randn([4])
    target = ox.randn([4])
    var = ox.Tensor([1.0, 1.0, 1.0, 1.0], shape=[4])
    loss_g = gnll(pred, target, var)
    assert float(loss_g.to_numpy()) > 0.0
