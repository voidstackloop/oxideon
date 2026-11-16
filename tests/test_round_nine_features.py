"""Unit tests for Oxideon Round 9 high-impact AI features:
  1. Paged and standard KV Cache (ox.PagedKVCache, ox.StandardKVCache)
  2. Speculative decoding (ox.SpeculativeDecoder, ox.SpeculativeStats)
  3. FlashAttention-2 online tiling (ox.flash_attention_forward, ox.TiledAttention)
  4. DoRA and LoRA+ fine-tuning (ox.DoRALinear, ox.LoRAPlusLinear)
  5. Weisfeiler-Lehman and Random Walk graph kernels (ox.weisfeiler_lehman_kernel, ox.random_walk_kernel)
  6. Sparse matrix linear algebra (ox.CSRMatrix)
  7. Byte-Pair Encoding subword tokenizer (ox.BPETokenizer)
  8. Direct Preference Optimization (ox.DPOLoss, ox.ORPOLoss)
  9. Multimodal fusion (ox.CrossAttentionFusion, ox.GatedMultimodalFusion)
  10. Vision Transformer (ox.PatchEmbedding, ox.VisionTransformer)
"""

import math
import numpy as np
import pytest
import os
import tempfile

import oxideon as ox


def test_paged_and_standard_kv_cache():
    # 1. PagedKVCache
    cache = ox.PagedKVCache(num_blocks=16, block_size=4, num_heads=2, head_dim=8)
    assert cache.free_blocks_count == 16

    # Allocate sequence of length 7 (needs 2 blocks of 4)
    block_table = cache.allocate(num_tokens=7)
    assert len(block_table) == 2
    assert cache.free_blocks_count == 14

    # Append 5 tokens sequentially
    for t in range(5):
        k = ox.Tensor.randn([2, 8])
        v = ox.Tensor.randn([2, 8])
        cache.append(block_table, current_len=t, key=k, value=v)

    # Read back sequence
    k_seq, v_seq = cache.read_sequence(block_table, seq_len=5)
    assert k_seq.shape == [5, 2, 8]
    assert v_seq.shape == [5, 2, 8]

    # Attention over paged KV
    q = ox.Tensor.randn([2, 8])
    attn_out = cache.paged_attention(q, block_table, seq_len=5)
    assert attn_out.shape == [2, 8]

    # Free blocks
    cache.free(block_table)
    assert cache.free_blocks_count == 16

    # 2. StandardKVCache with sliding window
    std_cache = ox.StandardKVCache(max_batch_size=2, max_seq_len=8, num_heads=2, head_dim=4, sliding_window=6)
    k_chunk = ox.Tensor.randn([2, 4, 2, 4])
    v_chunk = ox.Tensor.randn([2, 4, 2, 4])
    k_out, v_out = std_cache.update(k_chunk, v_chunk)
    assert k_out.shape == [2, 4, 2, 4]

    # Append another chunk of length 4 (total 8 <= 8)
    k_out, v_out = std_cache.update(k_chunk, v_chunk)
    assert k_out.shape == [2, 8, 2, 4]


def test_speculative_decoding():
    vocab_size = 32

    # Deterministic mock logits: peak on token (len(seq) % vocab_size)
    def mock_draft(seq):
        logits = np.zeros(vocab_size, dtype=np.float32)
        target = (len(seq) + 1) % vocab_size
        logits[target] = 5.0
        return logits

    def mock_target(seq):
        logits = np.zeros(vocab_size, dtype=np.float32)
        target = (len(seq) + 1) % vocab_size
        logits[target] = 5.0
        return logits

    decoder = ox.SpeculativeDraftDecoder(mock_draft, mock_target, gamma=3, temperature=0.1)
    prompt = [1, 2, 3]

    new_tokens = decoder.step(prompt)
    assert len(new_tokens) >= 1
    assert decoder.stats.target_forward_passes >= 1
    assert decoder.stats.draft_tokens_proposed == 3

    generated = decoder.generate(prompt, max_new_tokens=6)
    assert len(generated) == len(prompt) + 6
    assert decoder.stats.speedup >= 1.0


def test_flash_attention_forward():
    np.random.seed(42)
    b, h, sq, sk, d = 2, 2, 8, 8, 16

    q_np = np.random.randn(b, h, sq, d).astype(np.float32)
    k_np = np.random.randn(b, h, sk, d).astype(np.float32)
    v_np = np.random.randn(b, h, sk, d).astype(np.float32)

    q = ox.from_numpy(q_np)
    k = ox.from_numpy(k_np)
    v = ox.from_numpy(v_np)

    # 1. Non-causal FlashAttention vs Standard Matmul Attention
    fa_out = ox.flash_attention_forward(q, k, v, block_size_r=4, block_size_c=4, causal=False)
    
    scale = 1.0 / math.sqrt(d)
    scores = np.matmul(q_np, k_np.swapaxes(-1, -2)) * scale
    m = np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(scores - m)
    weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    expected_out = np.matmul(weights, v_np)

    np.testing.assert_allclose(fa_out.to_numpy(), expected_out, atol=1e-4)

    # 2. Causal FlashAttention
    fa_causal = ox.flash_attention_forward(q, k, v, block_size_r=4, block_size_c=4, causal=True)
    assert fa_causal.shape == [b, h, sq, d]

    # 3. TiledAttention module
    tiled_mod = ox.TiledAttention(embed_dim=32, num_heads=4, block_size=4)
    x = ox.Tensor.randn([2, 8, 32])
    mod_out = tiled_mod(x, x, x)
    assert mod_out.shape == [2, 8, 32]


def test_dora_and_loraplus():
    # 1. DoRALinear
    dora = ox.DoRALinear(in_features=16, out_features=8, r=4, lora_alpha=8.0)
    x = ox.Tensor.randn([4, 16])
    y_before = dora(x)
    assert y_before.shape == [4, 8]

    # Merge weights
    dora.merge()
    assert dora.merged is True
    y_after = dora(x)
    np.testing.assert_allclose(y_before.to_numpy(), y_after.to_numpy(), atol=1e-5)

    # 2. LoRAPlusLinear
    lora_plus = ox.LoRAPlusLinear(in_features=16, out_features=8, r=4, b_ratio=4.0)
    y_plus = lora_plus(x)
    assert y_plus.shape == [4, 8]


def test_graph_kernels():
    # Triangle graph (3-cycle)
    adj_triangle = ox.from_numpy(np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ], dtype=np.float32))

    # Path graph (3-path: 0-1-2)
    adj_path = ox.from_numpy(np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=np.float32))

    # 1. Weisfeiler-Lehman kernel
    gram = ox.weisfeiler_lehman_kernel([adj_triangle, adj_path], num_iterations=2)
    gram_np = gram.to_numpy()
    assert gram_np.shape == (2, 2)
    # Self-similarities must be positive
    assert gram_np[0, 0] > 0
    assert gram_np[1, 1] > 0
    # Symmetric
    assert abs(gram_np[0, 1] - gram_np[1, 0]) < 1e-6

    # 2. Random Walk kernel
    rw_self = ox.random_walk_kernel(adj_triangle, adj_triangle, num_steps=2, decay=0.1)
    rw_cross = ox.random_walk_kernel(adj_triangle, adj_path, num_steps=2, decay=0.1)
    assert rw_self > 0
    assert rw_cross > 0


def test_sparse_csr_matrix():
    dense = np.array([
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 3.0],
        [4.0, 5.0, 0.0, 0.0]
    ], dtype=np.float32)

    csr = ox.CSRMatrix.from_dense(dense)
    assert csr.nnz == 5
    assert abs(csr.sparsity - (1.0 - 5.0 / 12.0)) < 1e-5
    assert csr.shape == (3, 4)

    # Reconstruct dense
    dense_rec = csr.to_dense().to_numpy()
    np.testing.assert_allclose(dense, dense_rec)

    # Sparse-Dense matrix multiplication (SpMM)
    x = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ], dtype=np.float32)

    y_spmm = csr.spmm(x).to_numpy()
    expected_spmm = dense @ x
    np.testing.assert_allclose(y_spmm, expected_spmm, atol=1e-5)

    # Sparse-Sparse matrix multiplication (SpSpMM)
    b_dense = np.array([
        [0.0, 1.0],
        [2.0, 0.0],
        [0.0, 0.0],
        [3.0, 4.0]
    ], dtype=np.float32)
    csr_b = ox.CSRMatrix.from_dense(b_dense)

    csr_c = csr.spspmm(csr_b)
    assert csr_c.shape == (3, 2)
    expected_c = dense @ b_dense
    np.testing.assert_allclose(csr_c.to_dense().to_numpy(), expected_c, atol=1e-5)


def test_bpe_tokenizer():
    corpus = [
        "low low lower lowest",
        "newest newest wider widest"
    ]
    tok = ox.BPETokenizer()
    tok.train(corpus, max_vocab_size=60, min_freq=1)

    assert tok.vocab_size > len(tok.special_tokens)

    text = "lower widest"
    encoded = tok.encode(text, add_special_tokens=True)
    assert encoded[0] == tok.bos_token_id
    assert encoded[-1] == tok.eos_token_id

    decoded = tok.decode(encoded, skip_special_tokens=True)
    assert decoded == text

    # Serialization test
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "bpe.json")
        tok.save(save_path)
        loaded_tok = ox.BPETokenizer.load(save_path)
        assert loaded_tok.vocab_size == tok.vocab_size
        assert loaded_tok.encode(text) == tok.encode(text)


def test_bpe_train_is_deterministic():
    corpus = [
        "low low lower lowest",
        "newest newest wider widest"
    ]
    a = ox.BPETokenizer()
    a.train(corpus, max_vocab_size=60, min_freq=1)
    b = ox.BPETokenizer()
    b.train(corpus, max_vocab_size=60, min_freq=1)
    assert a.merges == b.merges
    assert a.vocab == b.vocab


def _naive_apply_merges(symbols, merges):
    """Reference: sequential rank-order merge application (pre-optimization algorithm)."""
    symbols = list(symbols)
    for p0, p1 in merges:
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == p0 and symbols[i + 1] == p1:
                new_symbols.append(p0 + p1)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols


def test_bpe_encode_matches_sequential_reference():
    import itertools
    import random
    rng = random.Random(7)
    words = ["".join(rng.choice("abcdefgh") for _ in range(rng.randint(1, 10)))
             for _ in range(800)]
    corpus = [" ".join(words[i:i + 20]) for i in range(0, len(words), 20)]
    tok = ox.BPETokenizer()
    tok.train(corpus, max_vocab_size=400)
    rank = {pair: i for i, pair in enumerate(tok.merges)}

    probe = words + ["".join(t) for n in range(1, 6)
                     for t in itertools.product("abc", repeat=n)]
    for w in probe:
        expected = _naive_apply_merges(list(w), tok.merges)
        assert tok._apply_merges(list(w), rank) == expected

    # encode() end-to-end agrees with the reference path too
    text = " ".join(words[:50])
    expected_ids = []
    split_words = text.split(" ")
    for w_idx, word in enumerate(split_words):
        for sw in _naive_apply_merges(list(word), tok.merges):
            expected_ids.append(tok.vocab.get(sw, tok.unk_token_id))
        if w_idx < len(split_words) - 1 and " " in tok.vocab:
            expected_ids.append(tok.vocab[" "])
    assert tok.encode(text) == expected_ids


def test_dpo_and_orpo_losses():
    # DPOLoss (alignment)
    dpo = ox.DPOLoss(beta=0.1)
    
    # Policy prefers chosen (w) over rejected (l) more than reference model does
    pi_w = ox.from_numpy(np.array([-1.0, -1.2], dtype=np.float32))
    pi_l = ox.from_numpy(np.array([-3.5, -4.0], dtype=np.float32))
    ref_w = ox.from_numpy(np.array([-2.0, -2.5], dtype=np.float32))
    ref_l = ox.from_numpy(np.array([-2.5, -2.8], dtype=np.float32))

    loss_t, chosen_r, rejected_r = dpo(pi_w, pi_l, ref_w, ref_l)
    assert float(loss_t.to_numpy().flat[0]) > 0
    assert float(chosen_r.to_numpy().flat[0]) > float(rejected_r.to_numpy().flat[0])

    # ORPOLoss
    orpo = ox.ORPOLoss(lambda_coeff=0.2)
    loss_orpo, orpo_metrics = orpo(pi_w, pi_l)
    assert float(loss_orpo.to_numpy().flat[0]) > 0
    assert "odds_ratio_loss" in orpo_metrics

    # KTOLoss
    kto = ox.KTOLoss(beta=0.1)
    labels = ox.from_numpy(np.array([1.0, 0.0], dtype=np.float32))
    kto_loss, kto_metrics = kto(pi_w, ref_w, labels)
    assert float(kto_loss.to_numpy().flat[0]) > 0
    assert "kto_loss" in kto_metrics


def test_multimodal_fusion():
    b, sq, skv, d = 2, 4, 6, 16

    # 1. CrossAttentionFusion
    cross_fuse = ox.CrossAttentionFusion(d_model=d, num_heads=2)
    query_tokens = ox.Tensor.randn([b, sq, d])
    kv_tokens = ox.Tensor.randn([b, skv, d])

    fused = cross_fuse(query_tokens, kv_tokens)
    assert fused.shape == [b, sq, d]

    # 2. GatedMultimodalFusion
    dim_a, dim_b, out_dim = 12, 20, 16
    gated_fuse = ox.GatedMultimodalFusion(dim_a=dim_a, dim_b=dim_b, out_dim=out_dim)
    xa = ox.Tensor.randn([b, dim_a])
    xb = ox.Tensor.randn([b, dim_b])

    gated_out = gated_fuse(xa, xb)
    assert gated_out.shape == [b, out_dim]


def test_vision_transformer():
    b, c, h, w = 2, 3, 16, 16
    patch_size = 4
    embed_dim = 32
    num_classes = 5

    # 1. PatchEmbedding
    patch_embed = ox.PatchEmbedding(img_size=h, patch_size=patch_size, in_channels=c, embed_dim=embed_dim)
    images = ox.Tensor.randn([b, c, h, w])
    patches = patch_embed(images)
    expected_num_patches = (h // patch_size) * (w // patch_size) + 1  # 16 + 1 = 17
    assert patches.shape == [b, expected_num_patches, embed_dim]

    # 2. VisionTransformer
    vit = ox.VisionTransformer(
        img_size=h,
        patch_size=patch_size,
        in_channels=c,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=2,
        num_heads=4,
    )
    logits = vit(images)
    assert logits.shape == [b, num_classes]
