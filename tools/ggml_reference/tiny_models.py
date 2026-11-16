#!/usr/bin/env python3
"""Writes tiny random-weight GGUFs for architectures that llama.cpp's own test tool (`test-llama-archs`) cannot save,
so the parity tests can still compare against llama.cpp's logits:

  apertus, bitnet

usage: tiny_models.py <test-models-dir> <out-dir> [path-to-llama.cpp/gguf-py]
Each model reuses the key/value block (tokenizer included) of an existing test model, renamed to the new architecture,
and writes exactly the tensors that architecture's loader creates (llama.cpp rejects unused tensors).
Make the reference logits with `llama_logits <out-dir>/<name>.gguf 1 2 ... 48`.
"""
import os
import sys

import numpy as np

models, out_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/projects/llama.cpp/gguf-py"))
from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402

os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(1234)
D, FF, V, HD, H, KV, L = 256, 384, 128, 64, 4, 1, 2


STD = float(os.environ.get("STD", "0.1"))


def w(*shape, std=None):
    return rng.normal(0.0, STD if std is None else std, shape).astype(np.float32)


def norm(n):
    return (1.0 + rng.normal(0.0, 0.05, (n,))).astype(np.float32)


def derive(template, arch, name, tensors, overrides=None, layers=None):
    """Copies the template's KVs under the new architecture prefix, applies `overrides`, writes `tensors`.
    `layers=(old, new)` stretches the template's per-layer arrays (length `old`) to `new` entries."""
    reader = GGUFReader(os.path.join(models, f"{template}.gguf"))
    old = reader.fields["general.architecture"].contents()
    out = GGUFWriter(os.path.join(out_dir, f"{name}.gguf"), arch)
    # debugging aid: ZERO=laurel_r.weight,attn_output.weight silences every tensor whose name ends so
    for suffix in filter(None, os.environ.get("ZERO", "").split(",")):
        for k in tensors:
            if k.endswith(suffix):
                tensors[k] = np.zeros_like(tensors[k])
    seen = set()
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        value = field.contents()
        if isinstance(value, (list, tuple)) and len(value) == 0:
            continue  # the writer rejects empty arrays
        if layers and isinstance(value, (list, tuple)) and len(value) == layers[0] and not key.startswith("tokenizer."):
            value = (list(value) * layers[1])[: layers[1]]
        if key.startswith(old + "."):
            key = arch + key[len(old):]
        if overrides and key in overrides:
            value = overrides[key]
        vt = field.types[0]
        sub = field.types[-1] if vt == GGUFValueType.ARRAY else None
        if isinstance(value, (list, tuple)) and vt != GGUFValueType.ARRAY:
            vt, sub = GGUFValueType.ARRAY, field.types[0]  # a per-layer override of a scalar key
        out.add_key_value(key, value, vt, sub_type=sub)
        seen.add(key)
    # overrides for keys the template does not carry: unsigned 32-bit arrays (per-layer patterns) or scalars
    for key, value in (overrides or {}).items():
        if key in seen:
            continue
        if isinstance(value, (list, tuple)):
            sub = GGUFValueType.STRING if value and isinstance(value[0], str) else GGUFValueType.UINT32
            out.add_key_value(key, list(value), GGUFValueType.ARRAY, sub_type=sub)
        elif isinstance(value, str):
            out.add_key_value(key, value, GGUFValueType.STRING)
        elif isinstance(value, bool):
            out.add_key_value(key, value, GGUFValueType.BOOL)
        elif isinstance(value, float):
            out.add_key_value(key, value, GGUFValueType.FLOAT32)
        else:
            out.add_key_value(key, int(value), GGUFValueType.UINT32)
    for tname, data in tensors.items():
        out.add_tensor(tname, data)
    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
    print("wrote", name)


def llama_layer(i, qk_norm=False, h=H, hd=HD, kv=KV):
    """One attention + FFN layer; the defaults match qwen3-dense (4 heads of 64, one KV head)."""
    t = {
        f"blk.{i}.attn_norm.weight": norm(D),
        f"blk.{i}.attn_q.weight": w(h * hd, D),
        f"blk.{i}.attn_k.weight": w(kv * hd, D),
        f"blk.{i}.attn_v.weight": w(kv * hd, D),
        f"blk.{i}.attn_output.weight": w(D, h * hd),
        f"blk.{i}.ffn_norm.weight": norm(D),
        f"blk.{i}.ffn_up.weight": w(FF, D),
        f"blk.{i}.ffn_down.weight": w(D, FF),
    }
    if qk_norm:
        t[f"blk.{i}.attn_q_norm.weight"] = norm(hd)
        t[f"blk.{i}.attn_k_norm.weight"] = norm(hd)
    return t


# Apertus: Qwen3-style attention with per-head Q/K norms, an ungated FFN with the xIELU activation.
t = {"token_embd.weight": w(V, D), "output_norm.weight": norm(D), "output.weight": w(V, D)}
for i in range(L):
    t.update(llama_layer(i, qk_norm=True))
an, ap, beta, eps = (float(v) for v in os.environ.get("XIELU", "0.8,0.6,0.5,-1e-6").split(","))
xielu = {
    "xielu.alpha_n": [an] * L,
    "xielu.alpha_p": [ap] * L,
    "xielu.beta": [beta] * L,
    "xielu.eps": [eps] * L,
}
derive("qwen3-dense", "apertus", "apertus", t, xielu)

# BitNet: llama attention and a gated SiLU FFN, each followed by its own RMS "sub norm" (over the attention output /
# over act(gate) * up) before the output projection; the output head is the token embedding.
t = {"token_embd.weight": w(V, D), "output_norm.weight": norm(D)}
for i in range(L):
    t.update(llama_layer(i, h=2, hd=128, kv=2))  # llama-dense geometry; BitNet's loader has no grouped-query heads
    t[f"blk.{i}.ffn_gate.weight"] = w(FF, D)
    t[f"blk.{i}.attn_sub_norm.weight"] = norm(D)
    t[f"blk.{i}.ffn_sub_norm.weight"] = norm(FF)
derive("llama-dense", "bitnet", "bitnet", t)


# Chameleon: llama attention with per-head LayerNorm Q/K norms; the graph hides the logits of ids 4..8195 (image
# tokens), so the vocabulary has to be larger than that. Tied output head. `chameleon-swin` sets swin_norm.
VC = 8200
for name, swin in (("chameleon", False), ("chameleon-swin", True)):
    t = {"token_embd.weight": w(VC, D), "output_norm.weight": norm(D)}
    for i in range(L):
        t.update(llama_layer(i, h=2, hd=128, kv=2))
        t[f"blk.{i}.ffn_gate.weight"] = w(FF, D)
        t[f"blk.{i}.attn_q_norm.weight"] = norm(2 * 128).reshape(2, 128)
        t[f"blk.{i}.attn_k_norm.weight"] = norm(2 * 128).reshape(2, 128)
        t[f"blk.{i}.attn_q_norm.bias"] = w(2, 128, std=0.1)
        t[f"blk.{i}.attn_k_norm.bias"] = w(2, 128, std=0.1)
        if swin:
            # llama.cpp's swin layout feeds the FFN the normed attention output instead of the residual sum (the
            # reference implementation uses the sum), so the FFN is silenced here and only the attention half is compared.
            t[f"blk.{i}.ffn_down.weight"] = np.zeros_like(t[f"blk.{i}.ffn_down.weight"])
    derive("llama-dense", "chameleon", name, t, {
        "chameleon.vocab_size": VC,
        "chameleon.swin_norm": swin,
        "tokenizer.ggml.tokens": [f"tok_{i}" for i in range(VC)],
        "tokenizer.ggml.token_type": [1] * VC,
        "tokenizer.ggml.scores": [0.0] * VC,
    })

# Gemma 3n: Gemma 3 blocks on four AltUp streams with LAuReL side paths, per-layer input embeddings, activation
# sparsity on the first ten layers, and K/V shared from layers 18/19 by every layer from the twentieth on (hard-wired in
# llama.cpp, so the model needs more than 20 layers). Widths that llama.cpp hard-codes: 4 streams, LAuReL rank 64,
# per-layer input width 256.
GL, NA, RANK, WIDTH = int(os.environ.get("GEMMA3N_LAYERS", "24")), 4, 64, 256
t = {
    "token_embd.weight": w(V, D),
    "output_norm.weight": norm(D),
    "altup_proj.weight": w(NA - 1, D, D, std=0.05),
    "altup_unembd_proj.weight": w(NA - 1, D, D, std=0.05),
    "per_layer_token_embd.weight": w(V, WIDTH * GL),
    "per_layer_model_proj.weight": w(WIDTH * GL, D, std=0.05),
    "per_layer_proj_norm.weight": norm(WIDTH),
}
for i in range(GL):
    t.update(llama_layer(i, qk_norm=True, h=2, hd=128, kv=2))
    # Gemma 3n scales attention by 1.0, so unit Q/K norm weights give scores in the hundreds and a one-hot softmax that
    # amplifies float noise layer after layer; small norm weights keep the random model well conditioned.
    for nm in ("q", "k"):
        t[f"blk.{i}.attn_{nm}_norm.weight"] = t[f"blk.{i}.attn_{nm}_norm.weight"] * 0.1
    t.update({
        f"blk.{i}.ffn_gate.weight": w(FF, D),
        f"blk.{i}.post_attention_norm.weight": norm(D),
        f"blk.{i}.post_ffw_norm.weight": norm(D),
        f"blk.{i}.inp_gate.weight": w(WIDTH, D),
        f"blk.{i}.proj.weight": w(D, WIDTH),
        f"blk.{i}.post_norm.weight": norm(D),
        f"blk.{i}.altup_correct_coef.weight": w(NA, NA),
        f"blk.{i}.altup_correct_scale.weight": norm(D),
        f"blk.{i}.altup_predict_coef.weight": w(NA * NA, NA),
        f"blk.{i}.altup_router.weight": w(NA, D),
        f"blk.{i}.altup_router_norm.weight": norm(D),
        f"blk.{i}.laurel_l.weight": w(RANK, D),
        f"blk.{i}.laurel_r.weight": w(D, RANK),
        f"blk.{i}.laurel_post_norm.weight": norm(D),
    })
derive("gemma3-dense", "gemma3n", "gemma3n", t, {
    "gemma3n.block_count": GL,
    "gemma3n.final_logit_softcapping": 30.0,
    "gemma3n.attention.sliding_window": 16,
    "gemma3n.attention.sliding_window_pattern": [0 if i % 5 == 4 else 1 for i in range(GL)],
}, layers=(2, GL))

# RWKV-6: attention-free layers (time mix with a data-dependent token shift and a per-head WKV state, then a channel
# mix), LayerNorms with bias everywhere, norm after the embedding. Per-layer 0..1 interpolation offsets, decays around
# exp(-exp(-0.5)).
HS, EX, EXD = 64, 32, 64
t = {
    "token_embd.weight": w(V, D),
    "token_embd_norm.weight": norm(D),
    "token_embd_norm.bias": w(D, std=0.05),
    "output_norm.weight": norm(D),
    "output_norm.bias": w(D, std=0.05),
    "output.weight": w(V, D),
}
for i in range(L):
    lerp = lambda: rng.uniform(0.0, 1.0, (1, 1, D)).astype(np.float32)  # noqa: E731
    t.update({
        f"blk.{i}.attn_norm.weight": norm(D), f"blk.{i}.attn_norm.bias": w(D, std=0.05),
        f"blk.{i}.attn_norm_2.weight": norm(D), f"blk.{i}.attn_norm_2.bias": w(D, std=0.05),
        f"blk.{i}.time_mix_w1.weight": w(5 * EX, D, std=0.1),
        f"blk.{i}.time_mix_w2.weight": w(5, D, EX, std=0.1),
        f"blk.{i}.time_mix_lerp_x.weight": lerp(),
        f"blk.{i}.time_mix_lerp_w.weight": lerp(), f"blk.{i}.time_mix_lerp_k.weight": lerp(),
        f"blk.{i}.time_mix_lerp_v.weight": lerp(), f"blk.{i}.time_mix_lerp_r.weight": lerp(),
        f"blk.{i}.time_mix_lerp_g.weight": lerp(),
        f"blk.{i}.time_mix_first.weight": w(D // HS, HS, std=0.3),
        f"blk.{i}.time_mix_decay.weight": rng.normal(-0.5, 0.5, (D,)).astype(np.float32),
        f"blk.{i}.time_mix_decay_w1.weight": w(EXD, D, std=0.1),
        f"blk.{i}.time_mix_decay_w2.weight": w(D, EXD, std=0.1),
        f"blk.{i}.time_mix_key.weight": w(D, D), f"blk.{i}.time_mix_value.weight": w(D, D),
        f"blk.{i}.time_mix_receptance.weight": w(D, D), f"blk.{i}.time_mix_gate.weight": w(D, D),
        f"blk.{i}.time_mix_ln.weight": norm(D), f"blk.{i}.time_mix_ln.bias": w(D, std=0.05),
        f"blk.{i}.time_mix_output.weight": w(D, D),
        f"blk.{i}.channel_mix_lerp_k.weight": lerp(), f"blk.{i}.channel_mix_lerp_r.weight": lerp(),
        f"blk.{i}.channel_mix_key.weight": w(FF, D), f"blk.{i}.channel_mix_value.weight": w(D, FF),
        f"blk.{i}.channel_mix_receptance.weight": w(D, D),
    })
derive("llama-dense", "rwkv6", "rwkv6", t, {
    "rwkv6.wkv.head_size": HS,
    "rwkv6.time_mix_extra_dim": EX,
    "rwkv6.time_decay_extra_dim": EXD,
    "rwkv6.rescale_every_n_layers": 2,
    "rwkv6.attention.layer_norm_epsilon": 1e-5,
    # real RWKV files carry no head counts; a template's would send llama.cpp down its grouped-KV path
    "rwkv6.attention.head_count": 0,
    "rwkv6.attention.head_count_kv": 0,
})

# RWKV-7: the same layout with the delta-rule recurrence, low-rank decay / in-context-learning-rate / value-residual /
# output-gate paths and a fused 6-way interpolation. Three layers, so that layers 1 and 2 blend layer 0's values in.
RD, RI, RV, RG = 24, 20, 16, 28
t = {
    "token_embd.weight": w(V, D),
    "token_embd_norm.weight": norm(D),
    "token_embd_norm.bias": w(D, std=0.05),
    "output_norm.weight": norm(D),
    "output_norm.bias": w(D, std=0.05),
    "output.weight": w(V, D),
}
for i in range(3):
    rv = RI if i == 0 else RV
    t.update({
        f"blk.{i}.attn_norm.weight": norm(D), f"blk.{i}.attn_norm.bias": w(D, std=0.05),
        f"blk.{i}.attn_norm_2.weight": norm(D), f"blk.{i}.attn_norm_2.bias": w(D, std=0.05),
        f"blk.{i}.time_mix_w0.weight": rng.normal(-0.5, 0.5, (D,)).astype(np.float32),
        f"blk.{i}.time_mix_w1.weight": w(RD, D), f"blk.{i}.time_mix_w2.weight": w(D, RD),
        f"blk.{i}.time_mix_a0.weight": w(D, std=0.5),
        f"blk.{i}.time_mix_a1.weight": w(RI, D), f"blk.{i}.time_mix_a2.weight": w(D, RI),
        f"blk.{i}.time_mix_v0.weight": w(D, std=0.5),
        f"blk.{i}.time_mix_v1.weight": w(rv, D), f"blk.{i}.time_mix_v2.weight": w(D, rv),
        f"blk.{i}.time_mix_g1.weight": w(RG, D), f"blk.{i}.time_mix_g2.weight": w(D, RG),
        f"blk.{i}.time_mix_lerp_fused.weight": rng.uniform(0.0, 1.0, (6, 1, 1, D)).astype(np.float32),
        f"blk.{i}.time_mix_k_k.weight": rng.uniform(0.5, 1.5, (D,)).astype(np.float32),
        f"blk.{i}.time_mix_k_a.weight": rng.uniform(0.5, 1.5, (D,)).astype(np.float32),
        f"blk.{i}.time_mix_r_k.weight": w(D, std=0.3),
        f"blk.{i}.time_mix_key.weight": w(D, D), f"blk.{i}.time_mix_value.weight": w(D, D),
        f"blk.{i}.time_mix_receptance.weight": w(D, D),
        f"blk.{i}.time_mix_ln.weight": norm(D), f"blk.{i}.time_mix_ln.bias": w(D, std=0.05),
        f"blk.{i}.time_mix_output.weight": w(D, D),
        f"blk.{i}.channel_mix_lerp_k.weight": rng.uniform(0.0, 1.0, (1, 1, D)).astype(np.float32),
        f"blk.{i}.channel_mix_key.weight": w(FF, D), f"blk.{i}.channel_mix_value.weight": w(D, FF),
    })
derive("llama-dense", "rwkv7", "rwkv7", t, {
    "rwkv7.block_count": 3,
    "rwkv7.wkv.head_size": HS,
    "rwkv7.attention.decay_lora_rank": RD,
    "rwkv7.attention.iclr_lora_rank": RI,
    "rwkv7.attention.value_residual_mix_lora_rank": RV,
    "rwkv7.attention.gate_lora_rank": RG,
    "rwkv7.attention.layer_norm_epsilon": 1e-5,
    "rwkv7.attention.head_count": 0,
    "rwkv7.attention.head_count_kv": 0,
}, layers=(2, 3))

# ARWKV7: the RWKV-7 time mix (here without the group norm, with the output gate) behind RMS norms, an embedding that is
# not normed, a SwiGLU FFN and a single token-shift slot.
t = {"token_embd.weight": w(V, D), "output_norm.weight": norm(D), "output.weight": w(V, D)}
for i in range(3):
    rv = RI if i == 0 else RV
    t.update({
        f"blk.{i}.attn_norm.weight": norm(D),
        f"blk.{i}.time_mix_w0.weight": rng.normal(-0.5, 0.5, (D,)).astype(np.float32),
        f"blk.{i}.time_mix_w1.weight": w(RD, D), f"blk.{i}.time_mix_w2.weight": w(D, RD),
        f"blk.{i}.time_mix_a0.weight": w(D, std=0.5),
        f"blk.{i}.time_mix_a1.weight": w(RI, D), f"blk.{i}.time_mix_a2.weight": w(D, RI),
        f"blk.{i}.time_mix_v0.weight": w(D, std=0.5),
        f"blk.{i}.time_mix_v1.weight": w(rv, D), f"blk.{i}.time_mix_v2.weight": w(D, rv),
        f"blk.{i}.time_mix_g1.weight": w(RG, D), f"blk.{i}.time_mix_g2.weight": w(D, RG),
        f"blk.{i}.time_mix_lerp_fused.weight": rng.uniform(0.0, 1.0, (6, 1, 1, D)).astype(np.float32),
        f"blk.{i}.time_mix_k_k.weight": rng.uniform(0.5, 1.5, (D,)).astype(np.float32),
        f"blk.{i}.time_mix_k_a.weight": rng.uniform(0.5, 1.5, (D,)).astype(np.float32),
        f"blk.{i}.time_mix_r_k.weight": w(D, std=0.3),
        f"blk.{i}.time_mix_key.weight": w(D, D), f"blk.{i}.time_mix_value.weight": w(D, D),
        f"blk.{i}.time_mix_receptance.weight": w(D, D),
        f"blk.{i}.time_mix_output.weight": w(D, D),
        f"blk.{i}.ffn_norm.weight": norm(D),
        f"blk.{i}.ffn_gate.weight": w(FF, D), f"blk.{i}.ffn_up.weight": w(FF, D), f"blk.{i}.ffn_down.weight": w(D, FF),
    })
derive("llama-dense", "arwkv7", "arwkv7", t, {
    "arwkv7.block_count": 3,
    "arwkv7.wkv.head_size": HS,
    "arwkv7.attention.decay_lora_rank": RD,
    "arwkv7.attention.iclr_lora_rank": RI,
    "arwkv7.attention.value_residual_mix_lora_rank": RV,
    "arwkv7.attention.gate_lora_rank": RG,
    "arwkv7.attention.layer_norm_rms_epsilon": 1e-5,
    "arwkv7.token_shift_count": 1,
    "arwkv7.attention.head_count": 0,
    "arwkv7.attention.head_count_kv": 0,
}, layers=(2, 3))

# QRWKV6 (rwkv6qwen2): RWKV-6's interpolation and decay feeding gated linear attention with grouped key / value heads
# (4 heads, 2 key / value heads), biases on r / k / v, a sigmoid gate, RMS norms and a SwiGLU FFN.
KVD = 2 * HS
t = {"token_embd.weight": w(V, D), "output_norm.weight": norm(D), "output.weight": w(V, D)}
for i in range(L):
    t.update({
        f"blk.{i}.attn_norm.weight": norm(D),
        f"blk.{i}.time_mix_w1.weight": w(5 * EX, D, std=0.1),
        f"blk.{i}.time_mix_w2.weight": w(5, D, EX, std=0.1),
        f"blk.{i}.time_mix_lerp_x.weight": rng.uniform(0.0, 1.0, (1, 1, D)).astype(np.float32),
        f"blk.{i}.time_mix_lerp_fused.weight": rng.uniform(0.0, 1.0, (5, 1, 1, D)).astype(np.float32),
        f"blk.{i}.time_mix_decay.weight": rng.normal(-0.5, 0.5, (D,)).astype(np.float32),
        f"blk.{i}.time_mix_decay_w1.weight": w(EXD, D, std=0.1),
        f"blk.{i}.time_mix_decay_w2.weight": w(D, EXD, std=0.1),
        f"blk.{i}.time_mix_key.weight": w(KVD, D), f"blk.{i}.time_mix_key.bias": w(KVD, std=0.2),
        f"blk.{i}.time_mix_value.weight": w(KVD, D), f"blk.{i}.time_mix_value.bias": w(KVD, std=0.2),
        f"blk.{i}.time_mix_receptance.weight": w(D, D), f"blk.{i}.time_mix_receptance.bias": w(D, std=0.2),
        f"blk.{i}.time_mix_gate.weight": w(D, D),
        f"blk.{i}.time_mix_output.weight": w(D, D),
        f"blk.{i}.ffn_norm.weight": norm(D),
        f"blk.{i}.ffn_gate.weight": w(FF, D), f"blk.{i}.ffn_up.weight": w(FF, D), f"blk.{i}.ffn_down.weight": w(D, FF),
    })
derive("llama-dense", "rwkv6qwen2", "rwkv6qwen2", t, {
    "rwkv6qwen2.wkv.head_size": HS,
    "rwkv6qwen2.time_mix_extra_dim": EX,
    "rwkv6qwen2.time_decay_extra_dim": EXD,
    "rwkv6qwen2.token_shift_count": 1,
    "rwkv6qwen2.attention.layer_norm_rms_epsilon": 1e-5,
    "rwkv6qwen2.attention.head_count": D // HS,
    "rwkv6qwen2.attention.head_count_kv": KVD // HS,
})

# Step 3.5: four layers with different head counts (4/2/4/2 heads, 2/1/2/1 KV heads), sliding-window layers 0 and 2
# rotating 32 dims and full layers 1 and 3 half of `rope.dimension_count` (= 64 of 128), the head-wise output gate on
# layers 1 and 2 only, a dense first FFN, then sigmoid-routed experts with a shared expert and (different) SwiGLU clamps.
SL, SHEADS, SKV, SE, SEU, SFE, SFS = 4, [4, 2, 4, 2], [2, 1, 2, 1], 4, 2, 192, 384
t = {"token_embd.weight": w(V, D), "output_norm.weight": norm(D), "output.weight": w(V, D)}
for i in range(SL):
    h, kv = SHEADS[i], SKV[i]
    t.update({
        f"blk.{i}.attn_norm.weight": norm(D),
        f"blk.{i}.attn_q.weight": w(h * 128, D), f"blk.{i}.attn_k.weight": w(kv * 128, D),
        f"blk.{i}.attn_v.weight": w(kv * 128, D), f"blk.{i}.attn_output.weight": w(D, h * 128),
        f"blk.{i}.attn_q_norm.weight": norm(128), f"blk.{i}.attn_k_norm.weight": norm(128),
        f"blk.{i}.ffn_norm.weight": norm(D),
    })
    if i in (1, 2):
        t[f"blk.{i}.attn_gate.weight"] = w(h, D)
    if i == 0:
        t.update({
            f"blk.{i}.ffn_gate.weight": w(FF, D), f"blk.{i}.ffn_up.weight": w(FF, D), f"blk.{i}.ffn_down.weight": w(D, FF),
        })
    else:
        t.update({
            f"blk.{i}.ffn_gate_inp.weight": w(SE, D),
            f"blk.{i}.ffn_gate_exps.weight": w(SE, SFE, D), f"blk.{i}.ffn_up_exps.weight": w(SE, SFE, D),
            f"blk.{i}.ffn_down_exps.weight": w(SE, D, SFE),
            f"blk.{i}.exp_probs_b.bias": w(SE, std=0.2),
            f"blk.{i}.ffn_gate_shexp.weight": w(SFS, D), f"blk.{i}.ffn_up_shexp.weight": w(SFS, D),
            f"blk.{i}.ffn_down_shexp.weight": w(D, SFS),
        })
derive("qwen3moe-moe", "step35", "step35", t, {
    "step35.block_count": SL,
    "step35.leading_dense_block_count": 1,
    "step35.attention.head_count": SHEADS,
    "step35.attention.head_count_kv": SKV,
    "step35.attention.sliding_window": 16,
    "step35.attention.sliding_window_pattern": [1, 0, 1, 0],
    "step35.rope.dimension_count": 128,
    "step35.rope.dimension_count_swa": 32,
    "step35.rope.freq_base": 10000.0,
    "step35.rope.freq_base_swa": 5000.0,
    "step35.expert_count": SE,
    "step35.expert_used_count": SEU,
    "step35.expert_feed_forward_length": SFE,
    "step35.expert_shared_feed_forward_length": SFS,
    "step35.expert_gating_func": 2,
    "step35.expert_weights_norm": True,
    "step35.expert_weights_scale": 1.5,
    "step35.swiglu_clamp_exp": [0.0, 2.0, 2.0, 2.0],
    "step35.swiglu_clamp_shexp": [0.0, 3.0, 3.0, 3.0],
}, layers=(2, SL))

# Gemma 4 (dense): six layers, sliding-window layers 0,1,2,4 (64-wide heads, 4 heads / 2 KV heads, 64 rotary dims) and
# full layers 3,5 (128-wide heads, 2 heads / 1 KV head, proportional RoPE through `rope_freqs`); layer 3 has no V
# projection (values = keys); layers 4 and 5 (`shared_kv_layers = 2`) have no K/V at all and attend to layers 2 and 3;
# per-layer input embeddings (width 64) gate every layer and each layer has an output scalar.
G4L, G4W = 6, 64
G4SWA = [1, 1, 1, 0, 1, 0]
G4H = [4 if s else 2 for s in G4SWA]
G4KV = [2 if s else 1 for s in G4SWA]
G4HD = [64 if s else 128 for s in G4SWA]
t = {
    "token_embd.weight": w(V, D),
    "output_norm.weight": norm(D),
    "per_layer_token_embd.weight": w(V, G4W * G4L),
    "per_layer_model_proj.weight": w(G4W * G4L, D, std=0.05),
    "per_layer_proj_norm.weight": norm(G4W),
    "rope_freqs.weight": rng.uniform(1.0, 3.0, (64,)).astype(np.float32),
}
for i in range(G4L):
    h, kv, hd = G4H[i], G4KV[i], G4HD[i]
    has_kv = i < G4L - 2
    t.update({
        f"blk.{i}.attn_norm.weight": norm(D),
        f"blk.{i}.attn_q.weight": w(h * hd, D),
        f"blk.{i}.attn_q_norm.weight": norm(hd) * 0.3,
        f"blk.{i}.attn_output.weight": w(D, h * hd),
        f"blk.{i}.post_attention_norm.weight": norm(D),
        f"blk.{i}.ffn_norm.weight": norm(D),
        f"blk.{i}.ffn_gate.weight": w(FF, D), f"blk.{i}.ffn_up.weight": w(FF, D), f"blk.{i}.ffn_down.weight": w(D, FF),
        f"blk.{i}.post_ffw_norm.weight": norm(D),
        f"blk.{i}.inp_gate.weight": w(G4W, D), f"blk.{i}.proj.weight": w(D, G4W), f"blk.{i}.post_norm.weight": norm(D),
        f"blk.{i}.layer_output_scale.weight": rng.uniform(0.5, 1.2, (1,)).astype(np.float32),
    })
    if has_kv:
        t[f"blk.{i}.attn_k.weight"] = w(kv * hd, D)
        t[f"blk.{i}.attn_k_norm.weight"] = norm(hd) * 0.3
        if i != 3:
            t[f"blk.{i}.attn_v.weight"] = w(kv * hd, D)
G4_OVER = {
    "gemma4.block_count": G4L,
    "gemma4.final_logit_softcapping": 30.0,
    "gemma4.attention.head_count": G4H,
    "gemma4.attention.head_count_kv": G4KV,
    "gemma4.attention.key_length": 128,
    "gemma4.attention.value_length": 128,
    "gemma4.attention.key_length_swa": 64,
    "gemma4.attention.value_length_swa": 64,
    "gemma4.attention.sliding_window": 16,
    "gemma4.attention.sliding_window_pattern": G4SWA,
    "gemma4.attention.shared_kv_layers": 2,
    "gemma4.embedding_length_per_layer_input": G4W,
    "gemma4.rope.dimension_count": 128,
    "gemma4.rope.dimension_count_swa": 64,
    "gemma4.rope.freq_base": 10000.0,
    "gemma4.rope.freq_base_swa": 5000.0,
}
derive("gemma3-dense", "gemma4", "gemma4", t, G4_OVER, layers=(2, G4L))

# Gemma 4 MoE: layers 1 and 3 add routed experts (softmax routing, GELU, per-expert scales) next to the dense MLP, with
# the router reading the normed residual and extra pre / post norms.
GE, GEU, GFE = 4, 2, 192
t = dict(t)
for i in (1, 3):
    t.update({
        f"blk.{i}.ffn_gate_inp.weight": w(GE, D),
        f"blk.{i}.ffn_gate_inp.scale": norm(D),
        f"blk.{i}.ffn_gate_exps.weight": w(GE, GFE, D),
        f"blk.{i}.ffn_up_exps.weight": w(GE, GFE, D),
        f"blk.{i}.ffn_down_exps.weight": w(GE, D, GFE),
        f"blk.{i}.ffn_gate_exps.scale": rng.uniform(0.5, 1.5, (GE,)).astype(np.float32),
        f"blk.{i}.ffn_up_exps.scale": rng.uniform(0.5, 1.5, (GE,)).astype(np.float32),
        f"blk.{i}.ffn_down_exps.scale": rng.uniform(0.5, 1.5, (GE,)).astype(np.float32),
        f"blk.{i}.pre_ffw_norm_2.weight": norm(D),
        f"blk.{i}.post_ffw_norm_1.weight": norm(D),
        f"blk.{i}.post_ffw_norm_2.weight": norm(D),
    })
derive("gemma3-dense", "gemma4", "gemma4-moe", t, {
    **G4_OVER,
    "gemma4.expert_count": GE,
    "gemma4.expert_used_count": GEU,
    "gemma4.expert_feed_forward_length": GFE,
}, layers=(2, G4L))

# PLM: MLA with one Q projection (no low-rank Q), the rotary part trailing each head, an ungated ReLU^2 FFN and the
# token embedding as output head. Heads: 2 x (64 nope + 64 rope) keys, 128-wide values, 512-wide latent.
NOPE, ROPE, VD, KVL = 64, 64, 128, 512
t = {"token_embd.weight": w(V, D), "output_norm.weight": norm(D)}
for i in range(L):
    t.update({
        f"blk.{i}.attn_norm.weight": norm(D),
        f"blk.{i}.attn_q.weight": w(2 * (NOPE + ROPE), D),
        f"blk.{i}.attn_kv_a_mqa.weight": w(KVL + ROPE, D),
        f"blk.{i}.attn_kv_a_norm.weight": norm(KVL),
        f"blk.{i}.attn_kv_b.weight": w(2 * (NOPE + VD), KVL, std=0.05),
        f"blk.{i}.attn_output.weight": w(D, 2 * VD),
        f"blk.{i}.ffn_norm.weight": norm(D),
        f"blk.{i}.ffn_up.weight": w(FF, D),
        f"blk.{i}.ffn_down.weight": w(D, FF),
    })
derive("minicpm3-dense", "plm", "plm", t,
       {"plm.rope.dimension_count": ROPE, "plm.rope.dimension_count_swa": ROPE, "plm.attention.q_lora_rank": 0})

# BERT: post-LayerNorm encoder, learned positions + one token-type row, biases everywhere, GELU FFN. Pooling variants:
# `bert` (mean), `bert-cls`, `bert-none` (a vector per token). Compare with llama_embed, not llama_logits.
NPOS = 64
for name, pooling in (("bert", 1), ("bert-cls", 2), ("bert-none", 0)):
    t = {
        "token_embd.weight": w(V, D),
        "token_types.weight": w(2, D),
        "position_embd.weight": w(NPOS, D),
        "token_embd_norm.weight": norm(D),
        "token_embd_norm.bias": w(D, std=0.05),
    }
    for i in range(L):
        for p, (o, n) in {"attn_q": (D, D), "attn_k": (D, D), "attn_v": (D, D), "attn_output": (D, D),
                          "ffn_up": (FF, D), "ffn_down": (D, FF)}.items():
            t[f"blk.{i}.{p}.weight"] = w(o, n)
            t[f"blk.{i}.{p}.bias"] = w(o, std=0.05)
        for nm in ("attn_output_norm", "layer_output_norm"):
            t[f"blk.{i}.{nm}.weight"] = norm(D)
            t[f"blk.{i}.{nm}.bias"] = w(D, std=0.05)
    derive("llama-dense", "bert", name, t, {
        "bert.context_length": NPOS,
        "bert.attention.causal": False,
        "bert.pooling_type": pooling,
        "bert.attention.layer_norm_epsilon": 1e-12,
        "tokenizer.ggml.token_type_count": 2,
    })

# nomic-bert: the same post-LN encoder with RoPE instead of learned positions, a fused QKV projection and a SwiGLU FFN
# (no biases on those); mean pooling.
t = {
    "token_embd.weight": w(V, D),
    "token_types.weight": w(2, D),
    "token_embd_norm.weight": norm(D),
    "token_embd_norm.bias": w(D, std=0.05),
}
for i in range(L):
    t[f"blk.{i}.attn_qkv.weight"] = w(3 * D, D)
    t[f"blk.{i}.attn_output.weight"] = w(D, D)
    t[f"blk.{i}.ffn_up.weight"] = w(FF, D)
    t[f"blk.{i}.ffn_gate.weight"] = w(FF, D)
    t[f"blk.{i}.ffn_down.weight"] = w(D, FF)
    for nm in ("attn_output_norm", "layer_output_norm"):
        t[f"blk.{i}.{nm}.weight"] = norm(D)
        t[f"blk.{i}.{nm}.bias"] = w(D, std=0.05)
derive("llama-dense", "nomic-bert", "nomic-bert", t, {
    "nomic-bert.context_length": NPOS,
    "nomic-bert.attention.causal": False,
    "nomic-bert.pooling_type": 1,
    "nomic-bert.attention.layer_norm_epsilon": 1e-12,
    "tokenizer.ggml.token_type_count": 2,
})

# jina-bert-v2: ALiBi instead of positions; `jina-bert-v2` has a fused GEGLU `ffn_up` (2 * FF rows), LayerNorms over the
# whole Q / K width and a second post-attention norm; `jina-bert-v2-gate` has a separate `ffn_gate` and none of those.
for name, fused in (("jina-bert-v2", True), ("jina-bert-v2-gate", False)):
    t = {
        "token_embd.weight": w(V, D),
        "token_types.weight": w(2, D),
        "token_embd_norm.weight": norm(D),
        "token_embd_norm.bias": w(D, std=0.05),
    }
    for i in range(L):
        for p, (o, n) in {"attn_q": (D, D), "attn_k": (D, D), "attn_v": (D, D), "attn_output": (D, D),
                          "ffn_up": (2 * FF if fused else FF, D), "ffn_down": (D, FF)}.items():
            t[f"blk.{i}.{p}.weight"] = w(o, n)
            t[f"blk.{i}.{p}.bias"] = w(o, std=0.05)
        names = ["attn_output_norm", "layer_output_norm"] + (["attn_q_norm", "attn_k_norm", "attn_norm_2"] if fused else [])
        for nm in names:
            t[f"blk.{i}.{nm}.weight"] = norm(D)
            t[f"blk.{i}.{nm}.bias"] = w(D, std=0.05)
        if not fused:
            t[f"blk.{i}.ffn_gate.weight"] = w(FF, D)
    derive("llama-dense", "jina-bert-v2", name, t, {
        "jina-bert-v2.context_length": NPOS,
        "jina-bert-v2.attention.causal": False,
        "jina-bert-v2.pooling_type": 1,
        "jina-bert-v2.attention.layer_norm_epsilon": 1e-12,
        "tokenizer.ggml.token_type_count": 2,
    })

# bert reranker heads: pooling RANK (4); `bert-rank` pools the CLS row, `bert-rank-mean` the mean (classifier.pooling_type),
# both through `cls` (+ tanh / gelu) and `cls.output` with 3 labels.
for name, cls_pool, act in (("bert-rank", 2, "tanh"), ("bert-rank-mean", 1, "gelu")):
    t = {
        "token_embd.weight": w(V, D),
        "token_types.weight": w(2, D),
        "position_embd.weight": w(NPOS, D),
        "token_embd_norm.weight": norm(D),
        "token_embd_norm.bias": w(D, std=0.05),
        "cls.weight": w(D, D),
        "cls.bias": w(D, std=0.05),
        "cls.output.weight": w(3, D),
        "cls.output.bias": w(3, std=0.05),
    }
    for i in range(L):
        for p, (o, n) in {"attn_q": (D, D), "attn_k": (D, D), "attn_v": (D, D), "attn_output": (D, D),
                          "ffn_up": (FF, D), "ffn_down": (D, FF)}.items():
            t[f"blk.{i}.{p}.weight"] = w(o, n)
            t[f"blk.{i}.{p}.bias"] = w(o, std=0.05)
        for nm in ("attn_output_norm", "layer_output_norm"):
            t[f"blk.{i}.{nm}.weight"] = norm(D)
            t[f"blk.{i}.{nm}.bias"] = w(D, std=0.05)
    derive("llama-dense", "bert", name, t, {
        "bert.context_length": NPOS,
        "bert.attention.causal": False,
        "bert.pooling_type": 4,
        "bert.classifier.pooling_type": cls_pool,
        "bert.classifier.activation": act,
        "bert.classifier.output_labels": ["a", "b", "c"],
        "bert.attention.layer_norm_epsilon": 1e-12,
        "tokenizer.ggml.token_type_count": 2,
    })
