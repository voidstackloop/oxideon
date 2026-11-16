# High-Throughput LLM Inference & Efficient Systems in Oxideon

Oxideon provides a comprehensive suite of hardware-efficient inference architectures and sub-quadratic attention algorithms engineered for high-throughput LLM serving, local edge deployment, and parameter-efficient fine-tuning.

---

## Table of Contents
1. [Paged Attention & Key-Value Caching (`ox.kv_cache`)](#1-paged-attention--key-value-caching)
2. [Speculative Decoding (`ox.speculative`)](#2-speculative-decoding)
3. [FlashAttention-2 Memory-Efficient Tiling (`ox.flash_attn`)](#3-flashattention-2-memory-efficient-tiling)
4. [Ultra-Low-Bit K-Quants & BitNet 1.58b (`ox.kquants`)](#4-ultra-low-bit-k-quants--bitnet-158b)
5. [Byte-Pair Encoding Subword Tokenizer (`ox.tokenizers`)](#5-byte-pair-encoding-subword-tokenizer)
6. [Parameter-Efficient Fine-Tuning: DoRA & LoRA+ (`ox.dora`)](#6-parameter-efficient-fine-tuning-dora--lora)
7. [KV Cache Eviction Policies](#7-kv-cache-eviction-policies)
8. [MXFP4 Microscaling Quantization](#8-mxfp4-microscaling-quantization)
9. [MLA & Native Sparse Attention](#9-mla--native-sparse-attention)
10. [Radix Tree Prefix KV-Cache (`ox.prefix_cache`)](#10-radix-tree-prefix-kv-cache)
11. [Advanced FP8 & Block-Wise INT4 Quantization (`ox.quantization_advanced`)](#11-advanced-fp8--block-wise-int4-quantization)
12. [Selective State Space Architectures: Mamba SSM (`ox.state_space`)](#12-selective-state-space-architectures-mamba-ssm)
13. [Sparse Long-Context Multi-Head Attention (`ox.sparse_attn`)](#13-sparse-long-context-multi-head-attention)

---

## 1. Paged Attention & Key-Value Caching

Oxideon implements both non-contiguous paged memory management inspired by vLLM's PagedAttention and standard contiguous sliding-window KV caches.

### `ox.PagedKVCache`
Partitions KV memory into fixed-size physical blocks of tokens. This eliminates external memory fragmentation and allows dynamic memory allocation, prefix sharing, and parallel request serving.

```python
import oxideon as ox

# Initialize a physical KV block pool
# 64 blocks, each holding 16 tokens, with 8 attention heads and 64 head dimension
cache = ox.PagedKVCache(
    num_blocks=64,
    block_size=16,
    num_heads=8,
    head_dim=64,
)

# Allocate physical blocks for a new sequence of 40 tokens (needs 3 blocks of 16)
block_table = cache.allocate(num_tokens=40)
print(f"Allocated blocks: {block_table}, Free blocks left: {cache.free_blocks_count}")

# Append single-token Key and Value vectors during autoregressive generation
new_key = ox.Tensor.randn([8, 64])
new_value = ox.Tensor.randn([8, 64])
block_table = cache.append(block_table, current_len=40, key=new_key, value=new_value)

# Compute multi-head attention directly over non-contiguous physical pages
query_token = ox.Tensor.randn([8, 64])
attn_output = cache.paged_attention(query_token, block_table, seq_len=41)
print("Paged Attention Output Shape:", attn_output.shape) # [8, 64]

# Free blocks when generation completes
cache.free(block_table)
```

### `ox.StandardKVCache`
Contiguous sliding-window and append-only cache designed for streaming and bounded-memory transformers.

```python
import oxideon as ox

# Fixed buffer with sliding-window eviction of 512 tokens
std_cache = ox.StandardKVCache(
    max_batch_size=4,
    max_seq_len=2048,
    num_heads=8,
    head_dim=64,
    sliding_window=512,
)

keys_chunk = ox.Tensor.randn([4, 64, 8, 64])
vals_chunk = ox.Tensor.randn([4, 64, 8, 64])

# Updates cache and automatically evicts oldest tokens when exceeding sliding window
k_out, v_out = std_cache.update(keys_chunk, vals_chunk)
```

---

## 2. Speculative Decoding

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) accelerates autoregressive text generation by leveraging a small, lightweight **draft model** to propose $K$ candidate tokens in parallel, which are verified in a single forward pass of the larger **target model**.

$$\text{Acceptance Rule: Accept candidate token } x \text{ with probability } \min\left(1, \frac{P_{\text{target}}(x)}{Q_{\text{draft}}(x)}\right)$$

If rejected, Oxideon samples from the normalized residual distribution $\max(0, P_{\text{target}}(x) - Q_{\text{draft}}(x))$, strictly recovering the exact target probability distribution without quality degradation.

```python
import numpy as np
import oxideon as ox

# Define draft and target model forward functions
def draft_model(token_ids):
    # Fast lightweight model returning logits over vocab
    return np.random.randn(32000)

def target_model(token_ids):
    # Full large verifier model returning logits over vocab
    return np.random.randn(32000)

decoder = ox.SpeculativeDraftDecoder(
    draft_model_fn=draft_model,
    target_model_fn=target_model,
    gamma=4,             # Speculate 4 lookahead tokens per step
    temperature=0.8,
)

prompt = [1, 1024, 450]
generated_tokens = decoder.generate(prompt, max_new_tokens=64)

# Inspect execution telemetry
print(f"Total tokens generated: {decoder.stats.total_generated}")
print(f"Draft tokens proposed:  {decoder.stats.draft_tokens_proposed}")
print(f"Draft tokens accepted:  {decoder.stats.draft_tokens_accepted}")
print(f"Acceptance rate:        {decoder.stats.acceptance_rate * 100:.1f}%")
print(f"Empirical speedup:      {decoder.stats.speedup:.2f}x")
```

---

## 3. FlashAttention-2 Memory-Efficient Tiling

Standard attention materializes an intermediate $N \times N$ attention matrix, incurring $O(N^2)$ memory traffic to high-bandwidth memory (HBM). Oxideon implements the **FlashAttention-2** online softmax tiling algorithm (Dao, 2023), computing exact scaled dot-product attention in blocks while maintaining running maximum $m_i$ and running sum of exponents $l_i$.

```python
import oxideon as ox

# Tensor shapes: [batch, heads, seq_len, head_dim]
B, H, S, D = 2, 8, 1024, 64
query = ox.Tensor.randn([B, H, S, D])
key   = ox.Tensor.randn([B, H, S, D])
value = ox.Tensor.randn([B, H, S, D])

# 1. Functional interface with block tiling and causal masking
attn_out = ox.flash_attention_forward(
    query,
    key,
    value,
    block_size_r=64,   # Query block row size Br
    block_size_c=64,   # Key/Value block column size Bc
    causal=True,       # Autoregressive lower-triangular causal mask
)

# 2. PyTorch-style Module interface
tiled_attn = ox.TiledAttention(
    embed_dim=512,
    num_heads=8,
    block_size=64,
    causal=True,
)

x = ox.Tensor.randn([2, 512, 512]) # [B, S, embed_dim]
out = tiled_attn(x, x, x)
```

---

## 4. Ultra-Low-Bit K-Quants & BitNet 1.58b

Oxideon supports sub-4-bit weight compression, GGUF/llama.cpp-compatible K-quants, and 1.58-bit ternary quantization:

| Quantization Type | Bits / Weight | Number of Levels | Storage Density | Supported Layer |
|---|---|---|---|---|
| `Q2_K` | 2-bit | 4 levels | 4 weights / byte | Packed block quantization |
| `Q3_K` | 3-bit | 8 levels | Variable packing | Block affine quantization |
| `Q5_K` | 5-bit | 32 levels | Low-loss compression | Block affine quantization |
| `Q6_K` | 6-bit | 64 levels | Near-FP16 fidelity | Block affine quantization |
| `BitNet b1.58` | 1.58-bit | 3 levels $\{-1, 0, 1\}$ | Ternary matrix math | `ox.BitLinear` |

### Block Quantization Workflow
```python
import oxideon as ox

weights = ox.Tensor.randn([512, 512])

# Quantize to 2-bit packed K-quant (Q2_K)
q2_block = ox.quantize_k_quant(weights, quant_type="q2_k", block_size=32)
print("Packed byte array shape:", q2_block.data.shape)

# Dequantize back to high-precision Tensor
recon_weights = ox.dequantize_k_quant(q2_block)
```

### Ternary `BitLinear` (1.58-bit Inference)
Weights are constrained to $\{-1, 0, 1\}$ using mean absolute scale normalization, while activations are dynamically quantized to 8-bit integers:

$$W_{\text{ternary}} = \text{round}\left(\text{clip}\left(\frac{W}{\gamma + \epsilon}, -1, 1\right)\right), \quad \gamma = \frac{1}{nm}\sum_{ij} |W_{ij}|$$

```python
import oxideon as ox

bit_linear = ox.BitLinear(in_features=256, out_features=128, bias=True)

x = ox.Tensor.randn([8, 256])
output = bit_linear(x) # [8, 128]
```

### Loading a GGUF model

`GGUFCausalLM.from_file` reads the architecture from the GGUF metadata and loads every weight by its
llama.cpp tensor name (`token_embd.weight`, `blk.N.attn_q.weight`, ..., `output.weight`, falling back to
the tied embedding). Dequantization covers `F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_NL IQ4_XS TQ1_0 TQ2_0 MXFP4 NVFP4 Q1_0 Q2_0` (all
checked bit-for-bit against llama.cpp's own dequantizers) plus plain `F64` / `I8`..`I64` tensors.
Llama-style grouped-query attention, the real FFN width and RMS epsilon are honored; a missing or
mis-shaped tensor is an error, never a silently random weight.

```python
import os
import oxideon as ox

# Needs a real checkpoint, e.g. Qwen2.5-0.5B-Instruct Q4_0 saved as model.gguf
# (see examples/gguf_generate.rs); prints a hint and skips when it is absent.
if not os.path.exists("model.gguf"):
    print("model.gguf not found; download a GGUF checkpoint to run this snippet")
else:
    model = ox.GGUFCausalLM.from_file("model.gguf")
    logits = model.forward(ox.Tensor.from_i64([1, 15043], [1, 2]))  # [1, 2, vocab]
    tok = ox.GGUFTokenizer.from_file("model.gguf")  # vocabulary + scores stored in the file
    ids = model.generate(tok.encode("Hello"), max_new_tokens=16, eos_token=tok.eos_id)
    print(tok.decode(ids))
```

`GGUFTokenizer` implements both GGUF tokenizer models: SentencePiece (`llama`) and byte-level BPE
(`gpt2`, with the GPT-2, Llama-3 and Qwen2 word-split patterns). Control tokens written inside the text
(`<|eot_id|>`, ...) are not parsed; add them by id. `generate` decodes with a per-layer KV cache (`forward_cached` in Rust). Architectures: `llama` (Llama 1/2/3, Mistral, Mixtral-style MoE), `mistral3`, `internlm2`, `exaone`, `smollm3`,
`granite`, `granitemoe`, `olmo`, `olmo2`, `olmoe`, `phi2`, `phi3`, `stablelm`, `starcoder2`, `command-r`, `orion`, `nemotron`, `jais2`, `gptneox`, `falcon`, `codeshell`, `minicpm`, `phimoe`, `pangu-embedded`,
`hunyuan-dense`, `dbrx`, `hunyuan-moe`, `dots1`, `bailingmoe`, `bailingmoe2`, `ernie4_5`, `ernie4_5-moe`, `gpt-oss` (attention sinks, clamped SwiGLU, expert biases), `cohere2`, `granite_swa`, `deci` and `openelm` (per-layer head counts from array metadata; attention-free layers are refused), `glm4` and `chatglm` (fused gate/up; GLM4's M-RoPE sections are treated as half-split rotation for text), `glm4moe` (sigmoid router with selection bias, shared expert, optional Q/K norms), `qwen` (Qwen v1), `deepseek` (DeepSeek v1 MoE), `plamo`, `mellum`, `maple`, `minimax-m2`, `exaone-moe`, `afmoe`, `spark2_5` (sigmoid attention output gate), `mimo2` (optional sinks; its graph ignores Q/K/V biases and scales), `laguna` (softplus attention gate), `kolibri1` (experts chosen on logits + bias), `flint` (per-layer rope flags, shared-expert scales), `cohere2moe` (parallel residual, rope on SWA + dense-lead layers, averaged MoE output), `grok` (attention-logit softcap, embedding / output scales, GELU experts + dense FFN), `arctic` (dense FFN in parallel with an MoE fed through `ffn_norm_exps`), `qwen2vl`, `qwen3vl`, `qwen3vlmoe` (text side only: M-RoPE / interleaved M-RoPE with text positions, so pairs on the unused fourth channel stay unrotated; no vision tower or deepstack), `paddleocr`, `hunyuan_vl` (text towers; NTK `rope.scaling.alpha`), `gpt2`, `starcoder` (learned absolute positions), `bloom` (embedding LayerNorm), `mpt` (clamped QKV, full-width LayerNorm Q/K norms, FFN activation scales), `refact`, `jais` (all with ggml's ALiBi slopes and no RoPE), `mistral4` (DeepSeek-2 layout), `llama4` (weight-before-FFN routing, chunked attention, NoPE query temperature, weightless Q/K norms; chunking and temperature only matter past 8192 tokens and are covered by a unit test, not a reference), `hy_v3`, `plamo3` (tolerates converters that write `post_*_norm` / `exp_probs_b` without a suffix), `smallthinker` (router fed by the raw block input, ReLU experts, NoPE every n-th layer counted from 0), `grovemoe` (chunk experts on the routed output; the reference uses one expert per group, so the `expert / group` index mapping for larger groups follows llama.cpp's code but is unverified), `minicpm3` (MLA with a combined `attn_kv_b`, hard-coded MiniCPM scalings; the reference has an all-rotary head, so a non-empty NoPE slice is covered by the DeepSeek-2 path only), `maion-coder` (latent MoE: routing on the full-width input, experts in the latent space), `mamba` and `mamba2` (selective state-space mixers: causal depthwise convolution with rolling state, sequential scan on the host, gated by `silu(z)`; decoding carries the conv and SSM state in the per-layer cache), `nemotron_h` and `nemotron_h_moe` (hybrid: each layer is one mixer - Mamba-2, attention without RoPE, or a ReLU² FFN - chosen by the per-layer head / FFN arrays; the dense layers are verified against the reference; the sparse layers of `nemotron_h_moe` - sigmoid router with a selection bias, ungated ReLU² experts in an optional latent space, ungated shared expert on the full-width input - are checked by a unit test against scalar arithmetic only, because the reference model has no sparse layer), `granitehybrid` (each layer swaps attention for a Mamba-2 mixer when it has no KV heads and keeps its FFN, Granite scalings as in `granite`), `falcon-h1` (attention and a Mamba-2 mixer in parallel on the same input, then the FFN; the SSM state lives in an extra cache slot after the per-layer ones, so always create caches with `new_cache()`), `jamba` (Mamba v1 mixers with their dt / B / C norms between attention layers without RoPE, each followed by a dense or MoE FFN), `lfm2` and `lfm2moe` (gated short-convolution layers - `out_proj(c * conv(b * x))` with a rolling state - between attention layers with Q/K norms, then a dense or MoE FFN), `qwen35`, `qwen35moe`, `qwen3next` (gated DeltaNet linear-attention layers - causal convolution over `[q | k | v]`, the delta rule with per-head state, gated RMS output norm - alternating with full-attention layers whose Q projection also carries an output gate; interleaved M-RoPE for Qwen3.5. The references have equal key and value head counts and use the plain `attn_qkv` tensors, so llama.cpp's key-head tiling for unequal counts and the fused `ssm_in` layout follow its code without a reference), `deepseek32` and `glm-dsa` (DeepSeek sparse attention: exact while the context fits the indexer's `top_k`, which is 2048 for the real models; the lightning indexer itself is not implemented, so longer contexts are refused with an error), `muse-glimmer` (weightless RMS norm over the embeddings, sigmoid attention gate, post-norms with epsilon 1e-8, logit scale and final softcap, RoPE on the sliding-window layers only), `beam` (NoPE full layers with a power-law query temperature, per-head gate bias, weighted residual, sigmoid MoE without group routing), `talkie` (weightless norms, per-head Q gain, per-layer embedding skip), `hrm_text` (looped low / high stacks with per-slot KV caches and per-slot projection scales), `k2-horizon` (grouped norms, log2-softplus attention gate, MoE with routed value experts), `muse-spark` (alpha / beta residuals, per-head output norm, latent MoE with expert-space norms), `cogvlm` (text path), `qwen3tts`, `pockettts`, the diffusion LMs `dream` / `llada` / `llada-moe` / `rnd1` (bidirectional attention, one forward pass; no denoising loop), `arcee`, `xverse`, `baichuan`, `nanbeige`,
`maincoder`, `seed_oss`, `exaone4`, `deepseek2` (multi-head latent attention, sigmoid-gated MoE with a
selection bias, shared and fused experts), `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`, `gemma`, `gemma2` and `gemma3`;
any other `general.architecture` is refused with an error rather than run wrongly. Every one of them produces the same
logits as llama.cpp on its own reference models (`tests/gguf_llama_cpp_parity.rs`, see `tools/ggml_reference/README.md`),
and KV-cached decoding agrees with the full forward, including MLA's latent cache and sliding windows.

It has also run a real checkpoint: Qwen2.5-0.5B-Instruct (`Q4_0`, 409 MB). `llama-tokenize` and `GGUFTokenizer` produce the same
token ids, the logits of all 48 positions pick the same top-1 token as llama.cpp (worst relative error 8e-2: llama.cpp quantizes
the activations to Q8_0 before a Q4_0 dot product, this crate multiplies dequantized F32), and greedy decoding reproduces
llama.cpp's text for the first 10-16 tokens of three prompts before a near-tie can flip. `examples/gguf_generate.rs` runs that
loop (`cargo run --release --example gguf_generate -- model.gguf "The capital of France is" 16`); the opt-in test is
`qwen25_0_5b_q4_0_real_model_matches_llama_cpp` (`OXIDEON_LLAMA_REAL_MODELS`).

Also honored: grouped-query attention, any head size, partial rotary dims, Q/K norms, post-norms, embedding / residual /
logit scales, Gemma's GELU and attention / final soft-capping, per-layer sliding windows with their own RoPE base,
`rope.scaling.*` (`linear`, `yarn` including DeepSeek's log multiplier, LongRoPE long/short factor tensors),
`rope.scaling.attn_factor`, `rope_freqs.weight`, fused QKV and fused gate/up tensors, and sparse MoE routing (softmax or
sigmoid gating, selection bias, group-limited routing, dense layers mixed in, optional shared expert). Rotary tables cover at most 131,072
positions.

Sliding windows follow llama.cpp: they apply only to the architectures that implement them (Gemma 2/3, OLMo 2, and
Exaone 4's 64-layer model), so a window key in a plain `llama`/Mistral file is ignored, exactly as llama.cpp does.
Norms may be RMS or LayerNorm (with or without weight / bias, OLMo 1's non-parametric one included); FFNs may be gated or
plain, and attention / FFN may run in parallel (Phi-2, Command-R, StableLM's parallel variant).

Not covered: other hybrid and linear-attention models (PLaMo-2,
Kimi-Linear, BailingMoE-3, Qwen4-Exp; plain Mamba / Mamba-2 are supported), sparse-attention indexer models beyond their `top_k` (and dots-3, HY-V4, DeepSeek-V4), diffusion language
models (LLaDA, Dream, RND1), vision / audio towers, the recurrent HRM-Text, Beam and Muse (their residual / norm schemes), Talkie,
K2-Horizon, CogVLM, and the clamped-SwiGLU architectures (DeepSeek-V4, GLM-5, ...). Attention temperature scaling is implemented
for Llama 4 only. The DeepSeek YaRN path (log multiplier) follows llama.cpp's formulas but is not covered by a reference model,
unlike the unscaled case. Reference models are 48 tokens long, so behaviour that only starts beyond that (Llama 4's 8192-token
chunks and NoPE temperature, GroveMoE groups larger than one) follows llama.cpp's code without a reference.

---

## 5. Byte-Pair Encoding Subword Tokenizer

Oxideon includes a self-contained, trainable Byte-Pair Encoding (`ox.BPETokenizer`) with word-frequency pre-aggregation for sub-millisecond vocabulary training:

```python
import oxideon as ox

corpus = [
    "deep learning with rust and python",
    "high throughput inference engines and kv cache management",
    "zero copy tensors and autograd optimization",
]

tokenizer = ox.BPETokenizer()
tokenizer.train(corpus, max_vocab_size=256, min_freq=1)

# Encode text to subword IDs
tokens = tokenizer.encode("inference engines with autograd", add_special_tokens=True)
print("Token IDs:", tokens)

# Reconstruct text
decoded_str = tokenizer.decode(tokens, skip_special_tokens=True)
print("Decoded:", decoded_str)

# Save & Load tokenizer
tokenizer.save("bpe_model.json")
loaded_tokenizer = ox.BPETokenizer.load("bpe_model.json")
```

---

## 6. Parameter-Efficient Fine-Tuning: DoRA & LoRA+

### DoRA: Weight-Decomposed Low-Rank Adaptation
Weight-Decomposed Low-Rank Adaptation (Liu et al., ICML 2024) decouples weights into directional component $V + \Delta V$ and learnable magnitude vector $m$:

$$W = m \odot \frac{V + \frac{\alpha}{r} B A}{\|V + \frac{\alpha}{r} B A\|_2}$$

```python
import oxideon as ox

dora_layer = ox.DoRALinear(
    in_features=512,
    out_features=256,
    r=8,              # Rank
    lora_alpha=16.0,  # Scaling factor
)

x = ox.Tensor.randn([4, 512])
out = dora_layer(x)

# Analytically merge adapted weights into base weight matrix for 0-latency inference
dora_layer.merge()
assert dora_layer.merged is True
```

### LoRA+: Asymmetric Learning Rate Dynamics
LoRA+ (Hayou et al., 2024) sets asymmetric learning dynamics between adapter matrices $A$ and $B$, scaling updates to $B$ by $\lambda > 1$ to overcome low-rank feature collapse:

```python
import oxideon as ox

loraplus_layer = ox.LoRAPlusLinear(
    in_features=512,
    out_features=256,
    r=8,
    lora_alpha=16.0,
    b_ratio=4.0,      # Scale matrix B updates by 4x relative to matrix A
)
```

---

## 7. KV Cache Eviction Policies

Beyond sliding windows, three score-aware policies keep long generations inside
a fixed KV budget (see `docs/BENCHMARKS.md` for the <10% overhead numbers):

- `ox.StreamingSinkCache`: StreamingLLM attention sinks (first `n_sink` tokens)
  plus the recent window; no scores needed.
- `ox.ScoredEvictionCache(policy="h2o")`: heavy hitters by cumulative attention
  plus the newest `recent_window` positions.
- `policy="tova"`: last-query scores only. `policy="snap"`: max over the last
  `observe_window` queries with causal max-pooling before top-k.

```python
import numpy as np
import oxideon as ox

cache = ox.StreamingSinkCache(1, 1024, 8, 64, n_sink=4)
k = ox.Tensor.randn([1, 512, 8, 64]); v = ox.Tensor.randn([1, 512, 8, 64])
k_full, v_full = cache.update(k, v)  # [1, 512, 8, 64]

h2o = ox.ScoredEvictionCache(1, 1024, 8, 64, recent_window=64, policy="h2o")
scores = np.random.rand(1, 8, 512, 512).astype(np.float64)  # [B, H, new, total]
k_full, v_full = h2o.update(k, v, scores)
```

## 8. MXFP4 Microscaling Quantization

Exact OCP Microscaling MXFP4: E2M1 data nibbles with a shared E8M0
power-of-two scale per 32-element block — 4.25 bits/param (**7.53x** vs f32).
NaN is refused; infinities and out-of-range outliers saturate.

```python
import oxideon as ox

w = ox.Tensor.randn([1024, 1024])
codes, scales = ox.quantize_mxfp4(w)
w_hat = ox.dequantize_mxfp4(codes, scales)
err = (w - w_hat).abs().mean().item()
print(f"MXFP4 mean abs err: {err:.4f}")
```

## 9. MLA & Native Sparse Attention

Two reference attention blocks alongside MQA/GQA (numpy-backed, causal):

- `ox.MLAAttention`: low-rank query/KV latents with a decoupled shared RoPE
  projection (DeepSeek-V2 style).
- `ox.NativeSparseAttention`: compressed (block-mean) + top-k selected fine
  tokens + sliding window, mixed by a learned gate.

```python
import oxideon as ox

x = ox.Tensor.randn([2, 128, 128])
mla = ox.MLAAttention(128, 4, q_lora_rank=16, kv_lora_rank=16)
nsa = ox.NativeSparseAttention(128, 4, block_size=32, top_blocks=2, window_size=64)
print(mla(x).shape, nsa(x).shape)  # [2, 128, 128] each
```

---

## 10. Radix Tree Prefix KV-Cache (`ox.prefix_cache`)

Oxideon features a Radix-Tree-based prefix KV-cache manager inspired by vLLM and SGLang. Instead of re-computing attention states for repeated system prompts, few-shot exemplars, or multi-turn conversational history, `RadixPrefixCache` organizes tokens in a prefix trie where each node maps contiguous token sequences to physical KV block identifiers.

```python
import oxideon as ox
from oxideon.prefix_cache import RadixPrefixCache

# Initialize prefix cache with block_size=16 tokens
cache = RadixPrefixCache(block_size=16)

# Insert system prompt tokens (32 tokens -> 2 physical blocks)
system_tokens = list(range(100, 132))
system_blocks = [0, 1]
cache.insert(system_tokens, system_blocks)

# New prompt sharing the exact system prompt prefix
user_prompt = system_tokens + [200, 201, 202]
matched_tokens, matched_blocks = cache.match_prefix(user_prompt)

print(f"Matched {len(matched_tokens)} tokens, reusing physical blocks: {matched_blocks}")
# Matched 32 tokens, reusing physical blocks: [0, 1]

# Automatic LRU eviction when memory threshold is reached
freed_blocks = cache.evict_lru(num_blocks=2)
print(f"Freed blocks under LRU policy: {freed_blocks}")
```

- **Node Splitting:** When two prompt sequences diverge, common parent nodes are split transparently.
- **Microsecond Latency:** Prefix matching and insertion take $< 0.01$ ms across thousands of tokens.

---

## 11. Advanced FP8 & Block-Wise INT4 Quantization (`ox.quantization_advanced`)

Oxideon provides production-grade simulated low-bit quantization formats with Straight-Through Estimator (STE) training modules:

### FP8 (E4M3 and E5M2)
- **E4M3**: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits; max normal = 448.0. Ideal for activations and forward passes where dynamic range is moderate and precision is critical.
- **E5M2**: 1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits; max normal = 57344.0. Matches IEEE FP16 dynamic range, ideal for gradients and sensitive weight tensors.

```python
import oxideon as ox
from oxideon.quantization_advanced import quantize_fp8_e4m3, dequantize_fp8_e4m3

weights = ox.Tensor.randn([1024, 1024])

# Simulate FP8 E4M3 quantization with dynamic tensor scaling
q_weights, scale = quantize_fp8_e4m3(weights)
reconstructed = dequantize_fp8_e4m3(q_weights, scale)

mae = (weights - reconstructed).abs().mean().item()
print(f"FP8 E4M3 MAE: {mae:.6f}, Memory compression: 4.00x")
```

### Block-Wise Weight-Only INT4 Quantization
Groups weights into small blocks (e.g. 32 elements) and computes independent scaling factors and integer zero-points, eliminating degradation from cross-channel activation outliers:

```python
from oxideon.quantization_advanced import WeightOnlyInt4Quantizer

# 8.0x compression: map floating-point weights into 4-bit integers [0, 15]
q_int4, scales, zeros = WeightOnlyInt4Quantizer.quantize(weights, group_size=32)
recovered = WeightOnlyInt4Quantizer.dequantize(q_int4, scales, zeros, group_size=32)

print(f"INT4 effective bits/param: 4.00 b/p, Reconstruction MAE: {(weights - recovered).abs().mean().item():.4f}")
```

---

## 12. Selective State Space Architectures: Mamba SSM (`ox.state_space`)

Oxideon incorporates selective state-space sequence modeling (Gu & Dao, 2023), replacing quadratic $O(L^2)$ attention with an input-dependent recurrent scan that processes long sequences in strict linear $O(L)$ time and constant $O(1)$ memory per token step.

$$
\Delta_t = 	ext{softplus}(	ext{Parameter} + 	ext{Proj}(x_t)), \quad ar{A}_t = \exp(\Delta_t \cdot A), \quad ar{B}_t = \Delta_t \cdot B_t
$$
$$
h_t = ar{A}_t h_{t-1} + ar{B}_t x_t, \quad y_t = C_t \cdot h_t + D \cdot x_t
$$

```python
import numpy as np
import oxideon as ox
from oxideon.state_space import MambaBlock

# Full selective Mamba block with depthwise 1D conv and selective scan
mamba = MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2)

# Forward pass over long sequence (B=2, L=512, D=64)
x = ox.from_numpy(np.random.randn(2, 512, 64).astype(np.float32))
y = mamba(x)

print(f"Mamba output shape: {y.to_numpy().shape}")  # [2, 512, 64]
```

---

## 13. Sparse Long-Context Multi-Head Attention (`ox.sparse_attn`)

For long-context transformer modeling, Oxideon provides BigBird (Zaheer et al., NeurIPS 2020) and sliding-window sparse attention patterns that compress quadratic $O(L^2)$ computation into $O(L)$ sparse graph connectivity:

1. **Global Tokens:** Dedicated memory tokens attending to and attended by all sequence positions.
2. **Sliding Window:** Banded attention attending to immediate left/right neighbors within `window_radius`.
3. **Random Graph Edges:** Stochastic connections preserving graph diameter and expander properties.

```python
import numpy as np
import oxideon as ox
from oxideon.sparse_attn import SparseMultiheadAttention, build_bigbird_mask

# Generate a BigBird binary connectivity mask for L=1024
mask = build_bigbird_mask(seq_len=1024, num_global=2, window_radius=4, num_random=2)
print(f"Mask density: {mask.mean():.4f} (sparse vs dense 1.000)")

# Multi-head attention layer with BigBird sparse pattern
sparse_mha = SparseMultiheadAttention(embed_dim=128, num_heads=4, window_radius=4)
x = ox.from_numpy(np.random.randn(2, 1024, 128).astype(np.float32))
out = sparse_mha(x)
print(f"Sparse attention output shape: {out.to_numpy().shape}")  # [2, 1024, 128]
```
