# Oxideon Getting Started & Developer Tutorial

Welcome to the comprehensive, hands-on tutorial for **Oxideon**. This guide will take you step-by-step from core tensor fundamentals to training frontier neural networks, high-throughput LLM serving, cloud dataset streaming, and advanced scientific AI modeling.

---

## Table of Contents
1. [Installation & Setup](#1-installation--setup)
2. [Tensor Basics, DTypes & Strided Views](#2-tensor-basics-dtypes--strided-views)
3. [Functional Autograd & Higher-Order Derivatives](#3-functional-autograd--higher-order-derivatives)
4. [Building Neural Networks with Optimizers](#4-building-neural-networks-with-optimizers)
5. [Subword Tokenization with BPETokenizer](#5-subword-tokenization-with-bpetokenizer)
6. [Tiled FlashAttention-2 & Causal Masking](#6-tiled-flashattention-2--causal-masking)
7. [Paged KV Caching (PagedAttention)](#7-paged-kv-caching-pagedattention)
8. [Speculative Decoding Acceleration](#8-speculative-decoding-acceleration)
9. [Parameter-Efficient Fine-Tuning (DoRA & LoRA+)](#9-parameter-efficient-fine-tuning-dora--lora)
10. [Human Preference Alignment (DPO, ORPO, KTO)](#10-human-preference-alignment-dpo-orpo-kto)
11. [Cloud Object Streaming (AWS S3 & Google Cloud Storage)](#11-cloud-object-streaming-aws-s3--google-cloud-storage)
12. [Scientific AI, Vision Transformers & Causal Discovery](#12-scientific-ai-vision-transformers--causal-discovery)

---

## 1. Installation & Setup

### Python (via pip or local build)
```bash
# Clone the repository
git clone https://github.com/oxideon/oxideon.git
cd oxideon

# Build and install the high-performance abi3 wheel
pip install maturin
maturin develop --release
```

### Rust (Cargo dependency)
Add Oxideon to your `Cargo.toml`:
```toml
[dependencies]
oxideon = { version = "0.1.0", features = ["rayon", "wgpu"] }
```

---

## 2. Tensor Basics, DTypes & Strided Views

Oxideon tensors support zero-copy strided views, arbitrary stepped slicing, and automatic broadcasting.

```python
import oxideon as ox
import numpy as np

# Create tensors from Python lists, NumPy arrays, or factory functions
x = ox.Tensor.arange(0.0, 24.0, 1.0).reshape([2, 3, 4])
print("Original Tensor shape:", x.shape)  # [2, 3, 4]

# Zero-copy transposition and permutations
view = x.transpose(0, 2)
print("Transposed view shape:", view.shape)  # [4, 3, 2]
print("Shares storage with original?", view.shares_storage(x))  # True!

# Stepped slicing (including negative strides and flipping)
# Slices row indices with step 2
sub_view = x[:, ::2, :]
print("Sub-view shape:", sub_view.shape)

# Convert seamlessly to/from NumPy without data corruption
np_arr = x.to_numpy()
reconstructed = ox.Tensor.from_numpy(np_arr)
assert np.allclose(x.to_numpy(), reconstructed.to_numpy())
```

---

## 3. Functional Autograd & Higher-Order Derivatives

Oxideon provides a fully functional autograd engine with higher-order differentiation support ($1^{\text{st}}, 2^{\text{nd}}, 3^{\text{rd}}$ derivatives, Jacobians, and Hessians).

```python
import oxideon as ox

# Create leaf tensor tracking gradients
x = ox.Tensor.from_slice([3.0], [1], requires_grad=True)

# Define scalar function: y = x^4
y = x.powf(4.0)

# Compute 1st derivative: dy/dx = 4 * x^3 = 4 * 27 = 108.0
dy = ox.grad([y], [x], create_graph=True)[0]
print("1st derivative dy/dx:", dy.item())  # 108.0

# Compute 2nd derivative: d2y/dx2 = 12 * x^2 = 12 * 9 = 108.0
d2y = ox.grad([dy], [x], create_graph=True)[0]
print("2nd derivative d2y/dx2:", d2y.item())  # 108.0

# Compute 3rd derivative: d3y/dx3 = 24 * x = 24 * 3 = 72.0
d3y = ox.grad([d2y], [x], create_graph=False)[0]
print("3rd derivative d3y/dx3:", d3y.item())  # 72.0
```

---

## 4. Building Neural Networks with Optimizers

Oxideon provides modular neural network layers and state-of-the-art optimizers:

```python
import oxideon as ox
from oxideon.nn import Sequential, Linear, GELU, LayerNorm
from oxideon.optim import AdamW

# Construct a multi-layer perceptron (MLP)
model = Sequential(
    Linear(in_features=16, out_features=64),
    GELU(),
    LayerNorm(normalized_shape=64),
    Linear(in_features=64, out_features=4)
)

# Initialize AdamW with weight decay
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# Training step
inputs = ox.Tensor.randn([32, 16])
targets = ox.Tensor.randn([32, 4])

# Forward pass
predictions = model.forward(inputs)
loss = ((predictions - targets) ** 2).mean()

# Backward pass & optimization step
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Step Loss: {loss.item():.4f}")
```

---

## 5. Subword Tokenization with BPETokenizer

Oxideon features an ultra-fast subword Byte-Pair Encoding (BPE) tokenizer with word-frequency pre-aggregation ($98.5\times$ faster than naive scans).

```python
import oxideon as ox

# Corpus for training
corpus = [
    "deep learning with rust and python is incredibly fast and memory safe",
    "zero copy tensors and autograd make scientific computing effortless",
    "large language models utilize paged attention and speculative decoding"
]

# Initialize and train the BPE tokenizer
tokenizer = ox.BPETokenizer()
tokenizer.train(corpus, max_vocab_size=100)

# Encode text to token IDs
text = "deep learning models are fast"
encoded_ids = tokenizer.encode(text)
print("Encoded IDs:", encoded_ids)

# Decode token IDs back to text
decoded_text = tokenizer.decode(encoded_ids)
print("Decoded Text:", decoded_text)

# Save and reload tokenizer state
tokenizer.save("my_tokenizer.json")
loaded_tokenizer = ox.BPETokenizer.load("my_tokenizer.json")
```

---

## 6. Tiled FlashAttention-2 & Causal Masking

FlashAttention-2 computes scaled dot-product attention without materializing intermediate $O(S^2)$ attention matrices:

```python
import oxideon as ox
import numpy as np

batch_size = 2
num_heads = 4
seq_len = 128
head_dim = 64

# Shape: (batch, num_heads, seq_len, head_dim)
q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
k = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
v = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

# Exact tiled forward pass with causal autoregressive masking
output = ox.flash_attention_forward(
    ox.Tensor.from_numpy(q),
    ox.Tensor.from_numpy(k),
    ox.Tensor.from_numpy(v),
    causal=True,
    block_size_r=64,
    block_size_c=64,
)

print("FlashAttention output shape:", output.shape)  # (2, 4, 128, 64)
```

---

## 7. Paged KV Caching (PagedAttention)

Oxideon's `PagedKVCache` eliminates virtual memory fragmentation by allocating 16-token physical pages on demand:

```python
import oxideon as ox
import numpy as np

# Allocate a PagedKVCache with 128 physical blocks of 16 tokens each
cache = ox.PagedKVCache(
    num_blocks=128,
    block_size=16,
    num_heads=8,
    head_dim=64,
)

# Allocate blocks for a 24-token sequence, then append one token at a time
num_tokens = 24
block_table = cache.allocate(num_tokens=num_tokens)
for i in range(num_tokens):
    k_tok = ox.Tensor.randn([8, 64])
    v_tok = ox.Tensor.randn([8, 64])
    block_table = cache.append(block_table, current_len=i, key=k_tok, value=v_tok)

# Read contiguous sequence tokens (automatically stitches non-contiguous physical blocks)
k_seq_t, v_seq_t = cache.read_sequence(block_table, seq_len=num_tokens)
k_seq, v_seq = k_seq_t.to_numpy(), v_seq_t.to_numpy()
print("Retrieved KV sequence shapes:", k_seq.shape, v_seq.shape)  # (24, 8, 64)

# Free sequence resources when generation completes
cache.free(block_table)
```

---

## 8. Speculative Decoding Acceleration

Accelerate autoregressive inference using a fast draft model and parallel target verification:

```python
import oxideon as ox
import numpy as np

# Define draft and target model forward functions
def draft_model(token_ids):
    # Proposes next-token logits for the current prefix
    return np.random.randn(1000).astype(np.float32)

def target_model(token_ids):
    # Evaluates the prefix in parallel
    return np.random.randn(1000).astype(np.float32)

# Setup speculative decoder with lookahead gamma = 4
decoder = ox.SpeculativeDraftDecoder(
    draft_model_fn=draft_model,
    target_model_fn=target_model,
    gamma=4,
    temperature=0.7
)

# Generate accelerated completion
prompt = [101, 45, 89]
generated_tokens = decoder.generate(prompt, max_new_tokens=32)
stats = decoder.stats

print("Generated tokens:", generated_tokens)
print(f"Speculative Acceptance Rate: {stats.acceptance_rate * 100:.1f}%")
print(f"Effective Decode Speedup: {stats.speedup:.2f}x")
```

---

## 9. Parameter-Efficient Fine-Tuning (DoRA & LoRA+)

Fine-tune models using Weight-Decomposed Low-Rank Adaptation (DoRA) or Asymmetric LoRA+:

```python
import oxideon as ox
import numpy as np

# 1. DoRA: Decomposes magnitude and directional adaptation
dora = ox.DoRALinear(in_features=256, out_features=256, r=8, lora_alpha=16.0)
x = ox.Tensor.randn([16, 256])
out_dora = dora.forward(x)

# Merge DoRA adapters directly into base weights for zero-latency inference
dora.merge()

# 2. LoRA+: Asymmetric learning rates for matrix A and B
lora_plus = ox.LoRAPlusLinear(
    in_features=256,
    out_features=256,
    r=8,
    b_ratio=8.0  # lr_B / lr_A = 8.0
)
out_lora = lora_plus.forward(x)
```

---

## 10. Human Preference Alignment (DPO, ORPO, KTO)

Align foundation models directly with human feedback using modern preference loss functions:

```python
import oxideon as ox

# Log-probabilities of chosen vs rejected completions (policy and reference)
chosen_logps = ox.Tensor.from_slice([-0.25, -0.15, -0.30, -0.10], [4])
rejected_logps = ox.Tensor.from_slice([-1.80, -2.10, -1.45, -2.50], [4])
ref_chosen_logps = ox.Tensor.from_slice([-0.50, -0.80, -0.60, -1.00], [4])
ref_rejected_logps = ox.Tensor.from_slice([-1.20, -1.00, -1.60, -0.90], [4])

# 1. Direct Preference Optimization (DPO)
dpo = ox.DPOLoss(beta=0.1)
loss_dpo, chosen_rewards, rejected_rewards = dpo(
    chosen_logps, rejected_logps,
    ref_chosen_logps, ref_rejected_logps
)
print("DPO Loss:", loss_dpo.item())

# 2. Odds Ratio Preference Optimization (ORPO)
orpo = ox.ORPOLoss(lambda_coeff=0.1)
sft_loss = ox.Tensor.from_slice([0.45], [1])
loss_orpo, metrics_orpo = orpo(chosen_logps, rejected_logps, sft_loss=sft_loss)
print("ORPO Loss:", loss_orpo.item())

# 3. Kahneman-Tversky Optimization (KTO) for unpaired binary feedback
kto = ox.KTOLoss(beta=0.1)
labels = ox.Tensor.from_slice([1.0, 0.0, 1.0, 0.0], [4])  # 1 = desirable, 0 = undesirable
loss_kto, metrics_kto = kto(chosen_logps, ref_chosen_logps, labels)
print("KTO Loss:", loss_kto.item())
```

---

## 11. Cloud Object Streaming (AWS S3 & Google Cloud Storage)

Stream datasets and checkpoints directly to and from AWS S3 and Google Cloud Storage:

```python
import oxideon as ox
from oxideon.cloud import aws, gcp

# AWS S3: stream a checkpoint straight to object storage (no local temp files).
# Uses real AWS credentials when configured, otherwise a local mock store.
s3 = aws.S3Client()
ox.save_file({"weight": ox.Tensor.randn([8, 8])}, "s3://my-ml-models/checkpoints/demo.ox")
print("checkpoint on S3:", s3.exists("s3://my-ml-models/checkpoints/demo.ox"))

# Google Cloud Storage (GCS) exposes the same client API
gcs = gcp.GCSClient()
print("objects under gs://my-gcp-models/v1/:", gcs.list_objects("gs://my-gcp-models/v1/"))
```

---

## 12. Scientific AI, Vision Transformers & Causal Discovery

Combine deep learning with causal structure learning and geometric architectures:

```python
import oxideon as ox
import numpy as np

# 1. Vision Transformer (ViT)
vit = ox.VisionTransformer(
    img_size=32, 
    patch_size=4, 
    in_channels=3, 
    num_classes=10, 
    embed_dim=128, 
    depth=6, 
    num_heads=8
)
images = np.random.randn(8, 3, 32, 32).astype(np.float32)
logits = vit.forward(ox.Tensor.from_numpy(images))
print("ViT classification logits shape:", logits.shape)  # (8, 10)

# 2. Causal Structure Discovery via NOTEARS
# Learns DAG adjacency matrix W from continuous data X
data = np.random.randn(200, 10)  # 200 samples, 10 variables
adj_dag = ox.notears_linear(ox.Tensor.from_numpy(data), lambda1=0.1, max_iter=50)
print(f"Discovered DAG edges count: {np.count_nonzero(adj_dag.to_numpy())}")

# 3. Graph Topological Similarity with Weisfeiler-Lehman Subtree Kernel
adj_1 = ox.Tensor.from_numpy(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
adj_2 = ox.Tensor.from_numpy(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32))
gram = ox.weisfeiler_lehman_kernel([adj_1, adj_2], num_iterations=3)
print(f"Weisfeiler-Lehman Graph Similarity: {gram[0, 1].item():.4f}")
```

---

## Next Steps
- Consult the [Python API Reference Guide](PYTHON_API.md) for exhaustive symbol references.
- Read [Inference & LLM Acceleration](INFERENCE_AND_LLM.md) for large model deployment.
- Review [Benchmarks & Performance Analysis](BENCHMARKS.md) for architectural proofs.
