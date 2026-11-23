# Oxideon Python API Reference Manual

**Package:** `oxideon`  
**Version:** `0.1.0`  
**Python Compatibility:** Python $\ge 3.9$ (`abi3`)  
**Public Symbols:** 2,339 public symbols (4,835 typed signatures)

---

## Table of Contents
1. [Core Tensors & Autograd Engine](#1-core-tensors--autograd-engine)
2. [Neural Network Modules (`ox.nn`)](#2-neural-network-modules-oxnn)
3. [Functional API & Loss Functions (`ox.F`, `ox.losses`)](#3-functional-api--loss-functions)
4. [Optimizers & Learning Rate Schedulers (`ox.optim`)](#4-optimizers--learning-rate-schedulers)
5. [Data Pipeline & High-Throughput Loaders (`ox.data`)](#5-data-pipeline--high-throughput-loaders)
6. [Trainer, Metrics & Callbacks (`ox.training`, `ox.callbacks`)](#6-trainer-metrics--callbacks)
7. [Inference, KV Caches & Attention Acceleration](#7-inference-kv-caches--attention-acceleration)
8. [Low-Bit Quantization & PEFT](#8-low-bit-quantization--peft)
9. [Native Cloud Storage & Compute (`ox.cloud`)](#9-native-cloud-storage--compute)
10. [Logging, Tensor Audits & Debug Mode (`ox.logging`)](#10-logging-tensor-audits--debug-mode)
11. [Scientific AI, Graph & Multimodal Subsystems](#11-scientific-ai-graph--multimodal-subsystems)
12. [Millennial Singularity Milestone Engines](#12-millennial-singularity-milestone-engines)
13. [Frontier AI & Scientific Modules (52 Features, 10 Flagship Systems)](#13-frontier-ai--scientific-modules-52-features-10-flagship-systems)
14. [Evaluation Harness (`ox.evals`)](#14-evaluation-harness-oxevals)
15. [Speech, Vision Toolkits & 3D (`ox.speech`, `ox.vision`, `ox.nerf`, `ox.splat`)](#15-speech-vision-toolkits--3d-oxspeech-oxvision-oxnerf-oxsplat)
16. [High-Performance Inference & Serving Systems (`ox.prefix_cache`, `ox.flash_attn`, `ox.sparse_attn`, `ox.sparse_matrix`)](#16-high-performance-inference--serving-systems)
17. [Advanced Quantization & Architecture Optimization (`ox.quantization_advanced`, `ox.ao`)](#17-advanced-quantization--architecture-optimization)
18. [Canonical Deep Architectures & State Space Models (`ox.models`, `ox.state_space`)](#18-canonical-deep-architectures--state-space-models)
19. [Signal Processing, 3D Geometry & Video Analytics (`ox.signal`, `ox.geometry`, `ox.video`)](#19-signal-processing-3d-geometry--video-analytics)
20. [Frontier Optimizers, Loss Functions & Synthetic Data (`ox.optim_ext`, `ox.losses_ext`, `ox.synthetic_data`, `ox.uncertainty_quant`, `ox.interpret`, `ox.func`)](#20-frontier-optimizers-loss-functions--synthetic-data)

---

## 1. Core Tensors & Autograd Engine

### Tensor Construction
```python
import numpy as np
import oxideon as ox

# Primitive factory functions
zeros = ox.zeros([3, 4], dtype="float32")
ones  = ox.ones([3, 4], dtype="float32")
rand  = ox.rand([3, 4])
randn = ox.randn([3, 4])

# Construction from Python lists or NumPy arrays
t1 = ox.Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
t2 = ox.from_numpy(np_array)

# Conversions
np_arr = t1.to_numpy()
```

### Strided Zero-Copy Views
Oxideon supports zero-copy strided views across transpose, slicing, negative steps, and broadcasting:
```python
import oxideon as ox

x = ox.Tensor.arange(0.0, 24.0, 1.0).reshape([2, 3, 4])

# Zero-copy transformations
x_transposed = x.transpose(1, 2)
x_slice = x[:, 1:3, ::-1]          # Negative stride zero-copy reversal
x_broadcast = x.unsqueeze(1)       # Stride 0 broadcasting

# Flattening
flat = x.reshape([-1])
```

### Automatic Differentiation
Exact reverse-mode autograd with support for arbitrary higher-order derivatives (`create_graph=True`):
```python
import oxideon as ox

x = ox.Tensor([3.0], shape=[1]).requires_grad_(True)
y = x ** 3 + 2.0 * x

# 1st-order gradient: dy/dx = 3*x^2 + 2 = 29.0
y.backward(create_graph=True)
grad1 = x.grad()
print("First derivative:", grad1.to_numpy())

# 2nd-order derivative: d^2y/dx^2 = 6*x = 18.0
grad1.backward()
print("Second derivative:", x.grad().to_numpy())
```

---

## 2. Neural Network Modules (`ox.nn`)

All layers subclass `ox.nn.Module`, providing parameter tracking, device migration, and forward execution:

### Feed-Forward & Convolutions
```python
import oxideon as ox

linear = ox.nn.Linear(in_features=128, out_features=64, bias=True)
conv2d = ox.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
```

### Normalization & Activations
```python
import oxideon as ox

layer_norm = ox.nn.LayerNorm(normalized_shape=64)
batch_norm = ox.nn.BatchNorm2d(num_features=32)
relu       = ox.nn.ReLU()
gelu       = ox.nn.GELU()
silu       = ox.nn.SiLU()
dropout    = ox.nn.Dropout(p=0.1)
```

### Sequence & Transformer Architectures
```python
import oxideon as ox

# Embedding
embedding = ox.nn.Embedding(num_embeddings=32000, embedding_dim=512)

# Multi-Head Attention
mha = ox.nn.MultiheadAttention(embed_dim=512, num_heads=8)

# End-to-End Vision Transformer (ViT)
vit = ox.VisionTransformer(
    img_size=32,
    patch_size=4,
    in_channels=3,
    num_classes=10,
    embed_dim=64,
    depth=2,
    num_heads=4,
)

# Sequential Container
model = ox.Sequential(
    ox.Linear(64, 128),
    ox.ReLU(),
    ox.Linear(128, 10),
)
```

---

## 3. Functional API & Loss Functions

Available under `ox.functional` and alias `ox.F`:

### Loss Functions
```python
# Standard Classification & Regression
import numpy as np
import oxideon as ox

logits = ox.Tensor.randn([4, 10])
targets = ox.from_numpy(np.array([1, 2, 3, 4], dtype=np.int64))
float_targets = ox.Tensor.randn([4])
predictions = ox.Tensor.randn([4])
anchor = ox.Tensor.randn([4, 8])
positive = ox.Tensor.randn([4, 8])
negative = ox.Tensor.randn([4, 8])
log_probs = ox.Tensor.randn([8, 2, 5])
ctc_targets = ox.from_numpy(np.array([[1, 2], [3, 1]], dtype=np.int64))

loss_ce  = ox.cross_entropy(logits, targets)
loss_bce = ox.binary_cross_entropy_with_logits(logits[:, 0], float_targets)
loss_mse = ox.mse_loss(predictions, float_targets)
loss_l1  = ox.l1_loss(predictions, float_targets)
loss_focal = ox.focal_loss(logits, targets, gamma=2.0)
loss_huber = ox.huber_loss(predictions, float_targets, delta=1.0)

# Specialized Ranking & Alignment Losses
loss_triplet = ox.triplet_margin_loss(anchor, positive, negative, margin=1.0)
loss_ctc     = ox.ctc_loss(log_probs, ctc_targets)
```

---

## 4. Optimizers & Learning Rate Schedulers (`ox.optim`)

### First- and Second-Order Optimizers
```python
import oxideon as ox

model = ox.Sequential(ox.Linear(8, 4), ox.ReLU(), ox.Linear(4, 2))

# Stochastic Gradient Descent with momentum
opt_sgd = ox.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam & AdamW with decoupled weight decay
opt_adam  = ox.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
opt_adamw = ox.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Advanced SOTA Optimizers
opt_muon  = ox.optim.Muon(model.parameters(), lr=0.02, momentum=0.95)
opt_radam = ox.optim.RAdam(model.parameters(), lr=1e-3, weight_decay=0.01)
opt_sophia = ox.optim.Sophia(model.parameters(), lr=1e-4)
```

### Learning Rate Schedulers
```python
import oxideon as ox

model = ox.Sequential(ox.Linear(8, 4), ox.Linear(4, 2))
optimizer = ox.optim.SGD(model.parameters(), lr=0.01)

# Cosine Annealing
scheduler = ox.optim.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Cyclic Learning Rate
cyclic_sched = ox.optim.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=20)

# Step scheduler each epoch
scheduler.step()
```

---

## 5. Data Pipeline & High-Throughput Loaders (`ox.data`)

Oxideon features high-performance data pipelines optimized with direct contiguous slice streaming and vectorized epoch permutation:

```python
import oxideon as ox

# Wrap tensors into dataset
x_data = ox.Tensor.randn([10000, 64])
y_data = ox.Tensor.randn([10000, 1])
dataset = ox.TensorDataset(x_data, y_data)

# High-throughput DataLoader
loader = ox.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,       # Uses single-permutation array slicing (9x faster)
    drop_last=False,
)

model = ox.Sequential(ox.Linear(64, 1))

# Iterate across batches with zero per-sample indexing overhead
for batch_x, batch_y in loader:
    preds = model(batch_x)
```

### Dataset Splitting
```python
import oxideon as ox

dataset = ox.TensorDataset(ox.Tensor.randn([100, 8]), ox.Tensor.randn([100, 1]))
train_set, val_set = ox.data.random_split(dataset, [0.8, 0.2], seed=42)
```

---

## 6. Trainer, Metrics & Callbacks (`ox.training`, `ox.callbacks`)

### Production `Trainer`
Includes automatic gradient clipping, real-time visual progress bars with throughput (`samples/sec`), ETA tracking, tensor health verification, and debug audits:

```python
import numpy as np
import oxideon as ox

model = ox.Sequential(ox.Linear(8, 4), ox.ReLU(), ox.Linear(4, 2))
optimizer = ox.optim.Adam(model.parameters(), lr=1e-3)
train_loader = ox.DataLoader(
    ox.TensorDataset(ox.Tensor.randn([64, 8]), ox.from_numpy(np.zeros(64, dtype=np.int64))),
    batch_size=16,
)
val_loader = ox.DataLoader(
    ox.TensorDataset(ox.Tensor.randn([32, 8]), ox.from_numpy(np.zeros(32, dtype=np.int64))),
    batch_size=16,
)

trainer = ox.Trainer(
    model=model,
    optimizer=optimizer,
    criterion=ox.cross_entropy,
    callbacks=[
        ox.callbacks.EarlyStopping(patience=5, monitor="val_loss"),
        ox.callbacks.ModelCheckpoint("best_model.ox", monitor="val_loss", save_best_only=True),
    ],
    max_grad_norm=1.0,
    debug=False,         # Set debug=True for deep tensor audits (NaN/Inf, grad norms)
)

# Execute training loop with validation
history = trainer.fit(train_loader, val_loader=val_loader, epochs=2)
```

---

## 7. Inference, KV Caches & Attention Acceleration

See [INFERENCE_AND_LLM.md](INFERENCE_AND_LLM.md) for full architectural documentation.

- `ox.PagedKVCache`: Block-based memory manager for PagedAttention.
- `ox.StandardKVCache`: Sliding-window and append-only multi-head KV cache.
- `ox.SpeculativeDraftDecoder`: Draft-and-verify speculative autoregressive acceleration.
- `ox.flash_attention_forward`: Exact FlashAttention-2 online softmax tiling.
- `ox.TiledAttention`: FlashAttention-2 `Module` interface.
- `ox.BPETokenizer`: Subword tokenizer with frequency-aggregated merge training.
- `ox.StreamingSinkCache`: Attention sinks + recent window eviction.
- `ox.ScoredEvictionCache`: H2O / TOVA / SnapKV score-guided eviction.
- `ox.MLAAttention`: Low-rank KV attention with decoupled RoPE.
- `ox.NativeSparseAttention`: Compressed + top-k + sliding gated attention.

---

## 8. Low-Bit Quantization & PEFT

- **Ultra-Low-Bit K-Quants (`ox.kquants`)**: `Q2_K`, `Q3_K`, `Q5_K`, `Q6_K` block quantization.
- **Ternary Inference (`ox.BitLinear`)**: 1.58-bit weights $\{-1, 0, 1\}$ with 8-bit dynamic activation quantization.
- **DoRA (`ox.DoRALinear`)**: Weight-Decomposed Low-Rank Adaptation with `.merge()`.
- **LoRA+ (`ox.LoRAPlusLinear`)**: Asymmetric low-rank learning rate dynamics.
- **MXFP4 (`ox.quantize_mxfp4`)**: Exact OCP microscaling, 4.25 bits/param.

---

## 9. Native Cloud Storage & Compute (`ox.cloud`)

See [CLOUD_AND_DEPLOYMENT.md](CLOUD_AND_DEPLOYMENT.md) for full cloud guides.

- Direct remote URI streaming: `ox.save_file(tensors, "s3://bucket/model.ox")`, `ox.load_file("gs://bucket/model.ox")`.
- `ox.cloud.aws`: `S3Client`, `is_sagemaker()`, `SageMakerCallback`.
- `ox.cloud.gcp`: `GCSClient`, `is_vertex_ai()`, `VertexAICallback`, `is_tpu_available()`.
- Unified detection: `ox.cloud.is_cloud_environment()`, `ox.cloud.get_cloud_provider()`.

---

## 10. Logging, Tensor Audits & Debug Mode (`ox.logging`)

Oxideon includes a thread-safe unified logging engine with global and contextual debug controls:

```python
import oxideon as ox

# 1. Global Debug Mode
ox.set_debug(True)
print("Is debug enabled?", ox.is_debug())

# 2. Contextual Debug Scope
model = ox.Sequential(ox.Linear(4, 2))
x = ox.Tensor.randn([3, 4])
with ox.debug_mode(True):
    out = model(x)

# 3. Dedicated Oxideon Logger
logger = ox.get_logger("inference")
logger.info("Initializing KV cache blocks")
logger.debug("Allocated physical block ID: 12")

# 4. In-Flight Tensor Health Verification
t = ox.Tensor.randn([10, 10])
ox.check_nan(t, name="activation_layer_1")
ox.debug_tensor(t, name="latent_vectors")
```

---

## 11. Scientific AI, Graph & Multimodal Subsystems

See [SCIENTIFIC_AI_MODULES.md](SCIENTIFIC_AI_MODULES.md) for full scientific documentation.

- **Causal Discovery (`ox.causal_discovery`)**: `notears_linear`, `correlation_skeleton`, `is_dag`, `topological_sort`.
- **Structural Biology (`ox.protein`)**: `compute_contact_map`, `EGNNCoordinateUpdate`.
- **Graph Kernels (`ox.graph_kernels`)**: `weisfeiler_lehman_kernel`, `random_walk_kernel`.
- **Sparse Matrix (`ox.sparse_matrix`)**: `CSRMatrix` with BLAS `spmm` and `spspmm`.
- **Survival Analysis (`ox.survival`)**: `CoxProportionalHazardsLoss`, `concordance_index`, `KaplanMeierEstimator`.
- **Preference Alignment (`ox.dpo`)**: `DPOLoss`, `ORPOLoss`, `KTOLoss`.
- **Multimodal Fusion (`ox.multimodal_fusion`)**: `CrossAttentionFusion`, `GatedMultimodalFusion`.
- **Clustering (`ox.clustering`)**: `KMeans`, `PCA`, `pairwise_distances`.
- **Audio DSP (`ox.audio`, `ox.audio_synth`)**: `Spectrogram`, `MelSpectrogram`, `MFCC`, `BiquadFilter`, `EnvelopeADSR`.
- **3D Geometry (`ox.geometry`)**: `PointNetEncoder`, `chamfer_distance`, `farthest_point_sampling`.

---

## 12. Millennial Singularity Milestone Engines

Provides access to all 100 decadal milestone synthesis engines:

| Engine Class | Engine no. | Output Type |
|---|---|---|
| `GrandCentennialSingularityEngineX` | 100 | `GrandCentennialSingularityReportX` |
| `GrandBicentennialSingularityEngineXX` | 200 | `float` invariant |
| `GrandOmniMillennialEngineC` | 1000 | `float` Millennial Invariant |

```python
import oxideon as ox

engine = ox.GrandOmniMillennialEngineC(100)
invariant = engine.execute_millennial_synthesis()
```

---

## 13. Frontier AI & Scientific Modules (52 Features, 10 Flagship Systems)

Oxideon natively integrates 52 high-level features across 10 flagship systems and 7 domain modules:

### 10 Flagship Systems

1. **Radix Prefix Cache (`ox.RadixPrefixCache`, `ox.RadixNode`)**
   - Radix tree prefix matching for KV cache reuse across shared prompts.
   - Methods: `cache.insert(token_ids, block_ids)`, `cache.match_prefix(token_ids) -> (matched_tokens, matched_blocks)`.

2. **Monte Carlo Tree Search (`ox.MCTS`, `ox.MCTSNode`, `ox.MCTSConfig`)**
   - UCT-guided chain-of-thought exploration for reasoning LLMs.
   - Methods: `mcts.search(initial_state, policy_fn, value_fn, is_terminal_fn) -> root`, `mcts.best_action(root)`.

3. **Graph Transformer (`ox.GraphTransformer`, `ox.GraphTransformerLayer`)**
   - Full all-pair attention with edge feature bias and spectral Laplacian eigenvector positional encodings.
   - Forward: `model(node_features, laplacian_pe=None, edge_attr=None) -> logits`.

4. **BitNet b1.58 Ternary Linear Layer (`ox.BitLinear158b`, `ox.WeightQuantizerSTE`, `ox.AbsmaxQuantizer`)**
   - 1.58-bit ternary $\{-1, 0, +1\}$ linear layer with Straight-Through Estimator and INT8 activation scaling.
   - Forward: `bit_linear(x) -> out`.

5. **Frontier Diffusion Schedulers (`ox.EulerDiscreteScheduler`, `ox.DPMSolverMultistepScheduler`, `ox.DDIMScheduler`)**
   - 2nd-order DPM-Solver++ multistep ODE solver, Euler discrete scheduler, and deterministic DDIM reverse sampling.
   - Methods: `scheduler.set_timesteps(num_steps)`, `scheduler.step(model_output, timestep, sample) -> prev_sample`.

6. **Audio Neural Codec & Residual VQ (`ox.AudioResidualVQ`, `ox.VectorQuantizer`, `ox.AudioEncoder1D`, `ox.AudioDecoder1D`)**
   - Multi-codebook residual vector quantization with commitment loss and strided 1D conv encoder/decoder.
   - Forward: `rvq(z) -> (quantized, stage_indices, total_loss)`.

7. **Adaptive Conformal Prediction Sets (`ox.AdaptivePredictionSets`, `ox.ConformalizedQuantileRegression`, `ox.MondrianConformalGroup`)**
   - Adaptive non-conformity prediction sets (APS), conformalized quantile regression (CQR) intervals, and group coverage.
   - Methods: `aps.calibrate(probs, labels)`, `aps.predict(probs) -> List[List[int]]`.

8. **Synaptic Intelligence Continual Learning (`ox.PathIntegralRegularizer`, `ox.MemoryReplayBuffer`)**
   - Path-integral synaptic importance tracking $\Omega$ and reservoir sampling replay buffer.
   - Methods: `si.on_step()`, `si.consolidate_task()`, `si.penalty_loss() -> Tensor`.

9. **Multimodal Vision-Language Connector (`ox.VisionLanguageConnector`, `ox.PerceiverResampler`, `ox.VLMProjection`)**
   - Flamingo-style Perceiver Resampler with learnable latent cross-attention and LLaVA-style MLP projection.
   - Forward: `connector(visual_features, text_embeddings) -> multi_modal_tokens`.

10. **Capacity-Constrained Sparse MoE (`ox.CapacityConstrainedMoE`, `ox.TopKRouterWithAuxLoss`)**
    - Switch Transformer / DeepSeek-V3 load-balanced Top-2 router with auxiliary balancing loss and capacity factor limits.
    - Forward: `moe(x) -> (output_tokens, aux_loss)`.

### 42 Extended Modules & Solvers

* **Modern Attention (`ox.attention_ext`)**:
  - `AlibiPositionalBias(num_heads)`: Sloped positional biases for length extrapolation.
  - `SpatiotemporalRoPE(dim)`: 3D spatiotemporal rotary embeddings.
  - `MultiQueryAttention(hidden_dim, num_heads)`: Shared KV head multi-query attention.
  - `SlidingWindowAttention(hidden_dim, num_heads, window_size)`: Local-global window attention.
  - `DifferentialAttention(hidden_dim, num_heads)`: Dual-softmax noise-canceling attention.
  - `LogitLens.inspect_layer(hidden_state, lm_head_weight, top_k)`: Unembed intermediate representations.

* **Frontier Optimizers (`ox.optim_ext`)**:
  - `ShampooOptimizer`: Matrix preconditioned 2nd-order optimizer with Kronecker Gram factor statistics.
  - `AdEMAMix`: Dual fast/slow EMA optimizer (Apple 2024).
  - `ProdigyOptimizer`: Learning-rate-free adaptive D-adaptation optimizer.
  - `SophiaOptimizer`: Second-order stochastic curvature-clipped optimizer.
  - `ScheduleFreeSGD`: Schedule-free optimizer without learning rate decay.
  - `LookaheadOptimizer`: $k$-step slow-fast weight interpolation wrapper.
  - `SGDRScheduler`: Cosine annealing with warm restarts.

* **Loss Functions (`ox.losses_ext`)**:
  - `DiceLoss`: Sørensen-Dice loss for semantic segmentation.
  - `AdaptiveFocalLoss`: Modulated focal loss down-weighting well-classified examples.
  - `LabelSmoothingCrossEntropy`: Regularized cross-entropy with uniform smoothing.
  - `TripletMarginWithDistanceLoss`: Metric learning triplet distance loss.
  - `RankNetLoss`: Pairwise ranking cross-entropy loss.
  - `InfoNCELoss`: Contrastive representation temperature loss.

* **Graph Neural Networks (`ox.gnn_ext`)**:
  - `GatedGraphConv`: Gated graph sequence neural networks with GRU cell updates.
  - `GraphSAGE`: Inductive neighborhood representation learning (mean, sum, max).
  - `GINConv`: Graph Isomorphism Network with trainable $\epsilon$ parameter and MLP.
  - `LaplacianEigenmaps.fit_transform(adj, n_components)`: Spectral graph embedding.
  - `AdamicAdarIndex.score_matrix(adj)`: Link prediction metric.
  - `JaccardSimilarityCoeff.score_matrix(adj)`: Neighborhood overlap similarity matrix.

* **Numerical ODE & PDE Solvers (`ox.ode_pde_solvers`)**:
  - `RungeKutta4.integrate(f, y0, t_span, steps)`: 4th-order explicit RK4.
  - `DormandPrince54.integrate(f, y0, t_span, rtol, atol)`: Adaptive step size DOPRI5.
  - `SymplecticEuler.integrate(grad_v, grad_t, q0, p0, t_span)`: Symplectic Hamiltonian integrator.
  - `Burgers1DEquation.solve(u0, dx, dt, nu, num_steps)`: Viscous 1D Burgers' PDE.
  - `Wave1DEquation.solve(u0, v0, c, dx, dt, num_steps)`: 1D acoustic wave FDTD solver.
  - `Diffusion1DEquation.solve(u0, alpha, dx, dt, num_steps)`: 1D heat diffusion solver.

* **Quantization & Compression (`ox.quant_compression`)**:
  - `SmoothQuant.smooth_layer(weight, act_sample, alpha)`: Activation outlier smoothing for INT8.
  - `GPTQQuantizer.quantize(weight, hessian, num_bits)`: Second-order inverse Hessian column quantization.
  - `SparseMagnitudePruner.prune_unstructured`, `prune_n_m_structured`: Fine-grained 2:4 sparsity.
  - `VectorQuantizedVAE(in_dim, hidden_dim, num_embeddings)`: Discrete codebook autoencoder.
  - `HQQQuantizer.quantize`, `dequantize`: Half-Quadratic calibration-free low-bit quantization.

* **Signal Processing & Metrics (`ox.signal_metrics`)**:
  - `MelFilterbank.create(sr, n_fft, n_mels)`: Triangular linear-to-Mel filterbank matrix.
  - `FastDTW.distance(s1, s2, radius)`: Dynamic Time Warping with Sakoe-Chiba band.
  - `ExponentialMovingAverage(params, decay)`: Polyak-Ruppert shadow parameter averaging.
  - `Autocorrelation1D.correlate`, `detect_pitch_period`: 1D autocorrelation and pitch detection.
  - `PeakSignalToNoiseRatio.compute(y_true, y_pred, data_range)`: PSNR in decibels (dB).
  - `StructuralSimilarityIndex.compute(x, y, data_range)`: SSIM comparing luminance, contrast, and structure.

---

## 14. Evaluation Harness (`ox.evals`)

- **Metrics (`ox.evals`)**: `accuracy`, `exact_match`, `token_f1`, `pass_at_k`,
  `math_equal`, `rouge_l`, `corpus_bleu`, `perplexity_from_nll`.
- **Tasks (`ox.evals`)**: `MultipleChoiceTask` (loglikelihood or generate),
  `MathTask`, sandboxed `CodeTask` (pass@k), synthetic `NeedleTask`.
- **Runner (`ox.evals`)**: `run_suite` aggregates tasks into `EvalReport`
  (dict/JSON/text); `demo_choices`/`demo_math`/`demo_code` fixtures included.

---

## 15. Speech, Vision Toolkits & 3D (`ox.speech`, `ox.vision`, `ox.nerf`, `ox.splat`)

- **Speech (`ox.speech`)**: `energy_vad` (hangover + min-duration),
  `SpeakerDiarizer` (caller embedder + greedy clustering),
  `spectral_subtraction`, `wiener_filter`, `ideal_ratio_mask`,
  `audio_fingerprint`/`fingerprint_matches` (constellation hashes).
- **Detection & metrics (`ox.vision.ops`)**: `yolo_decode`,
  `mean_average_precision` (101-point, multi-IoU), `mask_iou`,
  `panoptic_quality`, `oks`, `pck`, `heatmap_to_keypoints`.
- **Tracking & geometry (`ox.vision.ops`)**: `linear_sum_assignment`,
  `KalmanBoxTracker`, `IoUTracker`, `ByteTracker`, `lucas_kanade`,
  `block_match_disparity`.
- **OCR (`ox.vision.ops`)**: `ctc_greedy_decode`, `ctc_beam_decode`,
  `edit_distance`, `cer`, `wer`.
- **Segmentation (`ox.vision.models`)**: `DeepLabHead` with ASPP and
  `DilatedConv2d` (numpy reference, no autograd).
- **NeRF (`ox.nerf`)**: `positional_encoding`, `get_rays`,
  `stratified_samples`, differentiable `volume_render`, `sample_pdf`,
  `NeRFField`, hierarchical `render_rays`.
- **Splatting (`ox.splat`)**: `quat_to_matrix`, `project_gaussians`
  (EWA + low-pass), `rasterize`, `GaussianSplatScene`.

---

## 16. High-Performance Inference & Serving Systems

### Radix Tree Prefix KV-Cache (`ox.prefix_cache`)
- `RadixPrefixCache(block_size=16)`: Manages token prefix matching with $O(1)$ subtree block reuse.
  - `match_prefix(token_ids)`: Finds longest cached token prefix; returns `(matched_token_ids, matched_block_ids)`.
  - `insert(token_ids, block_ids)`: Inserts sequences into the radix tree with automatic node splitting.
  - `evict_lru(num_blocks)`: Evicts least-recently-accessed unreferenced leaf nodes in LRU order.

### FlashAttention-2 Exact Forward Tiling (`ox.flash_attn`)
- `flash_attention_forward(query, key, value, block_size_r=64, block_size_c=64, scale=None, causal=False)`:
  Exact online softmax tiling (Dao, 2023) keeping attention statistics in registers without materializing $S 	imes S$ scores in global memory.
- `TiledAttention(embed_dim, num_heads, block_size=64, causal=False)`:
  Multi-head attention module powered by FlashAttention-2 tiling.

### Sparse Compressed Row Matrix (`ox.sparse_matrix`)
- `CSRMatrix(values, col_indices, row_ptrs, shape)`: High-performance CSR representation.
  - `from_dense(matrix, eps=1e-7)`: Converts 2D dense tensor/array into CSRMatrix.
  - `to_dense()`: Materializes dense 2D tensor.
  - `spmm(x)`: Vectorized SpMV / SpMM ($Y = A X$) with BLAS-accelerated dot product and `bincount` scatter.
  - `spspmm(other)`: Sparse-sparse matrix multiplication ($C = A B$) with sorted accumulation.

### Sparse Long-Context Attention (`ox.sparse_attn`)
- `build_sliding_window_mask(seq_len, window_radius)`: Binary banded attention mask.
- `build_bigbird_mask(seq_len, num_global=2, window_radius=3, num_random=2, seed=None)`: BigBird $O(L)$ mask.
- `SparseMultiheadAttention(embed_dim, num_heads=4, window_radius=4, num_global=2, num_random=2)`: Sub-quadratic sparse transformer attention layer.

---

## 17. Advanced Quantization & Architecture Optimization

### Simulated Low-Bit Precision Formats (`ox.quantization_advanced`)
- `quantize_fp8_e4m3(x)`, `dequantize_fp8_e4m3(x_q, scale)`: FP8 E4M3 (1S 4E 3M) with dynamic scaling.
- `quantize_fp8_e5m2(x)`, `dequantize_fp8_e5m2(x_q, scale)`: FP8 E5M2 (1S 5E 2M) matching IEEE FP16 range.
- `WeightOnlyInt4Quantizer`: Block-wise 4-bit integer quantization with group scales and zero-points.
  - `quantize(weight, group_size=32)`: Returns `(q_weight, scales, zeros)`.
  - `dequantize(q_weight, scales, zeros, group_size=32)`: Reconstructs unquantized weights.
- `QATLinear(in_features, out_features, num_bits=8)`: Linear layer with Straight-Through Estimator (STE).

### Architecture Optimization & Calibration Observers (`ox.ao.quantization`)
- `MinMaxObserver(dtype="qint8", qscheme="per_tensor_affine")`: Computes scale/zero-point from observed min/max.
- `MovingAverageMinMaxObserver(averaging_constant=0.01)`: Running min/max tracking via exponential moving average.
- `FakeQuantize(observer=None, quant_min=-128, quant_max=127, dtype="qint8")`: QAT simulation module.
- `prepare(model, qconfig=None, inplace=False)`: Prepares model by inserting observers or fake-quant modules.
- `convert(model, inplace=False)`: Freezes calibration parameters and converts to quantized inference representation.

---

## 18. Canonical Deep Architectures & State Space Models

### Canonical Vision & Language Models (`ox.models`)
- `ResNet18(in_channels=3, num_classes=10, hidden_dim=32)`: ResNet architecture with residual skip connections.
- `VisionTransformer(img_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=64, depth=2, num_heads=4)`:
  ViT classifier with vectorized patch extraction, class token prepending, and transformer encoder layers.
- `MiniGPT(vocab_size=1000, max_seq_len=128, embed_dim=64, num_layers=2, num_heads=4)`:
  Autoregressive causal language model with text generation via `.generate(prompt_ids, max_new_tokens, temperature)`.

### Selective State Space Models (`ox.state_space`)
- `SelectiveScan.forward(x, delta, A, B, C, D_param)`: Core input-dependent selective discretization recurrence.
- `MambaBlock(d_model, d_state=16, d_conv=4, expand=2)`:
  Mamba selective state-space block combining gated linear projection, depthwise 1D convolution, and selective scan in $O(L)$ linear time.

---

## 19. Signal Processing, 3D Geometry & Video Analytics

### Digital Signal Processing (`ox.signal`)
- `windows`: Analytical window functions (`hann`, `hamming`, `blackman`, `bartlett`, `kaiser`, `gaussian`, `exponential`, `chebyshev`).
- `stft(input, n_fft, hop_length, win_length, window)`: Short-Time Fourier Transform.
- `istft(complex_spec, n_fft, hop_length, win_length, window)`: Inverse Short-Time Fourier Transform with overlap-add.
- `hilbert(input, n=None, axis=-1)`: Analytic signal computation via Hilbert transform.
- `correlate(input, other, mode="valid")`: 1D cross-correlation along target axis.
- `convolve(input, other, mode="full")`: 1D convolution along target axis.
- `resample_poly(input, up, down, axis=-1)`: Rational polyphase resampling in the frequency domain.

### 3D Point Cloud Processing & Geometry (`ox.geometry`)
- `PointNetEncoder(in_channels=3, hidden_dim=64, global_dim=128)`: Symmetric max-pooling point cloud feature encoder.
- `farthest_point_sampling(points, num_samples)`: Uniform point cloud downsampling via iterative FPS.
- `chamfer_distance(p1, p2)`: Symmetric squared Chamfer Distance computed via BLAS GEMM expansion.
- `icp(source, target, max_iters=50, tol=1e-6)`: Iterative Closest Point registration with Umeyama step.
- `voxel_downsample(points, voxel_size)`: Occupied voxel grid centroid downsampling.
- `estimate_normals(points, k=10, viewpoint=None)`: PCA normal estimation from nearest neighbors.
- `earth_mover_distance(p1, p2)`: Hungarian minimum-cost pairing distance between point sets.
- `hausdorff_distance(p1, p2)`: Symmetric Hausdorff distance.

### Video Analytics (`ox.video`)
- `MOGBackgroundSubtractor(n_modes=3, learning_rate=0.01)`: Mixture-of-Gaussians foreground segmentation.
- `detect_shot_boundaries(frames, thresh=0.5, bins=32)`: Histogram shot-cut detection with flash veto.
- `interpolate_frames(prev, next, block=16, search=7)`: Block-matching motion-compensated mid-frame (t=0.5).
- `estimate_translation(ref, mov)`: Phase-correlation subpixel translation estimation.
- `stabilize_frames(frames, radius=5)`: Translation-only camera path stabilization with edge-cropping.

---

## 20. Frontier Optimizers, Loss Functions & Synthetic Data

### Frontier Optimizers & Schedulers (`ox.optim_ext`)
- `ShampooOptimizer(params, lr=1e-3, beta2=0.999, epsilon=1e-4)`: 2nd-order matrix preconditioned optimizer.
- `AdEMAMix(params, lr=1e-3, beta1=0.9, beta2=0.999, beta3=0.9999, alpha=5.0)`: Fast/slow dual exponential moving average.
- `SophiaOptimizer(params, lr=1e-3, beta1=0.96, beta2=0.99, rho=0.04)`: Second-order curvature-clipped optimizer.
- `ScheduleFreeSGD(params, lr=0.01, momentum=0.9)`: Schedule-free SGD without explicit learning rate decay.
- `LookaheadOptimizer(base_optimizer, k=5, alpha=0.5)`: Slow-fast weight interpolation wrapper.
- `SGDRScheduler(optimizer, t_0=10, t_mult=2, eta_min=0.0)`: Cosine annealing with warm restarts.

### Machine Learning Loss Functions (`ox.losses_ext`)
- `DiceLoss(smooth=1e-6)`: Multi-class Sørensen-Dice loss for image segmentation.
- `AdaptiveFocalLoss(gamma=2.0, alpha=0.25)`: Focal loss for imbalanced classification.
- `LabelSmoothingCrossEntropy(smoothing=0.1)`: Cross-entropy with uniform label smoothing.
- `TripletMarginWithDistanceLoss(margin=1.0)`: Metric learning triplet loss.
- `RankNetLoss(sigma=1.0)`: Pairwise ranking loss for information retrieval.
- `InfoNCELoss(temperature=0.07)`: Self-supervised contrastive representation loss.

### Tabular Synthetic Data Generation (`ox.synthetic_data`)
- `GaussianCopulaSynthesizer()`: Generates tabular data preserving marginal distributions and Gaussian copula rank correlation.
- `CTGANGenerator(latent_dim=16, output_dim=10, hidden_dim=32)`: Conditional tabular GAN generator network.
- `DifferentialPrivacySynthesizer(epsilon=1.0, num_bins=10)`: Differentially private noisy histogram synthesizer under Laplace mechanism.

### Evidential Deep Learning & Uncertainty (`ox.uncertainty_quant`)
- `DirichletLoss(kl_weight=0.01)`: Places a Dirichlet prior over class predictions (Sensoy et al., NeurIPS 2018).
- `decompose_uncertainty(ensemble_probs)`: Decomposes ensemble variance into total entropy, aleatoric risk, and epistemic mutual information.
- `brier_score(probs, targets_onehot)`: Multi-class Brier calibration score.

### Model Interpretability (`ox.interpret`)
- `TCAV(cav_vector)`: Testing with Concept Activation Vectors (Kim et al., ICML 2018).
  - `train_cav(concept_activations, random_activations)`: Fits linear probe separating concept features.
  - `tcav_score(activation_grads)`: Computes fraction of inputs with positive directional derivative.
- `IntegratedHessians(steps=20)`: Integrated Hessians for pairwise non-linear feature interaction attributions.
- `difference_from_reference_multipliers(model_fn, x, reference)`: DeepLIFT-style feature attributions.

### High-Level Functional Transforms (`ox.func`)
- `vmap(func, in_dims=0, out_dims=0)`: Vectorizing map that batches execution across tensor dimensions.
- `grad(func, argnums=0)`: Computes function gradients w.r.t specified arguments.
- `jacrev(func, argnums=0)`, `jacfwd`: Reverse-mode and forward-mode Jacobian matrices.
- `hessian(func, argnums=0)`: Second-order Hessian matrix of scalar functions.
