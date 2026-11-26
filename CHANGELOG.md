# Changelog

## 0.1.0 - 2026-11-23

### Added
- **Workflow, tree-mapping, parameterization, data, profiling and vision utilities:** nested container transformations (`ox.treemap`: `tree_flatten`, `tree_unflatten`, `tree_map`, `tree_map_only`, `tree_all`, `tree_any`, `broadcast_tree_prefix`, `flatten_nested`, `assert_same_structure`, `pack_sequence_as`, `map_structure`, custom node registration); neural parameterizations, fusion and pruning (`ox.nn.utils`: `weight_norm`, `spectral_norm`, `bind_reparam` / `unbind_reparam` / `cache_reparam`, `fuse_conv_batchnorm`, `fuse_linear_batchnorm`, `ln_structured_prune`, `mask_prune`, `identity_prune`, `fast_init`, `apply_with_weights`, `total_norm`, `freeze_module_types`, `freeze_by_prefix`); functional `Dataset` combinators (`.map()`, `.filter()`, `.cache()`, `.repeat()`, `.take()`, `.skip()`, `.window()`), `IterableDataset` streaming in `DataLoader`, `padded_collate`, `to_tensors`, `stratified_split`, `BalancedClassSampler`, `LengthBucketSampler`, `InfiniteDataLoader` / `cycle_batches`, and `padding_side` on `pad_sequence` / `unpad_sequence`; multimodal & structured loaders (`image_folder_dataset`, `text_folder_dataset`, `audio_folder_dataset`, `csv_dataset`, `jsonl_dataset`, `sliding_window_dataset`, `split_dataset`, pure-Python `read_wav_tensor` / `save_wav_tensor`, `VocabularyIndex` / `build_vocabulary`, `one_hot_encode`, `axis_normalize`, `pad_sequences`, `compute_class_weight`, `compute_sample_weight`, `pack_batch` / `unpack_batch`); checkpointing & config registry (`save_sharded` / `load_sharded`, `CheckpointVault`, `ShadowWeightEMA`, `fetch_artifact` / `verify_checksum`, `SerializationScope`, `register_serializable`, `serialize_config` / `deserialize_config`); profiling, graph rendering & numerical assertions (`FlopProfiler` / `count_flops`, `MemoryFootprint` / `estimate_memory_footprint`, `MicroTimer` / `TimingReport` / `TimingComparison`, `collect_runtime_info`, `HookHandle` / `ephemeral_hook`, `model_to_dot`, `model_to_mermaid`, `chrome_trace_handler`, `ProgressBar`, `SummaryWriter` histograms/images/hparams/graphs, `gradgradcheck`, `assert_finite`, `assert_shapes`); and `ox.vision.utils` (`image_grid`, `save_image`, `draw_bounding_boxes`, `draw_segmentation_masks`, `draw_keypoints`).
- `oxideon.tensor_utils`: 71 PyTorch-named tensor utilities, all exported at the top level and covered by `tests/test_tensor_utils.py`
  (compared against NumPy and PyTorch): `ravel`, `movedim` / `moveaxis`, `swapaxes` / `swapdims`, `tensor_split`, `hsplit` / `vsplit` /
  `dsplit`, `broadcast_shapes` / `broadcast_to` / `broadcast_tensors`, `combinations`, `msort`, `lerp`, `float_power`, `addcmul`, `addcdiv`,
  `signbit`, `gcd`, `lcm`, `ldexp`, `frexp`, `nextafter`, `sgn`, `positive`, `negative`, `trunc` / `fix`, `frac`, `square`, `lgamma`,
  `mvlgamma`, `logspace`, `vander`, `isnan` / `isinf` / `isfinite` / `isposinf` / `isneginf` / `isreal`, `is_tensor`, `numel`, `result_type`,
  `logsumexp`, `cumulative_logsumexp`, `var_mean`, `std_mean`, `aminmax`, `count_nonzero`, `renorm`, `scatter_reduce`, `histogramdd`,
  `chain_matmul`, `matrix_power`, `matrix_exp`, `pinverse`, `cholesky_solve`, `lu`, `det`, `inverse`, `slogdet`, `ger`, the logical
  (`and` / `or` / `xor` / `not`) and bitwise operators. See [docs/PYTHON_API.md](docs/PYTHON_API.md) section 21.
- `oxideon.tensor_methods`: 149 PyTorch-style `Tensor` methods, every one checked against PyTorch: method forms (`x.logsumexp(1)`,
  `x.amax(0)`, `x.movedim(0, -1)`, `x.unique()`, `x.trace()`, ...), comparison and division aliases (`x.gt(y)`, `x.true_divide(y)`,
  `x.divide(y, rounding_mode="floor")`, `x.dist(y, p)`, ...), and 75 in-place forms (`x.exp_()`, `x.clamp_min_(0)`, `x.lerp_(y, w)`,
  `x.addcdiv_(a, b, value=)`, ...) that refuse shape changes, float results in integer tensors, tensors that require grad outside
  `no_grad`, and views (`tests/test_tensor_methods.py`).
- `oxideon.distributions`: 19 distributions (`Chi2`, `InverseGamma`, `Pareto`, `Kumaraswamy`, `FisherSnedecor`, `GeneralizedPareto`,
  `NegativeBinomial`, `OneHotCategorical`, `OneHotCategoricalStraightThrough`, `LogitRelaxedBernoulli`, `RelaxedBernoulli`,
  `ExpRelaxedCategorical`, `RelaxedOneHotCategorical`, `LogisticNormal`, `MixtureSameFamily`, `VonMises`, `Wishart`, `LKJCholesky`,
  `ExponentialFamily`) and 18 transforms (`Exp`, `Affine`, `Sigmoid`, `Tanh`, `Softplus`, `Abs`, `Power`, `Softmax`, `StickBreaking`,
  `Compose`, `Independent`, `Reshape`, `Cat`, `Stack`, `CumulativeDistribution`, `CorrCholesky`, `LowerCholesky`, `PositiveDefinite`);
  `TransformedDistribution` gains `log_prob` and `cdf`; `mean` / `variance` / `stddev` on every distribution with a closed form. All
  checked against `torch.distributions` (`tests/test_distributions_ext.py`).
- `oxideon.diagnostics`: `detect_anomaly(model)` (the layer that first produces NaN / Inf, forward or backward), `precision_audit`
  (float32 error added by each layer, against a float64 run), `estimate_memory`, `lr_find` (learning-rate range test),
  `check_determinism`, `diff_checkpoints`, `TrainingMonitor` (gradient norms, update ratios, activation statistics, warnings),
  `hessian_top_eigenvalue` and `hessian_trace` (sharpness). `tests/test_diagnostics.py`.
- More PyTorch parity, each checked against PyTorch (`tests/test_torch_parity_extra.py`): max pooling with `dilation`, `ceil_mode`
  and `return_indices` (functional and `nn`, 1-3 D, plus the `*_with_indices` forms; one gather-based implementation replaces the
  Python window loops: a Rust kernel, parallel over planes, finds the maxima and the values are gathered, so they stay
  differentiable), `lp_pool3d` / `LPPool3d`, `multi_margin_loss`,
  `multilabel_margin_loss`, `multilabel_soft_margin_loss`, `triplet_margin_with_distance_loss`, `linear_cross_entropy` /
  `LinearCrossEntropyLoss`, in-place functional activations, `upsample*`; `linalg.cholesky_ex`, `inv_ex`, `solve_ex`,
  `lu_factor_ex`, `ldl_factor(_ex)`, `ldl_solve` (new Rust module `oxideon-core/src/lapack.rs`: getrf / getrs / potrf /
  Bunch-Kaufman sytrf / sytrs with LAPACK's info codes), `matrix_sqrth`, `polar`, `matmul`; `special.hermite_polynomial_he`,
  `shifted_chebyshev_polynomial_t/u/v/w`, `modified_bessel_i0/i1`; `fft.hfft2/hfftn/ihfft2/ihfftn`; `nn.LazyConv1d/3d`,
  `LazyConvTranspose1d/2d/3d`, `CrossMapLRN2d`, `Buffer`, `UninitializedParameter`, `UninitializedBuffer`, `RNNCellBase`,
  `NLLLoss2d`, `Container`; `optim.SparseAdam` (its step runs in Rust); `data.DistributedSampler`, `data.StackDataset`.

### Speed (accuracy unchanged unless noted)
- Balanced `gelu` 150 -> 135 us per 1M elements (PyTorch 111 us): the tail exponent `poly - z^2` is never positive, so it uses an `exp` without the
  upper clamp and the NaN restore (a NaN input still gives NaN through `x * cdf`), the 0.5 of the cdf folds into the quotient factor, and
  two vectors run per iteration. Fastest `gelu` 113 -> 101 us. The measured errors are identical.
- Balanced conv2d: the TwoSum error of each channel chain starts the next chain instead of living in a register set of its own (that
  spilled 24 vectors at every chain boundary), and from 32 input channels the chains are 8 deep. 4x64x56x56 conv2d 1690 -> 1340-1470 us
  (PyTorch 1529 us); 8x16x64x64 665 -> 640-660 us (PyTorch 512 us). Accuracy vs PyTorch (worst / mean): 16 channels 3.05x / 1.55x -> 2.69x /
  1.54x, 64 channels 4.64x / 2.99x -> 4.58x / 2.67x.

### Changed
- Compute moved from NumPy to Rust: the Python optimizers (`RMSprop`, `Adagrad`, `Adadelta`, `Adamax`, `NAdam`, `RAdam`, `ASGD`,
  `Lion`, `Lamb`, `Adafactor`, `Rprop`, `SparseAdam`) run their updates in `oxideon-core/src/optim_kernels.rs` and keep their
  state as tensors (NumPy state from older checkpoints is converted on load), `LBFGS` does its two-loop recursion with tensor
  operations; `linalg.qr` (any shape, both modes), `svd(full_matrices=True)`, `lu`, `norm`, `pinv`, `eig`, `eigvals` no longer call
  NumPy; `tensor_utils.gcd` / `lcm` / `frexp` / `ldexp` / `nextafter` / `signbit` / `isposinf` / `isneginf` / `histogramdd` /
  `logspace`, all orthogonal polynomials (`special.*_polynomial_*`, now also with a tensor degree), the Hermitian FFTs, and the
  sampling and densities of the new distributions use Rust kernels (`numeric_kernels.rs`, `lapack.rs`, samplers in
  `distributions_ext.rs`); distribution samples now come from the oxideon generator, so `oxideon.manual_seed` makes them
  reproducible.

### Fixes
- Bug hunt against PyTorch (regression tests in `tests/test_bug_hunt.py`): method-form binary ops accept Python scalars; `//`, `%`, `<<`, `>>`, unary `+`; integer `/` and `div` give floats; `x += y` writes in place (out-of-place when autograd is involved); `nan_to_num` keeps 0-d shape and gradient; `angle(-0.0)`; `x.to(y.dtype)` / `.to("float64")` / `.to(np.float64)`.
- `max`/`min` with a dim return `(values, indices)`; `sort`/`topk`/`median`/`mode`/`kthvalue`/`cummax` return named tuples; `max(other)` is the elementwise maximum.
- Reductions over several dims (`sum`, `mean`, `var`, `std`, `amax`, `amin`, `logsumexp`, `nansum`, `nanmean`), `var`/`std` `correction=`, differentiable `amax`/`amin` (ties share the gradient), full `prod`/`all`/`any`/`count_nonzero` return 0-d tensors.
- Differentiable `repeat`, `tile`, `repeat_interleave`, `hstack`/`vstack`/`dstack`, `diag_embed`, `take`, `index_add`, `index_copy`, `meshgrid`, `cumulative_trapezoid`; `einsum` is the native differentiable kernel (keeps dtype); `where` with a Python scalar; `put`, `scatter(value)`; varargs `permute`/`expand`/`flip`, `T`/`mT`/`H`/`mH`; `triu_indices`/`tril_indices` return one 2xN tensor.
- Advanced indexing (`x[[0, 2]]`, `x[[0, 1], [1, 2]]`, `x[:, idx]`, masks) is one differentiable gather with PyTorch's placement rules (replaces the O(n^2) one-hot path).
- 0-d NumPy results no longer become shape `(1,)` (`np.ascontiguousarray` replaced by `_contig`).
- Optimizers that disagreed with PyTorch: `RMSprop` added `eps` inside the square root (`sqrt(v + eps)` instead of
  `sqrt(v) + eps`), `RAdam` switched to the rectified update at rho > 4 instead of 5 and scaled `eps` differently, `ASGD` never
  applied its decay `lambd` to the parameters, `Adamax` added `eps` after the max instead of inside it; `Rprop` keyed its state by
  the tensor instead of its id, so `state_dict` / `load_state_dict` lost it. All now match `torch.optim` step for step
  (`tests/test_rust_kernels.py`).
- `linalg.lu_factor` returned LAPACK's 0-based swaps when SciPy was installed and a plain row permutation when it was not, and
  `lu_solve` guessed which one it got from whether SciPy imports; neither matched PyTorch. Both now use PyTorch's 1-based pivots
  in every environment (no SciPy needed), work on batches, and accept factors computed by `torch.linalg.lu_factor`.
- `ox.autograd.grad` ran a backward pass and returned the inputs' `.grad`, so it added to (and returned) gradients left by
  earlier calls, left `.grad` changed, and made every second derivative wrong (`grad` of `w * w` with `create_graph=True`, then
  `grad` again, gave 2 + 2w instead of 2). It now uses the native graph traversal: the gradients are returned and `.grad` is not
  touched, like `torch.autograd.grad` (custom `autograd.Function` outputs keep the old path, now with `.grad` saved and restored).
- `ox.save(state_dict, "model.weights")`: a file name without a checkpoint extension was written with pickle, which `ox.load`
  refuses to read from such a name, so the file could never be loaded. Tensor dicts are now written as safetensors; other objects
  raise an error that names the extensions to use.
- `oxideon.distributions`: a distribution with scalar parameters (`Normal(0.0, 1.0)`, `HalfNormal(1.0)`, ...) failed in `sample()`
  (`np.random.randn()` returns a Python float), and scalar results came back with shape `[1]` instead of `[]`.
- CUDA: `where_cond` with a broadcast mask (kernel declined) returned a host-backed tensor labelled `cuda:0`, so the next matmul failed with
  `device mismatch (cuda:0 vs cuda:0)` (attention with a causal mask on the composed path); it now returns CUDA storage.
- NVRTC is loaded with the wide-character Windows loader, so it is found under a user directory with non-ASCII characters
  (`C:\Users\Jürgen\...`); before, the CUDA tests quietly skipped and every `.cu` kernel fell back to the host.
- `ox.compile` sets `_compiled` once a call has been traced into a graph.
- Examples: `gguf_generate` and `gguf_info`/`accuracy_dump` report a bad path, an unsupported tokenizer or a missing argument instead of
  panicking; `browser_demo` says that it is a simulation; the duplicate `tools/core_ops_bench_*.py` scripts are gone (use `benchmarks/`).

### Numerical accuracy (see `docs/NUMERICAL_ACCURACY.md`)
- `PrecisionMode { Fastest, Balanced, Highest }`: one application-wide switch for `f32` matrix products (default `Balanced`; `Highest` does f64
  accumulation at ~1.9x the time of `Fastest`). Set with `oxideon_core::set_precision_mode`, the `OXIDEON_PRECISION` environment
  variable, or `oxideon.set_precision_mode("balanced")` in Python.
- Float32 results measured against a float64 reference next to PyTorch (`benchmarks/accuracy_vs_pytorch.py`, 40 cases, worst and
  mean error together). Large wins on long sums/means/dot/norm (f64 accumulators), matmul, softmax/log_softmax, layer/batch/group
  norm and variance of offset data (accurate centring, `Tensor::centered_by_mean`), `cosine_similarity`, `softplus`, `elu`.
- Default `Balanced` matmul: 48-deep f32 chains (matmul 2.35x / 1.66x better than PyTorch on worst / mean error); sigmoid, silu and tanh
  use a hi/lo `exp` (accurate to ~0.03 ulp; tanh now ties PyTorch on worst and mean, sigmoid/silu worst 2.5-2.8x better); Winograd convolution sums channels in TwoSum-combined chains (conv2d 3.0x / 1.5x and 4.6x / 3.0x); `normalize`
  (L2) and `var_axis` run in f64 on the host fast path.
- `Fastest` matmul runs through the same block grid in 256-deep `f32` slices (was one `gemm` call: a 1x4096x4096 product took 2.5 ms, now 1.2 ms; matmul 512 / 1024 / k=4096 now 352 / 2823 / 707 us, all at or under PyTorch), and its k=512 error falls from about 6 to about 2.6 half-ulps. Balanced products inside a rayon worker run their chains serially instead of opening a parallel region per chain.
- `Fastest` keeps plain `f32` accumulation for any `k` (the `f64` group combine of the long-`k` chain loop is `Balanced` only), the wide chain loop reuses per-thread work buffers, and the GELU mode is read once per tensor.
- The buffer pool reads its limit and updates its byte accounting under one lock (a racing `give` could overshoot a freshly lowered limit, which made `typed_buffers_stay_inside_the_global_byte_limit` flaky).
- `Fastest` mode evaluates GELU with a cheaper five-term erfc (150 -> 113 us per 1M elements, level with PyTorch; abs. CDF error ~1.5e-7).
- Faster `Balanced` matmul (bit-identical results): the accumulation chains of one product run inside a single parallel region over output
  blocks instead of one region per chain (matmul 512: 568 -> 391 us, 1024: 4.5 -> 3.3 ms, k=4096: 2.05 -> 0.96 ms, MLP step 1.35 -> 1.0 ms;
  PyTorch 392 / 3045 / 804 / 832 us; with a 2-D block grid: 371 / 2813 / 866 / 860 us). The erfc polynomial of `gelu` and the plain `exp` polynomial use Estrin evaluation (gelu 184 -> 150 us, same error). Fused `layer_norm` variance is vectorized (83 -> 52 us, now faster than PyTorch and more accurate).
- Faster: Winograd 3x3 convolution pipeline, fused unary backward, adaptive column blocks for leading-axis reductions.
- Image codecs: PNG Adam7 interlacing and progressive JPEG decoding.
- Fixed: `f32_row_broadcast` used x86 stubs on aarch64 (wrong layer-norm output on ARM); `c_char` portability on aarch64;
  stale `Linear` graph test.

### Added
- Training and serving infrastructure (oxideon-core), each with tests:
  - `loader`: `PrefetchLoader` (worker threads, output order independent of the worker count, per-epoch order fixed by
    `(seed, epoch)`, resume from any batch), `StreamingLoader` (sharded streaming with a shuffle buffer and per-rank
    shard split), `NpyDataset` (rows of `.npy` files read on demand).
  - `checkpoint`: full training state (sharded model, optimizer, RNG, epoch/step/batch) written atomically, optional
    background writes, `CheckpointManager` (keep last N, find latest). A resumed run is bit-identical.
  - `train_loop`: step-based `TrainLoop` (LR schedule function, gradient accumulation and clipping, EMA weights used for
    evaluation, callbacks, TensorBoard logger, automatic exact resume).
  - `profiler` (operator timings, Chrome trace, NaN/inf anomaly detection), `summary_writer` (TensorBoard event files),
    `memory` (live / peak tensor bytes), `hub` (Hugging Face snapshot cache via `curl`, sharded safetensors loading),
    `dlpack` (DLManagedTensor export / import).
  - `serving`: `PrefixCache` (prompt-prefix KV block reuse, LRU eviction) wired into `ServingEngine`, KV block growth
    during generation, regex-constrained requests; `grammar::regex` (linear-time regex matcher and logit masker,
    cross-checked against Python `re`).
  - dtypes: `U64` (exact above 2^53), `F8E4M3` and `F8E5M2` 8-bit floats (encodings match PyTorch), next to the
    earlier `I8`/`I16`/`U16`/`U32`.
  - tests: finite-difference gradient checks for 57 linalg and core ops, bit-identical results across thread counts.
- Rust library gap-filling (oxideon-core), each module with tests; several checked against external references
  (llama.cpp, onnxruntime, NumPy):
  - `special`: erfc, erfinv, lgamma, digamma, polygamma, zeta, igamma/igammac, ndtr/log_ndtr/ndtri, entr, xlog1py,
    i0/i0e/i1/i1e, Bessel j0/j1/y0/y1, mvlgamma, all with autograd (higher order through the gamma chain).
  - `ops_extra`, `tensor_misc`, `inplace`, `linalg_extra`: logaddexp, float_power, frexp/ldexp, gcd/lcm/shifts,
    index_put/index_copy/select|slice|diagonal_scatter, scatter_reduce, nonzero/unique/bucketize, mm/bmm/addmm family,
    multi_dot, rot90, pdist, in-place `add_`/`mul_`/`fill_`/... with PyTorch's autograd safety rules, batched
    svd/eigh/qr/lu/pinv, svdvals, logdet, vector/matrix norms, general `eig`/`eigvals`, tensorinv/tensorsolve.
  - `complex::Complex` (autograd-aware complex tensors, FFT integration, `eig_complex`), `signal` (windows, fftfreq,
    fftshift, fftn with norm modes, stft/istft, DCT), `numeric_ext` (interp, polyfit/roots, convolve/correlate,
    sqrtm/logm).
  - `optim_extra`: Adadelta, Adamax, NAdam, RAdam, ASGD, Rprop, Adafactor, LARS, AdaBelief, Yogi, SAM; composable LR
    schedules; `ParamGroup`/`GroupedOptimizer`; `state_dict`/`load_state_dict` with safetensors save/load for all of
    them and for `Sgd`/`Adam`.
  - `nn`: `loss_ext`/`loss_modules` (reductions, class weights, ignore_index, label smoothing, probability targets),
    `functional_ext` (scaled_dot_product_attention, functional norms, reflect/replicate/circular padding, interpolate),
    `attention_ext` (PyTorch-style MultiheadAttention with masks, norm_first Transformer layers and the full
    encoder-decoder Transformer), `conv_ext`, `layers_ext` (embedding padding_idx/max_norm, fuse_conv_bn),
    `activations_ext`, `init_ext`, `utils` (EMA, SWA, clip_grad_norm_p, pruning), `hooks`.
  - `distributions_ext`/`distributions_transforms`: 23 more distributions, bijectors and `TransformedDistribution`.
  - `quant_aware` (fake quantization with straight-through gradients, observers, QatLinear), `autograd_utils`
    (gradcheck/gradgradcheck), `data_ext`/`data_transforms` (subsets, samplers, augmentation, mixup/cutmix),
    `metrics_ext`, `training_utils`.
  - `serialize::npy` (NumPy .npy/.npz incl. compressed archives) and `serialize::onnx` (ONNX export of MLP, CNN,
    residual/attention blocks, LSTM/GRU/BiLSTM; verified in onnxruntime).
  - `GGUF` encoders: `BertEncoder` for `bert`, `nomic-bert` and `jina-bert-v2` with reranker heads, parity-tested
    against llama.cpp; parity fixes for the per-tensor scales current llama.cpp applies.
  - Later additions: `nn::gnn_ext` (SAGE/GIN/GCN/GAT on `edge_index`, pooling, graph utilities), `nn::module_tools`,
    `nn::layers_misc` (PadLayer, GaussianNoise, ParameterList, ConvLstmCell), `sparse_ext`, `flops`, `ops_stats`
    (trapezoid, isin, histogram), `Lamb`, `samplers_ext`, `privacy` (per-sample grads, DP-SGD, RDP accountant),
    `functional_ext` (jacfwd, vmap, ...), `image_aug` (RandAugment, RandomAffine, ...), `metrics_more` (clustering,
    calibration, text metrics), and pure-Rust classical ML: `classical*` (scalers, k-means, DBSCAN, PCA, KNN, naive
    Bayes, OLS/ridge, logistic regression, trees, forest, boosting, GMM, SVM), `manifold` (MDS, Isomap, t-SNE),
    `timeseries` (ACF/PACF, AR/ARIMA, Holt-Winters, decomposition), `preprocessing`, `model_selection`. Classical
    models are checked on synthetic data and hand-computed values, not against scikit-learn.
  - Also: `inspect` (permutation importance, PDP/ICE, Shapley sampling, LIME), `checkpointing` (atomic, rotating,
    resumable), `tuning_ext` (successive halving, Hyperband, TPE), `gp` (Gaussian process, Bayesian optimization),
    `scientific` (root finding, BFGS/Nelder-Mead, quadrature, cubic spline, curve_fit), `optimization` (simplex LP,
    NNLS, Hungarian), `graph_algos` (shortest paths, SCC, MST, PageRank, betweenness, communities, max flow),
    `stats_tests` (t/chi2/ANOVA/KS/rank tests, correlations, bootstrap), `survival` (Kaplan-Meier, log-rank, Cox PH),
    `inference` (MH, HMC, Kalman, HMM, particle filter), `text_ir` (TF-IDF, BM25, string metrics), `spatial` (KdTree,
    hull, closest pair), `numtheory`, `dsp` (filters, Butterworth, Savitzky-Golay, Welch), `ml_extras` (AdaBoost,
    isotonic/Platt calibration, SMOTE, Sinkhorn, feature selection).
  - And: `decomp` (NMF, FastICA, kernel PCA, LDA, agglomerative/spectral/mean-shift clustering), `nn::implicit`
    (SIREN, Fourier features, LayerScale, NT-Xent), `dataio` (CSV/libsvm/IDX), `anomaly` (isolation forest, LOF,
    CUSUM), `recsys` (matrix factorization, item-kNN, Apriori), `glm` (IRLS GLMs, Huber, LOWESS, KDE),
    `rl::classic` (bandits, MDP planning, Q-learning/SARSA, replay buffers), `causal` (multiple-testing
    corrections, power analysis, IPW, diff-in-diff, 2SLS), `sketches` (Bloom, Count-Min, HyperLogLog, streaming
    quantiles), `iterative` (CG, GMRES, BiCGSTAB, Lanczos, randomized SVD), `frame` (DataFrame), `numerics`
    (grid/Chebyshev interpolation, finite differences). All covered by unit tests on synthetic data or
    hand-computed values at the time of writing; see the next entry for what was later cross-checked externally.
  - `Sgd` gained `nesterov`, `dampening` and `maximize`; `Adam`/`AdamW` gained `amsgrad` and `maximize` (all match
    PyTorch in cross-validation batch 4, including the new state-dict entry `max_exp_avg_sq`). `RmsProp` gained
    `alpha`/`eps`/`weight_decay`/`momentum`/`centered` (and a state dict); `Adagrad` gained `lr_decay`,
    `initial_accumulator_value`, `eps` (and a state dict). `ReduceLROnPlateau` gained `maximize`, `threshold` (relative
    or absolute), `cooldown` and `eps`. Every scheduler is now cross-checked against PyTorch (batch 18). `LBFGS` now follows PyTorch's per-iteration rules
    (damped first step `min(1, 1/|g|_1) * lr`, unclamped `H_diag`, `tolerance_grad`/`tolerance_change` exits) and
    matches `torch.optim.LBFGS(max_iter=1)`; `Adafactor` is cross-checked as well (batch 19).
  - Fixed: `Binomial::entropy` and `Poisson::entropy` returned constants with no gradient; both now carry the exact
    derivative w.r.t. the logits / rate (batch 28 checks 89 distribution gradients against PyTorch).
  - Fixed: `BatchNorm3d` updated `running_var` with the biased batch variance (PyTorch and the 1-D / 2-D layers use the
    unbiased one), so eval-mode outputs drifted from PyTorch (batch 29).
  - Review follow-ups: `norm_p(inf)`, `matrix_norm` and `aminmax` share the gradient among ties again (only
    `max_axis`/`min_axis` pick the first); `clamp` gives NaN inputs zero gradient; `multi_margin_loss` has zero gradient
    at an exactly-zero hinge; `celu` handles negative `alpha` (and rejects 0); SVD gradients through `U`/`Vt` are NaN
    for repeated or zero singular values instead of silently wrong; `jn` keeps relative accuracy for tiny values and
    no longer scales its cost with `|x|` (series / Hankel expansion); `kve` no longer overflows for large orders;
    `fresnel`, `sici`, `lambertw`, `ellipk`/`ellipe` return their limits at infinity.
  - Changed (PyTorch parity, batch 26): `clamp` now passes the gradient at exactly `min`/`max`; `max_axis`/`min_axis`
    route the gradient to the first extremal element on ties while `amax`/`amin` share it evenly; `diag_embed`
    accepts batched `[..., N]` input.
  - Fixed: `Tensor::svd()` only backpropagated through the singular values; `U` and `Vt` are now differentiable
    (Townsend backward, tall and wide inputs), so losses on singular vectors train. Gradients of 90 further tensor and
    linear-algebra ops were verified against PyTorch (batches 24-25).
  - Fixed: `ctc_loss` and `multilabel_margin_loss` produced no gradient (calling `backward()` failed); both are now
    differentiable and match PyTorch (CTC uses the alpha-beta recursion). `max_unpool2d` overwrote-vs-summed
    duplicate indices, `celu` had gradient 0 at 0, `triplet_margin_loss` lacked `eps` (batches 20-23).
  - `special_ext`: scalar special functions SciPy has and the library lacked — tail-accurate `erfc`/`erfcx`, complete
    elliptic integrals, `E1`/`Ei`, Lambert W (both real branches), Fresnel, Dawson, `Si`/`Ci`, dilogarithm (`spence`),
    Bessel `J_n`, scaled `I_n`, `K_v` of real order, and the classical orthogonal polynomials. Cross-checked against
    `scipy.special` on 763 values (batch 17, worst relative error 3e-13).
  - Cross-validation harness (`tools/crossval`, `crates/oxideon-core/tests/crossval_emit*.rs`): ~10,000 values from the
    statistics, metrics, clustering, classical-ML, graph, geometry, signal, special-function, distribution, loss,
    optimizer, tensor-op, attention, convolution, recurrent-layer, survival, HMM, Kalman, GP and image code are compared against SciPy, scikit-learn,
    NetworkX, NumPy and PyTorch (run on demand with `--ignored`; agreement is typically 1e-9 to 1e-13, 1e-6 for float32
    modules). It found and fixed: Butterworth high-pass pole scaling, FIR high-pass design (now `scipy.signal.firwin`),
    an infinite loop in `i1`/`i1e` at 0, `multilabel_margin_loss` counting target-vs-target pairs, CTC `mean` reduction
    (now divides by target length like PyTorch), `scatter_reduce(sum|mean, include_self=false)`, and `ComplementNb`
    (new `norm` flag; `false` matches scikit-learn). Batches 13-15 added string metrics, BM25, the RDP accountant,
    sketches, Apriori, item-kNN, evolutionary optimizers, particle filter, TPE, Hyperband and Bayesian optimization,
    and found two more bugs: `jaro_winkler` applied the prefix bonus below Winkler's 0.7 threshold, and
    `expected_improvement` used a 1e-7-accurate erf (now the exact normal CDF).
- Fixed: BatchNorm running statistics were missing from `state_dict`/`save`/`load` (new `Module::named_buffers`).
- Bitwise/logical/predicate tensor ops (3 new `ops::tests`, 1 new pytest, all
  torch-verified): `bitwise_and/or/xor/not` (exact native-width integer/bool
  kernels, `&`/`|`/`^`/`~` dunders included), `logical_and/or/xor/not` with
  `!= 0` truthiness on any dtype (mixed dtypes allowed: no promotion rule is
  needed), and `isnan`/`isinf`/`isfinite` (integers/bools never NaN/inf,
  always finite). Float bitwise operands are rejected like torch; mixed-dtype
  bitwise fails closed while torch promotes (deliberate, matches `binary_op`).
- Offline-first eval harness (`oxideon.evals`, `tests/test_evals.py`): text metrics
  (`accuracy`, `exact_match`, `token_f1`, `pass_at_k`, `math_equal`, `rouge_l`,
  `corpus_bleu`, `perplexity_from_nll`), four task families (`MultipleChoiceTask`
  with loglikelihood-or-generate scoring, `MathTask`, sandboxed `CodeTask` with
  pass@k, synthetic `NeedleTask` long-context recall), plus `run_suite`/`EvalReport`
  aggregation with JSON export and demo fixtures. Namespace-only import, so the
  existing top-level `accuracy` from `metrics` is untouched.
- Inference-efficiency additions (`tests/test_inference_efficiency.py`): KV eviction
  policies `StreamingSinkCache` (attention sinks + recent window) and
  `ScoredEvictionCache` (H2O cumulative, TOVA last-query, SnapKV pooled-window
  selection); exact OCP Microscaling MXFP4 (`quantize_mxfp4`/`dequantize_mxfp4`,
  E2M1 nibbles + E8M0 power-of-two scales); reference `MLAAttention` (low-rank
  KV, decoupled RoPE) and `NativeSparseAttention` (compressed + top-k selected
  + sliding branches with learned gating).
- Docs and benchmarks for the new stack: `tools/efficiency_bench.py` measures MXFP4,
  KV eviction, MLA/NSA and evals overhead (numbers recorded in
  `docs/BENCHMARKS.md` section 6); README gains an evals quickstart;
  `docs/INFERENCE_AND_LLM.md` gains eviction/MXFP4/MLA-NSA sections;
  `docs/PYTHON_API.md` gains a section 14 plus updated inference entries. Every new
  snippet was executed against the built extension before landing.
- Post-training stack (groups 4a-4d): RL alignment extras in `alignment.py`
  (`RLOOLoss`, `ReMaxLoss`, `DrGRPOLoss`, `DAPOLoss` with clip-higher/dynamic
  sampling/overlong shaping, `PPOClipLoss`, `ProcessRewardHead`/`PRMLoss`);
  weight-space merging in new `oxideon.merge` (`average_merge`,
  `task_arithmetic_merge`, `ties_merge`, `dare_merge`, `slerp_merge`);
  knowledge editing + unlearning in new `oxideon.edit` (`rome_update`,
  `memit_layer_update`, `NPOLoss`, `GALoss`); sparsity in `prune.py`
  (`prune_wanda`, exact sequential `prune_sparsegpt`, `prune_nm_structured`,
  `block_influence`/`select_layers_to_drop`/`drop_sequential_blocks`).
  Tests: `test_rl_alignment.py`, `test_model_merging.py`,
  `test_edit_unlearn.py`, `test_sparsity.py`.
- Data and retrieval stack (groups 5a-5b): curation in new `oxideon.curate`
  (`MinHashLSH` near-dup LSH with union-find clusters, `NGramDecontaminator`,
  `gopher_flags`/`passes_gopher`, `c4_flags`/`passes_c4`, `doremi_weights`,
  `regmix_fit`/`regmix_search`, `temperature_mix`); retrieval in new
  `oxideon.retrieval` (`BM25Index`, `reciprocal_rank_fusion`, `HybridRetriever`
  over BM25 + dense indexes, `CrossEncoderReranker`, `EntityGraph` with
  label-propagation `communities` and `summarize_communities`); RAGAS-style
  `rag_context_precision`/`rag_context_recall`/`rag_faithfulness`/
  `rag_answer_relevancy` in `oxideon.evals`. Tests: `test_data_curation.py`,
  `test_retrieval.py`, plus RAG cases in `test_evals.py`.
- Modalities stack (groups 6a-6c): speech front-end in new `oxideon.speech`
  (`energy_vad` with hangover/min-duration, `SpeakerDiarizer` with greedy
  average-linkage clustering, `spectral_subtraction`/`wiener_filter`/
  `ideal_ratio_mask` enhancement over `signal.stft`, Shazam-style
  `audio_fingerprint`/`fingerprint_matches`); vision toolkits in
  `oxideon.vision.ops` (`yolo_decode`/`yolo_decode_batch`, COCO-style
  `mean_average_precision`, `mask_iou`, `panoptic_quality`, `oks`, `pck`,
  `heatmap_to_keypoints`, `linear_sum_assignment`, `KalmanBoxTracker`,
  `IoUTracker`, two-stage `ByteTracker`, pyramidal `lucas_kanade`,
  `block_match_disparity`, `ctc_greedy_decode`/`ctc_beam_decode`,
  `edit_distance`/`cer`/`wer`) plus a numpy-reference `DeepLabHead` with
  ASPP in `oxideon.vision.models`; differentiable NeRF in new
  `oxideon.nerf` (`positional_encoding`, `get_rays`, `stratified_samples`,
  `volume_render`, `sample_pdf`, `NeRFField`, hierarchical `render_rays`
  with end-to-end autograd) and a CPU EWA `GaussianSplatScene` renderer
  in new `oxideon.splat` (`project_gaussians`, `rasterize`).
  Tests: `test_speech_frontend.py`, `test_vision_toolkit.py`,
  `test_nerf_splat.py`.
- Classical science stack (additive-only, no behavior change to existing
  code; SciPy/PyTorch used as test oracles only, never imported at
  runtime):
  - `oxideon.numerical`: adaptive `quad`/`dblquad`, `gauss_legendre`/
    `fixed_quad`, `Interp1D`, `CubicSpline`, `minimize_scalar`,
    `curve_fit`, `physical_constants` table.
  - `oxideon.stats`: `ttest_1samp`/`ttest_ind`/`ttest_rel`, `chisquare`/
    `chi2_contingency`, `pearsonr`/`spearmanr`, `mannwhitneyu`,
    `kstest`, `skew`/`kurtosis`/`jarque_bera`, `ecdf`/`ECDF`,
    `p_adjust` (bonferroni/holm/bh).
  - `oxideon.physics`: `Wave2DEquation`/`Diffusion2DEquation` PDE
    solvers, `simulate_nbody` (+ energy/momentum, Kepler helpers),
    `hamiltonian_leapfrog`.
  Tests: `test_numerical.py` (26), `test_stats.py` (33),
  `test_physics.py` (12), several cross-checked against SciPy.
- Docs: `docs/API_STABILITY.md` (tiered stability statement with
  per-module classification), `docs/SCIENCE_API.md` (science API guide
  with executed snippets).
- Examples (all `main()`-runnable and gate-enforced via
  `test_new_example_scripts_execute_cleanly`):
  `science_numerical_methods.py`, `science_statistics.py`,
  `science_physics_sims.py`, `train_with_scheduler_and_checkpoint.py`.

### Fixed
- nn autograd restoration (Batch CV, `tests/test_bug_hunt_cv.py`, 21
  regression tests, PyTorch oracle for values/gradients/validation):
  - `BatchNorm1d/2d/3d` and `GroupNorm` now propagate input gradients
    (previously silently None below any norm); `running_var` tracks the
    unbiased estimate (torch ddof parity).
  - `MultiMarginLoss`, `MultiLabelMarginLoss`,
    `MultiLabelSoftMarginLoss` rewritten with Tensor ops (grad flows;
    bad reductions and out-of-range labels are `ValueError`).
  - `Flatten` honors `start_dim`/`end_dim`; `CTCLoss` accepts
    torch-style lengths/targets (validated native subset, N*S `sum`
    rescaling); `PReLU` broadcasts per-channel weights with grad.
  - `adaptive_max_pool1d/2d/3d`, `lp_pool1d/2d` (also fixes signed
    odd-power values), `F.pad` (all modes), and
    `local_response_norm` (also fixes even sizes) keep autograd.
- Collective/parallel-container hardening (Batch CU,
  `tests/test_bug_hunt_cu.py`, 18 regression tests, torch gloo oracle
  for op/dtype domains):
  - `all_reduce` list path and `DataParallel` numpy gather no longer
    force float32 (f64/i64 preserved); unknown collective ops raise
    `ValueError` instead of silently summing or shuffling garbage.
  - Bitwise `BAND`/`BOR`/`BXOR` reductions implemented in `all_reduce`
    and `ring_all_reduce`; bitwise-on-float and `AVG`-on-int raise
    `TypeError` (torch parity).
  - `init_process_group` rejects out-of-range ranks before mutating
    state; `scatter`/`reduce_scatter` require `world_size` entries;
    `reduce_scatter_tensor` chunks the input and reduces per rank
    (previously always crashed on legitimate shapes).
  - `DataParallel` gathers tuple/list outputs elementwise (torch
    parity); `PipelineParallel` forwards kwargs on the microbatched
    path and rejects batch-0 input with `ValueError`.
  - `KVMemoryBank.read` on empty memory returns grad-connected zeros
    so `backward()` works; `all_gather` accepts array-like lists.
- Optimizer/scheduler/serialization hardening (Batch CT,
  `tests/test_bug_hunt_ct.py`, 11 regression tests, torch-oracle
  verified where applicable):
  - Optimizer: bare-Tensor/empty/set/non-Tensor params rejected
    (`TypeError`/`ValueError`, torch parity); `load_state_dict` maps
    hyper-parameters and per-param state onto the live params by order
    instead of adopting checkpoint objects; `state_dict` deep-copies.
  - LBFGS honors `max_iter` (inner loop) and `tolerance_change`
    (early stop); non-None `line_search_fn` raises `ValueError`.
  - NAdam uses the scheduled momentum-decay correction (mirrors the
    in-repo Rust schedule) instead of a constant-beta1 term.
  - Schedulers read/write per-param-group LRs (previously a single
    `.lr`/1e-3 base that never moved native optimizers); `LambdaLR`
    validates lambda-list length; StepLR/MultiStepLR preserve external
    LR edits on non-decay steps.
  - `serialization.load` no longer masks strict state-dict mismatches
    as unpickle errors (parse-then-apply); loading a non-dict object
    into a model raises `TypeError` instead of silently returning the
    untouched model.
- Index-fill audit (engine + binding + shim unwiring, 1 new `ops::tests`,
  2 new pytest in `tests/test_masked.py`, all torch-verified): new engine
  `index_fill` (axis mask + `where_cond` composition, so gradients flow
  around filled slices) replacing the numpy shim. Negatives wrap, empty
  and 0-d indices work, 0-d inputs behave as size-1, and indices must be
  int64 vectors/scalars like torch. Tensor values coerce through item()
  at the module level, like torch Scalar. Deliberate divergences:
  out-of-range indices and bad dims raise the engine's `RuntimeError`
  (torch: `IndexError`).
- Masked-select/scatter audit (engine + bindings + shim unwiring, 1 new
  `ops::tests`, 3 new pytest in `tests/test_masked.py`, all
  torch-verified): new engine `masked_select`/`masked_scatter` composed
  from `index_select`/`scatter` (masks broadcast, must be boolean;
  N-D sources flatten, tails drop, short sources fail closed), so both
  operands differentiate (select scatters grads back; scatter splits
  base/complement with trailing source zeros). The module functions
  delegate (the in-place `masked_scatter_` keeps working through
  delegation) and the `_Tensor` monkeypatches are deleted. `index_fill`
  stays a shim (follow-up).
- Mode audit (engine + binding + shim unwiring, 1 new `ops::tests`, 8
  new pytest in `tests/test_mode.py`, all torch-verified): new engine
  `mode_axis` (per-slice counts with i64 keys past 2^53, values gathered
  so gradients flow) replacing the scipy/numpy shim; ties resolve to
  the smallest value, NaN groups and sorts smallest, +-0.0 group, 0-d
  and bool inputs work like torch. The module function delegates (keeps
  the `_ModeResult` namedtuple) and the `_Tensor.mode` monkeypatch is
  deleted. Deliberate divergence: indices address the FIRST occurrence
  (the shim's documented contract) while torch follows its internal
  unstable sort order (content-dependent, usually last); empty/bad-dim
  raise the engine's `RuntimeError` (torch: `IndexError`).
- Quantile audit (engine + bindings + shim unwiring, 2 new `ops::tests`,
  13 new pytest in `tests/test_quantile.py`, all torch-verified): new
  engine `quantile_axis`/`quantile`, `nanquantile_axis`/`nanquantile`
  composed from stable `sort` + `gather` with all five interpolations
  (linear lerps `lo+(hi-lo)*frac` in-width; midpoint averages the
  overflow-safe way; nearest rounds half up). Rank positions run in f64
  with q in the input width (near-integral f32 hits like 0.9*140 bracket
  downwards, torch-verified); `quantile` propagates NaN (any NaN wins,
  gradients address the first NaN) while `nanquantile` positions each
  slice by its own valid count. Scalar q drops the q dim; tensor q must
  match the input dtype and be 0/1-D. The module functions delegate and
  the `_Tensor.quantile`/`_Tensor.nanquantile` monkeypatches are deleted,
  so gradients flow (linear splits across brackets, midpoint evenly).
  Deliberate extension: list/tuple q converts with the input dtype
  (torch rejects it with `TypeError`).
- Median audit (engine + binding + shim unwiring, 1 new `ops::tests`, 6
  new pytest in `tests/test_median.py`, all torch-verified): new engine
  `median_axis`/`median` composed from stable `sort` + `gather` with
  torch's lower-middle rule for even counts (the old numpy shim averaged
  them) and NaN propagation (any NaN wins; values/indices/gradients
  address the FIRST NaN via an `argmax(isnan)` + `where_cond` branch).
  The module function delegates and the `_Tensor.median` monkeypatch is
  deleted, so gradients flow to the winner only. Deliberate divergences:
  empty reductions raise the engine's `RuntimeError` (torch: `IndexError`);
  `dim` form returns a plain tuple (torch: namedtuple); `keepdim` without
  `dim` is accepted ones-shaped (torch rejects it); bool input sorts
  (torch: `NotImplementedError`).
- NaN-reduction audit (engine + bindings + shim unwiring, 2 new
  `ops::tests`, 13 new pytest in `tests/test_nan.py`, all torch-verified):
  new engine `nanmean_axes`/`nanmean`, `nansum_axes`/`nansum`,
  `nanmedian_axis`/`nanmedian` composed from `masked_fill`/`isnan`,
  `sum_axes`, and stable `sort` + `gather` (counts cast to i64 before
  summing: `sum_axes` casts back to the input dtype, so a bool sum would
  collapse to true/false). The module functions delegate, so gradients
  flow (1/count on valid entries for `nanmean`, chosen-element-only for
  `nanmedian`); `nanmedian` with `dim` returns a `(values, indices)`
  tuple with torch's lower-middle rule for even counts, and `nanmean`/
  `nansum` accept tuple dims via the multi-axis kernels. Deliberate
  divergences: integer `nanmean` and empty `nanmedian` raise the engine's
  `RuntimeError` (torch: `NotImplementedError`/`IndexError`); `nanmedian`
  returns a plain tuple (torch: namedtuple); `nanmedian` accepts
  `keepdim` without `dim` (ones-shaped, torch rejects it). `median`,
  `quantile`/`nanquantile`, `mode`, `gradient`, and `trapz` stay numpy
  shims (follow-up).
- Histogram audit (engine weights/range + bindings + shim unwiring, 6 new
  pytest in `tests/test_histogram.py`, all torch-verified): engine
  `bincount` gains `weights` (output follows floating weights dtypes,
  else F64 — the old numpy shim always downcast to F32) with torch's
  1-D same-length rejections, plus new `bincount`/`histc` bindings; the
  module functions delegate, so `bincount` now rejects negatives, N-D
  input, and float input with torch's `RuntimeError` instead of silently
  flattening/casting, and negative `minlength` raises torch's
  `RuntimeError`; engine `histc` implements the `min == max` data-range
  rule (all-equal expands +/-0.5, empty reads zeros, any NaN raises)
  instead of erroring on every `min == max`. `histogram` keeps its numpy
  path (counts exact; edges carry f32 linspace noise, pinned with
  tolerance). Deliberate superset: integer `histc` input is binned
  although torch rejects it. (Existing `bincount` Rust call sites updated
  for the new `weights` parameter.)
- Cumulative ops audit (bindings + shim unwiring + diff edge fix, 4 new
  pytest in `tests/test_cumulative.py`, all torch-verified): `cumprod`
  and `diff` had engine methods but no bindings, served by numpy-shim
  monkeypatches with no autograd — new bindings (`cumprod(dim)`,
  `diff(n=1, dim=-1)`) serve the methods (the diff wrapper keeps
  `prepend`/`append` via cat plus torch's `RuntimeError` for negative n),
  and the module functions delegate so gradients flow (cumprod backward
  is division-free, hence exact with zeros in the row). Engine `diff`
  reads empty on short/empty dims like torch instead of raising; `n = 0`
  returns the input. No `cummax`/`cummin`/`logcumsumexp` yet (follow-up).
- Masked/gather/scatter audit (shim unwiring + engine `scatter`, 1 new
  `ops::tests`, 4 new pytest in `tests/test_masked.py`, all torch-verified):
  the `masked_fill` Tensor method was monkeypatched with a numpy shim
  (exact-shape mask only, no autograd) shadowing the native binding — the
  patch is deleted so the native method serves (broadcasting mask,
  differentiable, int masks keep nonzero-truthiness); `gather` and
  `scatter_add` module functions delegated from numpy to the native
  methods (gradients now flow); new engine `Tensor::scatter` (backend
  `scatter_vals`, binding, delegated module function) with last-write-wins
  duplicates matching torch CPU and a backward that zeroes base grads at
  scattered positions while src gathers grad; new module-level
  `masked_fill`; removed a dead duplicate `masked_select` definition.
  Deliberate notes: negative indices wrap (the pinned gather-family
  divergence, now documented on `scatter` too); `masked_select`,
  `masked_scatter`, and `index_fill` stay value-correct numpy shims with
  no autograd (engine follow-up).
- `kthvalue` implemented natively (engine + binding + module function, 1 new
  `ops::tests`, 4 new pytest in `tests/test_kthvalue.py`, all torch-verified):
  the op existed nowhere (no engine method, binding, or shim). O(n)-average
  quickselect per row with `total_cmp` (NaN largest, torch parity) and an
  I64-native path past 2^53; 1-based k with `keepdim`, differentiable values
  via `gather`, torch-matching `RuntimeError` on out-of-range k/dim. The
  module function is pure delegation (no legacy `ValueError` shell: none
  existed to preserve).
- Sort/topk/searchsorted audit (native bindings + shim unwiring + 2 engine
  fixes, 3 new `ops::tests`, 10 new pytest in `tests/test_sort.py`, all
  torch-verified): the native `sort`/`argsort`/`topk`/`searchsorted` engine
  methods had no Python surface — `primitives.py` monkeypatched the Tensor
  methods with numpy shims (unstable/anti-stable ties, no autograd) and the
  module functions shadowed the engine. New bindings use torch's `dim`
  names; the monkeypatches are deleted and the module functions keep
  their exact `ValueError` validation shells (`non-negative`, `out of
  range`, `above the limit`) while delegating compute to the engine
  (stable ties, gradients flow, existing error-contract tests green).
  Engine fixes: `topk`/`searchsorted` compare I64 natively (f64 keys
  rounded past 2^53 and misordered/misplaced); `searchsorted` grows N-D
  row support with torch's exact shape rule (same ndim, exactly matching
  leading dims, output = values shape — the old numpy fallback raised a
  cryptic error on any N-D input) and reads NaN queries as the row
  length. Deliberate notes: no `stable` flag (always stable, a valid
  refinement of torch's undefined tie order) and no `sorter` variant;
  bool topk sorts False < True although torch CPU rejects bool topk.
- Special-functions audit (12 fixes in `python/oxideon/special.py`, 15 new
  pytest in `tests/test_special.py`, all torch-verified): `erfinv`/`ndtri`
  now return ±inf at the closed endpoint and NaN outside instead of
  silently clamping (ndtri also keeps f64 end to end, no f32 roundtrip);
  `gammaln` uses Euler reflection left of 0 with +inf at poles instead of
  clamping to 1e-12 (shared with `multigammaln`, which raised on invalid
  input and now reads +inf like torch); `digamma` uses reflection with
  NaN/-inf poles instead of 1e15-scale garbage; `polygamma`/`zeta` gain
  Euler-Maclaurin tails (0.2%/5% low before); `entr` reads -inf for x<0
  and `rel_entr` matches the `xlogy(x,x)-xlogy(x,y)` convention (NaN for
  negatives); `lambert_w` reads NaN below -1/e with 25 Halley steps for
  the branch point; `i1`/`bessel_j0`/`bessel_j1` move to Miller's
  downward recurrence (fixes a small-branch aliasing bug that zeroed J1
  and a 0.5% large-x I1 error). Deliberate divergence: Miller agrees
  with the Taylor series to 1e-15 at x=8 while torch's cephes port reads
  ~1e-8 high, so the Bessel tolerance covers torch's noise; `zeta`
  reads NaN for x<1 (no analytic continuation) and Hankel leading-order
  remains past |x|=200 for J (both documented in the shims).
- Activations audit (libm erf upgrade + 7 new Python bindings, 2 new
  `ops::tests`, 10 new pytest in `tests/test_activations.py`): the CPU
  `Erf`/`Gelu` kernels now call `libm::erf`/`erff` (new direct `libm`
  dependency, already in the lockfile) instead of the A&S 7.1.26
  approximation (max error 1.5e-7, `erf(0)` returned 1e-9). `erf` and exact
  GELU now match reference values at 1e-12 in f64, forward and backward
  (GELU grad calls `.erf()`). New bindings: `relu6`, `hardtanh`,
  `log_sigmoid`, `softshrink`, `hardshrink`, `elu`, `erf` (torch defaults;
  `softplus(beta)` already covered `softplus_beta`). Deliberate divergence:
  the f32 GELU kernel evaluates erf in f64 internally, so at x=-2 it lands
  ~500x closer to the true value than torch's pure-f32 gelu; pytest
  tolerance (2e-5) covers torch's own f32 noise. GPU (WGSL) kernels keep
  the f32 approximation (documented, single-precision only).
- Autograd engine edges (1 fix + 5 new `tests/autograd.rs` tests): `grad()`
  with duplicate inputs now returns a gradient per entry like torch (the
  remove-based result lookup starved every repeat after the first to `None`).
  Also pinned: duplicate-output seed accumulation, `None` for off-graph
  inputs, `grad()` never writing `.grad` slots, `backward_with_grad`
  second-order flow, `no_grad`/`enable_grad` nesting restore +
  thread-locality, and hook-`None` passthrough.
- RNN audit (1 fix + first `nn::rnn::tests` module, 6 tests, torch-verified):
  `RNN`/`LSTM`/`GRU` `forward_seq` now validates initial states exactly like
  torch (`[num_layers, batch, hidden]`; extra layers used to be silently
  ignored). Also pinned: cell + 2-layer batch-first sequence parity against
  `torch.nn` with gate-distinct weights (catches any gate-order/wiring
  mixup), `h_n`-equals-last-output and zero-state/`None` equivalence, and
  fail-closed malformed input (non-3D, zero layers, empty sequences).
  (Engine-level; Python already serves RNNs via the native-op `nn.py` cells.)
- Quantization audit (3 fixes + first `quant::tests` module, 11 tests): Q4_0
  packing is now GGML split-half nibble order (byte j packs x[j] low +
  x[j+B/2] high), interchangeable with GGUF Q4_0 blocks and the in-repo GGUF
  writer — the old adjacent-pair packing produced bytes no GGML tool could
  read back under the same Q4_0 name (existing tests only pinned roundtrip
  values, never the layout, so they stay green); FP8 E4M3/E5M2 now round ties
  to even per OCP (was half-away); `ternary_matmul`/`PackedBitLinear` compute
  in the input dtype like the other quantized linears (was a hard
  mixed-dtype error on F64 input). Also pinned: Q8 exact bytes, FP8 spec bit
  patterns, FP8Tensor error bounds + non-finite safety, ternary radix-3
  packing, BitLinear hand-computed integer path, and AWQ/GPTQ/FP8/Linear
  forward tracking. (Engine-level only; quant has no Python bindings yet.)
- Distributions audit (4 fixes + 7 new `distributions::tests` + 3 new native
  classes + `tests/test_distributions.py`, torch-verified): `Normal`/`Laplace`
  now broadcast `loc`/`scale` like torch (the struct doc already promised
  broadcastable tensors but `new` rejected mismatched shapes; entropy is
  broadcast to the union batch shape too), `Categorical::sample` on 1D logits
  returns a scalar like torch (was `[1]`), and `Categorical::log_prob` works
  for 1D logits (gather rank mismatch used to error). New `sample()` shorthands
  on Normal/Laplace. The engine is exposed to Python for the first time as
  `oxideon.oxideon.Normal/Categorical/Laplace` (the only differentiable
  distribution path; top-level `ox.Normal` stays the 18-class numpy shim by
  the package's shims-win wiring rule). Also pinned: torch cross-checked
  closed-form values and `rsample` gradients.
- Sparse audit (2 new Python classes + `tests/test_sparse.py`, all
  torch.sparse-verified): the Rust COO/CSR core was already correct and is now
  exposed to Python for the first time (`ox.SparseCoo` with coalescing
  construction/`from_dense`/`transpose`/`to_dense`/`to_csr`, `ox.SparseCsr`
  with layout accessors/`transpose`/`to_dense`/differentiable `spmm`), pinned
  against `torch.sparse_coo_tensor` values, CSR row pointers, rectangular
  matmuls, and autograd gradients. Also documented the deliberate `coalesce()`
  divergence (oxideon drops exact zeros from storage; values are identical,
  only `nnz` differs).
- Sampling HF parity (3 fixes + 7 new `sampling::tests` +
  `tests/test_sampling.py`, all verified against HF `transformers` warper
  semantics): top-p now cuts after the first exceeding rank *including* rank 0
  (a lone dominant token collapses the nucleus to itself; the old `rank > 0`
  guard leaked a second token), warper order is now top-p *then* min-p (the two
  do not commute once renormalized), and degenerate knobs match HF
  (`top_k=0` disables filtering instead of collapsing to argmax, `min_p=1.0`
  keeps only the max, `top_p=0.0` keeps only the argmax). Also pinned:
  presence/frequency/bias math with out-of-vocab ids ignored, unfiltered-NaN
  deterministic fallback, and negative-temperature clamping to greedy.
- FFT audit (2 fail-closed guards + 7 new Python bindings + 5 new `fft::tests` +
  `tests/test_fft_bindings.py`, all torch-verified): `fft`/`ifft` now reject
  non-floating dtypes like torch instead of silently truncating the spectrum
  back to int, and `complex_mul`/`complex_abs` require a trailing complex-pair
  dim 2 instead of silently operating on the wrong lanes. New bindings expose
  the differentiable Rust path to Python (`fft2d`/`ifft2d`/`rfft`/`irfft`/
  `stft`/`complex_mul`/`complex_abs`; only `fft`/`ifft` were bound before).
  Also pinned: hand-computed DFT values, the Hann DC-bin == n_fft/2 property,
  and the deliberate `rfft` divergence (Rust returns the full two-sided
  spectrum for the internal `stft`/mel frontend; the numpy-backed
  `oxideon.fft.rfft` shim keeps true torch one-sided semantics).
- Safetensors spec strictness (2 fixes + 5 new `safetensors::tests`, spec-verified
  against the safetensors format doc): duplicate header keys are now rejected
  (the spec disallows them; the reader used to last-win silently) and
  `__metadata__` is validated instead of silently truncated (non-string values
  and malformed pairs used to stop parsing and return a partial map). Also
  pinned: bit-exact round trips for all 8 dtypes (signed zeros, inf, NaN
  payloads, integer extremes, i64 beyond 2^53), scalar/empty/strided tensors,
  escape-heavy tensor names, byte-for-byte spec layout (the 8-byte space
  padding the writer emits is explicitly permitted by the spec), and the
  `load_state_dict` contract (missing keys and shape mismatches error,
  unexpected keys are ignored like torch `strict=False`, dtype mismatches
  cast like torch `copy_`).
- Indexing torch parity (3 new `ops::tests` + 1 gradcheck + 1 pytest,
  torch-verified): `index_select`/`index_add`/`gather`/`scatter_add` are now
  exposed to Python and run on exact-width
  typed kernels instead of an f64 roundtrip (values beyond 2^53 survived
  selection corrupted; integer adds now wrap and bool adds OR instead of
  inheriting float-cast rules); add/scatter sources must match dtype like
  torch raises (the numpy `index_add` shim, which shadows the kernel to add
  `alpha`/inplace support, got the same check). Negative indices keep wrapping
  (torch rejects them here:
  deliberate, pinned by three sibling tests, consistent with
  take/advanced-indexing); non-bool `masked_fill`/`where` masks keep
  `!= 0` coercion (torch errors: deliberate, documented).
- Linalg values verification (1 new `linalg::tests`, all torch float64, no
  bugs found): `pinv` (full-rank + rank-deficient rcond), `lstsq`, 10th-order
  `matrix_exp` (small + scaled norms), `matrix_power` (0/3), `cond`, `kron`
  (1D/2D), general `cdist`, and negative-determinant `slogdet`/`det` now lock
  torch numerics. `matrix_rank`'s `tol<=0` fallback scales by 1e-6 while
  torch's default scales by eps (deliberate, documented).
- Linalg backward verification (1 new gradcheck sweep, 15/15 pass, no bugs
  found): `triangular_solve`, `slogdet`, `pinv`, `lstsq`, `matrix_exp`,
  `matrix_power` (+3/-2), `cond`, `kron`, `cdist`, `norm_p` (L1/L2/L3/inf)
  and `normalize` now lock their analytic gradients against central finite
  differences. Joins the existing cholesky/solve/inv/det/qr/eigh/svd
  gradchecks; every differentiable linalg op is now covered.
- Comparison torch parity (4 new `ops::tests`, all torch-verified): `cmp`
  compares I64 operands exactly instead of through f64 (2^53+1 no longer
  equals 2^53); `allclose` broadcasts singleton dims, treats same-signed inf
  as close and unequal infinities as never close, and errors on incompatible
  shapes instead of returning false. NaN ordering, bool operands,
  `logical_not` on float/int/bool, cross-dtype `allclose`, and empty inputs
  were already correct and are now locked. Mixed-dtype comparisons still fail
  closed (torch promotes: deliberate, matches `binary_op`); bitwise ops,
  `logical_and/or/xor`, and `isnan/isinf/isfinite` do not exist yet.
- Scheduler torch parity (5 new `optim::tests`, 5 new pytest, all torch-verified):
  `CosineAnnealingLR` oscillates past `T_max` instead of clamping at `eta_min`;
  `OneCycleLR` uses torch's float spans (`up = pct*total - 1`) with a cosine
  warmup ramp (a linear ramp only matches at the midpoint) and raises past
  `total_steps` instead of clamping; `ExponentialLR` compounds the live lr so
  external edits are picked up; `ReduceLROnPlateau` defaults to torch's
  relative `threshold_mode='rel'` (new opt-in `'abs'` mode). StepLR, SGDR,
  LinearLR, and warmup-cosine values were already correct and are now locked.
- Shape-manip torch parity (4 new `ops::tests`, all torch-verified): `split`
  of an empty dim yields one empty view (was zero pieces), `chunk` yields
  `chunks` empty views with fan-out capped, and nonzero-constant `pad` works
  on bool tensors via an equality mask (was an unsupported-dtype error).
  Uneven split/chunk tails, pad backward, diagonals, `repeat_interleave`,
  and `diag_embed` are now locked by tests.
- Dtype casting torch parity (4 new `dtype::tests`, all torch-verified):
  integer narrowing wraps low bits exactly beyond 2^53 (was losing precision
  through f64); float→int truncates with NaN/inf/overflow landing on INT_MIN
  like torch-CPU, and float→u8 wraps through i64 (negatives no longer
  saturate to 0). Bool/half/int→float edges were already correct and are now
  locked by tests.
- Matmul/einsum torch parity (5 new `ops::tests`, all torch-verified): `matmul`
  now accepts 1D vector operands (row/column semantics, batched included) with
  gradients flowing through the views; two-operand `einsum` backward zeroes
  off-diagonal gradient for repeated target labels (was broadcasting the
  diagonal value across the full matrix). Batch broadcasting, empty
  contractions, einsum values/implicit outputs, and 3-operand limits are now
  locked by tests.
- Embedding and RNG coverage (4 new `nn::layers::tests` + 4 new `rng::tests`,
  torch-verified): `EmbeddingBag` pools empty bags to zeros in every mode
  instead of erroring; decreasing offsets fail closed (torch silently
  computes garbage there: deliberate). Embedding selection/grads, dropout
  scaling/modes/mask-consistent backward, and RNG determinism, state
  roundtrip, fork isolation, and shuffle permutation are now locked by tests.
- Pooling torch parity (6 new `ops::tests`, all torch-verified): `max_pool2d`
  (and `max_pool1d` via it) now excludes padding from the max instead of
  letting zero-fill win over all-negative windows, for float and integer
  dtypes; both poolings reject padding beyond half the kernel like torch.
  `avg_pool2d` padding, nearest/bilinear upsampling, adaptive avg/max, and
  Lp pooling were already correct and are now locked by tests.
- Attention coverage (5 new `nn::transformer::tests`, all torch-verified): the
  tiled `flash_attention` kernel matches full-matrix SDPA across block sizes,
  causal/non-causal modes, KV prefixes, single queries/keys, and sharp
  softmax rows; causal masking provably never peeks at future KV; GQA-style
  head broadcasting works and is locked; empty KV returns zeros like torch.
  Empty queries now fail with a pointed error instead of a `cat` internal.
- Optimizer torch parity (4 new `optim::tests`, all torch-verified): Adam and
  RiemannianAdam bias correction is now per-parameter (a shared step counter
  under-corrected params that skipped steps, e.g. fresh params joining late);
  RiemannianAdam's pub-field length check covers the new counter. SGD with
  momentum/decay, Adam/AdamW/L2, beta1=0, RMSprop, Adagrad, Lion, and both
  gradient clippers were already correct and are now locked by tests.
- Loss coverage (4 new `nn::loss::tests` + 3 gradcheck cases, all torch-verified):
  `cross_entropy` values/grads, label smoothing (incl. the exact ε=0 collapse),
  `bce_with_logits` stability at ±50, focal (incl. the γ=0 reduction),
  `kl_div` batchmean, huber/mse/l1/cosine values, GAE/DPO/PPO hand values with
  PPO's detach routing, and fail-closed validation (empty batch errors where
  torch returns NaN: deliberate). No bugs found; all lock-ins.
- Transposed convolution torch parity (4 new `ops::tests` + 2 gradcheck cases,
  all torch-verified): `conv_transpose2d` no longer rejects valid
  `output_padding` + stride-1 + dilation configs (the shared `fold2d` size
  check was too strict); forward scatters over the input grid with zero fringe
  cells and backward narrows the unfolded gradient to the live subgrid, both
  verified against torch values and finite differences. Also validates
  stride/dilation positivity and torch's output_padding rule explicitly.
  `conv2d`/`conv1d`/grouped/depthwise/`unfold2d`/`fold2d` values were already
  correct and are now locked by tests.
- BatchNorm torch parity (5 new `nn::layers::tests`, all torch-verified):
  running variance now tracks the Bessel-corrected batch variance, training
  with a single value per channel errors like torch, `BatchNorm1d` eval mode
  and affine parameters broadcast per channel on 3-D input (previously
  errored or silently misaligned when `C == L`), and both norms reject
  wrong-rank input. `LayerNorm`/`GroupNorm`/`RMSNorm` values were already
  correct and are now locked by tests.
- Autograd engine validation parity (3 new `tests/autograd.rs` cases): `grad()`
  now rejects non-grad outputs, wrong-size seeds, and hook-resized gradients
  with pointed errors instead of silent `None`s or downstream noise, matching
  `backward_with_grad`. Double-`backward` accumulation, diamond fan-out,
  retain-grad opt-in, lenient/strict grad flags, detach sharing, and `no_grad`
  tracking are now locked by tests.
- Elementwise edge parity (6 new `ops::tests`, all torch-verified): `maximum`/
  `minimum` propagate NaN on CPU (f32/f64/f16/bf16) and WGSL instead of
  returning the non-NaN operand, ties split the gradient 0.5/0.5 and NaN
  inputs route full gradient to both sides; `clamp` gradients flow strictly
  inside `(min, max)` (zero at exact bounds); `pow`/`powf` give gradient 0
  at the `0^0` / `0^e (e >= 0)` corners instead of NaN/`-inf`. Singular
  `sqrt`/`ln`/`recip`/`rsqrt` grads at 0, `div` by zero, and `sign(NaN) = 0`
  were already correct and are now locked by tests.
- Sparse formats hardening (`sparse::tests`, 7 tests, new): `SparseCoo::new`
  and `from_dense` no longer silently drop NaN values, `to_csr` merges
  duplicate entries into canonical CSR, and CSR `to_dense` accumulates so
  hand-built duplicates agree with `spmm`. Coalescing, transposes,
  validation errors, `spmm` values/gradients, and empty shapes locked by tests.
- Generation sampling coverage (`sampling::tests`, 5 tests, new): greedy
  argmax/ties/empty errors, top-k support restriction with seed
  reproducibility, and repetition-penalty behavior. The vocab sort now uses
  `total_cmp`, so NaN logits rank first deterministically; greedy decoding
  deliberately skips NaN logits (best finite token wins) rather than
  argmax-first-NaN.
- `squeeze_axis` on a non-1 dimension now errors like PyTorch instead of
  silently returning the input unchanged (all in-repo callers squeeze
  known size-1 dims). Indexing coverage (`ops::tests`): hand-checked
  `index_select`/`index_add`/`gather`/`scatter_add` values, negative and
  out-of-bounds indices, empty selections, duplicate-index gradient
  accumulation, and `narrow`/`select`/`slice`/`flip`/`roll` behavior.
- `var` with `unbiased=True` over a single element now yields NaN like
  PyTorch instead of silently falling back to the biased (zero) estimate;
  `std`'s `1e-12` stability floor is now documented. Reductions coverage
  (`ops::tests`): hand-checked `var`/`std`/`cumsum`/`cumprod` values and
  gradients (including the division-free `cumprod` backward at zeros),
  empty-tensor `mean`/`sum`, and duplicate-axis divisor handling.
- Linear algebra hardening in `oxideon-core` (`linalg::tests`, 13 tests,
  new): `qr` rewritten from modified Gram-Schmidt to Householder
  reflections (orthonormal `Q` to machine precision, including
  rank-deficient inputs); the Giles (2008) QR backward had a dropped
  diagonal and a transposed solve — both branches now verified against
  finite differences. `svd` propagates NaN into `S` instead of silencing
  it to 0, `eigh` orders NaN eigenvalues last deterministically via
  `total_cmp`, and `pinv` NaN propagation is locked by tests.
- Non-finite max handling in `logsumexp`/`softmax`/`log_softmax`: the
  `x - max` shift turned any non-finite max into NaN (`inf - inf`).
  `logsumexp` now returns `+inf`/`-inf` for +inf-max/all-`-inf` slices
  (values verified against `torch.logsumexp`), `softmax` returns a uniform
  one-hot over `+inf` entries (PyTorch returns NaN there), and `log_softmax`
  returns `ln(1/k)`/`-inf` accordingly — all with zero gradients instead of
  NaN poison. NaN inputs still propagate, and all-`-inf` `softmax`/
  `log_softmax` stay NaN like PyTorch. Locked by `ops::tests::logsumexp_*`,
  `softmax_*`, `log_softmax_*` (values cross-checked against torch).
- NaN ordering in `topk`/`sort`/`argsort`: NaN compared equal to everything
  and lodged mid-slice; it now sorts as the largest value via `total_cmp`,
  verified value- and index-for-index against `torch.topk`/`torch.sort`.
- Removed the OpenAI-compatible serving layer (`oxideon.serve`,
  `tests/test_serve.py`) per owner direction: no HTTP/REST server, SSE
  streaming, metrics, adapter registry, or Modelfile export. The
  inference-engine kernels (paged KV, speculative decoding) are untouched;
  docs, `tools/efficiency_bench.py`, and the stub generator were updated.
- SSM width-consistency hardening (batch AP, 2026-11-05; closes the Batch AN
  open item): every `SsmBlock` sub-forward (Mamba, Mamba-2, PLaMo-2, gated
  delta-net, Kimi delta attention, lightning attention, LFM2 short conv)
  now validates its projection tensors and parameter vectors against the
  config before indexing them, so a width-swapped or truncated weight
  returns an error instead of panicking mid-scan (or silently truncating
  in a norm zip). Also from the deferred-module audit (batch AQ):
  `MixtureOfDepths::new` rejects NaN/infinite capacity instead of
  silently routing a single token. Head geometry is checked at the same entry
  (`dt_rank * d_state == d_inner` for KDA/lightning, `d_inner % dt_rank`
  for Mamba-2/PLaMo-2/gated delta-net, Mamba-2 heads divisible by groups),
  mirroring the GGUF config loader. LFM2 short-conv blocks built by the
  loader (`d_state`/`dt_rank` legitimately zero) forward again instead of
  being rejected by the blanket positivity check. 19 regression tests
  (`ap1`-`ap19` in `tests/robustness.rs`); clippy back to zero warnings
  under `-D warnings`.
- Eval answer checking (batch BG): `math_equal` no longer accepts a
  non-finite prediction (e.g. `\boxed{inf}` or `\boxed{1e999}`) against
  a finite reference — the magnitude-scaled tolerance became infinite
  and matched anything.
- Deep symbolic dimensions and schemas (batch AW): `SymDim::resolve`/
  `bind_dim` reject dimensions nested past 1024 (iteratively measured)
  instead of recursing, and `SymDim`/`SchemaType` clone, drop, and
  display iteratively — deep hand-built expressions no longer overflow
  the stack.
- `PropExpr` deep-formula safety (batch AV): cloning or dropping a
  deep hand-built formula recursed per connective and could abort the
  process (the prover entry points were already capped in batch AR);
  `Clone` is now an iterative post-order rebuild and `Drop` tears the
  tree down iteratively, mirroring `SymbolicNode` (batch AS).
- TDA Wasserstein distance (batch AU): identical persistence diagrams
  containing an essential class (death = +inf, the standard encoding)
  scored NaN instead of 0 (`inf - inf` on the matched pair), and NaN
  deaths poisoned the sum despite the module's NaN policy in
  `persistence()`. Endpoints now follow that policy and equal endpoints
  differ by 0; the filtration/persistence sort comparators are
  unwrap-free.
- Symbolic-GP tree recursion (batch AS): `SymbolicNode::depth`,
  `complexity`, and `to_formula` recursed per tree level, so a deep
  hand-built tree overflowed the stack (evaluation was already capped);
  all three are iterative now, and dropping a deep tree no longer
  recurses either. CI also runs the robustness suite in release
  profile, gating the batch-AR divergence class.
- Release-only stack overflow in the sequent prover (batch AR):
  `SequentProver` cloned the remaining formula subtree at every search
  rule, so a 15k-deep formula aborted release builds (debug passed)
  despite the search depth cap. Formula depth is now measured
  iteratively and rejected at the entry points before any recursive
  clone/search. Locked by the existing `prover_rejects_extreme_nesting`
  test, now green in both profiles.
- 5-hour hardening run, batches S-AD (2026-11-05, test-first with gates;
  `tests/robustness.rs` 372/372, workspace green, clippy clean, pytest 699
  passed): autograd rejects mismatched upstream/hook grads (was: index panic
  or silently wrong grads); `IrGraph::differentiate` re-sorts to topological
  order after unfusing and sweeps consumers-first (was: silent zero grads /
  unexecutable graphs); `eliminate_dead_code` and `plan_memory` handle
  non-topological/OOB hand-built graphs without panicking; iterative
  `trace_to_ir` lowering and iterative `TensorInner` drop (20k chains no
  longer SIGABRT); `SymDim::resolve` checked arithmetic; `gemm_fast`
  saturating output guard; poison-recovering locks in `HardwareManager`,
  `GpuRegistry`, `ComputeStream`; fallible `PrefillDecodeInterleaver::new`
  and `RingAttention::new`; `i128` index normalization plus loud Err for
  un-addressable slice dims; plus batches T-Z component fixes (quantum,
  ode, graphs, riemannian, wavelet) — see `BUG_HUNT_2026-11-05.md`.
- 5-hour hardening run, batches AE-AM (2026-11-05, test-first with gates;
  `tests/robustness.rs` 415/415): entry-guard caps for tamperable loop
  bounds (DEQ iters, spectral power iters, Euler steps, NAS
  generations/population, all hang-class pre-fix); fallible reserves for
  hostile GGUF counts (457 MB abort pre-fix) and fallible
  `GGMLType::tensor_bytes`, `VQCodebookStage::quantize_vec`,
  `initialize_population` (panics pre-fix); checked arithmetic for
  xavier fans, bilinear/quadratic patch products, tuning-grid counting
  (17.6 TB abort pre-fix); loud Errs for zero-fan conv bias NaNs,
  div-by-zero MoE token counts, OOB conv-weight/rank reads, empty-dim
  elastic slicing, width-swapped DiT unpatchify, and zero NAS
  populations; poison-recovering spectral-norm locks; iterative
  graph-stats traversal (SIGABRT pre-fix); conv3d backward
  `try_filled` buffers — see `BUG_HUNT_2026-11-05.md` (AL3/AM3
  retracted with evidence).
- 5-hour hardening run, batches AN-AO (2026-11-05, test-first with gates;
  `tests/robustness.rs` 442/442): single-dispatch entry guards for
  `SsmBlock` (zero dims underflowed `d_conv - 1`, trapped head/group
  divisions), checked channels/state/cache/output/dt products plus
  `try_filled` scratch buffers across all SSM forwards; shape checks for
  Titans gate loops and STDP/neuromorphic tables; fallible
  `adapt_stddp`, SPIR-V emitter, and NaN-dropout rejection at every
  fallible `Dropout::new` forwarder (LoRA/QLoRA/transformer/MLP,
  transitively transfusion); poison-recovering spiking locks; iteration
  caps for Poisson/NAS-style reserves; `grid_sample` corner clamping
  for infinite grids; empty-spatial `roi_align` rejection; checked
  pairwise `box_iou` — see `BUG_HUNT_2026-11-05.md` (AO1/AO2 defended
  with locks; gguf_ssm width-consistency left as documented OPEN).
- 5-hour hardening sweep (2026-11-04, four scoped workers plus integration): every
  in-scope BUG_HUNT item triaged in its file body; the genuinely open ones fixed
  test-first and locked with new regression suites
  (`crates/oxideon-core/tests/core_tensor_improve.rs`, `tests/nn_train_improve.rs`,
  `tests/engines_serve_improve.rs`, `tests/test_nn_train_improve.py`,
  `tests/test_pybind_improve.py`; 146 tests total).
- Core tensor/ops: new fallible `try_gemm_f32_strided`, `try_batched_gemm_f32/f64`
  returning `Err` on OOB strided views, short outputs, overflowing dims, and giant
  outputs; raw GEMM entry points now fail loudly with `# Panics` docs pointing at the
  `try_` variants; `SparseCoo::transpose` is fallible like CSR (13 tests).
- NN/training: `batch_indices` clamps 0 to 1, new validated `TensorDataset::try_new`,
  `make_blobs`/`train_test_split`/`k_fold_split`/`Normalize` guards;
  `accuracy`/`top_k` validation (fixed OOB slice panic on extra targets),
  `rmse`/`r2` empty-input errors, `r2` sklearn parity on constant fits;
  `gumbel_softmax` temperature validation, positive-finite `Normal`/`Laplace`
  scales; trainer rejects empty train loaders and reports `None` for empty val
  loaders instead of fake 0.0; batch losses return `Err` on N==0 instead of silent
  NaN; `CosineAnnealingWarmRestarts` torch parity; 7 loss modules honor
  `reduction="sum"` (24 Rust + 7 Python tests).
- Engines/serving: fixed engine step truncation, `block_size==0`, seeded-sampling
  repeat draw, ZeRO3 `world_size==0`, DDP partial grads, quant 4x scalar unwrap,
  BitLinear validation, and k-diffusion degenerate sigmas; poison-recovering locks
  at 29 sites in `quant_kv.rs`/`async_pipeline.rs` (39 tests).
- Python bindings/docs: native negative-step slicing (`t[::-1]`, fuzz-verified
  against CPython on 60,840 cases); 26 doc blocks corrected to real API signatures
  with a 57-case doc runner; `tools/gen_stubs.py` now generates
  submodule/`__all__` coverage from runtime ground truth (140/140, +30 exported
  names); stale `v[::-1]` ValueError assertion in
  `tests/test_python_ergonomics.py` updated to the new behavior (63 tests).
- `Tensor.is_contiguous()` reports real layout contiguity (torch parity) via a native binding to
  `Tensor::is_contiguous`; the Python patch layer previously overwrote it with `lambda self: True`, so
  transposed views wrongly reported contiguous (`tests/test_is_contiguous.py`).
- Fuzz children run in a fresh `$FUZZ_OUT/scratch` directory instead of inheriting the caller's working
  directory, so fuzzed file calls (`BPETokenizer.save`, ...) can no longer drop pool-named junk files
  (`a`, `f32`, ...) into the repo (`tests/test_fuzz_hygiene.py`).
- Edge-value fuzzing (`tools/edge_fuzz.py`, 4638 APIs, zero panics): absurd structural repeat counts in pure-Python
  constructors now fail fast with `ValueError` instead of hanging (`AlibiPositionalBias`, `AudioResidualVQ`,
  `CapacityConstrainedMoE`, `RealNVP`, `GraphTransformer`, `PagedKVCache`, `MoELayer`, both `VisionTransformer`s,
  `MiniGPT`/`TransformerLM`, `CausalTransformerDecoder`; the limit mirrors `oxideon_core::limits::MAX_REPEATS`).
- Single-buffer sizes above 2**31 elements are refused up front (mirrors `limits::MAX_ELEMENTS`): the eight window
  functions, `SpatiotemporalRoPE`, `RotaryEmbedding`, `MFCC`, the diffusion schedulers (including `set_timesteps`),
  `add_self_loops`, `build_sliding_window_mask`, and `topk`. This also closes silent wrong results where
  `numpy.arange` wraps to an empty array for bounds at or past 2**63 - 1.
- `AlibiPositionalBias.forward` and `SpatiotemporalRoPE.forward` validate their size arguments with the same
  element limit, and `stft`/`istft` reject `hop_length < 1` (a zero hop stalled the frame loop forever).
- Triaged the remaining fuzz findings as expected cost, not bugs: giant-but-finite constructions (e.g. 1000-wide
  1000-layer recurrent stacks, 4 GiB convolutions) complete or fail through the fallible allocators,
  `MiniGPT.generate(1000)` terminates (~59 s in a debug build), and the single abort is a 4 GiB allocation
  failing under the fuzzer's 8 GiB address-space cap.
- Diffusion schedulers reject non-positive timesteps with `ValueError` (`nn._check_positive`): `set_timesteps(0)`
  previously raised a bare `ZeroDivisionError`, and negative counts silently produced empty schedules
  (`DDPMScheduler`, both `DDIMScheduler`s, `EulerDiscreteScheduler`, `DPMSolverMultistepScheduler`).
  `set_timesteps` also rejects counts above `num_train_timesteps` (a zero stride silently produced all-zero
  schedules).
- `RotaryEmbedding` (`new`, `from_fn`, `new_yarn`) and `SinusoidalPositionalEncoding` return errors instead of
  panicking on overflowing table sizes (`max_seq_len * dim / 2`); huge-but-valid sizes hit the element limit and
  allocation failures surface as errors through the fallible helpers (`tests/robustness.rs`).
- `GGUFDecoderBlock::new` and the full Q/K-norm path in `GGUFCausalLM::from_config` use checked `heads * head_dim`
  products (they panicked on overflow); the per-head product is now computed lazily inside the `Full` branch, and
  `QkNorm` is re-exported from `oxideon::nn` (`tests/robustness.rs`).
- `topk` rejects negative `k` with `ValueError` (`nn._check_non_negative`); it previously returned a silent empty
  result, while `k = 0` keeps returning empty (PyTorch parity).
- `DeviceMesh::new` uses a checked node count (overflowing shapes panicked) and `DeviceMesh::homogeneous` additionally
  honours the repeat limit instead of attempting absurd allocations (`tests/robustness.rs`).
- `ox.grad(tensor)` without `inputs` raises a clear `TypeError` naming the missing argument instead of
  `'NoneType' object is not iterable`; removed a dead `checkpoint` assignment shadowed by the submodule import.
  Both live in the `tools/gen_stubs.py` template (`__init__.py` is generated), and the generated file now says so.
- All eight window constructors match PyTorch at degenerate sizes: length 1 returns `[1.0]`, length 0 returns empty,
  and negative lengths raise `ValueError` (`nn._check_non_negative`); lengths 0 and below previously returned `[1.0]`.
- `topk` rejects out-of-range `k` and `dim` with descriptive `ValueError`s (previously bare `IndexError`/`AxisError`)
  and honours `sorted=False` (previously always returned sorted).
- `__array_interface__` no longer returns a dict pointing into a dropped temporary (use-after-free whenever NumPy
  consumed it, e.g. `np.asarray(t)` returned garbage); it now describes tensor storage directly (zero-copy,
  read-only, correct shape/strides/offset), with honest errors for bf16/non-host/big-endian cases. The latent bug was
  exposed by the `numpy()` speedup (`tests/test_interop.py`).
- `from_numpy` honours non-native byte order instead of silently decoding garbage (`dtype.name` drops the order, so
  big-endian input now converts to native order first) (`tests/test_interop.py`).
- The safetensors header reader resolves `dtype`/`shape`/`data_offsets` with a depth-aware scan instead of first-match
  text search, so extra nested fields can no longer shadow the real ones (foreign forward-compat).
- `CpuStorage::from_slice` is fallible: an allocation too large for memory is now an error instead of a process abort
  (found by `tools/edge_fuzz.py` as `Conv3d(64, 64, 64, 64, 64)` aborting; OOM itself is not CI-testable, the capped
  fuzzer run is the check).
- Triaged: multi-GB layer inits (`Conv3d`/`SpectralConv1d` at 4 GB per weight) remain slow-flagged by the fuzzer past
  5 s in unoptimized builds; that is inherent bandwidth/compute cost of touching gigabytes (release builds run ~4x
  faster), and the calls now succeed where memory fits or fail with a clean out-of-memory error - never abort or hang.
- `GatedGraphConv`, `GraphSAGE`, and `GINConv` (`gnn_ext`) are rewritten in differentiable oxideon ops: every forward
  previously round-tripped through NumPy, detaching the autograd graph (`backward()` errored on gated/SAGE outputs;
  GIN silently trained only its last layer). Forward values are pinned to PyTorch-validated goldens
  (`test_gnn_ext_forward_values`, `test_gnn_ext_layers_stay_differentiable`); integer adjacency now raises the
  standard matmul dtype error like every other op instead of promoting through NumPy.
- `ox.cat` delegates to the new differentiable native `Tensor.cat` (forward values unchanged); scalars, mixed
  dtypes, and non-tensor inputs keep the exact NumPy path (the native kernel needs rank >= 1 and coerces to the
  first dtype, while NumPy promotes). The carve-out is an explicit loop: this module shadows builtin `all`/`any`
  with tensor ops, so the first version's `all(...)` always took the native path (`test_cat_preserves_i64_exactness_and_gradients`).
- Core `Tensor::cat` stitches typed storage instead of round-tripping through f64, which silently rounded large
  i64 values (2**60 + 1 came back as 2**60); mixed-dtype inputs now coerce with exact per-element casts.
- `ox.stack` delegates to the new differentiable native `Tensor.stack` (it detached gradients before); mixed
  dtypes and non-tensor inputs keep the NumPy path (`test_stack_stays_differentiable`).
- `elu`/`selu`/`celu` no longer emit a NumPy overflow warning on huge positive inputs: the overflowed lanes are
  discarded by the `where`/`minimum` select, so the `exp` runs under `np.errstate(over="ignore")` with bit-identical
  results (`test_r2_functional_preserves_f64_precision` now asserts warning-free execution).
- Added the missing `LICENSE-MIT`/`LICENSE-APACHE` texts (the README license links pointed at nothing); maturin now
  bundles them into the wheel. README verification counts corrected to observed values (Rust suite 1105 passed;
  pytest 699 passed + 1 scipy skip), and `.gitignore` covers generated `runs/`, `mini_models_out/`, `.pytest_cache/`,
  `a.out`.
- Bug-hunt batch H (data/trainer/tuning/tokenizer/quant; 9 fixes, 10 tests, all failing pre-fix): `make_moons`/
  `make_circles`/`make_blobs` refuse giant sizes up front (one abort observed pre-fix); `TensorDataset::get`/
  `get_batch` reject huge indices instead of wrapping to negatives (silent last-row select); `Trainer::fit` and
  `Study::random_search` cap epoch/trial counts; `BpeTokenizer::encode_batch_padded` caps `max_len`;
  `QuantizedTensor::check_layout` requires exact scales/data lengths and no longer rejects `quantize()`'s own
  partial-block Q4 output; `compression_ratio_f32` uses a checked product; `TernaryPackedTensor::unpack` validates
  its element count (`BUG_HUNT_2026-11-05.md`).
- Bug-hunt batch I (sparse/graph/fft; 3 fixes, 3 tests, all failing pre-fix): `SparseCoo::transpose` sums
  duplicate hand-built entries like `new`/`to_dense` instead of keeping one; `analyze_graph` uses saturating
  arithmetic for meta-scale tensors; `Tensor::stft` survives giant `hop_length` without overflowing
  (`BUG_HUNT_2026-11-05.md`).
- Bug-hunt batch J (metrics/explain; 4 fixes, 4 tests, all failing pre-fix): `confusion_matrix`/
  `precision_recall_f1` are now fallible and refuse absurd class counts (were an abort vector; no in-repo
  callers); `auroc` normalizes over the truncated pairs it scores; `smooth_grad`/`integrated_gradients` cap
  sample/step counts (`BUG_HUNT_2026-11-05.md`).
- Bug-hunt batch K (audio/video; 4 fixes, 5 tests, all failing pre-fix): `ResidualVectorQuantizer`
  validates stage/config consistency up front; `RotaryEmbedding3D::forward` re-checks its partition
  invariant; `continuous_timestep_embedding` caps its table; `RectifiedFlowVideoDiT::sample_euler` caps
  step counts (`BUG_HUNT_2026-11-05.md`).
- Bug-hunt batch L (vision/vision_ops; 14 fixes, 17 tests, all failing pre-fix): `MiniImageDataset`
  caps generate sizes, validates its layout in `split`/`batch`, and saturates batch arithmetic; vision
  `forward`/`encode`/`decode` reject wrong ranks; `predict` slices by actual logit width; all three vision
  `fit` fns and `interpolate` cap counts; `affine_grid`/`roi_align` cap sizes, sampling, and reject NaN
  coordinates (`BUG_HUNT_2026-11-05.md`).
- Bug-hunt batch M (diffusion/rl; 5 fixes, 6 tests, all failing pre-fix): `DDPMScheduler::new` and
  `DiffusionForcingScheduler::new` are now fallible (zero rejected, counts capped); scheduler methods
  validate table consistency; `timestep_embedding` caps its table; EDM sigma building and rectified-flow
  euler cap step counts (`BUG_HUNT_2026-11-05.md`).
- Bug-hunt batch N (serving/distributed/long-context; 17 fixes + 1 backstop pin, 19 tests, all
  failing pre-fix): `BlockAllocator::new` is fallible with a block ceiling; `PagedKVCache` methods
  check store/dim agreement; Eagle draft depth is re-capped and token ids must fit `i64` (also on the
  chunked-prefill path); interleaver ack counts saturate; parallel-linear, ZeRO, DualPipe, and
  parallel-3d rank/count fan-outs are capped; Krum/Bulyan byzantine math saturates into its documented
  fallbacks; FlashAttention-2 backward rejects low-rank inputs; ring attention rejects tampered ranks
  (`BUG_HUNT_2026-11-05.md`).
- Bug-hunt batch O (optim/sampling/trainer/serialize; 9 fixes, 9 tests, all failing pre-fix):
  warmup-cosine construction saturates giant warmups; Lookahead tolerates a zeroed interval;
  Stiefel retract handles empty inputs; RiemannianAdam checks state agreement and saturates its
  counter; GGUF dequantize/fused-gemv validate tampered shapes before reserving; GGUF serialization
  rejects zero alignments and fallibly pads giant ones (`BUG_HUNT_2026-11-05.md`).
- Bug-hunt batch P (nn/indexing/backend; 14 fixes, 14 tests, all failing pre-fix): `SparseMoE`
  validates tampered config up front; `apply_view`/`view_scatter` bounds-check hand-built specs;
  generate/speculative/beam-search token budgets and beam widths are capped (quadratic candidate
  reserve is fallible); `KVCache::new` is fallible; GQA re-checks head divisibility; empty views
  with nonzero offsets no longer panic the CPU slicing fast paths
  (`BUG_HUNT_2026-11-05.md`).
- Bug-hunt batch Q (reasoning; 14 fixes, 15 tests, all failing pre-fix): the sequent prover
  hard-caps recursion depth; swarm debate, MCTS search, CRC grid search, and Bellman-Ford rounds
  are capped (`search` is now fallible); conformal constructors reject NaN alpha; normalized
  prediction rejects dispersion mismatch; CRC loss validates its bound (now fallible); SCM
  sampling checks its buffer size and graph invariants and wraps its seed stream; NBF/PGN
  validate graph endpoints (`BUG_HUNT_2026-11-05.md`).
- Bug-hunt batch R (reasoning cont.; 9 fixes, 9 tests, all failing pre-fix): symbolic-GP
  evaluation hard-caps nesting depth and fit checks target length plus population/
  generation budgets; topos verification is fallible with a lattice cap; Solomonoff
  construction rejects NaN lengths; ILP inference checks tampered rules, deductive
  closure is capped, and the loss uses a checked entity product
  (`BUG_HUNT_2026-11-05.md`).

### Performance
- Rust `Tensor::topk(..., sorted=false)` (the flag was previously ignored
  and always fully sorted) now selects via quickselect in O(n) average:
  1M-element top-10 drops from ~171 ms to ~11 ms (~15x) with identical
  winning sets. The `sorted=true` path is unchanged.
- The seven pure-Python window loops (`hann` through `exponential`) and the ALiBi slopes are vectorized with NumPy:
  a million-sample window builds in about 0.05 s, and legitimately large windows keep working with no repeat-count
  cap. Output matches the previous implementation bit-for-bit (`kaiser` now computes in float64, slightly more
  accurate than the old float32 intermediates).
- `Tensor.numpy()`/`to_numpy()` copies contiguous host storage as raw little-endian bytes in one memcpy (one
  allocation) instead of gathering element-by-element through four buffers: exporting 1M float32 drops from ~44 ms
  to ~0.2 ms, and every NumPy-backed primitive benefits. Strided views, f16/bf16/bool, and non-host backends keep the
  generic path (`tests/test_interop.py`).
- `topk` selects via `argpartition` (linear) instead of a full sort: 1M-element top-10 drops from ~300 ms to ~2 ms
  (~175x with the `numpy()` fast path; ~4x from partitioning alone).
- `from_numpy` reinterprets the contiguous buffer instead of converting element-by-element: importing 1M float32 drops
  from ~31 ms to ~5 ms.
- `rand`/`randn`/`rand_uniform` move the filled buffer into storage instead of copying it (peak memory halved on big
  inits; a 4 GB init no longer holds 8 GB transiently).
- `BPETokenizer.encode` applies merges rank-greedy (Sennrich reference algorithm, O(L^2) per word) instead of
  scanning the whole merge list per word: encoding 2000 words with ~2000 merges drops from ~716 ms to ~8 ms (~88x),
  and the gap widens with vocabulary size. Output is differentially pinned to the old sequential algorithm over
  realistic, exhaustive small-alphabet, and adversarial merge sets (`test_bpe_encode_matches_sequential_reference`).
- `BPETokenizer.train` micro-opts (hoisted lengths, short-word skips, first-maximum loop): ~14% faster with
  byte-identical merges and vocabulary (differentially verified; `test_bpe_train_is_deterministic` pins determinism).
- CPU elementwise/unary/binary/compare/where/reduce/copy kernels stay serial in debug builds and use coarse
  parallel tasks in release (`cpu::use_parallel` + `PAR_CHUNK_MIN`: debug always serial, release parallel above
  256k elements with >=256k elements per task and in-place writes; previously fine-grained parallel above 16k
  everywhere). Debug 16k/32k/131k-element f32 adds go 1.65/4.43/5.87 ms to 0.17/0.34/1.31 ms; release 1M/4M adds
  go 2.9/3.7 ms to 0.47/1.69 ms (serial: 0.6/2.4 ms) and 1M/4M sums go 0.88/1.2 ms to 0.26/0.89 ms - parallel now
  ties or beats serial everywhere instead of cliffing (`large_elementwise_matches_exact_reference` covers the
  serial path in debug and the parallel path in release).
- `Tensor.clone()` copies storage once via a typed copy (`memcpy` when contiguous) instead of gather-to-`Vec` plus
  rebuild: cloning 128k f32 drops from ~1.9 ms to ~0.014 ms, matching PyTorch (`test_clone_is_an_independent_copy`).
- safetensors save/load move tensor bytes in bulk on little-endian hosts instead of converting element-by-element:
  saving 1M float32 drops from ~31 ms to ~3.8 ms and loading from ~33 ms to ~1 ms. Big-endian keeps the portable
  loops, and an exact all-dtype roundtrip test (f32/f64/i32/i64/u8/bool/f16/bf16, huge i64, strided views) pins the
  behavior (`test_save_load_roundtrip_all_dtypes_exact`).

## 0.0.1

First semver version.

### Changed
- Split into a workspace: `oxideon-core` (tensors, autograd, nn, ...), `oxideon-engines` (science / physics / crypto-demo /
  synthesis chain) and the `oxideon` facade, which re-exports both and hosts the PyO3 bindings. Public paths are unchanged.
- Python bindings: the mechanical engine wrappers are generated by `py_engine!` / `millennium_engines!` (about 15k fewer lines);
  the Python API surface is identical.
- `python/oxideon/__init__.pyi` is generated by `tools/gen_stubs.py` with real signatures (properties, static methods, types);
  classes no longer fall back to `__getattr__ -> Any`.
- `[profile.test]` uses opt-level 1 (the slowest test dropped from ~56 s to ~12 s).
- PyO3 0.23.5 -> 0.29.3 (fixes RUSTSEC-2025-0020 and RUSTSEC-2026-0177); the bindings use the new
  `detach` / `attach` / `cast` APIs, an explicit `#[pyclass(from_py_object)]` opt-in, and `Py<PyAny>`
  for the removed `PyObject` alias. `abi3-py39` support and the Python API surface are unchanged.
- `cargo doc` builds with zero warnings in every feature combination (default, `python`, `wgpu`, `avx512`):
  LaTeX math brackets are escaped and the broken intra-doc links now point at the real items.

### Performance
- **GPU (wgpu), matmul:** two register-blocked, shared-memory kernels replace the 16 x 16 kernel that computed one output per
  thread: 64 x 64 blocks with 4 x 4 outputs per thread, and for large products (at least 48 workgroups of 128 x 128) 128 x 128
  blocks with 8 x 8 per thread. `OXIDEON_GPU_MATMUL=small|large` forces one. Measured with only a scalar read back to the host:
  on an RTX 4060 the original kernel gave 0.85 / 0.91 / 1.08 / 1.09 TFLOP/s at n = 1024 / 2048 / 4096 / 6144 and the new ones
  give 2.5 / 2.9 / 4.0 / 4.0 (3.0x to 3.7x); on an L4 they reach 4.1 / 6.3 / 5.8 TFLOP/s at n = 2048 / 4096 / 6144 (the original
  kernel was not timed that way on the L4).
- **GPU, everything else a training step needs:** transposed, broadcast, sliced and flipped views are gathered on the device by one
  strided-copy kernel, so `A.t() @ B`, `x + bias` and `permute` no longer round-trip through the host. `sum`, `mean`, `max` and
  `min` over any set of axes run on the device (tree reduction, several passes for long rows). The element-wise kernels use a
  two-dimensional dispatch (no 16.7-million-element ceiling) and cover every unary op (GELU, ELU, softplus, erf, sign, rsqrt,
  floor, ceil, round, ... used to fall back to the CPU). `x ** y` with a negative base follows `f32::powf` (WGSL leaves it
  undefined). `tests/gpu_kernels.rs` compares every kernel with the CPU on awkward sizes and counts host copies to prove that a
  chain of GPU ops never leaves the device. Against the 12-core Xeon of a Colab runtime, an L4 runs 10M-element `add`, `exp`,
  GELU and `sum` 9.7x, 10.7x, 16.4x and 10.8x faster, `softmax` (4096 x 1024) 18.6x, transpose + relu (4096 x 4096) 13.1x and
  f32 matmul 3.1x (1024), 12.3x (2048) and 17.2x (4096) faster; below 1024 x 1024 the CPU is still faster (matmul 512: 0.49x).
- **CPU:** `f32` and `f64` products above 64 x 64 x 64 go through the `gemm` crate (AVX2/FMA; AVX-512 with the opt-in `avx512`
  feature, which needs Rust 1.89; a NEON path for Apple Silicon that is type-checked but was never run; multi-threaded with
  `rayon`), which takes the same signed strides, so transposed and flipped views still multiply without a copy. `f64` products
  used to run an unblocked scalar loop. The in-house kernels stay as the fallback (`--no-default-features`) and for small
  matrices. Cargo feature `fast-gemm` (on by default) turns the `gemm` crate on. f32 matmul 4096 x 4096 on a 12-vCPU Xeon:
  774 ms with the in-house kernels, 428 ms with `gemm` (1.8x), 311 ms with AVX-512 (2.5x); on a 16-thread i7-13620H the
  throughput at 1024 / 2048 / 4096 went from 268 / 376 / 430 to 395 / 565 / 648 GFLOP/s.
- `tools/ops_bench.py` and `tools/gpu_bench.py` time the operations a training step is made of on both devices.

### Added (found by running the library as a user)
- Python tensors now behave like NumPy / PyTorch tensors for everyday code: scalar arithmetic on either side (`x * 2.0 + 1.0`,
  `1.0 / x`, `x ** 2`), `abs`, element-wise comparisons, `len`, `[]` indexing and iteration, `float()` / `bool()`, `reshape([-1])`,
  `dim=` / `keepdim=` on `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `var`, `std`, plus `log`, `abs`, `neg`, `clamp`, `pow`,
  `flatten`, `size`, `from_slice`, `full`, `rand` and the module functions `cat`, `stack`, `where`, `einsum`
  (58 of 133 everyday snippets worked before, 105 after; `tests/test_python_ergonomics.py`).
- Python bindings for the pieces that were missing from everyday training code: the layers `Embedding`, `LayerNorm`, `Dropout`,
  `ReLU`; the optimizers `SGD` and `Adam`; 18 loss functions (`mse_loss`, `l1_loss`, `huber_loss`, `nll_loss`, `cross_entropy` with
  label smoothing, `binary_cross_entropy_with_logits`, `kl_div_loss`, `focal_loss`, `info_nce_loss`, `dpo_loss`, ...);
  `Tensor.clone()` and `Tensor.numpy()`. Layers (`Linear`, `MLP`, `Sequential` included) are now callable, `model(x)`.
  `ox.nn` and `ox.optim` re-export them PyTorch-style. Bad hyper-parameters raise `ValueError`.
- `oxideon::limits`: shared upper bounds for user-chosen work (`MAX_ITERATIONS`, `MAX_REPEATS`, `MAX_ELEMENTS`) with
  `check_iterations` / `check_repeats` / `check_elements` helpers.
- CI runs the tests on Linux, Windows and macOS, and installs one wheel per OS under Python 3.9, 3.10, 3.11, 3.12 and 3.13; the
  release workflow now tests the wheels it built before it publishes them.
- `tests/robustness.rs`, `tests/test_python_ergonomics.py` and `tests/synthesis_golden.rs`.
- `tools/gpu_probe.py` (compares wgpu results with the CPU, refuses software renderers), `tools/gpu_soak.py` (forward and backward
  against the CPU over many shapes, VRAM leak check, very large allocations), `tools/colab_l4.sh` (the wgpu tests, the wheel and
  the probe on a Colab GPU runtime), `tools/python_matrix.sh` (one wheel, tests under Python 3.9 - 3.13) and `tools/edge_fuzz.py`
  (the edge-value fuzz harness).
- `examples/finetune_mini.py`: pretrain then fine-tune a mini conv model, a tiny language model and a raw-tensor MLP (on the GPU when
  there is one, compared with the CPU run), with pass/fail checks that fine-tuning beats training from scratch.
- `examples/true_false_news.py`: TRUE/FALSE news verdicts from the pretrained `fake_news_detector` weights
  (trains them when missing), with `tests/test_true_false_news.py`.

### Fixed
- `GGUFCausalLM` adds `muse-glimmer`; 113 reference tests agree with llama.cpp.
- `GGUFCausalLM` adds `beam` (NoPE full layers with a power-law query temperature, per-head attention-gate bias, weighted
  residual `h/sqrt(i) + r*sqrt((i-1)/i)`, sigmoid MoE that ignores the group keys as llama.cpp does); 114 reference tests agree.
- `GGUFCausalLM` adds `talkie` (weightless norms, Q/K norm after RoPE with a per-head Q gain, a normalized embedding that
  every layer adds back scaled by `layer_output_scale`, only the `ffn_down` scale applied); 115 reference tests agree.
- `GGUFCausalLM` adds `hrm_text` (low and high stacks re-run with shared weights, one KV slot per pass, the Q/K/V/O scales
  taken per slot rather than per weight set, as llama.cpp loads them); 116 reference tests agree.
- `GGUFCausalLM` adds `k2-horizon` (grouped main norms, per-head Q/K norm weights, `log2(1+2^x)` attention gate with its own
  scale, sigmoid MoE with a shared expert, and routed value experts on MoE layers); 118 reference tests agree.
- `GGUFCausalLM` adds `muse-spark` (per-head attention-output norm, `alpha`/`beta` norm-preserving residuals, weightless
  post norms, and a latent sigmoid MoE with pre- and post-expert norms on every layer); 119 reference tests agree.
- `GGUFCausalLM` adds `cogvlm` (text path; the stored `rope_freqs` is ignored because its graph applies none), `qwen3tts`
  (Qwen3-VL's graph) and `pockettts` (LayerNorm backbone, interleaved RoPE, ungated GELU FFN); 122 reference tests agree.
- `GGUFCausalLM` adds the diffusion LMs `dream`, `llada`, `llada-moe` and `rnd1`: their stacks are Qwen2 / Llama /
  Qwen3-MoE blocks with a bidirectional attention mask, so they are meant for one full-sequence forward pass (there is no
  denoising loop); 126 reference tests agree.
  `refact` with experts is deliberately not matched: llama.cpp's Refact graph has no MoE branch and runs a degenerate FFN.
- `GGUFCausalLM` adds `plamo2`: attention layers (per-head QK weights, post norms) interleaved with PLaMo-2 Mamba layers
  (per-head `[z|x]` split, RMS-normed B / C / dt, head-wise `dt_proj`); 127 reference tests agree. Fixed along the way:
  post-attention / post-FFN norm weights were never loaded for SSM+FFN layers (they stayed identity).
- `GGUFCausalLM` adds `kimi-linear`: Kimi Delta Attention layers (per-key-channel decay, separate q / k / v
  convolutions, sigmoid-gated RMS-normed output) interleaved with RoPE-free MLA layers and DeepSeek-style MoE; 128
  reference tests agree. Verified on whole prompts (llama.cpp's chunked path); cached token-by-token decoding agrees
  with this crate's full forward, but llama.cpp's single-token path was not compared.
- `GGUFCausalLM` adds `bailingmoe3`: the same delta-attention layers with a bounded ("safe") decay gate and full-width
  `f_a` / `g_a`, MLA layers with RoPE and a per-head sigmoid output gate, and `swiglu_clamp_exp` / `_shexp`-limited
  experts; 129 reference tests agree. The NextN / MTP draft layers are not run.
- `GGUFCausalLM` adds `kimi-k3`: Kimi delta-attention layers with a full-rank output gate and a bounded decay, gated
  RoPE-free MLA, the SITU activation (dense FFN, experts, shared expert), a latent MoE with a norm between the experts and
  the up-projection, and cross-layer attention residuals (a per-token stack of banked layer inputs mixed by a softmax
  over RMS-normed scores); 130 reference tests agree.
- `GGUFCausalLM` adds `dots3note`: MLA with a normed rotary key, a per-head sigmoid output gate and sliding-window layers
  (the window layers must share the full layers' MLA sizes; others are refused); the lightning indexer is not run, so
  contexts beyond its `top_k` are refused as for `deepseek32`; 131 reference tests agree.
- `GGUFCausalLM` adds `minimax-01`: lightning-attention layers (per-head exponential decay, per-head `q | k | v` with a
  SiLU, norm over all heads, sigmoid gate) mixed with RoPE softmax layers, MoE FFNs, and its residual of
  `residual_scale * normalised input`; 132 reference tests agree. Recurrent layers follow `attention.recurrent_layers`
  (files without it fall back to this crate's default interval, not llama.cpp's 8).
- `GGUFCausalLM` adds `minimax-m3`: per-head QK-normed GQA, SwiGLU-OAI dense / shared / routed FFNs and sigmoid routing.
  Its block-sparse attention is exact while the context fits in `indexer.top_k * indexer.block_size` tokens (longer
  contexts are refused); 133 reference tests agree.
- `GGUFCausalLM` adds `hy_v4`: MLA with attention sinks and a head output gate, iHC hyper-connections (several residual
  streams reduced and redistributed by learned sigmoid gates around every attention and FFN, collapsed by an output
  head), and experts whose SwiGLU clamps the gate before the SiLU. The lightning indexer is not run (dense attention
  up to `indexer.top_k` tokens); 134 reference tests agree.
- `GGUFCausalLM` adds `glm5-next`: Kimi delta-attention layers with a bounded decay gate, rope-free nope-only MLA
  layers, manifold-constrained hyper-connections (an `n x n` stream-mixing matrix made doubly stochastic with Sinkhorn
  iterations, streams averaged at the output) and gate-first clamped sigmoid MoE. The k-pool indexer is not run (dense
  attention up to `indexer.top_k` tokens) and the NextN draft layer is not run; 135 reference tests agree.
- `GGUFCausalLM` adds `qwen4exp`: a gated-DeltaNet (sigmoid output gate) / gated-attention hybrid in which low-rank
  hyper-connections replace every norm (including the output norm), per-layer n-gram hash embeddings (token history and
  conv history are carried in the cache's last slot), and QSA attention that is exact (dense) up to `indexer.top_k`
  tokens; the NextN draft layer is not run; 136 reference tests agree.
- `GGUFCausalLM` adds `deepseek4`: shared-KV attention (a per-head RMS-normed query, partial pair-wise RoPE, per-head
  sinks, a sliding window, compressed blocks of ratio 4 (overlapping softmax-pooled windows) or 128, and a rotated-back
  grouped low-rank output projection), iHC with Sinkhorn mixing, sqrt-softplus routing and gate-first clamped experts;
  137 reference tests agree. It also runs the compressed-block indexer (scores blocks with its own overlap-pooled keys
  and keeps the `top_k` best), hash-routed expert layers (a `[vocab, used]` table by token id) and YaRN scaling on the
  compressing layers. Those, and the ratio-128 pooling, are checked against llama.cpp with 200-token prompts on variants
  of the reference model (`tools/ggml_reference/deepseek4_variants.py`: `indexer.top_k = 3`, a hash-routed layer, YaRN
  factor 4); the MTP layer is not run.
- The lightning indexers of the token-level sparse-attention MLA models now run: `deepseek32`, `glm-dsa`, `dots3note`
  and `hy_v4` score every earlier token with `sum_h w_h relu(q_h . k)` over their own cached keys and attend only the
  best `indexer.top_k` (layers without an indexer of their own reuse the previous one's choice, as llama.cpp does), so
  these models no longer refuse contexts beyond `top_k`. The scores follow llama.cpp's Hadamard-rotated queries / keys and
  its f16 key cache, which keeps near-tied picks identical; checked against llama.cpp on a 100-token prompt with
  `indexer.top_k = 4` (`tools/ggml_reference/topk_variants.py`).
- The pooled / block indexers now run too (superseding the "dense up to `top_k`" notes above): `glm5-next` scores pools of
  `indexer.kpool` keys (a per-channel softmax over the pool's gate scores plus a positional bias) and attends the best
  `top_k / kpool` pools plus the incomplete tail; `minimax-m3` scores every position per KV group with its own RMS-normed,
  rotated indexer keys, takes the best position of each `block_size` block, forces the latest `local_blocks` blocks and
  attends the best `top_k` blocks; `qwen4exp` (QSA) pools the mean of each block's raw keys (normed, then rotated to the
  block's first position) and sums the rectified head scores. Checked on a 100-token prompt with reduced budgets; the
  minimax / qwen4exp variants scale the attention projections up (`OXIDEON_TENSOR_GAIN` in `topk_variants.py`) because the
  stock tiny models barely let attention reach the logits, which also exposed a bug: `qwen4exp` added the file's Q/K/V
  biases although llama.cpp's graph does not.
- `GGUFCausalLM` adds `apertus` (xIELU activation with per-layer parameters), `bitnet` (RMS "sub norms" over the attention
  output and the FFN's inner activation), `plm` (MLA with a single Q projection and a ReLU² FFN), `chameleon` (per-head
  LayerNorm Q/K norms, the image-token logits hidden; its `swin_norm` layout follows the reference implementation, not
  llama.cpp, whose graph feeds the FFN the normed attention output), `gemma3n` (four AltUp streams, LAuReL, per-layer
  input embeddings, activation sparsity on the first ten layers, K/V shared from layers 18/19 by layers 20 and up),
  `rwkv6` and `rwkv7` (new `nn::gguf_rwkv`: token shift, per-head WKV matrix state, group norm; the delta-rule recurrence
  and value residual of RWKV-7; also the distilled `arwkv7` and `rwkv6qwen2` variants with RMS norms, a SwiGLU FFN,
  one token-shift slot and, for QRWKV6, gated linear attention with grouped heads), `step35` (per-layer head counts and rotary widths, a head-wise attention gate on some
  layers, sigmoid-routed experts with a shared expert and per-layer SwiGLU clamps; NextN layers not run) and `gemma4`
  (per-layer head and rotary widths for the sliding-window layers, proportional RoPE through `rope_freqs`, values taken
  from the keys on layers without a V projection, the last `shared_kv_layers` layers reusing earlier K/V, per-layer input
  embeddings and output scalars, and MoE layers with softmax-routed GELU experts next to the dense MLP). llama.cpp's test
  A new `nn::gguf_bert::BertEncoder` runs `bert` encoders (post-LayerNorm blocks, learned positions, token-type row 0,
  mean / CLS / last / no pooling; also `nomic-bert`: RoPE, fused QKV, SwiGLU; and `jina-bert-v2`: ALiBi, Q/K norms, GEGLU) and matches `llama_encode` embeddings to 1e-4 on tiny models.
  tool cannot write models for these, so
  `tools/ggml_reference/tiny_models.py` derives tiny ones from a reference model's key/value block; their references are
  made with an f32 K/V cache and no flash attention (`LLAMA_LOGITS_KV_F32=1 LLAMA_LOGITS_NO_FA=1`), because llama.cpp's f16
  attention rounding would otherwise dominate the comparison. The parity tests can emulate ggml's f16 GELU lookup table
  (`gguf_model::emulate_ggml_gelu_table`, off by default).
- `nemotron_h_moe` sparse layers: ungated ReLU² experts, optional latent projection and an ungated shared expert on the full-width input (unit-tested against scalar arithmetic; the reference model has only dense layers).
- `GGUFCausalLM` adds `deepseek32` and `glm-dsa` for contexts up to the sparse-attention indexer's `top_k` (longer ones are refused); 111 reference models agree with llama.cpp.
- `GGUFCausalLM` adds the gated-DeltaNet architectures `qwen35`, `qwen35moe` and `qwen3next`; 109 reference models agree with llama.cpp, and cached decoding matches the full forward for two of them.
- `GGUFCausalLM` adds `lfm2` and `lfm2moe` (gated short convolution); 106 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `jamba`; 104 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `falcon-h1` (parallel attention + Mamba-2); `new_cache()` may now hold extra slots, so use it rather than building a cache by hand; 103 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `granitehybrid` (Mamba-2 or attention mixer plus FFN per layer); 102 reference models agree with llama.cpp.
- `GGUFCausalLM` adds the hybrid `nemotron_h` / `nemotron_h_moe` layouts (per-layer Mamba-2 / attention / FFN mixers); 101 reference models agree with llama.cpp.
- `GGUFCausalLM` adds the state-space architectures `mamba` and `mamba2` (new `nn::gguf_ssm`); 99 reference models agree with llama.cpp, and cached decoding matches the full forward for both.
- `GGUFCausalLM` + `GGUFTokenizer` verified on a real checkpoint (Qwen2.5-0.5B `Q4_0`): same token ids as `llama-tokenize`, same top-1 token at every position, matching greedy text; new `examples/gguf_generate.rs`.
- `GGUFCausalLM` adds `maion-coder` (latent MoE); 96 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `minicpm3`; 95 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `grovemoe`; 94 reference models agree with llama.cpp (`experts_per_group > 1` has no reference).
- `GGUFCausalLM` adds `smallthinker`; 93 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `hy_v3` and `plamo3`; 92 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `mistral4` and `llama4`; 90 reference models agree with llama.cpp.
- `GGUFCausalLM` adds ALiBi attention and the `bloom`, `mpt`, `refact` and `jais` architectures; 88 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `gpt2` and `starcoder` (learned absolute position embeddings); 84 reference models agree with llama.cpp.
- `GGUFCausalLM` adds the text towers of `paddleocr` and `hunyuan_vl`; 82 reference models agree with llama.cpp (the NTK `rope.scaling.alpha` path has no reference).
- `GGUFCausalLM` adds the text side of `qwen2vl`, `qwen3vl` and `qwen3vlmoe` (M-RoPE position channels modelled exactly); 80 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `grok` and `arctic`; 77 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `cohere2moe`; 75 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `laguna`, `kolibri1` and `flint`; 73 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `mimo2`; 70 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `afmoe` and `spark2_5` (attention output gate); 69 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `minimax-m2` and `exaone-moe`; 67 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `plamo`, `mellum` and `maple`; 65 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `qwen` and `deepseek`; 62 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `glm4moe`; 60 reference models agree with llama.cpp (the reference carries Q/K norms, so the norm-free real-model layout is untested).
- `GGUFCausalLM` adds `glm4` and `chatglm`; 59 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `deci` and `openelm` plus per-layer head / KV-head counts (array-valued metadata); 57 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `cohere2` and `granite_swa`; 55 reference models agree with llama.cpp.
- `GGUFCausalLM` runs `gpt-oss`: per-head attention sinks, OpenAI's clamped SwiGLU in the experts, router and expert
  biases, softmax over the selected experts' logits, alternating sliding windows; matches llama.cpp (53 reference models).
- GGUF: `TQ1_0`, `TQ2_0` (ternary), `MXFP4`, `NVFP4`, `Q1_0`, `Q2_0` dequantization, verified bit-for-bit against
  llama.cpp's own `dequantize_row_*` (25 block types in the fixture now), and plain `F64` / `I8`..`I64` tensors.
- `GGUFCausalLM` adds MoE architectures `dbrx`, `hunyuan-moe`, `dots1`, `bailingmoe`, `bailingmoe2`, `ernie4_5(-moe)` and
  MoE `mistral3`, plus group-limited expert routing (unit-tested; no reference model uses it) and gating /
  normalization taken from the file's keys where llama.cpp does so; 52 reference models agree with llama.cpp.
- `GGUFCausalLM` adds `gptneox` (parallel residual with its own FFN norm), `falcon` (optional second attention norm),
  `codeshell`, `minicpm` (Granite graph with MiniCPM's built-in scale defaults), `phimoe` (RMS norms with bias),
  `pangu-embedded` and `hunyuan-dense`; 44 llama.cpp reference models now agree.
- `GGUFCausalLM` adds `orion`, `nemotron`, `jais2` (squared-ReLU FFNs), `arcee`, `xverse`, `baichuan`, `nanbeige`,
  `maincoder` (Q/K norms after RoPE), `seed_oss` (FFN norm stored as `post_attention_norm`) and `exaone4`, each matching
  llama.cpp's logits. Sliding-window keys are now honored only where llama.cpp implements them (previously any
  architecture with the key was windowed, which diverged from llama.cpp for e.g. Mistral-style `llama` files).
- `GGUFCausalLM` runs the LayerNorm families `stablelm` (optional per-head Q/K LayerNorm, parallel variant), `phi2`
  (parallel residual, ungated GELU FFN, output bias), `starcoder2`, `command-r` (parallel residual, logit multiplier)
  and `olmo` (non-parametric LayerNorm), via a `Norm` type (RMS or LayerNorm, optional weight / bias) and ungated /
  parallel-residual block options; all five match llama.cpp's logits (27 reference models in the parity suite).
- `GGUFCausalLM` runs `deepseek2` (multi-head latent attention with a latent KV cache, low-rank Q, tail-row RoPE
  reordering, DeepSeek's attention scale; sigmoid / softmax-weight gating, `exp_probs_b` selection bias, fused
  `ffn_gate_up_exps`, shared experts, leading dense layers) and `olmoe`, plus partial-rotary and YaRN log-multiplier
  support; all checked against llama.cpp (22 reference models) and cached decoding is checked against the full forward on
  the MLA / sliding-window / MoE / NoPE models.
- `GGUFCausalLM` runs `phi3`: fused `attn_qkv` and fused gate/up tensors are split on load, LongRoPE
  (`rope_factors_long` beyond the original context, `rope_factors_short` within it), and `rope.scaling.attn_factor`
  now scales cos/sin for every scaling type as in llama.cpp. Matches llama.cpp's Phi-3 reference model.
- `GGUFCausalLM` also runs `mistral3`, `internlm2`, `exaone`, `smollm3` (RoPE skipped on every 4th layer), `granite` /
  `granitemoe` (embedding, residual and logit scales) and `olmo2` (post-norm only, full-projection Q/K norms), each
  checked against llama.cpp's own logits (18 reference models now).
- `GGUFCausalLM` generalized to a table of architecture differences (`ArchSpec`, mirroring llama.cpp's graph
  builders) and now also runs `qwen3`, `qwen3moe`, `gemma`, `gemma2` and `gemma3` with logits matching llama.cpp
  (parity test over 48 tokens, so 32-token sliding windows are exercised): any head size, partial rotary dims,
  per-head Q/K RMS norms, post-attention / post-FFN norms, GELU, embedding scaling, attention and final logit
  soft-capping, per-layer sliding windows with a separate local RoPE table, and dense layers inside MoE models.
- `GGUFCausalLM` now matches llama.cpp's own logits (about 1e-4 relative, float noise) on llama.cpp's real
  `llama`, Mixtral-style MoE, `qwen2` and `qwen2moe` test models (`tests/gguf_llama_cpp_parity.rs`, opt-in via
  environment variables). Getting there required: biases on every projection (llama.cpp's `llama` loader
  accepts them) with the Q/K ones permuted together with their weights, per-projection scalar weight scales
  in llama.cpp's operation order (before the bias for attention, after it for the FFN), `output.scale` /
  Qwen2 `output.bias`, `clamp_kqv`, per-expert MoE scales for `llama`, and MoE renormalization chosen by
  architecture (the file's `expert_weights_norm` key is not what llama.cpp consults).
- `GGUFCausalLM.from_gguf` now handles `rope.scaling.*` (`linear`, `yarn` with llama.cpp's exact math) and
  `rope_freqs.weight`, sliding-window attention (also with the KV cache), sparse mixture-of-experts
  (Mixtral-style `llama` + `expert_count`, and `qwen2moe` with a shared expert), and refuses partial
  rotary embeddings instead of running them wrongly.
- GGUF type ids: `BF16` was numbered 16 but GGML's id for BF16 is 30 (16 is `IQ2_XXS`), so genuine BF16
  files failed to load and `IQ2_XXS` files would have decoded as BF16 garbage. Fixed, and `IQ1_S`, `IQ1_M`,
  `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S` are now dequantized (codebooks generated from
  `ggml-common.h`, MIT). Every non-float type is verified bit-for-bit against llama.cpp's own
  `dequantize_row_*` through a fixture (`tools/ggml_reference/`), which also confirms the earlier K-quant /
  `IQ4_*` decoders.
- GGUF: `IQ4_NL` / `IQ4_XS` dequantization; `GGUFTokenizer` now also covers byte-level BPE (`gpt2`
  model: GPT-2, Llama-3 and Qwen2 word-split patterns); `GGUFCausalLM.from_gguf` supports `qwen2`
  (attention biases) and refuses unsupported architectures instead of running them wrongly.
- Added `GGUFTokenizer` (Rust and Python): SentencePiece tokenizer built from a GGUF file's own
  `tokenizer.ggml.*` metadata (score-ranked merges, byte fallback, BOS/EOS), plus
  `GGUFCausalLM.generate` (greedy). A model loaded with `from_file` can now go text -> text.
- `GGUFFile::dequantize_tensor` for `Q4_0` packed each byte's two nibbles as an adjacent pair
  (`x[2j], x[2j+1]`); GGML's actual on-disk layout is split-half (`x[j]` in the low nibble,
  `x[j+16]` in the high nibble of byte `j`). Decoding a real llama.cpp/GGML `Q4_0` file silently
  scrambled element order — undetectable by this crate's own round-trip tests, since its encoder
  made the same mistake symmetrically. Both the encoder and decoder now use the correct split-half
  layout. Also added decode support for `Q4_1`, `Q5_0`, `Q5_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K` and `Q6_K` (previously
  "not yet implemented"), the quant formats most real-world GGUF files (e.g. `Q4_K_M`) actually use;
  all follow GGML's reference bit layout, with regression tests against hand-built reference bytes
  (`crates/oxideon-core/src/serialize/gguf.rs`, `quant_layout_tests`), not just round-trip.
  `GGUFFile::fused_gemv`'s independent `Q4_0` fast path had the same adjacent-pair bug in its own
  unpacking loop and is fixed too, with a test cross-checking it against the generic
  dequantize-then-matmul path.
- `GGUFCausalLM::from_config` (what both the Rust and Python constructors used) only read
  hyperparameters from a GGUF file's metadata and built a model of the right shape — it never
  read the file's actual tensor data, despite `nn/gguf_model.rs`'s module doc literally claiming
  GGUF "quantized weights" are loaded directly. Every weight stayed at its random initialization;
  "loading a real GGUF model" silently produced an untrained model with the right shapes. Added
  `GGUFCausalLM::from_gguf` / Python `GGUFCausalLM.from_file`, which dequantizes and assigns every
  weight by its standard llama.cpp tensor name (`token_embd.weight`, `blk.N.attn_q.weight`, ...),
  erroring loudly on any missing or shape-mismatched tensor instead of falling back to noise.
- `GGUFCausalLM`'s attention had no causal mask (every position attended to the whole sequence,
  i.e. bidirectionally), no grouped-query attention (Llama-2/3, Mistral, ... use fewer KV heads
  than query heads, so their `attn_k/v` tensors did not even fit), a hard-coded `8/3 * d_model`
  FFN width (real models differ, e.g. 11008 vs 10922 for Llama-7B), a fixed RMSNorm epsilon, and
  `llama`-architecture Q/K weights (stored for interleaved RoPE) were fed to a half-split RoPE.
  All fixed: causal mask, `head_count_kv`, `feed_forward_length`, `layer_norm_rms_epsilon`, vocab
  and width from the embedding table, and a Q/K row re-ordering on load.
- The GGUF parser trusted file-supplied sizes: truncated files panicked on unchecked slicing, a
  header claiming 2^62 entries or a huge array drove a multi-terabyte `with_capacity`, tensor
  dimensions / offsets could overflow `usize`, `general.alignment = 0` divided by zero, and
  `fused_gemv` sliced past the data when the column count was not a multiple of the block size.
  All of these are now errors; `malformed_files_are_errors_not_panics` tries every truncation of a
  valid file plus the absurd-count cases.
- The `.safetensors` reader overflowed on a hostile header length (`8 + header_len`) and on shapes
  whose byte size exceeds `usize`; both are errors now, and a test truncates / bit-flips every
  byte of a valid file to prove none of them panic.
- `.safetensors` header parsing decoded JSON strings one byte at a time, casting each byte straight
  to `char`; a multi-byte UTF-8 sequence (any non-ASCII text in a tensor name or `__metadata__`
  value, valid unescaped in JSON, and common in real Hugging Face model metadata such as author
  names or licenses) turned into mojibake, e.g. `"café"` decoded back as `"cafÃ©"`. Now collects
  the raw bytes and decodes the whole string as UTF-8 once.
- `int64` tensors lost precision above 2^53: element reads and writes went through `f64`, so `Tensor.from_i64([2**60 + 1], [1])`
  read back as `2**60`. Reads and writes are now exact.
- A Rust panic no longer escapes as a `BaseException`. pyo3 raises panics as `PanicException`, which `except Exception` does not
  catch, so one overflow in an engine ended a script. The extension re-parents that class under `RuntimeError`, the generated engine
  wrappers (`py_engine!`, `millennium_engines!`) and the hand-written wrappers of the engines that overflowed on huge arguments
  (`TtCore`, `TropicalSpectralEngine`, `ClusterAlgebraQuiverEngine`, `AInfinityOperadEngine`, `ConnesSpectralTriple`,
  `DglaMaurerCartanEngine`, `EtaleWeilZetaEngine`, `MineEstimator`, `SE3EquivariantLayer`, `MikhalkinTropicalGeometryEngine`,
  `GirardLinearLogicResource`) turn panics into plain `RuntimeError`s without the stderr noise, and the engines crate keeps
  integer overflow checks on in release builds (a huge argument is an error instead of a silently wrapped result).
- `MultimodalVLM` with zero query heads divided by zero; its head, layer-count and sequence-length arguments are checked now. The
  same kind of unchecked size arithmetic was fixed in `SelectiveSSM` / `MambaBlock` (so `HybridJambaLM(d_state=2**63)` no longer
  hangs), `DeepSeekMoELM` (`num_mtp_heads`), `DeepSeekV3MoE` (total expert size) and `PointerGraphNetwork`, whose Rust `new` now
  returns a `Result` (API change for Rust callers).
- Absurd arguments are errors instead of panics, hangs or aborts. Sizes are overflow-checked and allocated fallibly in the
  `Photonic*`, `PrincipalBundleConnection`, `SherringtonKirkpatrickModel`, `TropicalMatrix`, `PoincareCnfFlow`,
  `DeepEquilibriumModel`, `STDPSynapse`, `PolychronousNeuralLayer`, `MelSpectrogram`, `QuantizedPagedKVCache`,
  `HypoellipticDiffusion3D`, `CoherentPhotonicCircuit`, `Tensor.eye`, `Tensor.linspace` and `Tensor.multinomial` constructors;
  layer, expert, head and timestep counts are bounded (`ConvNeXt`, `DiT`, `MLPMixer`, `Whisper`, `GGUFCausalLM`, `LSTM`,
  `MedusaHeads`, `DualPipeSchedule`, `TpuCluster`, ...); the iteration counts of `ChirikovStandardMapEngine`,
  `HardyLittlewoodCircleEngine`, `HubbellEwensBiodiversityEngine`, `KoopmanSpectralEngine`, `RelativisticLbm1D`,
  `RiemannSiegelZEngine`, `SchwingerEulerHeisenbergQED`, `SelbergSieveEngine` and `NonHermitianHamiltonian2x2.encircle_ep`
  are bounded per call; `MotivicHodgeEngine.projective_space_motive` sums a long series in closed form.
- Index and precondition mistakes that panicked are errors: `PipelineSchedule1F1B` with `stage_idx >= num_stages`,
  `AmplituhedronGrassmannianEngine.plucker_minor` out of range, `QuantizedPagedKVCache.read_token_dequantized` out of range,
  `RotaryEmbedding3D(dim < 4)`, `PhotonicMesh(0)`, `Tensor.clamp` with `min > max` or a NaN bound.
- Absurd sizes no longer kill the process or silently corrupt a tensor: `Tensor.zeros([10**12])` aborted the interpreter and
  `zeros([2**40, 2**40])` wrapped to an empty tensor. Shapes are now overflow-checked and allocations are fallible.
- `f32` sums and means on the CPU accumulated in a single `f32` running total. Once the total is large, adding a small value rounds
  the same way every time, so the error grew with the length: a sum of a million values came out up to 0.85% too high (the
  forward/backward soak showed 8.5e-3 between CPU and GPU, which sums by tree and was right). They now accumulate in `f64` in
  eight independent lanes, long rows are split into blocks across the rayon threads and added in block order (the result does not
  depend on the thread count), and many short rows run one per task. The same soak now shows 1.3e-6.
- Whether an absurd tensor size was an error depended on the operating system. `Tensor.zeros([10**12])` (4 TB) returned an error on
  Windows and WSL, but on Google Colab's Linux the allocation was accepted (permissive overcommit), the zero-fill touched real memory
  and the OOM killer ended the process (found by running the suite on a Colab L4). The tensor constructors `zeros` / `ones` / `full`
  / `randn` / `rand` / `arange` / `linspace` / `eye` now refuse more than `limits::MAX_ELEMENTS` (2^31, 8 GiB of `f32`) elements
  up front; meta tensors are exempt, and `full` on a meta device no longer builds a CPU buffer it throws away. **Behavior change:**
  a CPU tensor above 2^31 elements made with one of those constructors is now an error.
- The wgpu backend panicked (a wgpu validation error, raised by wgpu as a Rust panic) on any GPU tensor above 128 MiB, because the
  device was created with the default limits, and on any element-wise kernel above 16.7 million elements, which needs more than the
  65,535 workgroups one dispatch may use (found by `tools/gpu_soak.py`). The device now asks for the adapter's real buffer limits,
  sizes beyond what a buffer can hold are errors, and kernels that would not fit one dispatch run on the CPU path.
- Integer tensor division by zero (and `MIN / -1`) raised a Rust panic (`PanicException`, a `BaseException`); it is now a `RuntimeError`.
- Constructors that validated with `assert!` (odd RoPE head dim, `d_model` not divisible by `num_heads`, zero heads / groups /
  layers / LoRA rank, dropout `p >= 1`, NaN conformal bound, ...) returned a panic, and one panic inside `Sequential` poisoned the
  object for good. They now return ordinary errors.
- `CkksContext(2**30)` hung (O(N^2) keygen); the ring degree is bounded. `max` / `min` / `argmax` / `argmin` over an empty axis panicked.
- The hardware inventory presented simulated devices as detected hardware (`8 TPU core(s) [HBM: 128.0 GB ...]` on a machine without
  a TPU); it now names the real adapter, flags software renderers and calls the TPU pod simulated. GPU discovery tries Vulkan / Metal /
  DX12 first, which removes the `MESA: error: ZINK ...` / `libEGL warning` noise on Linux.
- `examples/compare_mini_models.py` and `tools/render_mini_images.py` wrote to a path from the original author's machine and
  required Pillow; the output directory is `OXIDEON_EXAMPLE_OUT` and Pillow is optional where possible. `tools/build_millennium.py`
  no longer hard-codes `/home/...`.
- The README quickstart and `docs/PYTHON_API.md` called `Tensor.from_slice`, which did not exist; it does now.
- `FedererFlemingCurrentsCoarea::sharp_isoperimetric_volume_bound` applied the exponent before the division (wrong for every `n`),
  used an absolute feasibility tolerance, and silently used the 4-ball volume for out-of-range `dim_n`.
- `RothBourgainCapSetBounds` now returns the exact Ellenberg-Gijswijt bound (`u128` DP, n <= 79) and the closed form beyond.
- `AllardVarifoldMonotonicityEngine` used `pi^(k/2)` as the unit-ball volume for `k >= 5`; `MeanCurvatureFlowEngine` used `2 pi`
  as the area of the unit n-sphere for `n >= 3`. Both now use the shared `unit_sphere_area` / `unit_ball_volume` helpers.
- Missing Python binding `SlepianWolfRateRegionEngine`; one Python test block used the wrong class name.
- Absurd integer sizes no longer panic: 31 fuzz-flagged constructor/math entry points (conv `fan_in`, RNN gate dims,
  patch counts, `GGUFModelConfig::basic`, `estimate_parameters`, `matrix_hafnian`, `tune_gemm`, graph wavelets,
  robotics policies, video/vision/world-model dims, and the engines math helpers) return ordinary errors or
  saturate/widen instead of overflowing `usize`/`i64`/`u32` (`tests/robustness.rs`,
  `tests/test_overflow_robustness.py`). Edge fuzzing went from 302 findings to zero panics; the remaining
  slow-init reports are legitimately large models.

### Clarified
- `crypto::*` is educational, not production cryptography: CKKS key generation and encryption are deterministic and the zkML
  "proof" is a hash transcript that is neither sound nor zero-knowledge (see the security section of the README).
- CUDA / Metal / Vulkan / TPU / NPU devices are simulated placement labels (host memory, CPU kernels); see `Device::is_simulated`.

### Known gaps
- Python API coverage: the Rust core has 255 `Tensor` methods and Python exposes about 140; most of the rest are rarely used.
- Edge-value fuzz (every plainly typed constructor, method and function, values such as 0, -1, NaN, inf, 1e300, 2**31, 2**63; 3,048 APIs;
  harness: `tools/edge_fuzz.py`, each call in a child process with an 8 GB address-space limit and a 20 s no-progress watchdog): a debug build
  before these fixes had findings in 113 APIs (80 panics, 32 aborts, 15 hangs, 4 slow); a release build of the final tree has findings
  in 12 APIs (3,048 scanned), none of them a panic or an exception that escapes `except Exception`.
- Those 12 are requests that are simply larger than the machine: `ActionChunkingTransformer`, `BiLSTM`, `CausalLM`, `DeepSeekMoELM`,
  `HybridJambaLM`, `TransfusionModel` and `Whisper` with 1000 layers of width 1000, `FourierNeuralOperator1d` / `SpectralConv1d`
  with 1000 channels and modes, `Conv3d` / `ConvTranspose3d` with a 64^3 kernel, and `GGUFCausalLM` (5 s). The first big tensor is
  allocated fallibly and returns an error, but later temporaries inside the CPU kernels allocate infallibly, so under a hard memory
  limit they abort and otherwise they run for a long time. Making every kernel allocation fallible is not done.
- Platforms: see the README table. Windows was verified with Python 3.11 only; macOS was type-checked, never run; Linux with a real
  GPU (wgpu) ran on Linux only on one Colab L4 (see the README table); the GitHub Actions workflows have never been executed.
- The wgpu matmul reaches 6.3 TFLOP/s on an L4 (about 20% of its fp32 peak) and 4.0 on an RTX 4060 (about a quarter of its peak).
  Vector loads, double buffering and subgroup operations are not used; below 1024 x 1024 the CPU is faster.
- There is no TPU backend: `Device::Tpu` is a placement label whose kernels run on the CPU. A real one needs a PJRT / XLA
  backend (loading `libtpu`, compiling StableHLO per op) and a TPU runtime to test on.
