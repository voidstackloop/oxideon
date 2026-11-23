# Oxideon Examples

Runnable showcases for tensors, autograd, neural nets, LLMs, vision, science,
and multi-framework interop. Python examples run with the built extension
(`maturin develop --release`, then `python3 examples/<name>.py`); Rust examples
run with `cargo run --release --example <name>`.

## Python

### Core tensors, autograd & modules

| Example | What it shows | Run |
|---|---|---|
| `python_showcase.py` | Higher-order autograd, differentiable `pinv`, fused kernels, whole-loop Rust training, frontier models | `python3 examples/python_showcase.py` |
| `class_style_neural_network.py` | Subclassing `ox.nn.Module`, residual blocks, `Sequential`/`ModuleList`/`ModuleDict`, hooks, freezing | `python3 examples/class_style_neural_network.py` |
| `save_and_load_ai_model.py` | `ox.save_file` / `.load_file` SafeTensors checkpointing, metadata, bit-for-bit resume | `python3 examples/save_and_load_ai_model.py` |

### LLMs, serving & state-space models

| Example | What it shows | Run |
|---|---|---|
| `train_llm.py` | BPE + `CausalLM` pre-training + KVCache generation, 100% in Rust | `python3 examples/train_llm.py` |
| `llm_paged_speculative_serving.py` | `PagedKVCache`, FlashAttention-2, `SpeculativeDraftDecoder` pipeline | `python3 examples/llm_paged_speculative_serving.py` |
| `radix_prefix_serving.py` | `RadixPrefixCache`, shared system prompt pre-warming, branch splitting, multi-turn context reuse, LRU eviction | `python3 examples/radix_prefix_serving.py` |
| `mamba_selective_ssm.py` | `MambaBlock`, `SelectiveScan`, linear $O(L)$ sequence scaling, bounded $O(1)$ recurrent step state | `python3 examples/mamba_selective_ssm.py` |
| `sparse_long_context_attention.py` | `SparseMultiheadAttention`, BigBird & sliding window masks, >98% sparsity at $L=2048$, 70x memory savings | `python3 examples/sparse_long_context_attention.py` |
| `peft_dora_and_alignment.py` | `DoRALinear`, `LoRAPlusLinear`, `ORPOLoss`, `KTOLoss` | `python3 examples/peft_dora_and_alignment.py` |
| `finetune_mini.py` | Pretrain → fine-tune vs from-scratch checks for vision + LM + MLP (exit 0 only if all pass) | `python3 examples/finetune_mini.py [cpu\|wgpu]` |

### Advanced quantization & sparse computing

| Example | What it shows | Run |
|---|---|---|
| `advanced_quantization_and_qat.py` | FP8 E4M3/E5M2, OCP MXFP4, block-wise Weight-Only INT4, `QATLinear` with STE | `python3 examples/advanced_quantization_and_qat.py` |
| `sparse_matrix_spmm.py` | `CSRMatrix` SpMM & SpSpMM, CSR storage invariants, memory & crossover benchmark vs dense GEMM | `python3 examples/sparse_matrix_spmm.py` |
| `python_autonomous_quant.py` | `DynamicLlmQuantizer` telemetry-driven FP32→1.58-bit→FP16 adaptation | `python3 examples/python_autonomous_quant.py` |

### Vision

| Example | What it shows | Run |
|---|---|---|
| `train_mini_images.py` | 1,000 procedural 16×16 images, 4-stage ConvNet, AdamW training table | `python3 examples/train_mini_images.py` |
| `compare_mini_models.py` | ConvNet vs ResNet vs AutoEncoder benchmark + latent morph strip | `python3 examples/compare_mini_models.py` |
| `vision_transformer_showcase.py` | ViT patch embed, CLS token, AdamW training, SafeTensors export | `python3 examples/vision_transformer_showcase.py` |

### Text classification

| Example | What it shows | Run |
|---|---|---|
| `fake_news_detector.py` | 10k-sample fake/real news classifier, training loop, `.safetensors` + vocab persistence | `python3 examples/fake_news_detector.py` |

### Interop, science, uncertainty & privacy

| Example | What it shows | Run |
|---|---|---|
| `uncertainty_and_interpretability.py` | `DirichletLoss` (Evidential DL), `decompose_uncertainty` (aleatoric vs epistemic), `TCAV`, `IntegratedHessians` | `python3 examples/uncertainty_and_interpretability.py` |
| `synthetic_tabular_generation.py` | `DifferentialPrivacySynthesizer` ($\epsilon$-Laplace mechanism), `GaussianCopulaSynthesizer`, correlation preservation | `python3 examples/synthetic_tabular_generation.py` |
| `python_interop_showcase.py` | NumPy, PyTorch (+DLPack, state_dict), PIL, Pandas, Polars, NetworkX, SymPy, unified pipeline | `python3 examples/python_interop_showcase.py` |
| `python_hardware_mixing.py` | `HardwareManager`, CPU/GPU/CUDA/TPU placement, `AutoDeviceMixer`, sharded pipeline | `python3 examples/python_hardware_mixing.py` |
| `python_physics_and_science.py` | Sedov-Taylor blast, LWE/BKZ lattices, Groth16 SNARK, Lawson fusion | `python3 examples/python_physics_and_science.py` |
| `scientific_causal_and_uncertainty.py` | NOTEARS DAG discovery, uncertainty decomposition, Cox survival, graph kernels | `python3 examples/scientific_causal_and_uncertainty.py` |
| `science_numerical_methods.py` | `ox.numerical`: adaptive quadrature, Gauss-Legendre, interpolation, `minimize_scalar`, `curve_fit`, constants | `python3 examples/science_numerical_methods.py` |
| `science_statistics.py` | `ox.stats`: t-tests, chi-square, correlation, Mann-Whitney, KS, ECDF, Holm correction | `python3 examples/science_statistics.py` |
| `science_physics_sims.py` | `ox.physics`: 2-D wave/diffusion PDEs, N-body orbit energy check, leapfrog oscillator | `python3 examples/science_physics_sims.py` |
| `train_with_scheduler_and_checkpoint.py` | Two-group NAdam + StepLR, external LR edits, safetensors save/strict-load round-trip | `python3 examples/train_with_scheduler_and_checkpoint.py` |

### Singularity milestones

| Example | What it shows | Run |
|---|---|---|
| `python_millennial_showcase.py` | Tensors, higher-order autograd, engines 100→1000 centennial cascade | `python3 examples/python_millennial_showcase.py` |
| `python_quincentennial_showcase.py` | Tensors, autograd, engines 100→500 cascade, quantization validation | `python3 examples/python_quincentennial_showcase.py` |

## Rust

### Core

| Example | What it shows | Run |
|---|---|---|
| `showcase` | Strided views, 8 dtypes, 1st/2nd/3rd-order autograd, einsum/Conv2d/MHA, GEMM bench | `cargo run --release --example showcase` |
| `xor` | Tiny MLP learning XOR | `cargo run --release --example xor` |
| `spiral` | 3-class spiral MLP with mini-batches + AdamW | `cargo run --release --example spiral` |

### LLMs & GGUF

| Example | What it shows | Run |
|---|---|---|
| `train_llm` | BPE + CausalLM pretrain + SimPO alignment + KVCache generation + DeepSeek MoE | `cargo run --release --example train_llm` |
| `tiny_llm` | Minimal LLaMA-style LM on a repeating sequence | `cargo run --release --example tiny_llm` |
| `gguf_generate` | Tokenize + greedy decode with a GGUF checkpoint (timing + tok/s) | `cargo run --release --example gguf_generate -- model.gguf "prompt" [n]` |
| `gguf_info` | GGUF metadata, tensor list, param/size summary | `cargo run --release --example gguf_info -- model.gguf` |

### Frontier, quantization & physics

| Example | What it shows | Run |
|---|---|---|
| `frontier_dl` | LoRA+merge, Q8/Q4 quant, GQA+KVCache, FFT/SVD, ViT/UNet, scaler/EMA | `cargo run --release --example frontier_dl` |
| `autonomous_quantization` | Telemetry-driven dynamic precision (FP32→1.58-bit→FP16) | `cargo run --release --example autonomous_quantization` |
| `physics_and_pde_solvers` | Sedov-Taylor, Bjorken QGP, reconnection, Lawson fusion | `cargo run --release --example physics_and_pde_solvers` |
| `quantum_and_lattice_crypto` | Regev LWE, BKZ root Hermite factor, Groth16 QAP SNARK | `cargo run --release --example quantum_and_lattice_crypto` |

### Singularity & bench

| Example | What it shows | Run |
|---|---|---|
| `millennial_singularity` | engine 1000 apex synthesis + coupled autograd | `cargo run --release --example millennial_singularity` |
| `quincentennial_singularity` | engine 500 synthesis + coupled autograd | `cargo run --release --example quincentennial_singularity` |
| `gpu_matmul_bench` | CPU vs wgpu square matmul (GFLOP/s + numerics check) | `cargo run --release --features wgpu --example gpu_matmul_bench` |

### Browser

`browser_demo/` — four live demos (matmul speed and accuracy, activation curves, overflow-safe softmax, a small network forward pass) running the real Oxideon library compiled to WebAssembly; build steps in its README, live at https://voidstackloop.github.io/oxideon/.
