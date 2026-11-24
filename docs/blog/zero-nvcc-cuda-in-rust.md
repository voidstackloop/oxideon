# Running 1.69 TFLOP/s CUDA Kernels in Pure Rust with Zero `nvcc` Dependencies

Most Rust deep learning frameworks face a painful build-time wall on Windows and Linux: you either shell out to `nvcc` inside `build.rs` (breaking `cargo install` / PyPI source builds whenever the user lacks the 4 GB CUDA Toolkit) or rely solely on WebGPU/Vulkan shaders.

In **Oxideon** (`oxideon-core`), we built a zero-build-dependency CUDA backend (`crates/oxideon-core/src/backend/cuda_driver.rs` & `cuda_kernels.rs`) that still runs native NVIDIA PTX/NVRTC kernels, CUDA Graphs, and zero-copy `kDLCUDA` DLPack directly on the GPU.

## 1. Dynamic Driver Loading (`nvcuda.dll` / `libcuda.so.1`)

Instead of linking against CUDA static stubs at compile time, `cuda_driver.rs` opens the NVIDIA driver library at runtime via OS dynamic loading (`LoadLibraryW` on Windows, `dlopen` on Linux) and resolves CUDA Driver API entry points:

- Device/context lifecycle (`cuInit`, `cuDevicePrimaryCtxRetain`, `cuDeviceGetAttribute`)
- VRAM allocation & transfers (`cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`, `cuMemcpyDtoDAsync_v2`)
- PTX JIT compilation & kernel launches (`cuModuleLoadDataEx` with JIT error log buffer, `cuModuleGetFunction`, `cuLaunchKernel`)
- Streams & CUDA Graphs (`cuStreamCreate`, `cuStreamBeginCapture`, `cuStreamEndCapture`, `cuGraphInstantiate`, `cuGraphLaunch`)

If no NVIDIA driver is present on the machine, Oxideon still builds identically on every OS and cleanly reports `ox.cuda.is_available() == False`.

## 2. Writing 32×32 Register-Blocked SGEMM Directly in PTX

For out-of-the-box execution even when `libnvrtc` isn't installed alongside the driver, `cuda_kernels.rs` embeds self-contained `.version 7.0 / .target sm_50` PTX kernels that the driver JITs straight to native SASS (`sm_89` on RTX 4060):

- **`matmul_tiled32_2x2_f32`**: A `[16, 16, 1]` block (256 threads) computes a `32×32` tile of `C` with `BK=16` in shared memory (`tileA[32][16]`, `tileB[16][32]`), where each thread accumulates a `2×2` register sub-tile (`4` FMAs per pair of shared loads).
- **`softmax_row_f32` & `layer_norm_row_f32`**: Block-per-row 256-thread shared-memory tree reductions so activations don't bounce through global memory between mean, variance, and normalization passes.
- **PTX Addressing Quirk Learned on Hardware**: Shared memory arrays in PTX cannot be indexed with `[sym + %reg]` directly—you must load the shared symbol base (`mov.u32 %r_base, tileA`), add the byte offset into a register (`add.u32`), and load/store `[%r_addr]`. Surfacing the `cuModuleLoadDataEx` JIT error log makes debugging PTX trivial.

### Measured Throughput (RTX 4060 Laptop GPU, JIT PTX, Zero `nvcc`)

| Matrix Size | Latency | Throughput |
|---|---:|---:|
| `CUDA matmul 1024×1024` | **1.404 ms** | **1529.9 GFLOP/s** |
| `CUDA matmul 2048×2048` | **10.148 ms** | **1692.9 GFLOP/s** |

When NVRTC *is* available at runtime (`backend/nvrtc.rs` + `crates/oxideon-core/cuda/*.cu`), higher-throughput `.cu` kernels (attention, conv2d, reductions, optimizers) are JIT-compiled on first use too.

## Try It Yourself
```bash
pip install oxideon
# or in Rust:
cargo add oxideon-core
```
Repo: https://github.com/voidstackloop/oxideon
