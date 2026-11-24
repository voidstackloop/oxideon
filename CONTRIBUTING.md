# Contributing to Oxideon

Thanks for your interest in contributing to **Oxideon**! Every kind of contribution is welcome—bug fixes, SIMD/CUDA kernels, Python/FFI ergonomics, docs, tests, and benchmarks.

## Getting Started

### Prerequisites
- **Rust** `1.85.0` or newer (`edition = "2024"`)
- **Python** `3.9+` with a virtual environment (`numpy`, `pytest`, `maturin`) if testing Python bindings

### 1. Run Rust Tests & Lints
```bash
cargo test -p oxideon-core --lib
cargo clippy --workspace --all-targets -- -D warnings
```

### 2. Build & Test Python Bindings
```bash
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install maturin numpy pytest
maturin develop --release
pytest tests/ -q
```

If you change any Python-facing signatures or docstrings in `src/python*.rs`, regenerate the type stubs:
```bash
python tools/gen_stubs.py
```

### 3. Run Benchmarks & Accuracy Checks
```bash
python benchmarks/bench_vs_pytorch.py
python benchmarks/accuracy_vs_pytorch.py
```

## Finding an Issue & Getting Help
- Browse [`good first issue`](https://github.com/voidstackloop/oxideon/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) or [`help wanted`](https://github.com/voidstackloop/oxideon/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).
- Comment on an issue if you'd like to claim it or ask design questions—mentorship and code-review feedback are happily provided!
- You can also start a thread in [GitHub Discussions](https://github.com/voidstackloop/oxideon/discussions).

## Pull Request Checklist
1. Keep changes focused and add unit tests covering new behavior or edge cases.
2. Run `cargo clippy --workspace --all-targets -- -D warnings` with zero warnings.
3. Verify existing tests pass (`cargo test -p oxideon-core --lib`).
4. If Python exports changed, run `python tools/gen_stubs.py`.

## License
By contributing, you agree that your contributions will be dual-licensed under **MIT OR Apache-2.0** (matching the repository license). No separate CLA is required.
