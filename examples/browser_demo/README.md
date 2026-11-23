# Browser demo

`index.html` runs the real Oxideon tensor library compiled to WebAssembly: a matrix multiply checked against a float64
reference, activation-function curves, an overflow-safe softmax, and a small neural-network forward pass.

Live: https://voidstackloop.github.io/oxideon/ (built by `.github/workflows/pages.yml`).

Build and run it locally:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <the wasm-bindgen version in Cargo.lock> --locked
cargo build --release --lib --target wasm32-unknown-unknown --no-default-features --features wasm
wasm-bindgen target/wasm32-unknown-unknown/release/oxideon.wasm --target web --out-dir examples/browser_demo/pkg
cd examples/browser_demo && python -m http.server 8000     # open http://localhost:8000
```

`pkg/` is generated and not committed. The build is single-threaded and uses the portable kernels (no SIMD, no `rayon`), so
speed is far below the native library; the WebGPU backend is not part of this build.
