#!/usr/bin/env bash
# Everything, on a Colab GPU runtime (the VM has more cores than a laptop): run after `tools/colab_l4.sh` has set up Rust.
#   !bash /content/oxideon/tools/colab_all.sh [SOAK_SECONDS]          (default soak: 600)
#   !ONLY="rust_gpu gpu_probe" bash /content/oxideon/tools/colab_all.sh 300      (just those stages)
# One line per stage, `PASS` or `FAIL`, and the log of every stage under /tmp/stage_<name>.log.
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
SOAK=${1:-600}
export PATH="$HOME/.local/bin:$PATH"
# A fresh runtime has neither a C toolchain, the Vulkan loader, nor Rust (tools/colab_l4.sh sets the same things up).
if ! command -v cargo > /dev/null && [ ! -f "$HOME/.cargo/env" ]; then
  apt-get update -qq && apt-get install -y -qq build-essential pkg-config libvulkan1 vulkan-tools curl > /dev/null
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null
fi
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
has_nvidia() { vulkaninfo --summary 2> /dev/null | grep -i deviceName | grep -qi nvidia; }
if ! has_nvidia; then
  # A compute-only driver install has no Vulkan driver file; the matching user-space package adds it.
  major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  for pkg in "libnvidia-gl-$major" "libnvidia-gl-$major-server"; do
    apt-get install -y -qq "$pkg" > /dev/null 2>&1 && has_nvidia && break
  done
fi
has_nvidia || echo "WARNING: Vulkan does not list the NVIDIA GPU; the GPU stages will see a software renderer and refuse to report it"
python3 -m pip install -q uv maturin 2> /dev/null
export PATH="$(python3 -m site --user-base)/bin:$PATH"
results=()
stage() {
  local name=$1; shift
  # ONLY="gpu_probe gpu_soak" runs just those stages (the rest are skipped, not reported)
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then return; fi
  local t0=$SECONDS
  echo "=== $name"
  "$@" > "/tmp/stage_$name.log" 2>&1
  local rc=$?
  tail -n 4 "/tmp/stage_$name.log" | cut -c1-200
  [ "$rc" = 0 ] && results+=("PASS  $name ($((SECONDS - t0)) s)") || results+=("FAIL  $name (exit $rc, $((SECONDS - t0)) s)")
}

rustup component add clippy > /dev/null 2>&1
# an interpreter with a shared libpython, for the Rust tests that embed Python
rm -rf /tmp/pyenv
uv venv -q --python 3.12 /tmp/pyenv && uv pip install -q --python /tmp/pyenv/bin/python numpy
export PYO3_PYTHON=/tmp/pyenv/bin/python
# Only the Rust tests that embed Python need this directory on the library path. Exported globally it hid the NVIDIA GPU from
# wgpu (the Vulkan driver loaded the wrong shared libraries and only the llvmpipe software renderer was left).
PYLIB=$(/tmp/pyenv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

stage clippy_default   cargo clippy --workspace --all-targets -- -D warnings
stage clippy_python    cargo clippy --workspace --all-targets --features python -- -D warnings
stage clippy_gpu       cargo clippy --all-targets --features rayon,wgpu,python -- -D warnings
stage rust_gpu         cargo test --workspace --no-fail-fast --features rayon,wgpu
stage rust_python      env LD_LIBRARY_PATH="$PYLIB:${LD_LIBRARY_PATH:-}" cargo test --workspace --no-fail-fast --features python

# one release wheel, tested under Python 3.9 - 3.13
mkdir -p /tmp/dist_rel
stage wheel_release    bash -c 'rm -f /tmp/dist_rel/*; maturin build --release --out /tmp/dist_rel'
stage python_matrix    bash tools/python_matrix.sh --release /tmp/dist_rel

# the wheel in the 3.12 environment for the fuzz and the GPU soak
uv pip install -q --python /tmp/pyenv/bin/python --reinstall --no-index --find-links /tmp/dist_rel oxideon
stage gpu_probe        /tmp/pyenv/bin/python tools/gpu_probe.py
stage ops_bench        /tmp/pyenv/bin/python tools/ops_bench.py
stage gpu_bench        /tmp/pyenv/bin/python tools/gpu_bench.py
stage matmul_bench     cargo run --release -q --features rayon,wgpu --example gpu_matmul_bench
# AVX-512 kernels for the CPU matmul, when this CPU has them: a second wheel, the same benchmark
if grep -q avx512f /proc/cpuinfo; then
  rm -rf /tmp/dist_avx512 /tmp/pyenv512; mkdir -p /tmp/dist_avx512
  stage wheel_avx512   maturin build --release --features rayon,wgpu,python-ext,avx512 --out /tmp/dist_avx512
  uv venv -q --python 3.12 /tmp/pyenv512 && uv pip install -q --python /tmp/pyenv512/bin/python numpy \
    && uv pip install -q --python /tmp/pyenv512/bin/python --reinstall --no-index --find-links /tmp/dist_avx512 oxideon
  stage ops_bench_avx512 /tmp/pyenv512/bin/python tools/ops_bench.py
else
  echo "this CPU has no AVX-512: skipping the AVX-512 comparison"
fi
stage gpu_soak         /tmp/pyenv/bin/python tools/gpu_soak.py "$SOAK"
stage finetune_mini    /tmp/pyenv/bin/python examples/finetune_mini.py wgpu
rm -rf /tmp/fuzz; mkdir -p /tmp/fuzz
stage edge_fuzz        env OXI_ROOT="$ROOT" FUZZ_OUT=/tmp/fuzz /tmp/pyenv/bin/python tools/edge_fuzz.py run
echo "--- edge fuzz summary (a finding is a panic, an abort, a hang or a call slower than 5 s)"; head -n 40 /tmp/stage_edge_fuzz.log | cut -c1-230
grep -E '^panic' /tmp/stage_edge_fuzz.log > /dev/null && results+=("FAIL  edge_fuzz_panics (see above)")

for bench in matmul_bench gpu_bench ops_bench ops_bench_avx512; do
  [ -f "/tmp/stage_$bench.log" ] && { echo; echo "--- $bench"; grep -v XDG_RUNTIME "/tmp/stage_$bench.log"; }
done

echo; echo "================ SUMMARY"
printf '%s\n' "${results[@]}"
printf '%s\n' "${results[@]}" | grep -q '^FAIL' && exit 1 || exit 0
