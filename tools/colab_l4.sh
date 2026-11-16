#!/usr/bin/env bash
# Google Colab (Runtime > Change runtime type > L4 GPU): does the wgpu backend run on a real NVIDIA GPU under Linux?
#
#   !tar xzf oxideon-colab.tar.gz && bash oxideon/tools/colab_l4.sh        (FULL=1 also runs the whole Rust suite)
#
# Exit 0 only if everything ran on a hardware adapter. The Rust GPU test skips itself (and passes) when wgpu finds no
# adapter, and Colab's driver may lack the Vulkan part, so this script checks for the NVIDIA adapter explicitly and
# reports "nothing about the L4 was tested" instead of a false pass.
cd "$(dirname "$0")/.." || exit 1
fail=0
say() { printf '\n== %s\n' "$*"; }

say "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || { echo "No NVIDIA GPU: pick a GPU runtime."; exit 2; }
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)

say "Packages and Rust"
apt-get update -qq && apt-get install -y -qq build-essential pkg-config libvulkan1 vulkan-tools curl > /dev/null
command -v cargo > /dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null
. "$HOME/.cargo/env"

say "Does Vulkan list the NVIDIA GPU?"
has_nvidia() { vulkaninfo --summary 2> /dev/null | grep -i deviceName | grep -qi nvidia; }
if ! has_nvidia; then
  # A compute-only driver install has no Vulkan driver file; the matching user-space package adds it.
  for pkg in "libnvidia-gl-$driver_major" "libnvidia-gl-$driver_major-server"; do
    apt-get install -y -qq "$pkg" > /dev/null 2>&1 && has_nvidia && break
  done
fi
vulkaninfo --summary 2>&1 | grep -iE "deviceName|driverName|apiVersion|error"
has_nvidia || {
  echo "FAIL: Vulkan does not list the NVIDIA GPU, so wgpu would only find a software renderer (or nothing)."
  echo "Nothing about the L4 was tested. Check the driver version above against the libnvidia-gl packages apt offers."
  exit 2
}

say "Rust: wgpu backend test on the adapter"
echo "Compiling: the first build takes several minutes, and nothing is printed until the test has run."
cargo test --features wgpu --test views_dtypes_ops wgpu_backend -- --nocapture > gpu_test.log 2>&1
cargo_rc=$?
grep -E "adapter|^test |test result|panicked" gpu_test.log
grep "Running WGPU test on adapter" gpu_test.log | grep -qi nvidia || { echo "FAIL: the test did not run on the NVIDIA adapter."; fail=1; }
{ [ "$cargo_rc" = 0 ] && grep -q "test result: ok" gpu_test.log && ! grep -qE "FAILED|panicked" gpu_test.log; } || { echo "FAIL: Rust GPU test (cargo exit $cargo_rc)."; fail=1; }
[ "$fail" = 0 ] || { echo "--- last lines of gpu_test.log ---"; tail -n 40 gpu_test.log; }

say "Python wheel: build, install, run the package tests and the GPU probe"
python3 -m pip install -q maturin
echo "Building the wheel: several minutes on Colab's CPUs, and nothing is printed until it finishes."
maturin build --out dist 2>&1 | tail -n 2
python3 -m pip install -q --force-reinstall --no-index --find-links dist oxideon || fail=1
# -u: a crash at interpreter exit must not swallow the buffered "passed" line; the exit code decides pass or fail
for t in tests/test_python_package.py tests/test_python_ergonomics.py; do
  python3 -u "$t" > "/tmp/$(basename "$t").log" 2>&1
  rc=$?
  tail -n 3 "/tmp/$(basename "$t").log"
  [ "$rc" = 0 ] || { echo "FAIL: $t exited with code $rc"; fail=1; }
done
python3 tools/gpu_probe.py || { echo "FAIL: GPU probe."; fail=1; }

if [ "${FULL:-0}" = 1 ]; then
  say "Rust: whole suite with the GPU features"
  cargo test --workspace --no-fail-fast --features rayon,wgpu 2>&1 | grep -E "^test result|FAILED|panicked at|^error" || true
fi

say "Result"
[ "$fail" = 0 ] && echo "PASS: oxideon ran on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) through wgpu." || echo "FAIL: see the messages above."
exit "$fail"
