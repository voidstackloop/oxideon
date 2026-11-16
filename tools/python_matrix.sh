#!/usr/bin/env bash
# One abi3 wheel, installed into a fresh virtualenv per Python version (3.9 - 3.13), with the Python tests run in each.
#   bash tools/python_matrix.sh [--release] [WHEEL_DIR]      (without WHEEL_DIR the wheel is built first)
# Interpreters come from `uv` (installed on demand). Exit code 0 only if every version passed both test files.
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
REL=""; [ "${1:-}" = "--release" ] && { REL="--release"; shift; }
DIST=${1:-}
export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
command -v uv > /dev/null || { python3 -m pip install -q uv && export PATH="$(python3 -m site --user-base)/bin:$PATH"; }
fail=0
if [ -z "$DIST" ]; then
  DIST=$(mktemp -d)
  python3 -m pip install -q maturin
  echo "== building one wheel ${REL:-(debug)}"
  maturin build $REL --out "$DIST" 2>&1 | tail -n 2 || fail=1
fi
ls "$DIST"
for v in 3.9 3.10 3.11 3.12 3.13; do
  echo "=================== Python $v"
  env_dir=$(mktemp -d)
  uv venv -q --python "$v" "$env_dir" || { echo "FAIL: no interpreter for $v"; fail=1; continue; }
  py="$env_dir/bin/python"
  uv pip install -q --python "$py" numpy
  uv pip install -q --python "$py" --reinstall --no-index --find-links "$DIST" oxideon || { echo "FAIL: install"; fail=1; continue; }
  "$py" -c "import sys, oxideon; print('python', sys.version.split()[0], '| oxideon', oxideon.__version__)"
  for t in test_python_package.py test_python_ergonomics.py; do
    (cd /tmp && "$py" -u "$ROOT/tests/$t" > "/tmp/matrix_$t.log" 2>&1)
    rc=$?
    tail -n 1 "/tmp/matrix_$t.log"
    [ "$rc" = 0 ] || { echo "FAIL: $t under Python $v (exit $rc)"; fail=1; }
  done
done
[ "$fail" = 0 ] && echo "MATRIX PASS" || echo "MATRIX FAIL"
exit "$fail"
