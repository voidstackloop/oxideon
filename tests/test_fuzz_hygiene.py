"""The edge fuzzer must not write junk into the caller's working directory.

Fuzzed file-writing APIs (e.g. ``BPETokenizer.save``) are called with pool
strings (``"a"``, ``"f32"``, ...) as paths; the fuzz children must run in a
scratch directory so those land under ``$FUZZ_OUT/scratch``, never in the
caller's CWD.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(sys.platform == "win32", reason="edge_fuzz.py needs POSIX resource limits")
def test_fuzz_run_writes_nothing_to_cwd():
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp) / "cwd"
        cwd.mkdir()
        out = Path(tmp) / "out"
        out.mkdir()
        env = dict(os.environ, FUZZ_OUT=str(out), OXI_ROOT=str(ROOT))
        r = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "edge_fuzz.py"), "run", "BPETokenizer"],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        leftovers = sorted(p.name for p in cwd.iterdir())
        assert leftovers == [], leftovers
        # The file-writing calls really ran (nothing was silently skipped):
        # their output must have landed in the scratch dir instead.
        scratch = out / "scratch"
        assert scratch.is_dir()
        assert sorted(p.name for p in scratch.iterdir()) != []
