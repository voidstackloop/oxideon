import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

DOC_CASES = [
    ('docs/TUTORIAL.md', [3, 4, 5, 6, 7, 8, 9, 10]),
    ('README.md', [3, 4, 5]),
    ('docs/CLOUD_AND_DEPLOYMENT.md', [0, 1, 2, 3, 4, 5, 6]),
    ('docs/PYTHON_API.md', list(range(14))),
    ('docs/SCIENTIFIC_AI_MODULES.md', list(range(11))),
    ('docs/INFERENCE_AND_LLM.md', list(range(13))),
]


def _python_blocks(md_path):
    fence = chr(96) * 3
    blocks, cur, in_block = [], [], False
    for line in Path(md_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not in_block and stripped.startswith(fence + 'python'):
            in_block, cur = True, []
        elif in_block and stripped.startswith(fence):
            in_block = False
            blocks.append(chr(10).join(cur))
        elif in_block:
            cur.append(line)
    return blocks


def _run_block(block, cwd):
    script = textwrap.dedent(block)
    return subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=280, cwd=cwd,
    )


CASE_IDS = [(doc, i) for doc, idxs in DOC_CASES for i in idxs]


@pytest.mark.parametrize('doc,idx', CASE_IDS)
def test_doc_block_runs(tmp_path, doc, idx):
    blocks = _python_blocks(ROOT / doc)
    assert len(blocks) > idx, 'missing block'
    proc = _run_block(blocks[idx], tmp_path)
    lines = proc.stderr.strip().splitlines() or ['?']
    assert proc.returncode == 0, doc + '#' + str(idx) + ': ' + lines[-1][:300]


def _grad_of(t):
    g = t.grad
    return g() if callable(g) else g


def test_negative_step_slice_reversal():
    import oxideon as ox
    t = ox.Tensor.arange(0.0, 8.0, 1.0)
    assert t[::-1].to_numpy().tolist() == [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    assert t[5:1:-1].to_numpy().tolist() == [5.0, 4.0, 3.0, 2.0]
    assert t[6::-2].to_numpy().tolist() == [6.0, 4.0, 2.0, 0.0]
    assert t[1:6:2].to_numpy().tolist() == [1.0, 3.0, 5.0]
    assert t[1:3:-1].to_numpy().tolist() == []
    m = ox.Tensor.arange(0.0, 24.0, 1.0).reshape([2, 3, 4])
    assert m[:, 1:3, ::-1].to_numpy().tolist() == [
        [[7.0, 6.0, 5.0, 4.0], [11.0, 10.0, 9.0, 8.0]],
        [[19.0, 18.0, 17.0, 16.0], [23.0, 22.0, 21.0, 20.0]],
    ]


def test_negative_step_slice_keeps_grad():
    import oxideon as ox
    t = ox.Tensor.arange(0.0, 8.0, 1.0).requires_grad_(True)
    t[::-1].sum().backward()
    assert _grad_of(t).to_numpy().tolist() == [1.0] * 8


def _top_submodules():
    import inspect
    import oxideon as ox
    out = []
    for s in dir(ox):
        if s.startswith('_') or s == 'oxideon':
            continue
        try:
            m = getattr(ox, s)
        except Exception:
            continue
        if inspect.ismodule(m) and getattr(m, '__name__', '') == 'oxideon.' + s:
            out.append(s)
    return sorted(out)


def test_stub_lists_all_submodules():
    from pathlib import Path
    pyi = (ROOT / 'python/oxideon/__init__.pyi').read_text(encoding="utf-8")
    missing = [s for s in _top_submodules() if (s + ' as ' + s + ',') not in pyi]
    assert missing == []
    assert 'functional as F,' in pyi
    for leak in ['np as np,', 'sys as sys,', 'math as math,', 'os as os,']:
        assert leak not in pyi


def test_all_lists_all_submodules():
    import oxideon as ox
    missing = [s for s in _top_submodules() if s not in ox.__all__]
    assert missing == []
    assert 'F' in ox.__all__


def test_gen_stubs_skips_unbalanced_input():
    import importlib.util
    spec = importlib.util.spec_from_file_location('gen_stubs', str(ROOT / 'tools/gen_stubs.py'))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert gen.match_close('impl Foo {', 9, '{', '}') == -1
    table = gen.RustTable()
    table.pyname['PyFoo'] = 'Foo'
    table._impl_blocks('#[pymethods]\nimpl PyFoo {')
    table._pyfunctions('#[pyfunction]\npub fn foo(')
    assert table.members == {}
    assert table.funcs == {}


def test_gen_stubs_idempotent():
    import subprocess
    import sys
    from pathlib import Path
    init_py = ROOT / 'python/oxideon/__init__.py'
    init_pyi = (ROOT / 'python/oxideon/__init__.pyi')
    before_py = init_py.read_text(encoding="utf-8")
    before_pyi = init_pyi.read_text(encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / 'tools/gen_stubs.py')],
            capture_output=True, text=True, timeout=280,
            encoding="utf-8",
        )
        assert proc.returncode == 0, proc.stderr[-500:]
        assert init_py.read_text(encoding="utf-8") == before_py, 'regen changed __init__.py'
        assert init_pyi.read_text(encoding="utf-8") == before_pyi, 'regen changed __init__.pyi'
    finally:
        init_py.write_text(before_py, encoding="utf-8")
        init_pyi.write_text(before_pyi, encoding="utf-8")
