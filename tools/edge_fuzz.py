"""Edge-value fuzz of every plainly typed constructor / method / function in the stub.

Needs an installed oxideon wheel (release build recommended) and reads python/oxideon/__init__.pyi under OXI_ROOT
(default ~/projects/oxideon). Every call runs in a child process with an 8 GB address-space limit and a 20 s no-progress
watchdog; a finding is a panic, an abort (rc -6/-9/-11), a hang or a call slower than 5 s.

parent:  python edge_fuzz.py run [ONLY-substring]      -> writes $FUZZ_OUT/results.jsonl, prints a summary
                                                   (fuzzed file output lands in $FUZZ_OUT/scratch, never the CWD)
child:   python edge_fuzz.py child START               (spawned by the parent; resumes after a crash/hang)
"""
import ast, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None

ROOT = Path(os.environ.get("OXI_ROOT", Path(__file__).resolve().parent.parent))
STUB = ROOT / "python/oxideon/__init__.pyi"
OUT = Path(os.environ.get("FUZZ_OUT", ROOT / "target" / "ut_fuzz"))
NAN, INF = float("nan"), float("inf")
POOLS = {
    "int": [0, 1, 2, 3, 7, 64, 1000, 2**31, 2**32, 2**40, 2**53, 2**62, 2**63 - 1, 2**63, 2**64 - 1],
    "float": [0.0, 1.0, -1.0, 0.5, 2.0, NAN, INF, -INF, 1e300, -1e300, 1e-300, 1e18, 9.3e18],
    "bool": [True, False],
    "str": ["", "a", "f32", "x" * 50],
    "list[float]": [[], [1.0], [NAN], [1e300] * 3, [0.0] * 4, [1.0, 2.0, 3.0, 4.0]],
    "list[int]": [[], [0], [1], [2**40], [2**63], [3, 3], [0, 0, 0]],
    "tuple[float, float]": [(0.0, 0.0), (NAN, 1.0), (1e300, -1e300), (1.0, 2.0)],
    "tuple[int, int]": [(0, 0), (1, 1), (2**63, 2**63), (2**40, 2)],
}
BASE = {"int": 2, "float": 0.5, "bool": True, "str": "f32", "list[float]": [1.0, 2.0, 3.0, 4.0], "list[int]": [2, 2],
        "tuple[float, float]": (1.0, 2.0), "tuple[int, int]": (2, 2)}


def ann_str(a):
    return ast.unparse(a) if a is not None else "Any"


def pool_of(t):
    t = t.replace("Optional[", "Optional[")
    if t.startswith("Optional[") and t.endswith("]"):
        inner = t[9:-1]
        return POOLS[inner] + [None] if inner in POOLS else None
    return POOLS.get(t)


def base_of(t):
    if t.startswith("Optional[") and t[9:-1] in BASE:
        return BASE[t[9:-1]]
    return BASE.get(t)


def params_of(fn, skip_self):
    """-> list of (name, annotation, default-or-MISSING) or None when a required parameter has an unsupported type."""
    a = fn.args
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    out = []
    for i, (arg, d) in enumerate(zip(pos, defaults)):
        if skip_self and i == 0:
            continue
        t = ann_str(arg.annotation)
        dv = "MISSING"
        if d is not None:
            try:
                dv = ast.literal_eval(d)
            except Exception:
                dv = "SKIP"
        out.append((arg.arg, t, dv))
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        out.append((arg.arg, ann_str(arg.annotation), "SKIP" if d is None else ast.literal_eval(d) if isinstance(d, ast.Constant) else "SKIP"))
    return out


def build_calls(params):
    """Argument dicts to try, or None when the API has a required parameter of an unsupported type."""
    base = {}
    for name, t, d in params:
        if d not in ("MISSING", "SKIP") and (pool_of(t) is None or d is not None):
            base[name] = d
        elif pool_of(t) is not None and base_of(t) is not None:
            base[name] = base_of(t)
        elif d == "MISSING":
            return None
        # else: leave out (default)
    calls = [dict(base)]
    for name, t, d in params:
        p = pool_of(t)
        if p is None:
            continue
        for v in p:
            c = dict(base)
            c[name] = v
            calls.append(c)
    for ty in ("int", "float"):
        names = [n for n, t, d in params if t == ty]
        if len(names) > 1:
            for v in POOLS[ty]:
                c = dict(base)
                c.update({n: v for n in names})
                calls.append(c)
    return calls


def collect():
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    items = []  # (kind, qualname, ctor-params, member, member-params, static)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            items.append(("func", node.name, None, None, params_of(node, False), False))
        elif isinstance(node, ast.ClassDef):
            ctor = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
            ctor_p = params_of(ctor, True) if ctor else None
            if ctor is not None:
                items.append(("ctor", node.name, ctor_p, "__init__", ctor_p, False))
            for n in node.body:
                if not isinstance(n, ast.FunctionDef) or n.name.startswith("__") or n.name == "forward":
                    continue
                decos = [ast.unparse(d) for d in n.decorator_list]
                if any(d.endswith(".setter") for d in decos):
                    continue
                static = "staticmethod" in decos
                prop = "property" in decos
                if ctor is None and not static:
                    # No constructor to build an instance with (e.g.
                    # static-constructor classes like SymbolicNode):
                    # statics are still callable without one.
                    continue
                items.append(("prop" if prop else "method", node.name, ctor_p, n.name, params_of(n, not static), static))
    return items


def child(start):
    import oxideon as ox
    if resource is not None and hasattr(resource, "setrlimit") and hasattr(resource, "RLIMIT_AS"):
        try:
            resource.setrlimit(resource.RLIMIT_AS, (8_000_000_000, 8_000_000_000))
        except (ValueError, OSError):
            pass
    only = os.environ.get("FUZZ_ONLY", "")
    log = open(OUT / "progress.log", "a", buffering=1, encoding="utf-8")
    res = open(OUT / "results.jsonl", "a", buffering=1, encoding="utf-8")
    sys.stderr = open(os.devnull, "w")
    items = collect()
    for idx in range(start, len(items)):
        kind, cls, ctor_p, member, params, static = items[idx]
        if only and only not in cls:
            continue
        log.write(f"API {idx} {cls}.{member}\n")
        calls = build_calls(params)
        if calls is None:
            continue
        ctor_calls = build_calls(ctor_p) if ctor_p is not None else None
        obj_args = ctor_calls[0] if ctor_calls else {}
        obj = None
        if kind in ("method", "prop") and not static:
            try:
                obj = getattr(ox, cls)(**obj_args)
            except BaseException:
                continue
        seen = set()
        for c in calls:
            key = repr(sorted(c.items(), key=lambda kv: kv[0]))
            if key in seen:
                continue
            seen.add(key)
            desc = f"{cls}.{member}({', '.join(f'{k}={v!r}' for k, v in c.items())})"
            log.write(f"CALL {desc[:300]}\n")
            t0 = time.time()
            try:
                if kind == "func":
                    getattr(ox, cls)(**c)
                elif kind == "ctor":
                    getattr(ox, cls)(**c)
                elif kind == "prop":
                    getattr(obj, member)
                elif static:
                    getattr(getattr(ox, cls), member)(**c)
                else:
                    getattr(obj, member)(**c)
            except BaseException as e:  # noqa: BLE001
                if type(e).__name__ == "PanicException":
                    res.write(json.dumps({"api": f"{cls}.{member}", "kind": "panic", "call": desc[:300], "msg": str(e)[:160]}) + "\n")
                    if kind in ("method", "prop") and not static:  # a panic may have poisoned the object
                        try:
                            obj = getattr(ox, cls)(**obj_args)
                        except BaseException:
                            break
            if time.time() - t0 > 5:
                res.write(json.dumps({"api": f"{cls}.{member}", "kind": "slow", "call": desc[:300], "msg": f"{time.time() - t0:.1f}s"}) + "\n")
        log.write(f"DONE {idx}\n")
    log.write("FINISHED\n")


def parent(only):
    OUT.mkdir(exist_ok=True, parents=True)
    for f in ("progress.log", "results.jsonl"):
        (OUT / f).write_text("", encoding="utf-8")
    # Fuzzed file calls use pool strings as paths; run children here so they
    # can never drop files into the caller's directory.
    scratch = OUT / "scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    start, total = 0, len(collect())
    env = dict(os.environ, FUZZ_ONLY=only or "")
    py = sys.executable  # the interpreter that has the wheel installed
    script = os.path.abspath(__file__)  # the child runs in scratch, so this must be absolute
    while start < total:
        p = subprocess.Popen([py, script, "child", str(start)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=scratch)
        last_size, last_t = -1, time.time()
        killed_for_stall = False
        while p.poll() is None:
            time.sleep(1)
            size = (OUT / "progress.log").stat().st_size
            if size != last_size:
                last_size, last_t = size, time.time()
            elif time.time() - last_t > 20:
                p.kill()
                p.wait()
                killed_for_stall = True
                break
        lines = (OUT / "progress.log").read_text(encoding="utf-8").splitlines()
        if lines and lines[-1] == "FINISHED":
            break
        api = next((l for l in reversed(lines) if l.startswith("API ")), None)
        call = next((l for l in reversed(lines) if l.startswith("CALL ")), "?")
        # SIGKILL we sent after a 20s stall is a hang; SIGKILL the child died
        # with on its own is the OOM killer (or an external kill), not a hang.
        if killed_for_stall or p.returncode is None:
            kind = "hang"
        elif p.returncode == -9:
            kind = "oom"
        else:
            kind = f"abort(rc={p.returncode})"
        if api is None:
            break
        with open(OUT / "results.jsonl", "a", encoding="utf-8") as r:
            r.write(json.dumps({"api": api.split(" ", 2)[2], "kind": kind, "call": call[5:], "msg": ""}) + "\n")
        start = int(api.split()[1]) + 1
        (OUT / "progress.log").open("a", encoding="utf-8").write(f"RESTART {start}\n")
    rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_api = {}
    for r in rows:
        by_api.setdefault((r["api"], r["kind"]), []).append(r)
    print(f"{total} APIs scanned; {len(rows)} findings in {len({a for a, _ in by_api})} APIs")
    for (api, kind), rs in sorted(by_api.items()):
        print(f"{kind:12} {api:70} x{len(rs):<3} e.g. {rs[0]['call'][:110]} :: {rs[0]['msg'][:60]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(int(sys.argv[2]))
    else:
        parent(sys.argv[1] if len(sys.argv) > 1 else "")
