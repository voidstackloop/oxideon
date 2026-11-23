#!/usr/bin/env python3
"""Regenerate python/oxideon/__init__.py (`__all__`) and python/oxideon/__init__.pyi from the built extension.

Names, parameter names, defaults and member kinds (method / staticmethod / property) come from runtime
introspection, so build and install the extension first (`maturin develop`). Types come from the Rust
signatures in src/python*.rs, falling back to `inspect.signature` annotations for pure-Python callables;
anything that cannot be mapped is `Any`.

    maturin develop && python tools/gen_stubs.py
"""
import ast
import inspect
import keyword
import math
import re
import types
import typing
from pathlib import Path

import oxideon as ox

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PKG = ROOT / "python" / "oxideon"


# ---------------------------------------------------------------- small Rust/text helpers
def split_top(s, sep=",", quotes="\"'"):
    """Split on `sep` outside brackets, angle brackets and quotes. Pass quotes='"' for Rust, where `'` is a lifetime."""
    out, depth, cur, quote = [], 0, "", None
    for ch in s:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in quotes:
            quote = ch
        elif ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [x.strip() for x in out]


def match_close(s, i, open_ch, close_ch):
    depth = 0
    for j in range(i, len(s)):
        if s[j] == open_ch:
            depth += 1
        elif s[j] == close_ch:
            depth -= 1
            if depth == 0:
                return j
    return -1


# ---------------------------------------------------------------- Rust signature table
class RustTable:
    def __init__(self):
        self.pyname = {}   # Rust wrapper struct -> Python class name
        self.members = {}  # (python class, member) -> (params {name: rust type}, ret rust type)
        self.setters = set()  # (python class, property)
        self.funcs = {}    # python function name -> (params, ret)

    def scan(self):
        files = sorted(SRC.glob("python*.rs"))
        texts = {f: f.read_text(encoding="utf-8") for f in files}
        for text in texts.values():
            for m in re.finditer(r'#\[pyclass\([^\]]*?name\s*=\s*"([^"]+)"[^\]]*\]\s*(?:#\[[^\n]*\]\s*)*pub struct (\w+)', text):
                self.pyname[m.group(2)] = m.group(1)
            for m in re.finditer(r"^\s+(Py\w+)\(\"([^\"]+)\"\) wraps", text, re.M):
                self.pyname[m.group(1)] = m.group(2)
            for m in re.finditer(r"^\s+(Py\w+)\((\w+), \"([^\"]+)\", ", text, re.M):
                self.pyname[m.group(1)] = m.group(3)
        for text in texts.values():
            self._impl_blocks(text)
            self._macro_engines(text)
            self._millennium(text)
            self._pyfunctions(text)
            self._loss_macro(text)

    def _fn_items(self, body):
        """Yield (attrs, name, params, ret) for each `pub fn` in an impl body."""
        for m in re.finditer(r"((?:[ \t]*(?:#\[[^\n]*\]|///[^\n]*)\n)*)[ \t]*pub fn (\w+)\s*(?:<[^>]*>)?\s*\(", body):
            open_i = m.end() - 1
            close_i = match_close(body, open_i, "(", ")")
            if close_i < 0:
                continue
            params = split_top(body[open_i + 1:close_i], quotes='"')
            rest = body[close_i + 1:close_i + 400]
            rm = re.match(r"\s*(?:->\s*([^{]+?))?\s*(?:where[^{]*)?\{", rest)
            ret = (rm.group(1) or "").strip() if rm else ""
            yield m.group(1), m.group(2), params, ret

    def _record(self, cls, name, params, ret, attrs):
        pd = {}
        for p in params:
            if p in ("&self", "&mut self", "self", "mut self") or ":" not in p:
                continue
            n, t = p.split(":", 1)
            n, t = n.strip().lstrip("_"), t.strip()
            if t.startswith("Python"):
                continue
            pd[n] = t
        nm = re.search(r'#\[pyo3\([^\]]*?\bname\s*=\s*"(\w+)"', attrs)
        if nm:
            name = nm.group(1)
        key = (cls, name)
        gm = re.search(r"#\[getter(?:\((\w+)\))?\]", attrs)
        if gm:
            key = (cls, gm.group(1) or name)
        sm = re.search(r"#\[setter(?:\((\w+)\))?\]", attrs)
        if sm:
            self.setters.add((cls, sm.group(1) or name.removeprefix("set_")))
            return
        if "#[new]" in attrs:
            key = (cls, "__init__")
        self.members[key] = (pd, ret)

    def _impl_blocks(self, text):
        for m in re.finditer(r"#\[pymethods\]\s*impl (\w+) \{", text):
            rust = m.group(1)
            cls = self.pyname.get(rust)
            if not cls:
                continue
            end = match_close(text, m.end() - 1, "{", "}")
            if end < 0:
                continue  # unbalanced block (file mid-edit): skip, do not misparse
            for attrs, name, params, ret in self._fn_items(text[m.end():end]):
                self._record(cls, name, params, ret, attrs)

    def _macro_engines(self, text):
        for m in re.finditer(r"py_engine! \{(.*?)\n\}\n", text, re.S):
            block = m.group(1)
            hm = re.search(r'(Py\w+)\("([^"]+)"\) wraps', block)
            if not hm:
                continue
            cls = hm.group(2)
            for line in block.splitlines():
                line = line.strip()
                nm = re.match(r"new\((.*?)\)( -> res)?;$", line)
                if nm:
                    self._record(cls, "new", split_top(nm.group(1)), "Self", "#[new]")
                    continue
                mm = re.match(r"(method|method_res|assoc|assoc_res) (\w+)\((.*?)\) -> (.+);$", line)
                if mm:
                    kind, name, ps, ret = mm.groups()
                    ret = f"PyResult<{ret}>" if kind.endswith("_res") else ret
                    self._record(cls, name, ([] if kind.startswith("assoc") else ["&self"]) + split_top(ps), ret, "")

    def _millennium(self, text):
        for m in re.finditer(r"^\s+(Py\w+)\((\w+), \"([^\"]+)\", (\w+)\);$", text, re.M):
            cls, fn = m.group(3), m.group(4)
            self._record(cls, "new", ["characteristic_scale: f64", "coupling_parameter: f64"], "Self", "#[new]")
            self._record(cls, fn, ["&self", "probe_energy: f64"], "PyResult<(f64, f64, bool)>", "")
        for m in re.finditer(r"^\s+(Py\w+)\((\w+), \"([^\"]+)\", (\d+), (\w+)\);$", text, re.M):
            cls, fn = m.group(3), m.group(5)
            self._record(cls, "new", ["num_layers: usize"], "Self", "#[new]")
            self._record(cls, fn, ["&self"], "PyResult<f64>", "")

    def _loss_macro(self, text):
        """`py_tensor_losses! { "name", fn_ident = path, (tensor args), float_kwarg = default, ...; }`"""
        for m in re.finditer(r"py_tensor_losses! \{(.*?)\n\}\n", text, re.S):
            for entry in m.group(1).split(";"):
                em = re.match(r'\s*"(\w+)",\s*\w+\s*=\s*[\w:]+,\s*\(([^)]*)\)((?:,\s*\w+\s*=\s*[-\w.]+)*)\s*$', entry)
                if em:
                    name, tensors, extras = em.groups()
                    params = {a.strip(): "&PyTensor" for a in tensors.split(",")}
                    params.update({k: "f64" for k in re.findall(r"(\w+)\s*=", extras)})
                    self.funcs[name] = (params, "PyResult<PyTensor>")

    def _pyfunctions(self, text):
        for m in re.finditer(r"#\[pyfunction\]\s*((?:#\[[^\n]*\]\s*)*)pub fn (\w+)\s*(?:<[^>]*>)?\s*\(", text):
            open_i = m.end() - 1
            close_i = match_close(text, open_i, "(", ")")
            if close_i < 0:
                continue  # unbalanced signature (file mid-edit): skip, do not misparse
            params = split_top(text[open_i + 1:close_i], quotes='"')
            rm = re.match(r"\s*(?:->\s*([^{]+?))?\s*\{", text[close_i + 1:close_i + 400])
            ret = (rm.group(1) or "").strip() if rm else ""
            pd = {}
            for p in params:
                if ":" in p:
                    n, t = p.split(":", 1)
                    if not t.strip().startswith("Python"):
                        n, t = n.strip().lstrip("_"), t.strip()
                        pd[n] = "ClassIndices" if n == "targets" and "PyAny" in t else t  # see `class_indices` in python.rs
            nm = re.search(r'\bname\s*=\s*"(\w+)"', m.group(1))
            self.funcs[nm.group(1) if nm else m.group(2)] = (pd, ret)


# ---------------------------------------------------------------- Rust type -> Python annotation
INTS = {"i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize"}


def py_type(rt, table, selfcls):
    rt = (rt or "").strip()
    if not rt:
        return "None"
    rt = re.sub(r"^&\s*(?:'\w+\s+)?(?:mut\s+)?", "", rt).strip()
    if rt == "()":
        return "None"
    if rt in INTS:
        return "int"
    if rt in ("f32", "f64"):
        return "float"
    if rt == "bool":
        return "bool"
    if rt in ("String", "str", "char"):
        return "str"
    if rt == "[u8]":  # PyO3 passes `&[u8]` as `bytes` (a list of ints is rejected); `Vec<u8>` stays a list
        return "bytes"
    if rt == "Self":
        return selfcls or "Any"
    if rt == "ClassIndices":
        return "Union[Tensor, list[int]]"
    if rt.startswith("(") and rt.endswith(")"):
        parts = split_top(rt[1:-1], quotes='"')
        return "tuple[" + ", ".join(py_type(p, table, selfcls) for p in parts) + "]" if parts else "None"
    if rt.startswith("[") and rt.endswith("]"):
        return "list[" + py_type(split_top(rt[1:-1], ";", quotes='"')[0], table, selfcls) + "]"
    gm = re.match(r"^([\w:]+)\s*<(.*)>$", rt, re.S)
    if gm:
        head, args = gm.group(1).split("::")[-1], split_top(gm.group(2), quotes='"')
        args = [a for a in args if not a.startswith("'")]
        if head == "PyResult" and args:
            return py_type(args[0], table, selfcls)
        if head == "Option" and args:
            inner = py_type(args[0], table, selfcls)
            return inner if inner == "Any" else f"Optional[{inner}]"
        if head in ("Vec", "VecDeque", "HashSet", "BTreeSet") and args:
            return f"list[{py_type(args[0], table, selfcls)}]"
        if head in ("HashMap", "BTreeMap") and len(args) == 2:
            return f"dict[{py_type(args[0], table, selfcls)}, {py_type(args[1], table, selfcls)}]"
        if head in ("PyRef", "PyRefMut", "Py", "Bound") and args:
            return py_type(args[-1], table, selfcls)
        if head in ("Box", "Arc", "Rc"):
            return py_type(args[0], table, selfcls) if args else "Any"
        return "Any"
    name = rt.split("::")[-1]
    if name in table.pyname:
        return table.pyname[name]
    if name == "PyBytes":
        return "bytes"
    if name == "PyDict":
        return "dict[str, Any]"
    if name == "PyList":
        return "list[Any]"
    return "Any"


# ---------------------------------------------------------------- runtime text signatures
def parse_textsig(sig):
    """'($self, a, b=1.0, /, *, c)' -> [(name_or_marker, default or None)] without self/type, or None."""
    if not sig or not (sig.startswith("(") and sig.endswith(")")):
        return None
    out = []
    for p in split_top(sig[1:-1]):
        if p in ("$self", "$type", "$module"):
            continue
        name, _, default = p.partition("=")
        out.append((name.strip(), default.strip() or None))
    return out


def clean_default(d):
    if d is None:
        return None
    if d in ("None", "True", "False", "..."):
        return d
    try:
        ast.literal_eval(d)
        return d
    except Exception:
        return "..."


def render_params(parsed, types, table, selfcls, with_self):
    """Return the parameter list text, or None when it cannot be rendered safely."""
    if parsed is None:
        return None
    parts = ["self"] if with_self else []
    for name, default in parsed:
        if name in ("/", "*"):
            parts.append(name)
            continue
        if name.startswith("**"):
            parts.append(f"{name}: Any")
            continue
        if name.startswith("*"):
            parts.append(f"{name}: Any")
            continue
        if keyword.iskeyword(name):
            return None
        ann = py_type(types[name], table, selfcls) if name in types else "Any"
        d = clean_default(default)
        if d == "None" and ann not in ("Any",) and not ann.startswith("Optional["):
            ann = f"Optional[{ann}]"
        parts.append(f"{name}: {ann}" + (f" = {d}" if d is not None else ""))
    return ", ".join(parts)


# ---------------------------------------------------------------- stub emission
DUNDERS = {
    "__repr__": "(self) -> str", "__str__": "(self) -> str", "__len__": "(self) -> int", "__bool__": "(self) -> bool",
    "__hash__": "(self) -> int", "__iter__": "(self) -> Any", "__next__": "(self) -> Any",
    "__getitem__": "(self, key: Any) -> Any", "__setitem__": "(self, key: Any, value: Any) -> None",
    "__call__": "(self, *args: Any, **kwargs: Any) -> Any", "__contains__": "(self, item: Any) -> bool",
    "__eq__": "(self, other: object) -> bool", "__ne__": "(self, other: object) -> bool",
    "__float__": "(self) -> float", "__int__": "(self) -> int",
    **{d: "(self, other: Any) -> Any" for d in (
        "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
        "__pow__", "__matmul__", "__lt__", "__le__", "__gt__", "__ge__")},
    "__neg__": "(self) -> Any", "__abs__": "(self) -> Any",
}
# `Tensor` operators accept a tensor or a number and return a tensor (comparisons: a boolean tensor).
TENSOR_DUNDERS = {
    **{d: "(self, other: Union[Tensor, float]) -> Tensor" for d in (
        "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
        "__pow__", "__lt__", "__le__", "__gt__", "__ge__")},
    "__matmul__": "(self, other: Tensor) -> Tensor",
    "__eq__": "(self, other: object) -> Tensor", "__ne__": "(self, other: object) -> Tensor",
    "__neg__": "(self) -> Tensor", "__abs__": "(self) -> Tensor",
    "__getitem__": "(self, key: Any) -> Tensor", "__len__": "(self) -> int",
}
SKIP_DUNDER = {"__new__", "__init__", "__doc__", "__module__", "__dict__", "__weakref__", "__text_signature__", "__class__", "__init_subclass__",
               "__subclasshook__", "__getattribute__", "__setattr__", "__delattr__", "__dir__", "__format__", "__reduce__", "__reduce_ex__",
               "__sizeof__", "__getstate__", "__lt__", "__le__", "__gt__", "__ge__"}


def has_constructor(cls):
    """pyo3 classes without #[new] are not constructible from Python (TypeError).

    pyo3 <= 0.23 raises 'No constructor defined for ...'; 0.24+ raises the
    standard "cannot create '...' instances". Accept both messages so the
    stubs stay correct across pyo3 versions.
    """
    try:
        cls.__new__(cls)
    except TypeError as e:
        msg = str(e)
        return "No constructor defined" not in msg and "cannot create" not in msg
    except Exception:
        pass
    return True


# ------------------------------------------- pure-Python signature fallback
_PY_SAFE_TYPES = {
    "int", "float", "bool", "str", "bytes", "complex", "list",
    "dict", "tuple", "set", "frozenset", "type", "object",
}


def _py_ann_to_str(ann):
    """Map a Python annotation to a stub type, or None when it is unsafe.

    Only builtin containers/scalars (plus oxideon Tensor) are trusted, so the
    emitted stub never references names the stub does not define.
    """
    if ann is inspect.Parameter.empty:
        return "Any"
    if ann is None or ann is type(None):
        return "None"
    if isinstance(ann, type):
        mod = getattr(ann, "__module__", "")
        if mod == "builtins" and ann.__name__ in _PY_SAFE_TYPES:
            return ann.__name__
        if ann.__name__ == "Tensor" and mod.split(".")[0] == "oxideon":
            return "Tensor"
        return None
    if isinstance(ann, str):
        m = re.fullmatch(r"([A-Za-z_]\w*)", ann.strip())
        if m and (m.group(1) in _PY_SAFE_TYPES or m.group(1) in ("Tensor", "Any")):
            return m.group(1)
        return None
    if isinstance(ann, types.UnionType):
        parts = [_py_ann_to_str(a) for a in ann.__args__]
        if any(p is None for p in parts):
            return None
        return " | ".join(parts)
    origin = getattr(ann, "__origin__", None)
    if origin is not None:
        if origin is typing.Union:
            parts = []
            for a in getattr(ann, "__args__", ()):
                if a is type(None):
                    parts.append("None")
                else:
                    m = _py_ann_to_str(a)
                    if m is None:
                        return None
                    parts.append(m)
            return " | ".join(parts) if parts else None
        oname = getattr(origin, "__name__", "")
        if oname in ("list", "dict", "tuple", "set", "frozenset"):
            args = [_py_ann_to_str(a) for a in getattr(ann, "__args__", ())]
            if not args or any(a is None for a in args):
                return None
            return f"{oname}[{', '.join(args)}]"
    return None


def _safe_default_repr(d):
    """repr() for simple default values, else None (caller emits `...`)."""
    if d is None or isinstance(d, (bool, int, str, bytes)):
        return repr(d)
    if isinstance(d, float) and math.isfinite(d):
        return repr(d)
    if isinstance(d, (list, tuple)):
        inner = [_safe_default_repr(v) for v in d]
        if any(v is None for v in inner):
            return None
        if isinstance(d, list):
            return "[" + ", ".join(inner) + "]"
        if len(inner) == 0:
            return "()"
        if len(inner) == 1:
            return f"({inner[0]},)"
        return "(" + ", ".join(inner) + ")"
    if isinstance(d, dict):
        items = [(_safe_default_repr(k), _safe_default_repr(v)) for k, v in d.items()]
        if any(k is None or v is None for k, v in items):
            return None
        return "{" + ", ".join(f"{k}: {v}" for k, v in items) + "}"
    if isinstance(d, (set, frozenset)):
        if not d:
            return None
        inner = [_safe_default_repr(v) for v in sorted(d, key=repr)]
        if any(v is None for v in inner):
            return None
        return "{" + ", ".join(inner) + "}"
    return None


def _py_default_to_str(d):
    if d is inspect.Parameter.empty:
        return ""
    r = _safe_default_repr(d)
    return f" = {r}" if r is not None else " = ..."


def render_py_sig(fn, skip_first):
    """Render (params, ret) for a pure-Python callable via inspect, else None.

    Used only when the callable has no `__text_signature__` (i.e. it is not a
    Rust binding). Unmappable annotations become `Any`, unsafe defaults `...`.
    """
    if fn is None or fn in (object.__init__, object.__new__):
        return None
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    P = inspect.Parameter
    items = []
    skipped = False
    try:
        for p in sig.parameters.values():
            if skip_first and not skipped and p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD):
                skipped = True
                continue
            if p.kind == P.VAR_POSITIONAL:
                items.append(("*args: Any", p.kind))
            elif p.kind == P.VAR_KEYWORD:
                items.append(("**kwargs: Any", p.kind))
            else:
                t = _py_ann_to_str(p.annotation) or "Any"
                items.append((f"{p.name}: {t}{_py_default_to_str(p.default)}", p.kind))
    except Exception:
        return None
    texts = [t for t, _ in items]
    kinds = [k for _, k in items]
    if P.POSITIONAL_ONLY in kinds:
        last = max(i for i, k in enumerate(kinds) if k == P.POSITIONAL_ONLY)
        texts.insert(last + 1, "/")
        kinds.insert(last + 1, None)
    if P.KEYWORD_ONLY in kinds and P.VAR_POSITIONAL not in kinds:
        first = next(i for i, k in enumerate(kinds) if k == P.KEYWORD_ONLY)
        texts.insert(first, "*")
    try:
        ret = _py_ann_to_str(sig.return_annotation) or "Any"
    except Exception:
        ret = "Any"
    return ", ".join(texts), ret


def doc_lines(obj):
    doc = inspect.getdoc(obj)
    if not doc:
        return []
    doc = doc.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return ['    """' + doc.replace("\n", "\n    ") + '"""']


def emit_class(name, cls, table, stats):
    lines = [f"class {name}:"]
    lines += doc_lines(cls)
    body = []
    types_of = lambda member: table.members.get((name, member), ({}, ""))
    # constructor: omitted for classes that cannot be built from Python
    ptypes, _ = types_of("__init__")
    sig = getattr(cls, "__text_signature__", None)
    rendered = render_params(parse_textsig(sig), ptypes, table, name, True)
    if sig is not None and rendered is not None:
        stats["typed"] += 1
        body.append(f"    def __init__({rendered}) -> None: ...")
    elif has_constructor(cls):
        py = render_py_sig(getattr(cls, "__init__", None), True)
        if py is not None:
            stats["typed"] += 1
            body.append(f"    def __init__(self{', ' + py[0] if py[0] else ''}) -> None: ...")
        else:
            stats["any"] += 1
            body.append("    def __init__(self, *args: Any, **kwargs: Any) -> None: ...")
    for member in sorted(vars(cls)):
        if member == "__call__" and hasattr(cls, "forward"):  # layers: `layer(x)` is `layer.forward(x)`
            fwd_types, fwd_ret = types_of("forward")
            r = render_params(parse_textsig(getattr(cls.forward, "__text_signature__", None)), fwd_types, table, name, True)
            if r is not None:
                body.append(f"    def __call__({r}) -> {py_type(fwd_ret, table, name)}: ...")
                continue
        if member.startswith("__"):
            sig = ({**DUNDERS, **TENSOR_DUNDERS} if name == "Tensor" else DUNDERS).get(member)
            if sig is not None and member in vars(cls):
                # element-wise == / != deliberately return a Tensor, which narrows object.__eq__'s bool
                ignore = "  # type: ignore[override]" if name == "Tensor" and member in ("__eq__", "__ne__") else ""
                body.append(f"    def {member}{sig}: ...{ignore}")
            continue
        if member.startswith("_") and member.endswith("_"):
            continue  # CPython Enum/internal sunder hooks vary across Python versions
        raw = inspect.getattr_static(cls, member)
        ptypes, ret = types_of(member)
        if isinstance(raw, staticmethod):
            fn = getattr(cls, member)
            r = render_params(parse_textsig(getattr(fn, "__text_signature__", None)), ptypes, table, name, False)
            ok = r is not None and getattr(fn, "__text_signature__", None) is not None
            if ok:
                rt = py_type(ret, table, name) if member in {k[1] for k in table.members if k[0] == name} else "Any"
            else:
                py = render_py_sig(fn, False)
                if py is not None:
                    r, rt = py
                    ok = True
                else:
                    r, rt = "*args: Any, **kwargs: Any", "Any"
            stats["typed" if ok else "any"] += 1
            body.append("    @staticmethod")
            body.append(f"    def {member}({r}) -> {rt}: ...")
        elif type(raw).__name__ == "getset_descriptor":
            rt = py_type(ret, table, name) if (name, member) in table.members else "Any"
            body.append("    @property")
            body.append(f"    def {member}(self) -> {rt}: ...")
            if (name, member) in table.setters:
                body.append(f"    @{member}.setter")
                body.append(f"    def {member}(self, value: {rt}) -> None: ...")
            stats["typed" if rt != "Any" else "any"] += 1
        elif callable(raw) or type(raw).__name__ in ("method_descriptor", "classmethod_descriptor") or isinstance(raw, classmethod):
            fn = getattr(cls, member)
            fsig = getattr(fn, "__text_signature__", None)
            r = render_params(parse_textsig(fsig), ptypes, table, name, True)
            ok = r is not None and fsig is not None
            if ok:
                rt = py_type(ret, table, name) if (name, member) in table.members else "Any"
            else:
                pyfn = raw.__func__ if isinstance(raw, classmethod) else fn
                py = render_py_sig(pyfn, True)
                if py is not None:
                    r = f"self, {py[0]}" if py[0] else "self"
                    rt = py[1]
                    ok = True
                else:
                    r = "self, *args: Any, **kwargs: Any"
                    rt = "Any"
            stats["typed" if (ok and rt != "Any") else "any"] += 1
            if isinstance(raw, classmethod):
                body.append("    @classmethod")
            body.append(f"    def {member}({r}) -> {rt}: ...")
    lines += body
    if len(lines) == 1:  # no doc, constructor or members: keep the class body valid
        lines.append("    ...")
    lines.append("")
    return lines


def main():
    table = RustTable()
    table.scan()
    public = sorted(s for s in dir(ox) if not s.startswith("_") and not inspect.ismodule(getattr(ox, s)))
    # Direct-child oxideon submodules from runtime ground truth.
    submods = []
    for _s in dir(ox):
        if _s.startswith("_") or _s == "oxideon":
            continue
        try:
            _m = getattr(ox, _s)
        except Exception:
            continue
        if inspect.ismodule(_m) and getattr(_m, "__name__", "") == "oxideon." + _s:
            submods.append(_s)
    submods.sort()

    init_py = [
        '"""Oxideon: Multi-DType Deep Learning, Autograd, Scientific PDEs, Quantum & Millennial Singularity Apex Engine."""',
        "# Generated by tools/gen_stubs.py; do not edit by hand (edit the template in main() instead).",
        "",
        "from .oxideon import *  # noqa: F401, F403",
        "from .oxideon import __version__",
        "",
        "# Top-level tensor constructor helpers",
        "ones = Tensor.ones",
        "zeros = Tensor.zeros",
        "randn = Tensor.randn",
        "rand = Tensor.rand",
        "",
        "from .serialization import (",
        "    _reconstruct_tensor,",
        "    save_file,",
        "    load_file,",
        "    load_file_metadata,",
        "    save,",
        "    load,",
        ")",
        "from . import nn, optim, data, callbacks, functional, metrics, utils, transforms, training, text, quantization, init, peft, models, amp, losses, explain, search, root, distributed, gnn, rl, diffusion, prune, audio, sparse, clustering, benchmark, export, alignment, timeseries, spatial, flow, calibration, adversarial, distill, active_learning, multimodal, meta, fairness, ssl, hpo, causal, privacy, genetic, graph, ensemble, rag, token_healing, anomaly, memory, synthetic_data, distill_attn, safety, symbolic, pde, compression, spectral, conformal, continual, vectorization  # noqa: E402  (hand-written namespaces, they import names from this package)",
        "F = functional",
        "Trainer = training.Trainer",
        "quantize_dynamic = quantization.quantize_dynamic",
        "from .nn import *  # noqa: F401, F403",
        "from .optim import *  # noqa: F401, F403",
        "from .data import *  # noqa: F401, F403",
        "from .callbacks import *  # noqa: F401, F403",
        "from .functional import *  # noqa: F401, F403",
        "from .metrics import *  # noqa: F401, F403",
        "from .utils import *  # noqa: F401, F403",
        "from .transforms import *  # noqa: F401, F403",
        "from .training import *  # noqa: F401, F403",
        "from .text import *  # noqa: F401, F403",
        "from .quantization import *  # noqa: F401, F403",
        "from .models import *  # noqa: F401, F403",
        "from .peft import *  # noqa: F401, F403",
        "from .amp import *  # noqa: F401, F403",
        "from .losses import *  # noqa: F401, F403",
        "from .explain import *  # noqa: F401, F403",
        "from .search import *  # noqa: F401, F403",
        "from .root import *  # noqa: F401, F403",
        "from .distributed import *  # noqa: F401, F403",
        "from .gnn import *  # noqa: F401, F403",
        "from .rl import *  # noqa: F401, F403",
        "from .diffusion import *  # noqa: F401, F403",
        "from .prune import *  # noqa: F401, F403",
        "from .audio import *  # noqa: F401, F403",
        "from .sparse import *  # noqa: F401, F403",
        "from .clustering import *  # noqa: F401, F403",
        "from .benchmark import *  # noqa: F401, F403",
        "from .export import *  # noqa: F401, F403",
        "from .alignment import *  # noqa: F401, F403",
        "from .timeseries import *  # noqa: F401, F403",
        "from .spatial import *  # noqa: F401, F403",
        "from .flow import *  # noqa: F401, F403",
        "from .calibration import *  # noqa: F401, F403",
        "from .adversarial import *  # noqa: F401, F403",
        "from .distill import *  # noqa: F401, F403",
        "from .active_learning import *  # noqa: F401, F403",
        "from .multimodal import *  # noqa: F401, F403",
        "from .meta import *  # noqa: F401, F403",
        "from .fairness import *  # noqa: F401, F403",
        "from .ssl import *  # noqa: F401, F403",
        "from .hpo import *  # noqa: F401, F403",
        "from .causal import *  # noqa: F401, F403",
        "from .privacy import *  # noqa: F401, F403",
        "from .genetic import *  # noqa: F401, F403",
        "from .graph import *  # noqa: F401, F403",
        "from .ensemble import *  # noqa: F401, F403",
        "from .rag import *  # noqa: F401, F403",
        "from .token_healing import *  # noqa: F401, F403",
        "from .anomaly import *  # noqa: F401, F403",
        "from .memory import *  # noqa: F401, F403",
        "from .synthetic_data import *  # noqa: F401, F403",
        "from .distill_attn import *  # noqa: F401, F403",
        "from .safety import *  # noqa: F401, F403",
        "from .symbolic import *  # noqa: F401, F403",
        "from .pde import *  # noqa: F401, F403",
        "from .compression import *  # noqa: F401, F403",
        "from .spectral import *  # noqa: F401, F403",
        "from .conformal import *  # noqa: F401, F403",
        "from .continual import *  # noqa: F401, F403",
        "from .graph_gen import *  # noqa: F401, F403",
        "from .bayes import *  # noqa: F401, F403",
        "from .energy import *  # noqa: F401, F403",
        "from .audio_synth import *  # noqa: F401, F403",
        "from .geometry import *  # noqa: F401, F403",
        "from .causal_discovery import *  # noqa: F401, F403",
        "from .federated import *  # noqa: F401, F403",
        "from .sparse_attn import *  # noqa: F401, F403",
        "from .rl_continuous import *  # noqa: F401, F403",
        "from .interpret import *  # noqa: F401, F403",
        "from .quantization_advanced import *  # noqa: F401, F403",
        "from .state_space import *  # noqa: F401, F403",
        "from .contrastive_learning import *  # noqa: F401, F403",
        "from .mixture_of_experts import *  # noqa: F401, F403",
        "from .protein import *  # noqa: F401, F403",
        "from .graph_matching import *  # noqa: F401, F403",
        "from .information_theory import *  # noqa: F401, F403",
        "from .survival import *  # noqa: F401, F403",
        "from .uncertainty_quant import *  # noqa: F401, F403",
        "from .meta_rl import *  # noqa: F401, F403",
        "from . import cloud",
        "from . import logging as logging",
        "from .logging import is_debug, set_debug, enable_debug, disable_debug, debug_mode, get_logger, debug_tensor, check_nan, LogLevel",
        "from .kquants import *  # noqa: F401, F403",
        "from . import kquants as kquants",
        "from .kv_cache import *  # noqa: F401, F403",
        "from .speculative import *  # noqa: F401, F403",
        "from .flash_attn import *  # noqa: F401, F403",
        "from .dora import *  # noqa: F401, F403",
        "from .graph_kernels import *  # noqa: F401, F403",
        "from .sparse_matrix import *  # noqa: F401, F403",
        "from .tokenizers import *  # noqa: F401, F403",
        "from .dpo import *  # noqa: F401, F403",
        "from .multimodal_fusion import *  # noqa: F401, F403",
        "from .vision_transformer import *  # noqa: F401, F403",
        "from .interop import *  # noqa: F401, F403",
        "from . import interop as interop",
        "from .prefix_cache import *  # noqa: F401, F403",
        "from . import prefix_cache as prefix_cache",
        "from .mcts import *  # noqa: F401, F403",
        "from . import mcts as mcts",
        "from .graph_transformer import *  # noqa: F401, F403",
        "from . import graph_transformer as graph_transformer",
        "from .bitnet import *  # noqa: F401, F403",
        "from . import bitnet as bitnet",
        "from .diffusion_sampler import *  # noqa: F401, F403",
        "from . import diffusion_sampler as diffusion_sampler",
        "from .audio_codec import *  # noqa: F401, F403",
        "from . import audio_codec as audio_codec",
        "from .adaptive_conformal import *  # noqa: F401, F403",
        "from . import adaptive_conformal as adaptive_conformal",
        "from .synaptic_intelligence import *  # noqa: F401, F403",
        "from . import synaptic_intelligence as synaptic_intelligence",
        "from .multimodal_vlm import *  # noqa: F401, F403",
        "from . import multimodal_vlm as multimodal_vlm",
        "from .expert_parallel import *  # noqa: F401, F403",
        "from . import expert_parallel as expert_parallel",
        "from .attention_ext import *  # noqa: F401, F403",
        "from . import attention_ext as attention_ext",
        "from .optim_ext import *  # noqa: F401, F403",
        "from . import optim_ext as optim_ext",
        "from .losses_ext import *  # noqa: F401, F403",
        "from . import losses_ext as losses_ext",
        "from .gnn_ext import *  # noqa: F401, F403",
        "from . import gnn_ext as gnn_ext",
        "from .ode_pde_solvers import *  # noqa: F401, F403",
        "from . import ode_pde_solvers as ode_pde_solvers",
        "from .quant_compression import *  # noqa: F401, F403",
        "from . import quant_compression as quant_compression",
        "from .numerical import *  # noqa: F401, F403",
        "from . import numerical as numerical",
        "from .stats import *  # noqa: F401, F403",
        "from . import stats as stats",
        "from .physics import *  # noqa: F401, F403",
        "from . import physics as physics",
        "from .signal_metrics import *  # noqa: F401, F403",
        "from . import signal_metrics as signal_metrics",
        "from .primitives import *  # noqa: F401, F403",
        "from . import primitives as primitives",
        "from .special import *  # noqa: F401, F403",
        "from . import special as special",
        "from .distributions import *  # noqa: F401, F403",
        "from . import distributions as distributions",
        "from .profiler import *  # noqa: F401, F403",
        "from . import profiler as profiler",
        "from .jit import *  # noqa: F401, F403",
        "from . import jit as jit",
        "from .hub import download_url_to_file, load_state_dict_from_url  # noqa: F401",
        "from . import hub as hub",
        "from . import fft as fft",
        "from . import linalg as linalg",
        "from .func import *  # noqa: F401, F403",
        "from . import func as func",
        "from .signal import *  # noqa: F401, F403",
        "from . import signal as signal",
        "from .nested import *  # noqa: F401, F403",
        "from . import nested as nested",
        "from .merge import *  # noqa: F401, F403",
        "from . import merge as merge",
        "from .edit import *  # noqa: F401, F403",
        "from . import edit as edit",
        "from .curate import *  # noqa: F401, F403",
        "from . import curate as curate",
        "from .retrieval import *  # noqa: F401, F403",
        "from . import retrieval as retrieval",
        "from .speech import *  # noqa: F401, F403",
        "from . import speech as speech",
        "from .nerf import *  # noqa: F401, F403",
        "from . import nerf as nerf",
        "from .splat import *  # noqa: F401, F403",
        "from . import splat as splat",
        "from .video import *  # noqa: F401, F403",
        "from . import video as video",
        "from .vectorization import *  # noqa: F401, F403",
        "from . import vectorization as vectorization",
        "from . import cuda as cuda",
        "from . import mps as mps",
        "from . import backends as backends",
        "from . import checkpoint as _checkpoint_mod",
        "from . import checkpoint as checkpoint",
        "from . import fx as fx",
        "from . import ao as ao",
        "from . import testing as testing",
        "from . import evals as evals",
        "from . import onnx as onnx",
        "from . import vision as vision",
        "from . import kernels as kernels",
        "",
        "# Ensure canonical serialization methods are not shadowed by submodules",
        "save = serialization.save",
        "load = serialization.load",
        "",
        "# Top-level autograd.grad with functional-grad dispatch",
        "from .oxideon import grad as _native_grad",
        "",
        "def grad(outputs_or_fn, inputs=None, *args, create_graph=False, **kwargs):",
        '    """Computes gradients either in functional style or in autograd tape style."""',
        '    if callable(outputs_or_fn) and not hasattr(outputs_or_fn, "shape") and not isinstance(outputs_or_fn, (list, tuple)):',
        "        return func.grad(outputs_or_fn, *args, **kwargs)",
        "    if inputs is None:",
        "        raise TypeError('grad() missing required argument: inputs')",
        "    out_list = [outputs_or_fn] if isinstance(outputs_or_fn, Tensor) else list(outputs_or_fn)",
        "    inp_list = [inputs] if isinstance(inputs, Tensor) else list(inputs)",
        "    return _native_grad(out_list, inp_list, create_graph=create_graph)",
        "",
        "import sys as _sys",
        '_sys.modules["oxideon.cuda"] = cuda',
        '_sys.modules["oxideon.mps"] = mps',
        '_sys.modules["oxideon.backends"] = backends',
        '_sys.modules["oxideon.checkpoint"] = _checkpoint_mod',
        '_sys.modules["oxideon.fx"] = fx',
        '_sys.modules["oxideon.ao"] = ao',
        '_sys.modules["oxideon.testing"] = testing',
        '_sys.modules["oxideon.evals"] = evals',
        '_sys.modules["oxideon.onnx"] = onnx',
        '_sys.modules["oxideon.vision"] = vision',
        '_sys.modules["oxideon.kernels"] = kernels',
        '_sys.modules["oxideon.utils.checkpoint"] = _checkpoint_mod',
        "",
        "__all__ = [",
        '    "__version__",',
        '    "cuda",',
        '    "mps",',
        '    "backends",',
        '    "checkpoint",',
        '    "fx",',
        '    "ao",',
        '    "testing",',
        '    "evals",',
        '    "onnx",',
        '    "vision",',
        '    "kernels",',
        '    "nn",',
        '    "optim",',
        '    "data",',
        '    "callbacks",',
        '    "functional",',
        '    "F",',
        '    "interop",',
        '    "from_networkx",',
        '    "to_networkx",',
        '    "from_sympy",',
        '    "to_sympy",',
        '    "from_scipy_sparse",',
        '    "to_scipy_sparse",',
        '    "load_torch_state_dict",',
        '    "export_torch_state_dict",',
        '    "from_jax",',
        '    "to_jax",',
        '    "from_arrow",',
        '    "metrics",',
        '    "utils",',
        '    "transforms",',
        '    "training",',
        '    "Trainer",',
        '    "text",',
        '    "quantization",',
        '    "quantize_dynamic",',
        '    "init",',
        '    "peft",',
        '    "models",',
        '    "amp",',
        '    "losses",',
        '    "explain",',
        '    "search",',
        '    "root",',
        '    "distributed",',
        '    "gnn",',
        '    "rl",',
        '    "diffusion",',
        '    "prune",',
        '    "audio",',
        '    "sparse",',
        '    "clustering",',
        '    "benchmark",',
        '    "export",',
        '    "alignment",',
        '    "timeseries",',
        '    "spatial",',
        '    "flow",',
        '    "calibration",',
        '    "adversarial",',
        '    "distill",',
        '    "active_learning",',
        '    "multimodal",',
        '    "meta",',
        '    "fairness",',
        '    "ssl",',
        '    "hpo",',
        '    "causal",',
        '    "privacy",',
        '    "genetic",',
        '    "graph",',
        '    "ensemble",',
        '    "rag",',
        '    "token_healing",',
        '    "anomaly",',
        '    "memory",',
        '    "synthetic_data",',
        '    "distill_attn",',
        '    "safety",',
        '    "symbolic",',
        '    "pde",',
        '    "compression",',
        '    "spectral",',
        '    "conformal",',
        '    "continual",',
        '    "graph_gen",',
        '    "bayes",',
        '    "energy",',
        '    "audio_synth",',
        '    "geometry",',
        '    "causal_discovery",',
        '    "federated",',
        '    "sparse_attn",',
        '    "rl_continuous",',
        '    "interpret",',
        '    "quantization_advanced",',
        '    "state_space",',
        '    "contrastive_learning",',
        '    "mixture_of_experts",',
        '    "protein",',
        '    "graph_matching",',
        '    "information_theory",',
        '    "survival",',
        '    "uncertainty_quant",',
        '    "meta_rl",',
        '    "cloud",',
        '    "logging",',
        '    "is_debug",',
        '    "set_debug",',
        '    "enable_debug",',
        '    "disable_debug",',
        '    "debug_mode",',
        '    "get_logger",',
        '    "debug_tensor",',
        '    "check_nan",',
        '    "LogLevel",',
        '    "kquants",',
        '    "QuantizedKBlock",',
        '    "Q2KQuantizer",',
        '    "Q3KQuantizer",',
        '    "Q5KQuantizer",',
        '    "Q6KQuantizer",',
        '    "quantize_k_quant",',
        '    "dequantize_k_quant",',
        '    "quantize_ternary_bitnet",',
        '    "dequantize_ternary_bitnet",',
        '    "BitLinear",',
        '    "PagedKVCache",',
        '    "StandardKVCache",',
        '    "SpeculativeStats",',
        '    "SpeculativeDraftDecoder",',
        '    "flash_attention_forward",',
        '    "TiledAttention",',
        '    "DoRALinear",',
        '    "LoRAPlusLinear",',
        '    "weisfeiler_lehman_kernel",',
        '    "random_walk_kernel",',
        '    "CSRMatrix",',
        '    "BPETokenizer",',
        '    "ORPOLoss",',
        '    "KTOLoss",',
        '    "CrossAttentionFusion",',
        '    "GatedMultimodalFusion",',
        '    "PatchEmbedding",',
        '    "TransformerEncoderBlock",',
        '    "VisionTransformer",',
        # Big Flagship 1: Prefix Cache
        '    "prefix_cache",',
        '    "RadixPrefixCache",',
        '    "RadixNode",',
        # Big Flagship 2: MCTS
        '    "mcts",',
        '    "MCTS",',
        '    "MCTSNode",',
        '    "MCTSConfig",',
        # Big Flagship 3: Graph Transformer
        '    "graph_transformer",',
        '    "GraphTransformerLayer",',
        '    "GraphTransformer",',
        # Big Flagship 4: BitNet 1.58b
        '    "bitnet",',
        '    "BitLinear158b",',
        '    "WeightQuantizerSTE",',
        '    "AbsmaxQuantizer",',
        # Big Flagship 5: Diffusion Samplers
        '    "diffusion_sampler",',
        '    "EulerDiscreteScheduler",',
        '    "DPMSolverMultistepScheduler",',
        '    "DDIMScheduler",',
        # Big Flagship 6: Audio Codec
        '    "audio_codec",',
        '    "VectorQuantizer",',
        '    "AudioResidualVQ",',
        '    "AudioEncoder1D",',
        '    "AudioDecoder1D",',
        # Big Flagship 7: Adaptive Conformal
        '    "adaptive_conformal",',
        '    "AdaptivePredictionSets",',
        '    "ConformalizedQuantileRegression",',
        '    "MondrianConformalGroup",',
        # Big Flagship 8: Synaptic Intelligence
        '    "synaptic_intelligence",',
        '    "PathIntegralRegularizer",',
        '    "MemoryReplayBuffer",',
        # Big Flagship 9: Multimodal VLM
        '    "multimodal_vlm",',
        '    "VLMProjection",',
        '    "PerceiverResampler",',
        '    "VisionLanguageConnector",',
        # Big Flagship 10: Expert Parallel MoE
        '    "expert_parallel",',
        '    "TopKRouterWithAuxLoss",',
        '    "CapacityConstrainedMoE",',
        # Attention Extensions
        '    "attention_ext",',
        '    "AlibiPositionalBias",',
        '    "SpatiotemporalRoPE",',
        '    "MultiQueryAttention",',
        '    "SlidingWindowAttention",',
        '    "DifferentialAttention",',
        '    "LogitLens",',
        # Optim Extensions
        '    "optim_ext",',
        '    "ShampooOptimizer",',
        '    "AdEMAMix",',
        '    "ProdigyOptimizer",',
        '    "SophiaOptimizer",',
        '    "ScheduleFreeSGD",',
        '    "LookaheadOptimizer",',
        '    "SGDRScheduler",',
        # Losses Extensions
        '    "losses_ext",',
        '    "DiceLoss",',
        '    "AdaptiveFocalLoss",',
        '    "LabelSmoothingCrossEntropy",',
        '    "TripletMarginWithDistanceLoss",',
        '    "RankNetLoss",',
        '    "InfoNCELoss",',
        # GNN Extensions
        '    "gnn_ext",',
        '    "GatedGraphConv",',
        '    "GraphSAGE",',
        '    "GINConv",',
        '    "LaplacianEigenmaps",',
        '    "AdamicAdarIndex",',
        '    "JaccardSimilarityCoeff",',
        # ODE/PDE Solvers
        '    "ode_pde_solvers",',
        '    "RungeKutta4",',
        '    "DormandPrince54",',
        '    "SymplecticEuler",',
        '    "Burgers1DEquation",',
        '    "Wave1DEquation",',
        '    "Diffusion1DEquation",',
        # Quant Compression
        '    "quant_compression",',
        '    "SmoothQuant",',
        '    "GPTQQuantizer",',
        '    "SparseMagnitudePruner",',
        '    "VectorQuantizedVAE",',
        '    "HQQQuantizer",',
        # Numerical Methods
        '    "numerical",',
        '    "quad",',
        '    "fixed_quad",',
        '    "gauss_legendre",',
        '    "dblquad",',
        '    "Interp1D",',
        '    "CubicSpline",',
        '    "minimize_scalar",',
        '    "ScalarOptResult",',
        '    "curve_fit",',
        '    "physical_constants",',
        '    "Constant",',
        # Classical Stats
        '    "stats",',
        '    "TtestResult",',
        '    "Chi2Result",',
        '    "CorrResult",',
        '    "MannWhitneyResult",',
        '    "KSTestResult",',
        '    "JarqueBeraResult",',
        '    "ttest_1samp",',
        '    "ttest_ind",',
        '    "ttest_rel",',
        '    "chisquare",',
        '    "chi2_contingency",',
        '    "pearsonr",',
        '    "spearmanr",',
        '    "mannwhitneyu",',
        '    "kstest",',
        '    "skew",',
        '    "kurtosis",',
        '    "jarque_bera",',
        '    "ECDF",',
        '    "ecdf",',
        '    "p_adjust",',
        # Physics Sims
        '    "physics",',
        '    "Wave2DEquation",',
        '    "Diffusion2DEquation",',
        '    "simulate_nbody",',
        '    "nbody_energy",',
        '    "nbody_momentum",',
        '    "hamiltonian_leapfrog",',
        '    "circular_velocity",',
        '    "orbital_period",',
        # Signal Metrics
        '    "signal_metrics",',
        '    "MelFilterbank",',
        '    "FastDTW",',
        '    "ExponentialMovingAverage",',
        '    "Autocorrelation1D",',
        '    "PeakSignalToNoiseRatio",',
        '    "StructuralSimilarityIndex",',
        *[f'    "{s}",' for s in public],
        "]",
        "",
    ]
    # __all__ submodule entries follow ground truth: append gaps.
    _have = set()
    for _e in init_py:
        if _e.startswith('    "') and _e.endswith('",'):
            _have.add(_e[5:-2])
    init_py[-2:-2] = ['    "' + s + '",' for s in submods if s not in _have]
    (PKG / "__init__.py").write_text("\n".join(init_py), encoding="utf-8", newline="\n")

    stats = {"typed": 0, "any": 0}
    stubs = [
        '"""PEP 561 type stubs for Oxideon. Generated by tools/gen_stubs.py; do not edit."""',
        "from typing import Any, Optional, Union",
        "",
        "from . import (",
        *['    ' + s + ' as ' + s + ',' for s in submods],
        '    functional as F,',
        ")",
        "",
        "__version__: str",
        "",
    ]
    for s in public:
        obj = getattr(ox, s)
        if inspect.isclass(obj):
            stubs += emit_class(s, obj, table, stats)
        elif callable(obj):
            ptypes, ret = table.funcs.get(s, ({}, ""))
            sig = getattr(obj, "__text_signature__", None)
            r = render_params(parse_textsig(sig), ptypes, table, None, False)
            ok = r is not None and sig is not None
            if ok:
                rt = py_type(ret, table, None) if s in table.funcs else "Any"
            else:
                py = render_py_sig(obj, False)
                if py is not None:
                    r, rt = py
                    ok = True
                else:
                    r, rt = "*args: Any, **kwargs: Any", "Any"
            stats["typed" if ok else "any"] += 1
            stubs += [f"def {s}({r}) -> {rt}: ...", ""]
        else:
            stubs += [f"{s}: Any", ""]
    text = "\n".join(stubs)
    ast.parse(text)  # the stub must at least be valid Python
    (PKG / "__init__.pyi").write_text(text, encoding="utf-8", newline="\n")
    total = stats["typed"] + stats["any"]
    print(f"{len(public)} public symbols; {stats['typed']}/{total} signatures typed, {stats['any']} fall back to Any")


if __name__ == "__main__":
    main()
