"""Batch 24: gradients of elementwise / binary / reduction tensor ops vs PyTorch (float64)."""
import re, sys, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")

arrs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    if not rest: continue
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass

D = torch.float64
UNIT = [0.15, 0.3, 0.42, 0.55, 0.61, 0.7, 0.82, 0.9]
WIDE = [-2.7, -1.3, -0.45, -0.1, 0.2, 0.9, 1.6, 2.8]
POS = [0.3, 0.7, 1.1, 1.9, 2.6, 3.4, 4.5, 6.1]
GT1 = [1.2, 1.5, 2.0, 2.6, 3.1, 4.0, 5.5, 7.0]
W = torch.tensor([0.5 + 0.37 * i for i in range(8)], dtype=D)
bad = n = 0; worst = 0.0

def check(name, t, tol):
    global bad, n, worst
    want = t.detach().double().flatten().numpy(); got = arrs[name]; n += 1
    if got.shape != want.shape: bad += 1; print("SHAPE", name, got.shape, want.shape); return
    err = np.max(np.abs(got - want) / np.maximum(1.0, np.abs(want)))
    worst = max(worst, err)
    if err > tol: bad += 1; print("MISMATCH", name, f"err {err:.2e}", got[:4], want[:4])

def bin_(name, a, b, f, tol=1e-9):
    ta = torch.tensor(a, dtype=D, requires_grad=True); tb = torch.tensor(b, dtype=D, requires_grad=True)
    y = f(ta, tb); (y * W).sum().backward()
    check(name + "_v", y, tol); check(name + "_ga", ta.grad, tol); check(name + "_gb", tb.grad, tol)

for name, x, f in [
    ("exp", WIDE, torch.exp), ("ln", POS, torch.log), ("sqrt", POS, torch.sqrt), ("rsqrt", POS, torch.rsqrt),
    ("sin", WIDE, torch.sin), ("cos", WIDE, torch.cos), ("tan", WIDE, torch.tan), ("asin", UNIT, torch.asin),
    ("acos", UNIT, torch.acos), ("atan", WIDE, torch.atan), ("sinh", WIDE, torch.sinh), ("cosh", WIDE, torch.cosh),
    ("tanh", WIDE, torch.tanh), ("asinh", WIDE, torch.asinh), ("acosh", GT1, torch.acosh), ("atanh", UNIT, torch.atanh),
    ("erf", WIDE, torch.erf), ("erfc", WIDE, torch.erfc), ("erfinv", UNIT, torch.erfinv), ("lgamma", POS, torch.lgamma),
    ("digamma", POS, torch.digamma), ("expm1", WIDE, torch.expm1), ("log1p", POS, torch.log1p), ("log2", POS, torch.log2),
    ("log10", POS, torch.log10), ("abs", WIDE, torch.abs), ("reciprocal", WIDE, torch.reciprocal), ("square", WIDE, torch.square),
    ("exp2", WIDE, torch.exp2), ("sigmoid", WIDE, torch.sigmoid), ("logit", UNIT, torch.logit), ("sinc", WIDE, torch.sinc),
    ("pow_scalar", POS, lambda t: t.pow(2.5)), ("pow_neg", POS, lambda t: t.pow(-1.3)),
    ("cumsum", WIDE, lambda t: t.cumsum(0)), ("cumprod", POS, lambda t: t.cumprod(0)),
    ("logcumsumexp", WIDE, lambda t: t.logcumsumexp(0)), ("logsumexp", WIDE, lambda t: t.logsumexp(0, keepdim=True)),
    ("softmax", WIDE, lambda t: t.softmax(0)), ("log_softmax", WIDE, lambda t: t.log_softmax(0)),
    ("prod", POS, lambda t: t.prod().reshape(1)), ("var", WIDE, lambda t: t.var(0, unbiased=True).reshape(1)),
    ("norm2", WIDE, lambda t: torch.linalg.vector_norm(t, 2).reshape(1)), ("norm3", WIDE, lambda t: torch.linalg.vector_norm(t, 3).reshape(1)),
    ("norm1", WIDE, lambda t: torch.linalg.vector_norm(t, 1).reshape(1)), ("amax", WIDE, lambda t: t.amax(0).reshape(1)),
    ("flip", WIDE, lambda t: t.flip(0)), ("roll", WIDE, lambda t: t.roll(3, 0)),
]:
    # unary helper: weight vector is W for 8-element outputs, [1.0*W0] semantics for single outputs
    t = torch.tensor(x, dtype=D, requires_grad=True)
    y = f(t)
    # the Rust emitter multiplies by an 8-vector, so a single-element output is broadcast against all weights
    (y * W).sum().backward()
    check(name + "_v", y, 1e-9); check(name + "_g", t.grad, 1e-9)

a = [0.3, 1.4, -0.7, 2.2, -1.5, 0.8, 3.1, -2.4]; b = [1.1, -0.6, 2.3, 0.4, -1.9, 1.7, 0.9, 2.6]; ap = [abs(v) for v in a]
for name, aa, bb, f in [
    ("add", a, b, torch.add), ("sub", a, b, torch.sub), ("mul", a, b, torch.mul), ("div", a, b, torch.div),
    ("pow", ap, b, torch.pow), ("atan2", a, b, torch.atan2), ("hypot", a, b, torch.hypot),
    ("maximum", a, b, torch.maximum), ("minimum", a, b, torch.minimum), ("logaddexp", a, b, torch.logaddexp),
    ("logaddexp2", a, b, torch.logaddexp2), ("fmod", a, b, torch.fmod), ("remainder", a, b, torch.remainder),
    ("copysign", a, b, torch.copysign), ("xlogy", ap, [abs(v) for v in b], torch.xlogy),
    ("lerp", a, b, lambda x, y: torch.lerp(x, y, 0.3)),
]:
    bin_(name, aa, bb, f)
print(f"{n} tensors compared, worst error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
