"""Batch 25: linear-algebra / contraction ops, values and gradients, vs PyTorch (float64)."""
import math, re, sys, warnings
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
def mat(r, c, salt):
    return [math.sin(0.83 * k + salt) + (2.5 if k // c == k % c else 0.0) for k in range(r * c)]
def wts(shape):
    n = int(np.prod(shape)); return torch.tensor([math.sin(1.3 * k + 0.4) for k in range(n)], dtype=D).reshape(shape)
def T(v, shape): return torch.tensor(v, dtype=D).reshape(shape)

av, bv, tv, bv2 = mat(4, 4, 0.2), mat(4, 3, 1.1), mat(5, 3, 2.3), mat(5, 2, 3.1)
m = mat(4, 4, 4.4)
sv = [(m[(k // 4) * 4 + k % 4] + m[(k % 4) * 4 + k // 4]) * 0.5 + (3.0 if k // 4 == k % 4 else 0.0) for k in range(16)]

bad = n = 0; worst = 0.0
SYM = {"cholesky", "logdet", "eigvalsh", "eigh_fn"}   # symmetric inputs: compare symmetrized gradients
def check(name, got, want, tol):
    global bad, n, worst
    want = want.detach().double().flatten().numpy(); n += 1
    if got.shape != want.shape: bad += 1; print("SHAPE", name, got.shape, want.shape); return
    err = np.max(np.abs(got - want) / np.maximum(1.0, np.abs(want)))
    worst = max(worst, err)
    if err > tol: bad += 1; print("MISMATCH", name, f"err {err:.2e}", got[:4], want[:4])

def case(name, specs, fn, tol=1e-8):
    leaves = [T(v, s).requires_grad_(True) for v, s in specs]
    outs = fn(*leaves)
    loss = sum((o * wts(list(o.shape))).sum() for o in outs)
    loss.backward()
    for k, o in enumerate(outs): check(f"{name}_out{k}", arrs[f"{name}_out{k}"], o, tol)
    for k, x in enumerate(leaves):
        g = x.grad
        if name in SYM:
            gr = arrs[f"{name}_g{k}"].reshape(g.shape); gr = (gr + gr.T) / 2
            check(f"{name}_g{k}(sym)", gr.flatten(), (g + g.T) / 2, tol)
        else:
            check(f"{name}_g{k}", arrs[f"{name}_g{k}"], g, tol)

A, B, S, Tt, B2 = (av, (4, 4)), (bv, (4, 3)), (sv, (4, 4)), (tv, (5, 3)), (bv2, (5, 2))
case("matmul", [A, B], lambda a, b: [a @ b])
case("det", [A], lambda a: [torch.linalg.det(a).reshape(1)])
case("inv", [A], lambda a: [torch.linalg.inv(a)])
case("solve", [A, B], lambda a, b: [torch.linalg.solve(a, b)])
case("slogdet", [A], lambda a: [torch.linalg.slogdet(a)[1].reshape(1)])
case("cholesky", [S], lambda s: [torch.linalg.cholesky(s)])
case("logdet", [S], lambda s: [torch.logdet(s).reshape(1)])
case("eigvalsh", [S], lambda s: [torch.linalg.eigvalsh(s)])
def eigh_fn(s):
    l, v = torch.linalg.eigh(s); return [(v * l.exp().reshape(1, 4)) @ v.T]
case("eigh_fn", [S], eigh_fn)
case("qr", [Tt], lambda t: list(torch.linalg.qr(t)))
case("svd_s", [Tt], lambda t: [torch.linalg.svdvals(t)])
def svd_fn(t):
    u, s, vh = torch.linalg.svd(t, full_matrices=False); return [(u * s.sqrt().reshape(1, 3)) @ vh]
case("svd_fn", [Tt], svd_fn)
def svd_fn_wide(t):
    u, s, vh = torch.linalg.svd(t, full_matrices=False); return [(u * s.sqrt().reshape(1, 3)) @ vh]
case("svd_fn_wide", [(mat(3, 5, 2.7), (3, 5))], svd_fn_wide)
case("pinv", [Tt], lambda t: [torch.linalg.pinv(t)])
case("lstsq", [Tt, B2], lambda t, b: [torch.linalg.lstsq(t, b, driver="gelsd").solution])
case("matrix_exp", [A], lambda a: [torch.linalg.matrix_exp(a * 0.2)])
case("matrix_power3", [A], lambda a: [torch.linalg.matrix_power(a * 0.3, 3)])
case("matrix_power-2", [A], lambda a: [torch.linalg.matrix_power(a, -2)])
case("trace", [A], lambda a: [torch.trace(a).reshape(1)])
case("kron", [(mat(2, 2, 0.5), (2, 2)), (mat(2, 3, 1.5), (2, 3))], lambda a, b: [torch.kron(a, b)])
case("cross", [(mat(4, 3, 0.7), (4, 3)), (mat(4, 3, 1.9), (4, 3))], lambda a, b: [torch.linalg.cross(a, b, dim=1)])
case("einsum_mm", [A, B], lambda a, b: [torch.einsum("ij,jk->ik", a, b)])
case("einsum_batch", [(mat(6, 4, 2.0), (2, 3, 4)), (mat(8, 3, 3.0), (2, 4, 3))], lambda a, b: [torch.einsum("bij,bjk->bik", a, b)])
case("einsum_trace_outer", [A, B], lambda a, b: [torch.einsum("ii->", a).reshape(1), torch.einsum("ij,kl->ikjl", a, b)])
case("tensordot", [(mat(6, 4, 2.2), (2, 3, 4)), (mat(12, 2, 3.3), (3, 4, 2))], lambda a, b: [torch.tensordot(a, b, dims=([1, 2], [0, 1]))])
case("cdist", [(mat(5, 3, 0.9), (5, 3)), (mat(4, 3, 2.9), (4, 3))], lambda a, b: [torch.cdist(a, b)])
case("cov", [(mat(3, 6, 1.7), (3, 6))], lambda a: [torch.cov(a, correction=1)])
case("tri_solve", [A, B], lambda a, b: [torch.linalg.solve_triangular(a, b, upper=False)])
case("tri_solve_t", [A, B], lambda a, b: [torch.linalg.solve_triangular(a.mT, b, upper=False)])
print(f"{n} tensors compared, worst error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
