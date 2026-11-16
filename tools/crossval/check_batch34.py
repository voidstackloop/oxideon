"""Batch 34: sparse COO / CSR ops vs scipy.sparse (float64)."""
import math, re, sys, warnings
import numpy as np
import scipy.sparse as sp
warnings.filterwarnings("ignore")

arrs = {}; shapes = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    m = re.match(r"^(\S+) \[([\d, ]*)\] (.*)$", line)
    if m:
        arrs[m.group(1)] = np.array([float(x) for x in m.group(3).split()]); shapes[m.group(1)] = tuple(int(v) for v in m.group(2).split(",") if v.strip())
        continue
    k, _, rest = line.partition(" ")
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass

M = (1 << 64) - 1
def u(i):
    z = ((i + 0x9E3779B97F4A7C15) & M) * 0xBF58476D1CE4E5B9 & M
    z = ((z ^ (z >> 30)) * 0x94D049BB133111EB) & M; z ^= z >> 31
    return ((z >> 11) + 0.5) / float(1 << 53)
def build(rows, cols, density, seed):
    idx, val = [], []
    for r in range(rows):
        for c in range(cols):
            if u(seed * 100000 + r * cols + c) < density:
                idx.append((r, c)); val.append((u(seed * 100000 + 50000 + r * cols + c) - 0.5) * 6.0)
    idx.append(idx[0]); val.append(1.25); idx.append((rows - 1, cols - 1)); val.append(0.0)
    rr, cc = zip(*idx)
    return sp.coo_matrix((val, (rr, cc)), shape=(rows, cols))   # duplicates are summed on conversion
bad = n = 0; worst = 0.0
def check(name, want, tol=1e-11):
    global bad, n, worst
    n += 1
    want = np.asarray(want.todense() if sp.issparse(want) else want, dtype=float)
    got = arrs[name]
    if name in shapes and shapes[name] and shapes[name] != want.shape and got.size != want.size:
        bad += 1; print("SHAPE", name, shapes[name], want.shape); return
    want = want.ravel()
    if got.shape != want.shape: bad += 1; print("SHAPE", name, got.shape, want.shape); return
    err = np.max(np.abs(got - want)) / max(1.0, np.abs(want).max()) if want.size else 0.0; worst = max(worst, err)
    if err > tol: bad += 1; print("MISMATCH", name, f"err {err:.2e}", got[:5], want[:5])

A, B, C, SQ = build(7, 6, 0.35, 1), build(6, 5, 0.4, 2), build(7, 6, 0.3, 3), build(6, 6, 0.4, 4)
for nm, m in [("a", A), ("b", B), ("c", C), ("sq", SQ)]: check(nm, m)
n += 1
if int(arrs["a_nnz"][0]) != int((np.asarray(A.todense()) != 0).sum()):
    bad += 1; print("NNZ", arrs["a_nnz"][0], int((np.asarray(A.todense()) != 0).sum()))
check("a_t", A.T); check("a_csr_t", A.T)
check("a_plus_c", A + C); check("a_minus_c", A - C); check("a_scaled", A * -2.5)
check("ab", A @ B); check("ab_csr", A @ B); check("a_had_c", A.multiply(C))
check("a_kron_b", sp.kron(build(2, 3, 0.7, 5), build(3, 2, 0.7, 6)))
D = np.asarray(SQ.todense())
for up, k in [(1, 0), (1, 1), (1, -2), (0, 0), (0, 2), (0, -1)]:
    check(f"tri_{up}_{k}", np.triu(D, k) if up else np.tril(D, k))
check("rows_sel", np.asarray(A.todense())[[3, 0, 3, 6]])
check("eye5", np.eye(5)); check("diag_from", np.diag([1.5, 0.0, -2.0, 4.0])); check("sq_diag", np.diag(D))
Ad = np.asarray(A.todense())
check("sum_axis0", Ad.sum(0)); check("sum_axis1", Ad.sum(1))
x = np.array([math.sin(i * 0.7) + 0.3 for i in range(6)]); check("spmv", Ad @ x)
check("a_norm", np.array([np.linalg.norm(Ad)])); check("a_sparsity", np.array([1.0 - (Ad != 0).sum() / Ad.size]))
sm = np.zeros_like(Ad)
for r in range(Ad.shape[0]):
    nz = Ad[r] != 0
    if nz.any(): e = np.exp(Ad[r][nz] - Ad[r][nz].max()); sm[r][nz] = e / e.sum()
check("a_softmax", sm)
X = np.array([math.cos(0.31 * i) for i in range(24)]).reshape(6, 4)
W = np.array([math.sin(0.77 * i) for i in range(28)]).reshape(7, 4)
check("spmm_out", Ad @ X); check("spmm_dx", Ad.T @ W); check("spmm_csr_out", Ad @ X); check("spmm_csr_dx", Ad.T @ W)
fd = np.array([[0.0, 2.0, 0.001], [-3.0, 0.0, 0.5], [1.0, 0.0, -0.0004]]); fd[np.abs(fd) <= 0.01] = 0.0
check("from_dense", fd)
print(f"{n} checks, worst scaled error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
