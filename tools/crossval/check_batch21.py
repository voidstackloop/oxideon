"""Batch 21: spatial ops and misc layers vs PyTorch."""
import re, sys
import numpy as np
import torch
import torch.nn.functional as F

arrs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    if not rest or rest.startswith("["): continue
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass

M = (1 << 64) - 1
def u(i):
    z = ((i + 0x9E3779B97F4A7C15) & M) * 0xBF58476D1CE4E5B9 & M
    z = ((z ^ (z >> 30)) * 0x94D049BB133111EB) & M; z ^= z >> 31
    return ((z >> 11) + 0.5) / float(1 << 53)
def data(seed, shape):
    n = int(np.prod(shape))
    return torch.tensor([(u(seed * 100000 + i) - 0.5) * 4.0 for i in range(n)], dtype=torch.float64).reshape(shape)

bad = n = 0; worst = 0.0
def check(name, want, tol=1e-9):
    global bad, n, worst
    want = want.detach().double().flatten().numpy()
    got = arrs[name]; n += 1
    if got.shape != want.shape:
        bad += 1; print("SHAPE MISMATCH", name, got.shape, want.shape); return
    err = np.max(np.abs(got - want) / np.maximum(1.0, np.abs(want))) if len(want) else 0.0
    worst = max(worst, err)
    if err > tol:
        bad += 1; print("MISMATCH", name, f"max err {err:.2e}", got[:4], want[:4])

a, b = data(1, [5, 6]), data(2, [5, 6])
a[1] = 0.0
check("cos_axis1", F.cosine_similarity(a, b, dim=1, eps=1e-8))
check("cos_axis0", F.cosine_similarity(a, b, dim=0, eps=1e-8))
for p, tag in [(2.0, "p2"), (1.0, "p1"), (3.0, "p3")]:
    check(f"pdist_{tag}", F.pairwise_distance(a, b, p=p, eps=1e-6))
    check(f"pdist_keep_{tag}", F.pairwise_distance(a, b, p=p, eps=1e-6, keepdim=True))
    check(f"normalize_{tag}", F.normalize(a, p=p, dim=1, eps=1e-12))
check("normalize_ax0", F.normalize(a, p=2.0, dim=0, eps=1e-12))
check("softmin1", F.softmin(b, dim=1))

x = data(3, [2, 3, 7, 6])
for tag, k, s, p, d in [("u_a", (3, 2), (1, 1), (0, 0), (1, 1)), ("u_b", (3, 3), (2, 1), (1, 2), (1, 1)), ("u_c", (2, 2), (1, 2), (1, 0), (2, 1))]:
    cols = F.unfold(x, k, dilation=d, padding=p, stride=s)
    check(tag, cols)
    check(tag + "_fold", F.fold(cols, (7, 6), k, dilation=d, padding=p, stride=s))

vals, idx = F.max_pool2d(x, (3, 2), (2, 2), (1, 1), return_indices=True)
check("mp_vals", vals); check("mp_idx", idx.double())
check("unpool", F.max_unpool2d(vals, idx, (3, 2), (2, 2), (1, 1), output_size=(7, 6)))
check("avgpool_pad", F.avg_pool2d(x, 3, 2, 1))
check("lppool2", F.lp_pool2d(x.abs(), 2.0, (2, 3), (2, 1)))
check("lppool15", F.lp_pool2d(x.abs(), 1.5, (3, 3), (2, 2)))
for h, w in [(1, 1), (3, 4), (4, 5), (7, 6), (5, 3)]:
    check(f"adamax_{h}x{w}", F.adaptive_max_pool2d(x, (h, w)))
    check(f"adaavg_{h}x{w}", F.adaptive_avg_pool2d(x, (h, w)))

for tag, align in [("gs_t", True), ("gs_f", False)]:
    theta = torch.tensor([0.9, 0.2, 0.1, -0.15, 1.1, -0.05, 0.7, -0.3, 0.2, 0.25, 0.8, 0.1], dtype=torch.float64).reshape(2, 2, 3).requires_grad_(True)
    inp = data(4, [2, 3, 5, 6]).requires_grad_(True)
    grid = F.affine_grid(theta, (2, 3, 4, 5), align_corners=align)
    check(tag + "_grid", grid)
    out = F.grid_sample(inp, grid, mode="bilinear", padding_mode="zeros", align_corners=align)
    check(tag + "_out", out)
    (out * data(5, list(out.shape))).sum().backward()
    check(tag + "_dinput", inp.grad); check(tag + "_dtheta", theta.grad, 1e-8)

W = data(6, [10, 4])
idx = torch.tensor([1, 4, 4, 9, 0, 2, 7, 7, 3]); offs = torch.tensor([0, 3, 4, 8])
for tag, mode in [("sum", "sum"), ("mean", "mean"), ("max", "max")]:
    check(f"ebag_{tag}", F.embedding_bag(idx, W.float().double(), offs, mode=mode), 1e-6)

f32d = lambda s, sh: data(s, sh).float()
bl = F.bilinear(f32d(9, [5, 3]), f32d(10, [5, 4]), f32d(7, [2, 3, 4]), f32d(8, [2]))
check("bilinear", bl, 1e-5)
check("pixshuf", F.pixel_shuffle(data(11, [2, 8, 3, 4]), 2))
check("pixunshuf", F.pixel_unshuffle(data(12, [2, 3, 6, 4]), 2))
print(f"{n} tensors compared, worst error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
