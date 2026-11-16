"""Batch 29: 3-D / transposed convs, 1-D / 3-D pooling, batch norm (train + eval), instance norm, PReLU, GLU, weight norm
vs PyTorch (float32)."""
import math, re, sys, warnings
import numpy as np
import torch
import torch.nn.functional as F
warnings.filterwarnings("ignore")

arrs = {}; shapes = {}; nograd = set()
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    m = re.match(r"^(\S+) \[([\d, ]*)\] (.*)$", line)
    if m:
        try:
            arrs[m.group(1)] = np.array([float(x) for x in m.group(3).split()])
            shapes[m.group(1)] = [int(v) for v in m.group(2).split(",") if v.strip()]
        except ValueError: pass
    elif line.endswith("NOGRAD"): nograd.add(line.split()[0])

def genv(n, salt): return [math.sin(0.37 * i + salt) * 0.8 + 0.1 * math.cos(0.11 * i * salt) for i in range(n)]
def t(shape, salt): return torch.tensor(genv(int(np.prod(shape)), salt), dtype=torch.float32).reshape(tuple(shape))
bad = n = 0; worst = 0.0
def check(name, tensor, tol=5e-5):
    global bad, n, worst
    n += 1
    if name in nograd: bad += 1; print("NO GRADIENT", name); return
    if name not in arrs: bad += 1; print("MISSING", name); return
    want = tensor.detach().double().numpy()
    # parameters / buffers may be stored broadcast-ready ([1, C, 1, 1]); only the element count must agree
    if int(np.prod(shapes[name])) != want.size: bad += 1; print("SHAPE", name, shapes[name], list(want.shape)); return
    got = arrs[name]; want = want.ravel()
    err = np.max(np.abs(got - want)) / max(1.0, np.abs(want).max()); worst = max(worst, err)
    if err > tol: bad += 1; print("MISMATCH", name, f"err {err:.2e}", got[:3], want[:3])

def run(name, x, out, params):
    check(name + "_out", out)
    (out * t(list(out.shape), 99.0)).sum().backward()
    check(name + "_dx", x.grad)
    for tag, p in params: check(f"{name}_d{tag}", p.grad)
R = lambda x: x.requires_grad_(True)

x = R(t([2, 2, 4, 5, 4], 1.0)); w = R(t([3, 2, 2, 3, 2], 2.0)); b = R(t([3], 2.5))
run("conv3d", x, F.conv3d(x, w, b, stride=(1, 2, 1), padding=(1, 0, 1)), [("w", w), ("b", b)])
x = R(t([2, 3, 6], 3.0)); w = R(t([3, 2, 3], 4.0)); b = R(t([2], 4.5))
run("convt1d", x, F.conv_transpose1d(x, w, b, stride=2, padding=1), [("w", w), ("b", b)])
x = R(t([1, 2, 3, 3, 2], 5.0)); w = R(t([2, 2, 2, 2, 2], 6.0)); b = R(t([2], 6.5))
run("convt3d", x, F.conv_transpose3d(x, w, b, stride=(2, 1, 2), padding=(0, 1, 0), output_padding=(1, 0, 1)), [("w", w), ("b", b)])

def pool(name, shape, salt, f):
    x = R(t(shape, salt)); run(name, x, f(x), [])
S1 = [2, 3, 11]
pool("maxpool1d", S1, 7.0, lambda v: F.max_pool1d(v, 3, 2)); pool("avgpool1d", S1, 7.0, lambda v: F.avg_pool1d(v, 3, 2))
pool("adaavg1d", S1, 7.0, lambda v: F.adaptive_avg_pool1d(v, 4)); pool("adamax1d", S1, 7.0, lambda v: F.adaptive_max_pool1d(v, 4))
pool("lppool1d", S1, 7.0, lambda v: F.lp_pool1d(v.abs() + 0.1, 2.0, 3, 2))
S3 = [1, 2, 4, 6, 4]
pool("maxpool3d", S3, 8.0, lambda v: F.max_pool3d(v, 2, 2)); pool("avgpool3d", S3, 8.0, lambda v: F.avg_pool3d(v, (2, 3, 2), (2, 3, 2)))

for tag, shape in [("bn1d_2d", [6, 4]), ("bn1d_3d", [3, 4, 5])]:
    w = R(t([4], 10.0)); b = R(t([4], 10.5)); rm, rv = torch.zeros(4), torch.ones(4)
    for step in range(3):
        xi = R(t(shape, 11.0 + step)); out = F.batch_norm(xi, rm, rv, w, b, training=True, momentum=0.1, eps=1e-5)
        if step == 2: run(f"{tag}_train", xi, out, [("w", w), ("b", b)])
    check(f"{tag}_rm", rm); check(f"{tag}_rv", rv)
    w.grad = None; b.grad = None
    xe = R(t(shape, 20.0)); run(f"{tag}_eval", xe, F.batch_norm(xe, rm, rv, w, b, training=False, eps=1e-5), [("w", w), ("b", b)])
w = R(t([3], 12.0)); b = R(t([3], 12.5)); rm, rv = torch.zeros(3), torch.ones(3)
for step in range(2): F.batch_norm(t([2, 3, 4, 3], 13.0 + step), rm, rv, w, b, training=True, momentum=0.1, eps=1e-5)
check("bn2d_rm", rm); check("bn2d_rv", rv)
xe = R(t([2, 3, 4, 3], 21.0)); run("bn2d_eval", xe, F.batch_norm(xe, rm, rv, w, b, training=False, eps=1e-5), [("w", w), ("b", b)])
w = R(t([2], 14.0)); b = R(t([2], 14.5)); rm, rv = torch.zeros(2), torch.ones(2)
xi = R(t([2, 2, 3, 2, 3], 15.0)); run("bn3d_train", xi, F.batch_norm(xi, rm, rv, w, b, training=True, momentum=0.1, eps=1e-5), [("w", w), ("b", b)])
check("bn3d_rm", rm); check("bn3d_rv", rv)
check("bn3d_eval_out", F.batch_norm(t([2, 2, 3, 2, 3], 22.0), rm, rv, w, b, training=False, eps=1e-5))

x = R(t([2, 3, 7], 16.0)); run("in1d", x, F.instance_norm(x, eps=1e-5), [])
x = R(t([2, 3, 4, 5], 17.0)); run("in2d", x, F.instance_norm(x, eps=1e-5), [])
x = R(t([2, 3, 5], 18.0)); w = R(t([3], 18.5)); run("prelu", x, F.prelu(x, w), [("w", w)])
x = R(t([2, 6, 5], 19.0)); run("glu", x, F.glu(x, dim=1), [])
v = R(t([3, 5], 23.0)); g = R(t([3, 1], 23.5)); b = R(t([3], 23.7)); x = R(t([4, 5], 24.0))
run("wnorm", x, F.linear(x, g * v / v.norm(dim=1, keepdim=True), b), [("v", v), ("g", g), ("b", b)])
print(f"{n} tensors compared, worst scaled error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
