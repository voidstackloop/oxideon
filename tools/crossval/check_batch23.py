"""Batch 23: loss-function values and gradients vs PyTorch (float64)."""
import re, sys
import numpy as np
import torch
import torch.nn.functional as F

arrs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    if not rest: continue
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass

D = torch.float64
T = lambda v, shape=None: torch.tensor(v, dtype=D).reshape(shape) if shape else torch.tensor(v, dtype=D)
bad = n = 0; worst = 0.0
def check(name, tensor, tol=1e-9):
    global bad, n, worst
    want = tensor.detach().double().flatten().numpy(); got = arrs[name]; n += 1
    if got.shape != want.shape: bad += 1; print("SHAPE", name, got.shape, want.shape); return
    err = np.max(np.abs(got - want) / np.maximum(1.0, np.abs(want))) if len(want) else 0.0
    worst = max(worst, err)
    if err > tol: bad += 1; print("MISMATCH", name, f"err {err:.2e}", got[:4], want[:4])

def run(name, loss_fn, inputs, tags):
    leaves = [x.clone().requires_grad_(True) for x in inputs]
    loss = loss_fn(*leaves); loss.backward()
    check(name + "_loss", loss)
    for tag, x in zip(tags, leaves): check(f"{name}_{tag}", x.grad)

pred = [0.3, -1.2, 2.5, 0.9, -0.4, 1.8]; targ = [0.0, -0.7, 1.9, 2.4, -0.4, 0.5]
t = T(targ, (2, 3)); lab = T([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], (2, 3)); P = T(pred, (2, 3))
run("mse", lambda p: F.mse_loss(p, t), [P], ["g"])
run("l1", lambda p: F.l1_loss(p, t), [P], ["g"])
run("huber", lambda p: F.huber_loss(p, t, delta=1.3), [P], ["g"])
run("smooth_l1", lambda p: F.smooth_l1_loss(p, t, beta=0.7), [P], ["g"])
run("smooth_l1_sum", lambda p: F.smooth_l1_loss(p, t, beta=0.7, reduction="sum"), [P], ["g"])
run("bce_logits", lambda p: F.binary_cross_entropy_with_logits(p, lab), [P], ["g"])
run("multilabel_soft_margin", lambda p: F.multilabel_soft_margin_loss(p, lab), [P], ["g"])
ml = torch.tensor([[1, 2, -1], [0, 2, -1]])
run("multilabel_margin", lambda p: F.multilabel_margin_loss(p, ml), [P], ["g"])
run("bce", lambda p: F.binary_cross_entropy(p, lab), [T([0.1, 0.7, 0.35, 0.92, 0.5, 0.03], (2, 3))], ["g"])
LG = T([1.2, -0.3, 0.8, 2.0, 0.1, -1.5, 0.4, 0.4, 1.1], (3, 3)); cls = torch.tensor([2, 0, 1])
run("ce_smooth", lambda l: F.cross_entropy(l, cls, label_smoothing=0.1), [LG], ["g"])
run("nll", lambda l: F.nll_loss(F.log_softmax(l, 1), cls), [LG], ["g"])
tp = T([0.2, 0.5, 0.3, 0.1, 0.1, 0.8, 0.6, 0.3, 0.1], (3, 3))
run("kl_mean", lambda l: F.kl_div(F.log_softmax(l, 1), tp, reduction="mean"), [LG], ["g"])
run("kl_batchmean", lambda l: F.kl_div(F.log_softmax(l, 1), tp, reduction="batchmean"), [LG], ["g"])
run("multi_margin1", lambda l: F.multi_margin_loss(l, cls, p=1, margin=1.0), [LG], ["g"])
run("multi_margin2", lambda l: F.multi_margin_loss(l, cls, p=2, margin=0.8), [LG], ["g"])
run("multi_margin_zero", lambda l: F.multi_margin_loss(l, torch.tensor([0, 2]), p=1, margin=1.0), [T([2.0, 1.0, 0.5, 0.0, 1.0, 3.0], (2, 3))], ["g"])
a, b, c = T([0.3, 1.1, -0.4, 0.9, -0.2, 0.7], (2, 3)), T([0.1, 0.8, -0.1, -0.5, 0.6, 0.2], (2, 3)), T([1.0, -0.4, 0.3, 0.2, 0.5, -0.9], (2, 3))
run("cosine_embedding", lambda a_, b_: F.cosine_embedding_loss(a_, b_, T([1.0, -1.0]), margin=0.2), [a, b], ["ga", "gb"])
run("triplet", lambda a_, b_, c_: F.triplet_margin_loss(a_, b_, c_, margin=1.0), [a, b, c], ["ga", "gb", "gc"])
x1, x2, y = T([0.4, 1.2, -0.3, 0.8]), T([0.1, 1.5, 0.2, -0.4]), T([1.0, -1.0, 1.0, -1.0])
run("margin_ranking", lambda p, q: F.margin_ranking_loss(p, q, y, margin=0.3), [x1, x2], ["g1", "g2"])
run("hinge_embedding", lambda p: F.hinge_embedding_loss(p, y, margin=0.9), [x1], ["g"])
run("soft_margin", lambda p: F.soft_margin_loss(p, y), [x1], ["g"])
counts, rate = T([0.0, 2.0, 5.0, 1.0]), T([0.4, 1.1, 1.6, -0.3])
run("poisson_log", lambda r: F.poisson_nll_loss(r, counts, log_input=True, full=False, eps=1e-8), [rate], ["g"])
run("poisson_full", lambda r: F.poisson_nll_loss(r, counts, log_input=True, full=True, eps=1e-8), [rate], ["g"])
run("gaussian_nll", lambda x, v: F.gaussian_nll_loss(x, x2, v, full=False, eps=1e-6), [x1, T([0.5, 1.2, 0.8, 2.0])], ["gx", "gv"])
raw = T([((i * 7 % 11) - 5.0) / 3.0 for i in range(48)], (6, 2, 4))
tg = torch.tensor([[1, 2, 3], [2, 2, 1]])
def ctc(r):
    return F.ctc_loss(F.log_softmax(r, 2), tg, torch.tensor([6, 6]), torch.tensor([3, 3]), blank=0, reduction="mean", zero_infinity=False)
run("ctc", ctc, [raw], ["g"])
print(f"{n} tensors compared, worst error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
