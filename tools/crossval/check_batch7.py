"""Seventh cross-validation batch: loss functions vs PyTorch.

    cargo test -p oxideon-core --test crossval_emit7 -- --nocapture --ignored > rust_values7.txt
    py -3.11 tools/crossval/check_batch7.py rust_values7.txt
"""
import sys

import torch
import torch.nn.functional as F

rust = {}
for line in open(sys.argv[1], encoding="utf-8"):
    parts = line.split()
    if len(parts) == 2:
        try:
            rust[parts[0]] = float(parts[1])
        except ValueError:
            pass

T = lambda v, shape=None: torch.tensor(v, dtype=torch.float64).reshape(shape) if shape else torch.tensor(v, dtype=torch.float64)
pred = T([0.3, -1.2, 2.5, 0.9, -0.4, 1.8], (2, 3))
targ = T([0.0, -0.7, 1.9, 2.4, -0.4, 0.5], (2, 3))
lab = T([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], (2, 3))
probs = T([0.1, 0.7, 0.35, 0.92, 0.5, 0.03], (2, 3))
logits = T([1.2, -0.3, 0.8, 2.0, 0.1, -1.5, 0.4, 0.4, 1.1], (3, 3))
cls = torch.tensor([2, 0, 1])
tp = T([0.2, 0.5, 0.3, 0.1, 0.1, 0.8, 0.6, 0.3, 0.1], (3, 3))
a = T([0.3, 1.1, -0.4, 0.9, -0.2, 0.7], (2, 3))
b = T([0.1, 0.8, -0.1, -0.5, 0.6, 0.2], (2, 3))
c = T([1.0, -0.4, 0.3, 0.2, 0.5, -0.9], (2, 3))
x1, x2, y = T([0.4, 1.2, -0.3, 0.8]), T([0.1, 1.5, 0.2, -0.4]), T([1.0, -1.0, 1.0, -1.0])
ml_t = torch.tensor([[1, 2, -1], [0, 2, -1]])
counts, rate, var = T([0.0, 2.0, 5.0, 1.0]), T([0.4, 1.1, 1.6, -0.3]), T([0.5, 1.2, 0.8, 2.0])
ref = {
    "mse": F.mse_loss(pred, targ),
    "l1": F.l1_loss(pred, targ),
    "huber": F.huber_loss(pred, targ, delta=1.3),
    "smooth_l1": F.smooth_l1_loss(pred, targ, beta=0.7),
    "smooth_l1_sum": F.smooth_l1_loss(pred, targ, beta=0.7, reduction="sum"),
    "bce_logits": F.binary_cross_entropy_with_logits(pred, lab),
    "bce": F.binary_cross_entropy(probs, lab),
    "ce": F.cross_entropy(logits, cls),
    "ce_smooth": F.cross_entropy(logits, cls, label_smoothing=0.1),
    "nll": F.nll_loss(F.log_softmax(logits, 1), cls),
    "kl_div_mean": F.kl_div(F.log_softmax(logits, 1), tp, reduction="mean"),
    "kl_div_batchmean": F.kl_div(F.log_softmax(logits, 1), tp, reduction="batchmean"),
    "cosine_embedding": F.cosine_embedding_loss(a, b, T([1.0, -1.0]), margin=0.2),
    "triplet": F.triplet_margin_loss(a, b, c, margin=1.0),
    "margin_ranking": F.margin_ranking_loss(x1, x2, y, margin=0.3),
    "hinge_embedding": F.hinge_embedding_loss(x1, y, margin=0.9),
    "soft_margin": F.soft_margin_loss(x1, y),
    "multilabel_margin": F.multilabel_margin_loss(pred, ml_t),
    "multilabel_soft_margin": F.multilabel_soft_margin_loss(pred, lab),
    "multi_margin1": F.multi_margin_loss(logits, torch.tensor([2, 0, 1]), p=1, margin=1.0),
    "multi_margin2": F.multi_margin_loss(logits, torch.tensor([2, 0, 1]), p=2, margin=0.8),
    "poisson_nll_log": F.poisson_nll_loss(rate, counts, log_input=True, full=False, eps=1e-8),
    "poisson_nll_full": F.poisson_nll_loss(rate, counts, log_input=True, full=True, eps=1e-8),
    "gaussian_nll": F.gaussian_nll_loss(x1, x2, var, full=False, eps=1e-6),
    "gaussian_nll_full": F.gaussian_nll_loss(x1, x2, var, full=True, eps=1e-6),
}
raw = T([((i * 7 % 11) - 5.0) / 3.0 for i in range(48)], (6, 2, 4))
lp = F.log_softmax(raw, 2)
ref["ctc"] = F.ctc_loss(lp, torch.tensor([[1, 2, 3], [2, 2, 1]]), torch.tensor([6, 6]), torch.tensor([3, 3]), blank=0, reduction="mean", zero_infinity=False)
lg = logits.clone().requires_grad_(True)
F.cross_entropy(lg, cls).backward()
out = {f"{k}_0": float(v) for k, v in ref.items()}
out = {k[:-2]: v for k, v in out.items()}
for i, v in enumerate(lg.grad.reshape(-1).tolist()): out[f"ce_g{i}"] = v

worst, bad = 0.0, []
for k, want in out.items():
    if k not in rust:
        bad.append((k, "missing", want)); continue
    err = abs(rust[k] - want) / max(1.0, abs(want))
    worst = max(worst, err)
    if err > 1e-6:
        bad.append((k, rust[k], want))
print(f"compared {len(out)} values, worst relative error {worst:.2e}")
for k, got, want in bad:
    print(f"MISMATCH {k}: rust={got} torch={want}")
print(f"{len(bad)} mismatches")
sys.exit(1 if bad else 0)
