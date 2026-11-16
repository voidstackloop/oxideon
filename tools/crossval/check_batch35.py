"""Batch 35: sampling filters vs the Hugging Face logits-processor semantics (computed with torch, same order)."""
import math, re, sys
import numpy as np
import torch

counts = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    try: counts[k] = np.array([int(x) for x in rest.split()])
    except ValueError: pass
DRAWS = 60000
def logits(vocab): return torch.tensor([math.sin(i * 1.7 + 0.4) * 2.2 + math.cos(i * 0.31) for i in range(vocab)], dtype=torch.float64)

def reference(lg, temperature=1.0, top_k=None, top_p=None, min_p=None, rep=1.0, presence=0.0, freq=0.0, hist=None, bias=None):
    """Probabilities after: bias, penalties, temperature, top-k, top-p, min-p (transformers' processor order)."""
    x = lg.clone()
    for tok, b in (bias or {}).items(): x[tok] += b
    if hist:
        cnt = {}
        for t in hist: cnt[t] = cnt.get(t, 0) + 1
        for t, c in cnt.items():
            z = x[t]; z = z / rep if z > 0 else z * rep   # RepetitionPenaltyLogitsProcessor
            x[t] = z - presence - freq * c
    x = x / temperature
    if top_k:
        kth = torch.topk(x, min(top_k, len(x))).values[-1]
        x = torch.where(x < kth, torch.tensor(-float("inf"), dtype=x.dtype), x)
    if top_p is not None and top_p < 1.0:   # TopPLogitsWarper
        s, idx = torch.sort(x, descending=False)
        cum = torch.softmax(s, -1).cumsum(-1)
        remove = cum <= (1 - top_p); remove[-1:] = False
        x = x.masked_fill(remove.scatter(0, idx, remove), -float("inf"))
    p = torch.softmax(x, -1)
    if min_p is not None:                    # MinPLogitsWarper
        p = torch.where(p < min_p * p.max(), torch.zeros_like(p), p); p = p / p.sum()
    return p.numpy()

bad = n = 0
def check(tag, p):
    global bad, n
    n += 1
    c = counts[tag]; f = c / DRAWS
    support_ref = p > 0; support_got = c > 0
    sigma = np.sqrt(np.maximum(p * (1 - p), 1e-12) / DRAWS)
    zmax = np.max(np.abs(f - p) / np.maximum(sigma, 1e-9) * (p > 1e-4))
    outside = c[~support_ref].sum()
    # tokens with expected mass < 2e-4 may legitimately not appear; extra tokens outside the reference support are errors
    if outside > 0 or zmax > 5.0:
        bad += 1; print("MISMATCH", tag, f"zmax={zmax:.1f}, draws outside reference support={outside}")
        print("   got ", np.round(f, 4)); print("   want", np.round(p, 4))

LG = logits(12)
check("plain", reference(LG)); check("temp05", reference(LG, temperature=0.5)); check("temp17", reference(LG, temperature=1.7))
check("topk4", reference(LG, top_k=4)); check("topk1", reference(LG, top_k=1))
for tag, tp in [("topp09", 0.9), ("topp05", 0.5), ("topp03", 0.3)]: check(tag, reference(LG, top_p=tp))
check("minp01", reference(LG, min_p=0.1)); check("minp04", reference(LG, min_p=0.4))
check("combo", reference(LG, temperature=0.8, top_k=6, top_p=0.85, min_p=0.05))
H = [2, 2, 5, 7, 7, 7]
check("rep13", reference(LG, rep=1.3, hist=H)); check("presence", reference(LG, presence=0.8, hist=H)); check("frequency", reference(LG, freq=0.5, hist=H))
check("bias", reference(LG, bias={3: 2.5, 9: -4.0}))
B = torch.log(torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64))
check("boundary_p05", reference(B, top_p=0.5)); check("boundary_p075", reference(B, top_p=0.75))
print(f"{n} configurations compared, {bad} mismatches")
sys.exit(1 if bad else 0)
