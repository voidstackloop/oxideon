"""Batch 53: SummaryWriter event files read back by TensorBoard's EventAccumulator (venv with tensorboard installed)."""
import math, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "tb_run")
ea = EventAccumulator(d, size_guidance={"scalars": 0, "histograms": 0}); ea.Reload()
tags = ea.Tags(); bad = n = 0
def check(cond, msg):
    global bad, n
    n += 1
    if not cond: bad += 1; print("FAIL", msg)
check(set(tags["scalars"]) == {"loss/train", "metrics/acc", "metrics/lr"}, f"scalar tags {tags['scalars']}")
loss = ea.Scalars("loss/train"); check(len(loss) == 20 and [e.step for e in loss] == list(range(20)), "loss steps")
check(all(abs(e.value - 1.0 / (1.0 + e.step)) < 1e-6 for e in loss), "loss values")
acc = ea.Scalars("metrics/acc"); check(all(abs(e.value - e.step / 20.0) < 1e-6 for e in acc), "acc values")
h = ea.Histograms("weights"); check(len(h) == 1 and h[0].step == 7, "histogram present")
hv = h[0].histogram_value
vals = [math.sin(i * 0.37) * 2 + 0.5 for i in range(1000)]
check(hv.num == 1000 and abs(hv.min - min(vals)) < 1e-9 and abs(hv.max - max(vals)) < 1e-9, "histogram min/max/num")
check(abs(hv.sum - sum(vals)) < 1e-6 and abs(hv.sum_squares - sum(v * v for v in vals)) < 1e-6, "histogram sums")
check(len(hv.bucket) == 30 and abs(sum(hv.bucket) - 1000) < 1e-9, "histogram buckets")
print(f"{n} checks, {bad} failures"); sys.exit(1 if bad else 0)
