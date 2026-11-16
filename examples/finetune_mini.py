#!/usr/bin/env python3
"""Pretrain a mini model, fine-tune it on a small new task, and check that fine-tuning actually helps.

    python examples/finetune_mini.py [cpu|wgpu]       (default: wgpu when an adapter exists, otherwise cpu)

A. MiniImageClassifier: pretrain with `fit` on the 1,000-image set, fine-tune on 64 images from another seed, compare with
   training on those 64 images from scratch, on 500 held-out images.
B. CausalLM: pretrain on "count up by 1 (mod 16)", fine-tune on "count up by 3" from four start tokens, test on twelve unseen
   start tokens; reports how much of the first task is forgotten.
C. A two-layer MLP written with raw tensors and trained with AdamW on the chosen device. The same run on the CPU and the GPU
   must give the same loss curve and about the same accuracy, and fine-tuning must beat training from scratch.

Exit code 0 only if every check passes.
"""
import sys
import time

import oxideon as ox

problems = []


def check(ok, text):
    print(("  ok    " if ok else "  FAIL  ") + text)
    if not ok:
        problems.append(text)


def banner(title):
    print("\n" + "=" * 72 + f"\n {title}\n" + "=" * 72)


def pick_device():
    if len(sys.argv) > 1:
        return sys.argv[1]
    try:
        ox.Tensor.zeros([1]).to_device("wgpu")
        return "wgpu"
    except Exception:
        return "cpu"


def part_a():
    banner("A. MiniImageClassifier: pretrain, fine-tune, compare with scratch")
    t0 = time.time()
    train, val = ox.MiniImageDataset.standard_1000().split(0.8)
    small = ox.MiniImageDataset.generate(num_images=64, seed=7)
    held = ox.MiniImageDataset.generate(num_images=500, seed=99)

    pre = ox.MiniImageClassifier.standard()
    untrained = pre.evaluate(held.images, held.labels)[1]
    pre.fit(train, val_dataset=val, epochs=3, batch_size=32, lr=0.005)
    zero_shot = pre.evaluate(held.images, held.labels)[1]
    pre.fit(small, epochs=2, batch_size=16, lr=0.001)  # fine-tune: continues from the pretrained weights, smaller step
    tuned = pre.evaluate(held.images, held.labels)[1]

    scratch = ox.MiniImageClassifier.standard()
    scratch.fit(small, epochs=2, batch_size=16, lr=0.001)
    from_scratch = scratch.evaluate(held.images, held.labels)[1]
    print(f"  held-out accuracy: untrained {untrained:.3f} | pretrained {zero_shot:.3f} | pretrained + fine-tuned {tuned:.3f} | scratch on the same 64 images {from_scratch:.3f}  ({time.time() - t0:.0f}s)")
    check(zero_shot > untrained + 0.3, "pretraining raises held-out accuracy far above the untrained model")
    check(tuned >= 0.8 and tuned > from_scratch, "the fine-tuned model beats training on the small set from scratch")


V, L = 16, 16


def counting(step, starts):
    return [[(s + step * i) % V for i in range(L)] for s in starts]


def next_token_accuracy(lm, data):
    tokens = ox.Tensor.from_i64([x for s in data for x in s], [len(data), L])
    pred = [int(p) for p in lm.forward(tokens).argmax(2).to_list()]
    hits = sum(pred[b * L + i] == s[i + 1] for b, s in enumerate(data) for i in range(L - 1))
    return hits / (len(data) * (L - 1))


def part_b():
    banner("B. CausalLM: pretrain on 'count by 1', fine-tune on 'count by 3'")
    t0 = time.time()
    ox.manual_seed(1)
    plus1, train3, test3 = counting(1, range(V)), counting(3, range(0, 4)), counting(3, range(4, V))
    lm = ox.CausalLM(V, 32, 2, 2, 64, 32)
    lm.train_ntp_epochs_in_rust(plus1, epochs=40, lr=0.01)
    task1 = next_token_accuracy(lm, plus1)
    zero_shot = next_token_accuracy(lm, test3)
    lm.train_ntp_epochs_in_rust(train3, epochs=40, lr=0.005)
    tuned, forgotten = next_token_accuracy(lm, test3), next_token_accuracy(lm, plus1)
    print(f"  task 1 after pretraining {task1:.2f} | task 2 before fine-tuning {zero_shot:.2f} -> after {tuned:.2f} on unseen starts | task 1 afterwards {forgotten:.2f}  ({time.time() - t0:.0f}s)")
    check(task1 >= 0.95, "the model learns the first task")
    check(zero_shot < 0.5 and tuned >= 0.95, "fine-tuning on four sequences teaches the new rule to unseen start tokens")
    print(f"  note: full fine-tuning forgot the first task (accuracy {task1:.2f} -> {forgotten:.2f}); a LoRA adapter or replay would keep it")


def mlp_run(device, steps_pre=150, steps_ft=25):
    train, _ = ox.MiniImageDataset.standard_1000().split(0.8)
    small = ox.MiniImageDataset.generate(num_images=64, seed=7)
    held = ox.MiniImageDataset.generate(num_images=500, seed=99)

    def init():
        ox.manual_seed(0)
        shapes = [([256, 32], 0.06), ([32], 0.0), ([32, 4], 0.2), ([4], 0.0)]
        return [(ox.Tensor.randn(s) * k).to_device(device).requires_grad_(True) for s, k in shapes]

    def forward(p, x):
        return (x @ p[0] + p[1]).relu() @ p[2] + p[3]

    def train_on(p, data, steps, lr):
        opt = ox.AdamW(p, lr=lr)
        losses = []
        for step in range(steps):
            x, y = data.batch(step % data.num_batches(32), 32)
            opt.zero_grad()
            loss = ox.cross_entropy(forward(p, x.reshape([len(y), 256]).to_device(device)), y)
            loss.backward()
            opt.step()
            losses.append(loss.cpu().item())
        return losses

    def accuracy(p, data):
        pred = forward(p, data.images.reshape([len(data), 256]).to_device(device)).argmax(1).cpu().to_list()
        return sum(int(a) == b for a, b in zip(pred, data.labels)) / len(data)

    params = init()
    pre_losses = train_on(params, train, steps_pre, 0.01)
    zero_shot = accuracy(params, held)
    ft_losses = train_on(params, small, steps_ft, 0.003)
    tuned = accuracy(params, held)
    scratch = init()
    train_on(scratch, small, steps_ft, 0.003)
    return pre_losses + ft_losses, zero_shot, tuned, accuracy(scratch, held), params[0].device


def part_c(device):
    banner(f"C. Raw-tensor MLP with AdamW on {device}, compared with the CPU")
    t0 = time.time()
    losses, zero_shot, tuned, scratch, where = mlp_run(device)
    t_dev = time.time() - t0
    print(f"  parameters live on {where}; held-out accuracy: pretrained {zero_shot:.3f} | fine-tuned {tuned:.3f} | scratch on the same 64 images {scratch:.3f}  ({t_dev:.0f}s)")
    print(f"  loss: first {losses[0]:.4f}, end of pretraining {losses[149]:.4f}, end of fine-tuning {losses[-1]:.4f}")
    check(tuned >= 0.8 and tuned > scratch, "the fine-tuned MLP beats training on the small set from scratch")
    if device != "cpu":
        t0 = time.time()
        cpu_losses, _, cpu_tuned, _, _ = mlp_run("cpu")
        gap = max(abs(a - b) for a, b in zip(losses, cpu_losses))
        print(f"  same run on the CPU: {time.time() - t0:.0f}s; largest difference in the loss curve {gap:.2e}; accuracy {cpu_tuned:.3f} vs {tuned:.3f}")
        check(gap < 5e-2 and abs(cpu_tuned - tuned) <= 0.03, "the GPU run follows the CPU run (loss curve and accuracy)")


if __name__ == "__main__":
    dev = pick_device()
    print("oxideon", ox.__version__, "| device for part C:", dev)
    part_a()
    part_b()
    part_c(dev)
    banner("RESULT: " + ("PASS" if not problems else "FAIL"))
    for p in problems:
        print("  -", p)
    sys.exit(1 if problems else 0)
