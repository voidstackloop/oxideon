"""Trains a GPT language model of about 4M, 20M or 50M parameters on a token file made by `prepare_code_corpus.py`.

    python examples/train_gpt.py --size 20m --data runs/data --out runs/gpt20m --device gpu --minutes 60

The run is budgeted by wall-clock time (`--minutes`) or by `--steps`; the learning-rate schedule (warm-up, then cosine to 10 %)
follows whichever is given, so a time-budgeted run ends at the bottom of the schedule.
"""
import argparse
import json
import math
import os
import time

import numpy as np

import oxideon as ox

# (embed_dim, layers, heads): 4.3 M, 21.1 M and 50.2 M parameters at a 4096-token vocabulary and 256 positions
SIZES = {"4m": (256, 4, 4), "20m": (512, 6, 8), "50m": (576, 12, 9)}


def put(x, device):
    """`'cuda'` is the native CUDA backend (`to_device`), `'gpu'` the wgpu one (`to`): both go with `model.to(device=...)`."""
    if device == "cpu":
        return x
    return x.to_device(device) if device.startswith("cuda") else x.to(device)


def batch(tokens, B, T, rng, device):
    starts = rng.integers(0, len(tokens) - T - 1, size=B)
    rows = np.stack([tokens[s:s + T + 1] for s in starts]).astype(np.int64)
    return put(ox.from_numpy(np.ascontiguousarray(rows)), device)


def evaluate(model, tokens, B, T, device, batches=20):
    rng = np.random.default_rng(1234)
    losses = []
    with ox.no_grad():
        for _ in range(batches):
            x = batch(tokens, B, T, rng, device)
            losses.append(float(model.loss(x[:, :-1], x[:, 1:]).item()))
    return float(np.mean(losses))


def generate(model, tok, prompt, n, device, temperature=0.8, top_k=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = tok.encode(prompt).ids
    for _ in range(n):
        ctx = ids[-model.max_seq_len:]
        x = put(ox.from_numpy(np.asarray([ctx], dtype=np.int64)), device)
        with ox.no_grad():
            logits = model(x)[0, -1].to("cpu").to_numpy().astype(np.float64) / temperature
        keep = np.argsort(logits)[-top_k:]
        p = np.exp(logits[keep] - logits[keep].max())
        ids.append(int(rng.choice(keep, p=p / p.sum())))
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(SIZES), required=True)
    ap.add_argument("--data", default="runs/data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="gpu", help="'cuda' (native CUDA), 'gpu' (wgpu) or 'cpu'")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None, help="peak learning rate (default: 1e-3 / 6e-4 / 4e-4 for 4m / 20m / 50m)")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.steps is None and a.minutes is None:
        ap.error("give --steps or --minutes")
    out = a.out or f"runs/gpt{a.size}"
    os.makedirs(out, exist_ok=True)
    tok = ox.Tokenizer.from_file(os.path.join(a.data, "tokenizer.json"))
    train = np.load(os.path.join(a.data, "train.npy"))
    val = np.load(os.path.join(a.data, "val.npy"))
    d, L, H = SIZES[a.size]
    ox.manual_seed(a.seed)
    model = ox.models.GPT(vocab_size=tok.get_vocab_size(), max_seq_len=a.seq_len, embed_dim=d, num_layers=L, num_heads=H)
    if a.device != "cpu":
        model.to(device=a.device)
    peak = a.lr or {"4m": 1e-3, "20m": 6e-4, "50m": 4e-4}[a.size]
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = ox.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                         lr=peak, betas=(0.9, 0.95))
    print(f"{a.size}: {model.num_params() / 1e6:.2f} M parameters, {len(train) / 1e6:.1f} M training tokens, device {a.device}", flush=True)
    log = open(os.path.join(out, "log.jsonl"), "a")
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    budget = a.minutes * 60 if a.minutes else None
    step, tokens_seen, smooth = 0, 0, None
    while True:
        frac = (time.time() - t0) / budget if budget else step / a.steps
        if frac >= 1.0:
            break
        lr = peak * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))
        for g in opt.param_groups:
            g["lr"] = lr
        x = batch(train, a.batch, a.seq_len, rng, a.device)
        opt.zero_grad()
        loss = model.loss(x[:, :-1], x[:, 1:])
        loss.backward()
        if a.grad_clip:
            ox.nn.utils.clip_grad_norm_(list(model.parameters()), a.grad_clip)
        opt.step()
        step += 1
        tokens_seen += a.batch * a.seq_len
        if step % 10 == 0:
            lv = float(loss.item())
            smooth = lv if smooth is None else 0.9 * smooth + 0.1 * lv
            if step % 50 == 0:
                rec = dict(step=step, loss=round(smooth, 4), lr=lr, tokens=tokens_seen, minutes=round((time.time() - t0) / 60, 2),
                           tok_per_s=int(tokens_seen / (time.time() - t0)))
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
        if step % a.eval_every == 0:
            vl = evaluate(model, val, a.batch, a.seq_len, a.device)
            print(json.dumps(dict(step=step, val_loss=round(vl, 4), val_ppl=round(math.exp(vl), 2))), flush=True)
            log.write(json.dumps(dict(step=step, val_loss=vl)) + "\n")
            model.save_file(os.path.join(out, "model.oxd"))
    vl = evaluate(model, val, a.batch, a.seq_len, a.device, batches=50)
    model.save_file(os.path.join(out, "model.oxd"))
    sample = generate(model, tok, "def fibonacci(n):\n", 120, a.device)
    summary = dict(size=a.size, params=model.num_params(), steps=step, tokens=tokens_seen, minutes=round((time.time() - t0) / 60, 1),
                   val_loss=round(vl, 4), val_ppl=round(math.exp(vl), 2), sample=sample)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("DONE", json.dumps({k: v for k, v in summary.items() if k != "sample"}), flush=True)
    print(sample, flush=True)


if __name__ == "__main__":
    main()
