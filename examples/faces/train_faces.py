"""Trains `ox.models.FaceFlow` (about 10 M parameters) on face images + their captions.

    python examples/faces/train_faces.py --data faces.npz --out runs/faces --device cuda --minutes 240

`faces.npz` holds `images` (uint8 `[N, 64, 64, 3]`) and `captions` (an array of strings). Words become token ids (0 = pad,
1 = the "no caption" null token used for classifier-free guidance, 2 = unknown). The weights are averaged (EMA) and the EMA
copy is what gets sampled and saved as `faces_ema.oxd`; a grid of samples for fixed captions is written every `--sample-every`.
"""
import argparse
import json
import math
import os
import re
import time

import numpy as np

import oxideon as ox

PAD, NULL, UNK = 0, 1, 2


def words(caption):
    return re.findall(r"[a-z]+|[,]", caption.lower())


def build_vocab(captions, min_count=1):
    from collections import Counter
    c = Counter(w for cap in captions for w in words(cap))
    return {w: i + 3 for i, w in enumerate(sorted(w for w, n in c.items() if n >= min_count))}


def encode(captions, vocab, max_tokens):
    ids = np.zeros((len(captions), max_tokens), np.int64)
    for i, cap in enumerate(captions):
        t = [vocab.get(w, UNK) for w in words(cap)][:max_tokens]
        ids[i, :len(t)] = t
    return ids


def put(x, device):  # 'cuda' is the native CUDA backend (`to_device`), 'gpu' the wgpu one (`to`)
    t = ox.from_numpy(np.ascontiguousarray(x))
    return t if device == "cpu" else t.to_device(device) if device.startswith("cuda") else t.to(device)


def save_grid(path, imgs):
    """imgs: float `[B, 3, H, W]` in [-1, 1] -> one PNG with up to 8 columns."""
    from PIL import Image
    a = np.clip((imgs.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
    cols = min(8, len(a))
    rows = math.ceil(len(a) / cols)
    h, w = a.shape[1:3]
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, im in enumerate(a):
        canvas[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = im
    Image.fromarray(canvas).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--drop", type=float, default=0.1, help="probability of dropping the caption (classifier-free guidance)")
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None)
    a = ap.parse_args()
    if a.steps is None and a.minutes is None:
        ap.error("give --steps or --minutes")
    os.makedirs(a.out, exist_ok=True)
    z = np.load(a.data, allow_pickle=False)
    images, captions = z["images"], [str(c) for c in z["captions"]]
    n = len(images)
    vocab_path = os.path.join(a.out, "vocab.json")
    if os.path.exists(vocab_path):
        vocab = json.load(open(vocab_path))
    else:
        vocab = build_vocab(captions)
        json.dump(vocab, open(vocab_path, "w"))
    tokens = encode(captions, vocab, a.max_tokens)
    size = images.shape[1]
    ox.manual_seed(a.seed)
    cfg = dict(vocab_size=len(vocab) + 3, max_tokens=a.max_tokens, size=size)
    model = ox.models.FaceFlow(**cfg)
    if a.init:
        model.load_file(a.init)
    ema_model = ox.models.FaceFlow(**cfg)
    ema_model.load_state_dict(model.state_dict())
    if a.device != "cpu":
        model.to(device=a.device)
        ema_model.to(device=a.device)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"))
    decay = [p for p in model.parameters() if p.ndim >= 2]
    other = [p for p in model.parameters() if p.ndim < 2]
    opt = ox.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": other, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.99))
    print(f"FaceFlow {model.num_params() / 1e6:.2f} M parameters; {n} images {size}px; {len(vocab)} words", flush=True)

    # fixed captions for the progress grids (distinct, from the training set)
    rng = np.random.default_rng(a.seed)
    probe = rng.choice(n, 16, replace=False)
    ptok, pmask = put(tokens[probe], a.device), put(tokens[probe] != PAD, a.device)
    with open(os.path.join(a.out, "probe_captions.txt"), "w") as f:
        f.write("\n".join(captions[i] for i in probe))
    save_grid(os.path.join(a.out, "probe_real.png"), images[probe].astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1)

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0, step, avg = time.time(), 0, None
    budget = a.minutes * 60 if a.minutes else None
    order = rng.permutation(n)
    pos = 0
    while True:
        frac = (time.time() - t0) / budget if budget else step / a.steps
        if frac >= 1.0:
            break
        if pos + a.batch > n:
            order, pos = rng.permutation(n), 0
        idx = order[pos:pos + a.batch]
        pos += a.batch
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        x = images[idx].astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1
        flip = rng.random(len(idx)) < 0.5  # a mirrored face is still a face with the same caption
        x[flip] = x[flip][..., ::-1]
        tok = tokens[idx].copy()
        drop = rng.random(len(idx)) < a.drop
        tok[drop] = 0
        tok[drop, 0] = NULL
        t = rng.beta(1.5, 1.0, len(idx)).astype(np.float32).clip(1e-3, 1.0)  # a little more weight on the noisy end
        noise = rng.standard_normal(x.shape).astype(np.float32)
        opt.zero_grad()
        loss = model.loss(put(x, a.device), put(noise, a.device), put(t, a.device), put(tok, a.device), put(tok != PAD, a.device))
        loss.backward()
        ox.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
        opt.step()
        step += 1
        d = min(a.ema, (1 + step) / (10 + step))
        with ox.no_grad():
            for pe, p in zip(ema_model.parameters(), model.parameters()):
                pe.copy_(pe * d + p * (1.0 - d))
        if step % 25 == 0:
            v = float(loss.item())
            avg = v if avg is None else 0.9 * avg + 0.1 * v
            if step % 100 == 0:
                rec = dict(step=step, loss=round(avg, 4), lr=lr, minutes=round((time.time() - t0) / 60, 2))
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
        if step % a.sample_every == 0:
            img = ema_model.sample(ptok, pmask, steps=20, guidance=3.0, seed=1).to("cpu").to_numpy()
            save_grid(os.path.join(a.out, f"samples_{step:06d}.png"), img)
            ema_model.save_file(os.path.join(a.out, "faces_ema.oxd"))
            model.save_file(os.path.join(a.out, "faces_raw.oxd"))
    ema_model.save_file(os.path.join(a.out, "faces_ema.oxd"))
    model.save_file(os.path.join(a.out, "faces_raw.oxd"))
    img = ema_model.sample(ptok, pmask, steps=30, guidance=3.0, seed=1).to("cpu").to_numpy()
    save_grid(os.path.join(a.out, "samples_final.png"), img)
    json.dump(dict(params=model.num_params(), steps=step, minutes=round((time.time() - t0) / 60, 1), final_loss=avg), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
