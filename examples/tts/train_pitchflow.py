"""Trains `ox.models.PitchFlow`: a small flow-matching model that *samples* the melody of a sentence, on top of a frozen TinyTTS.
The TinyTTS regression pitch head learns the average melody, which sounds flat; this model learns the distribution instead.

    python examples/tts/train_pitchflow.py --data voca/data --tts tinytts_BEST.oxd --out voca/pf --device cuda --minutes 50

With `--target mel` the same model learns the spectrogram DETAIL instead ("SpecFlow"): the residual (true - TinyTTS predicted
normalised log-mel) times MEL_FLOW_SCALE, 80 channels; at synthesis the sample is added back onto the predicted mel.

Pitch target per frame: [normalised log-F0 (interpolated through unvoiced frames), voicing sign +1 / -1]. Condition: the TinyTTS
decoder's hidden states under the true durations. Loss: rectified-flow velocity MSE, x_t = (1 - t) x0 + t x1, v = x1 - x0.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

import oxideon as ox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_tts import Data  # noqa: E402
from synthesize_voca import load, MEL_FLOW_SCALE  # noqa: E402


def flow_targets(data, idx):
    """`[B, T, 2]` targets and the `[B, T]` frame mask, both numpy."""
    T = max(int(data.frames[i]) for i in idx)
    x1, fm = np.zeros((len(idx), T, 2), np.float32), np.zeros((len(idx), T), np.float32)
    for b, i in enumerate(idx):
        lo, t = data.mel_off[i], int(data.frames[i])
        conf = data.conf[lo:lo + t]
        mu, sd = data.f0_stats[str(int(data.voice[i]))]
        voiced = np.flatnonzero(conf > 0.8)
        lf = (np.log(np.maximum(data.f0[lo:lo + t], 1.0)) - mu) / sd
        x1[b, :t, 0] = np.interp(np.arange(t), voiced, lf[voiced]) if len(voiced) else 0.0  # holds the end values at the edges
        x1[b, :t, 1] = np.where(conf > 0.7, 1.0, -1.0)
        fm[b, :t] = 1.0
    return x1, fm


def conditions(tts, b):
    """The frozen TinyTTS decoder states and its predicted normalised log-mel."""
    with ox.no_grad():
        enc, _ = tts.encode(b["ph"], b["ws"], b["voice"], b["ph_mask"])
        if getattr(tts, "variance", False):
            enc = tts.add_variance(enc, b["pbin"], b["ebin"])
        h = tts.decode_hidden(enc, b["A"], b["phase"], b["fmask"])
        return h, h @ tts.mel_w + tts.mel_b


def targets(target, data, idx, b, pred_mel):
    if target == "pitch":
        return flow_targets(data, idx)
    fm = b["fmask"].to("cpu").to_numpy().astype(np.float32)
    return ((b["mel"].to("cpu").to_numpy() - pred_mel.to("cpu").to_numpy()) * MEL_FLOW_SCALE * fm[..., None]).astype(np.float32), fm


def flow_loss(flow, cond, fmask, x1, fm, rng, device):
    B = x1.shape[0]
    x0 = rng.standard_normal(x1.shape).astype(np.float32)
    t = rng.random(B).astype(np.float32)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    to = lambda x: ox.from_numpy(np.ascontiguousarray(x)) if device == "cpu" else ox.from_numpy(np.ascontiguousarray(x)).to_device(device)
    v = flow(to(xt), to(flow.time_features(t, flow.dim)), cond, fmask)
    m = to(fm[..., None])
    return (((v - to(x1 - x0)) ** 2) * m).sum() / (m.sum() * float(x1.shape[-1]))


def evaluate(flow, tts, data, device, max_frames, target="pitch"):
    rng, out = np.random.default_rng(0), []
    with ox.no_grad():
        for idx in data.batches(rng, max_frames, shuffle=False):
            b = data.collate(idx, device)
            h, pm = conditions(tts, b)
            x1, fm = targets(target, data, idx, b, pm)
            out.append(float(flow_loss(flow, h, b["fmask"], x1, fm, rng, device).item()))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tts", required=True, help="the frozen TinyTTS checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--minutes", type=float, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--voice-only", type=int, default=None, help="train / validate on this voice id only (0 = F2, 1 = M2)")
    ap.add_argument("--target", choices=("pitch", "mel"), default="pitch")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tts, meta = load(a.data, a.tts, a.device)
    val_name = "val.npz" if os.path.exists(os.path.join(a.data, "val.npz")) else "validation.npz"
    train, val = Data(os.path.join(a.data, "train.npz"), meta), Data(os.path.join(a.data, val_name), meta)
    train.only = val.only = a.voice_only
    ox.manual_seed(a.seed)
    ch = 2 if a.target == "pitch" else meta["n_mels"]
    mk = lambda: ox.models.PitchFlow(cond_dim=tts.dim, dim=a.dim, layers=a.layers, channels=ch)
    flow, ema_flow = mk(), mk()
    ema_flow.load_state_dict(flow.state_dict())
    if a.device != "cpu":
        flow.to(device=a.device)
        ema_flow.to(device=a.device)
    decay = [p for p in flow.parameters() if p.ndim >= 2]
    other = [p for p in flow.parameters() if p.ndim < 2]
    opt = ox.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": other, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98))
    print(f"PitchFlow {flow.num_params() / 1e6:.3f} M parameters on TinyTTS {tts.num_params() / 1e6:.2f} M", flush=True)
    rng = np.random.default_rng(a.seed)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0, step, run, best = time.time(), 0, None, 1e9
    while time.time() - t0 < a.minutes * 60:
        for idx in train.batches(rng, a.max_frames):
            frac = (time.time() - t0) / (a.minutes * 60)
            if frac >= 1.0:
                break
            lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = lr
            b = train.collate(idx, a.device)
            h, pm = conditions(tts, b)
            x1, fm = targets(a.target, train, idx, b, pm)
            opt.zero_grad()
            loss = flow_loss(flow, h, b["fmask"], x1, fm, rng, a.device)
            loss.backward()
            ox.nn.utils.clip_grad_norm_(list(flow.parameters()), 1.0)
            opt.step()
            step += 1
            d = min(a.ema, (1 + step) / (10 + step))
            with ox.no_grad():
                for pe, p in zip(ema_flow.parameters(), flow.parameters()):
                    pe.copy_(pe * d + p * (1.0 - d))
            if step % 25 == 0:
                run = float(loss.item()) if run is None else 0.9 * run + 0.1 * float(loss.item())
                if step % 100 == 0:
                    rec = dict(step=step, flow_mse=round(run, 4), lr=lr, minutes=round((time.time() - t0) / 60, 2))
                    print(json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
            if step % a.eval_every == 0:
                for tag, m_ in (("raw", flow), ("ema", ema_flow)):
                    v = evaluate(m_, tts, val, a.device, a.max_frames, a.target)
                    print(json.dumps(dict(step=step, val_flow_mse=round(v, 4), which=tag)), flush=True)
                    if v < best:
                        best = v
                        m_.save_file(os.path.join(a.out, "pitchflow_best.oxd"))
                        print(json.dumps(dict(step=step, best=tag, best_val_flow_mse=round(v, 4))), flush=True)
    ema_flow.save_file(os.path.join(a.out, "pitchflow_ema.oxd"))
    v = evaluate(ema_flow, tts, val, a.device, a.max_frames, a.target)
    if v < best:
        best = v
        ema_flow.save_file(os.path.join(a.out, "pitchflow_best.oxd"))
    summary = dict(best_val_flow_mse=round(best, 4), final_ema_val=round(v, 4), params=flow.num_params(), steps=step, minutes=round((time.time() - t0) / 60, 1))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
