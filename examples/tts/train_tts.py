"""Trains `ox.models.TinyTTS` (about 4 M parameters) on the arrays made by `prepare_tts.py`.

    python examples/tts/train_tts.py --data runs/tts/data --out runs/tts/model --device gpu --minutes 120

Loss: L1 on the normalised log-mel frames plus an MSE on log(1 + frames) for the duration of every phoneme. Batches are
bucketed by length and capped by a frame budget so padding stays small.
"""
import argparse
import json
import math
import os
import time

import numpy as np

import oxideon as ox


class Data:
    def __init__(self, path, meta):
        z = np.load(path)
        self.mel, self.mel_off, self.ids, self.dur, self.ws, self.ph_off, self.voice = (z[k] for k in ("mel", "mel_off", "ids", "dur", "ws", "ph_off", "voice"))
        self.n = len(self.voice)
        self.mean, self.std = meta["mel_mean"], meta["mel_std"]
        self.frames = np.diff(self.mel_off)
        self.f0, self.conf = (z[k] if k in z else None for k in ("f0", "conf"))  # per-frame pitch (tiny-voca data only)
        self.f0_stats = meta.get("f0_stats")
        self.only = None  # a voice id: batches() then yields only that speaker's clips
        self.voiced_ids = None  # token ids that are voiced by definition (vowels): their frames get voicing target 1 (tiny-voca)

    def batches(self, rng, max_frames, shuffle=True):
        keep = np.arange(self.n) if self.only is None else np.flatnonzero(self.voice == self.only)
        order = keep[np.argsort(self.frames[keep] + (rng.random(len(keep)) * 40 if shuffle else 0))]
        out, cur, cur_max = [], [], 0
        for i in order:
            m = max(cur_max, int(self.frames[i]))
            if cur and m * (len(cur) + 1) > max_frames:
                out.append(cur)
                cur, m = [], int(self.frames[i])
            cur.append(int(i))
            cur_max = m
        if cur:
            out.append(cur)
        if shuffle:
            rng.shuffle(out)
        return out

    def vowel_frames(self, i, t):
        """Bool `[t]`: frames of clip `i` that belong to a token in `voiced_ids` (all False when it is None)."""
        if self.voiced_ids is None:
            return np.zeros(t, bool)
        a, e = self.ph_off[i], self.ph_off[i + 1]
        rows = np.repeat(np.arange(e - a), self.dur[a:e].astype(np.int64))[:t]
        rows = np.concatenate([rows, np.full(t - len(rows), e - a - 1)]) if len(rows) < t else rows
        return np.isin(self.ids[a:e], self.voiced_ids)[rows]

    def collate(self, idx, device):
        B = len(idx)
        N = max(int(self.ph_off[i + 1] - self.ph_off[i]) for i in idx)
        T = max(int(self.frames[i]) for i in idx)
        ph = np.zeros((B, N), np.int64)
        ws = np.zeros((B, N), np.int64)
        dur = np.zeros((B, N), np.float32)
        ph_mask = np.zeros((B, N), bool)
        A = np.zeros((B, T, N), np.float32)
        phase = np.zeros((B, T, 2), np.float32)
        mel = np.zeros((B, T, self.mel.shape[1]), np.float32)
        fmask = np.zeros((B, T), bool)
        f0n, vmask, vtgt = np.zeros((B, T), np.float32), np.zeros((B, T), bool), np.zeros((B, T), np.float32)
        pv = np.zeros((B, N, 2), np.float32)  # per-phoneme [pitch, energy] targets for a variance-adaptor model
        for b, i in enumerate(idx):
            a, e = self.ph_off[i], self.ph_off[i + 1]
            n = int(e - a)
            d = self.dur[a:e].astype(np.int64)
            t = int(self.frames[i])
            rows = np.repeat(np.arange(n), d)[:t]
            if len(rows) < t:  # rounding left a few frames over: they belong to the last phoneme
                rows = np.concatenate([rows, np.full(t - len(rows), n - 1)])
            ph[b, :n], ws[b, :n], dur[b, :n], ph_mask[b, :n] = self.ids[a:e], self.ws[a:e], d, True
            A[b, np.arange(t), rows] = 1.0
            start = np.concatenate([[0], np.cumsum(d)])[:-1]
            length = np.maximum(np.bincount(rows, minlength=n)[rows], 1)
            pos = np.arange(t) - start[rows]
            phase[b, :t, 0], phase[b, :t, 1] = pos / length, 1.0 / length
            mel[b, :t] = (self.mel[self.mel_off[i]:self.mel_off[i + 1]].astype(np.float32) - self.mean) / self.std
            fmask[b, :t] = True
            if self.f0 is not None:
                lo = self.mel_off[i]
                conf = self.conf[lo:lo + t]
                mu, sd = self.f0_stats[str(int(self.voice[i]))]
                voiced = conf > 0.8
                f0n[b, :t] = np.where(voiced, (np.log(np.maximum(self.f0[lo:lo + t], 1.0)) - mu) / sd, 0.0)
                vmask[b, :t], vtgt[b, :t] = voiced, (conf > 0.7) | self.vowel_frames(i, t)
                nv = np.bincount(rows, weights=voiced, minlength=n)  # mean normalised log-F0 of the voiced frames; 0 (= the speaker's
                pv[b, :n, 0] = np.where(nv > 0, np.bincount(rows, weights=f0n[b, :t] * voiced, minlength=n) / np.maximum(nv, 1), 0.0)  # mean) if none
            nf = np.bincount(rows, minlength=n)  # energy: mean normalised log-mel of the phoneme's frames
            pv[b, :n, 1] = np.bincount(rows, weights=mel[b, :t].mean(1), minlength=n) / np.maximum(nf, 1)
        def to(x):  # 'cuda' is the native CUDA backend (`to_device`), 'gpu' the wgpu one (`to`)
            t = ox.from_numpy(np.ascontiguousarray(x))
            return t if device == "cpu" else t.to_device(device) if device.startswith("cuda") else t.to(device)
        return dict(ph=to(ph), ws=to(ws), voice=to(self.voice[idx].astype(np.int64)), ph_mask=to(ph_mask), A=to(A), phase=to(phase),
                    mel=to(mel), fmask=to(fmask), dur=to(np.log1p(dur)), **(dict(f0=to(f0n), vmask=to(vmask), vtgt=to(vtgt)) if self.f0 is not None else {}),
                    pv=to(pv), pbin=to(ox.models.TinyTTS.var_bins(pv[..., 0])), ebin=to(ox.models.TinyTTS.var_bins(pv[..., 1])))


def losses(model, b, predicted_variance=False, var_pred_prob=0.0, rng=None):
    """`(mel L1, duration MSE, pitch L1, voicing BCE)`; the last two are None for a model without pitch outputs. A variance-adaptor
    model adds its per-phoneme pitch / energy MSE into the duration term; its decoder sees the true bins while training and,
    with `predicted_variance=True` (validation), its own quantised predictions - as at inference."""
    args = (b["ph"], b["ws"], b["voice"], b["ph_mask"], b["A"], b["phase"], b["fmask"])
    use_pred = predicted_variance or (var_pred_prob > 0 and rng is not None and rng.random() < var_pred_prob)  # scheduled sampling
    out = model(*args) if not getattr(model, "variance", False) else model(*args) if use_pred else model(*args, b["pbin"], b["ebin"])
    vpred = out[-1] if getattr(model, "variance", False) else None
    out = out[:-1] if vpred is not None else out
    mel, aux, log_dur = (out[0], out[1], out[2]) if len(out) == 3 else (out[0], None, out[1])
    fm = b["fmask"].to_dtype("f32").reshape([int(b["fmask"].shape[0]), int(b["fmask"].shape[1]), 1])
    pm = b["ph_mask"].to_dtype("f32")
    l1 = ((mel - b["mel"]).abs() * fm).sum() / (fm.sum() * float(mel.shape[-1]))
    ld = (((log_dur - b["dur"]) ** 2) * pm).sum() / pm.sum()
    if vpred is not None:
        ld = ld + (((vpred - b["pv"]) ** 2) * pm.reshape([int(pm.shape[0]), int(pm.shape[1]), 1])).sum() / (2.0 * pm.sum())
    if aux is None:
        return l1, ld, None, None
    B, T = int(b["fmask"].shape[0]), int(b["fmask"].shape[1])
    f0p, vl = aux[:, :, 0], aux[:, :, 1]
    vm = b["vmask"].to_dtype("f32")
    lf = ((f0p - b["f0"]).abs() * vm).sum() / vm.sum().clamp(1.0, 1e12)
    y, fm2 = b["vtgt"], fm.reshape([B, T])
    bce = vl.clamp(0.0, 1e9) - vl * y + (1.0 + (-vl.abs()).exp()).log()  # binary cross-entropy with logits
    lv = (bce * fm2).sum() / fm2.sum()
    return l1, ld, lf, lv


def evaluate(model, data, device, max_frames):
    rng = np.random.default_rng(0)
    cols = [[], [], [], []]
    with ox.no_grad():
        for idx in data.batches(rng, max_frames, shuffle=False):
            for c, v in zip(cols, losses(model, data.collate(idx, device), predicted_variance=True)):
                if v is not None:
                    c.append(float(v.item()))
    return tuple(float(np.mean(c)) if c else None for c in cols)


def build(meta, train, conv_kernel=0, variance=False, dim=192, enc_layers=4, dec_layers=5):
    return ox.models.TinyTTS(meta["n_phones"], n_mels=meta["n_mels"], n_flags=meta.get("n_flags", 2), n_aux=2 if train.f0 is not None else 0,
                             conv_kernel=conv_kernel, variance=variance, dim=dim, enc_layers=enc_layers, dec_layers=dec_layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="gpu", help="'cuda' (native CUDA), 'gpu' (wgpu) or 'cpu'")
    ap.add_argument("--max-frames", type=int, default=6000, help="padded frames per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--dur-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.0, help="decay of an exponential moving average of the weights (0 = off, e.g. 0.999); it is evaluated and saved next to the raw weights")
    ap.add_argument("--wd", type=float, default=0.01, help="AdamW weight decay on matrices")
    ap.add_argument("--init", default=None, help="start from the weights of this checkpoint (a .oxd from an earlier run of the same model)")
    ap.add_argument("--conv-kernel", type=int, default=0, help="odd width of a local temporal conv after every decoder block (0 = off)")
    ap.add_argument("--f0-weight", type=float, default=0.5, help="weight of the pitch L1 loss")
    ap.add_argument("--voicing-weight", type=float, default=0.2, help="weight of the voicing BCE loss")
    ap.add_argument("--variance", action="store_true", help="FastSpeech 2 style per-phoneme pitch + energy predictor feeding the decoder")
    ap.add_argument("--var-pred-prob", type=float, default=0.0, help="share of training steps where the decoder sees the predicted (not true) pitch / energy")
    ap.add_argument("--voice-only", type=int, default=None, help="train / validate on this voice id only (0 = F2, 1 = M2)")
    ap.add_argument("--dim", type=int, default=192, help="model width (192 = the 4 M model)")
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--dec-layers", type=int, default=5)
    a = ap.parse_args()
    size = dict(dim=a.dim, enc_layers=a.enc_layers, dec_layers=a.dec_layers)
    if a.steps is None and a.minutes is None:
        ap.error("give --steps or --minutes")
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    val_name = "val.npz" if os.path.exists(os.path.join(a.data, "val.npz")) else "validation.npz"
    train, val = Data(os.path.join(a.data, "train.npz"), meta), Data(os.path.join(a.data, val_name), meta)
    train.only = val.only = a.voice_only
    ox.manual_seed(a.seed)
    model = build(meta, train, a.conv_kernel, a.variance, **size)
    if a.init:
        model.load_file(a.init, strict=not (a.conv_kernel or a.variance))  # new conv weights are not in an older checkpoint
    ema_model = None
    if a.ema > 0:
        ema_model = build(meta, train, a.conv_kernel, a.variance, **size)
        ema_model.load_state_dict(model.state_dict())
    if a.device != "cpu":
        model.to(device=a.device)
        if ema_model is not None:
            ema_model.to(device=a.device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    other = [p for p in model.parameters() if p.ndim < 2]
    opt = ox.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": other, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98))
    kept = np.arange(train.n) if a.voice_only is None else np.flatnonzero(train.voice == a.voice_only)
    print(f"TinyTTS {model.num_params() / 1e6:.2f} M parameters; {len(kept)} training clips, {train.frames[kept].sum() * meta['hop'] / meta['sample_rate'] / 3600:.1f} h", flush=True)
    rng = np.random.default_rng(a.seed)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0, step, ema = time.time(), 0, None
    best_val, best_f0 = 1e9, 1e9
    budget = a.minutes * 60 if a.minutes else None
    done = False
    while not done:
        for idx in train.batches(rng, a.max_frames):
            frac = (time.time() - t0) / budget if budget else step / a.steps
            if frac >= 1.0:
                done = True
                break
            lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = lr
            b = train.collate(idx, a.device)
            opt.zero_grad()
            l1, ld, lf, lv = losses(model, b, var_pred_prob=a.var_pred_prob, rng=rng)
            (l1 + a.dur_weight * ld + (a.f0_weight * lf + a.voicing_weight * lv if lf is not None else 0.0)).backward()
            ox.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            opt.step()
            step += 1
            if ema_model is not None:
                d = min(a.ema, (1 + step) / (10 + step))
                with ox.no_grad():
                    for pe, p in zip(ema_model.parameters(), model.parameters()):
                        pe.copy_(pe * d + p * (1.0 - d))
            if step % 25 == 0:
                v = (float(l1.item()), float(ld.item()), float(lf.item()) if lf is not None else 0.0, float(lv.item()) if lv is not None else 0.0)
                ema = v if ema is None else tuple(0.9 * e + 0.1 * x for e, x in zip(ema, v))
                if step % 100 == 0:
                    rec = dict(step=step, mel_l1=round(ema[0], 4), dur_mse=round(ema[1], 4), f0_l1=round(ema[2], 4), voicing_bce=round(ema[3], 4), lr=lr, minutes=round((time.time() - t0) / 60, 2))
                    print(json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
            if step % a.eval_every == 0:
                for tag, m_ in [("raw", model)] + ([("ema", ema_model)] if ema_model is not None else []):
                    vl1, vld, vlf, vlv = evaluate(m_, val, a.device, a.max_frames)
                    print(json.dumps(dict(step=step, val_mel_l1=round(vl1, 4), val_dur_mse=round(vld, 4), val_f0_l1=vlf and round(vlf, 4), val_voicing_bce=vlv and round(vlv, 4), which=tag)), flush=True)
                    m_.save_file(os.path.join(a.out, "tinytts.oxd" if tag == "raw" else "tinytts_ema.oxd"))
                    if vlf is not None and vlf < best_f0:  # pitch generalises worse than the spectrogram: also keep the best-pitch weights
                        best_f0 = vlf
                        m_.save_file(os.path.join(a.out, "tinytts_best_f0.oxd"))
                        print(json.dumps(dict(step=step, best_f0=tag, best_val_f0_l1=round(vlf, 4), val_mel_l1=round(vl1, 4))), flush=True)
                    if vl1 < best_val:  # the best weights by validation loss are kept apart: later steps may overfit
                        best_val = vl1
                        m_.save_file(os.path.join(a.out, "tinytts_best.oxd"))
                        print(json.dumps(dict(step=step, best=tag, best_val_mel_l1=round(vl1, 4))), flush=True)
    vl1, vld, vlf, vlv = evaluate(model, val, a.device, a.max_frames)
    model.save_file(os.path.join(a.out, "tinytts.oxd"))
    if vl1 < best_val:
        best_val = vl1
        model.save_file(os.path.join(a.out, "tinytts_best.oxd"))
    ema_val = None
    if ema_model is not None:
        ema_val = evaluate(ema_model, val, a.device, a.max_frames)[0]
        ema_model.save_file(os.path.join(a.out, "tinytts_ema.oxd"))
        if ema_val < best_val:
            best_val = ema_val
            ema_model.save_file(os.path.join(a.out, "tinytts_best.oxd"))
    summary = dict(best_val_mel_l1=round(best_val, 4), ema_val_mel_l1=ema_val, params=model.num_params(), steps=step, minutes=round((time.time() - t0) / 60, 1), val_mel_l1=round(vl1, 4), val_dur_mse=round(vld, 4),
                   val_f0_l1=vlf, val_voicing_bce=vlv)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
