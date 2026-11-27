"""Re-times the forced alignment of the arrays made by `prepare_voca.py`: every phone boundary moves (by at most --max-shift
frames) to where the normalised log-mel actually changes (`ox.audio.refine_boundaries`, Rust). The CTC aligner behind
tiny-voca places boundaries a few frames off and leaves ~23 % of phones 1-2 frames long; a model trained on that timing learns
smeared transitions (its spectrum changes ~0.72x as fast as real speech).

    python examples/tts/realign_voca.py --data voca/data --out voca/data_ra --workers 16

Writes the same files with new `dur`; everything else is copied. Prints, per split, how far boundaries moved, the share of 1-2
frame phones, and how sharply the real spectrum changes at the boundaries (1.00 = like a random frame) before / after.
"""
import argparse
import json
import os
import shutil
from multiprocessing import Pool

import numpy as np

import oxideon as ox

_W = {}


def _init(path, mean, std, max_shift, min_len, penalty):
    z = np.load(path)
    _W.update(mel=z["mel"], mel_off=z["mel_off"], dur=z["dur"], ph_off=z["ph_off"], mean=np.asarray(mean, np.float32),
              std=np.asarray(std, np.float32), args=(max_shift, min_len, penalty))


def _one(i):
    w = _W
    feats = (w["mel"][w["mel_off"][i]:w["mel_off"][i + 1]].astype(np.float32) - w["mean"]) / w["std"]
    d = w["dur"][w["ph_off"][i]:w["ph_off"][i + 1]].astype(np.int64)
    if int(d.sum()) != len(feats):
        return d
    return ox.audio.refine_boundaries(feats, d, *w["args"])


def boundary_sharpness(mel, mel_off, dur, ph_off, n):
    """Mean spectral change INTO a boundary frame, relative to a random frame."""
    at, rnd = [], []
    rng = np.random.default_rng(0)
    for i in range(n):
        m = mel[mel_off[i]:mel_off[i + 1]].astype(np.float32)
        if len(m) < 20:
            continue
        flux = np.r_[0, np.abs(np.diff(m, axis=0)).mean(1)]
        b = np.cumsum(dur[ph_off[i]:ph_off[i + 1]].astype(int))[:-1]
        b = b[(b > 2) & (b < len(m) - 2)]
        at.extend(flux[b])
        rnd.extend(flux[rng.integers(2, len(m) - 2, max(len(b), 1))])
    return float(np.mean(at) / np.mean(rnd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shift", type=int, default=4)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--penalty", type=float, default=1.0, help="cost per frame a boundary moves (in squared normalised log-mel)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--splits", default="train,val,validation,test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    for f in os.listdir(a.data):
        if not f.endswith(".npz"):
            shutil.copy(os.path.join(a.data, f), os.path.join(a.out, f))
    for split in a.splits.split(","):
        src = os.path.join(a.data, split + ".npz")
        if not os.path.exists(src):
            continue
        z = dict(np.load(src))
        n = len(z["voice"])
        with Pool(a.workers, _init, (src, meta["mel_mean"], meta["mel_std"], a.max_shift, a.min_len, a.penalty)) as pool:
            new = list(pool.imap(_one, range(n), chunksize=64))
        old = z["dur"]
        z["dur"] = np.concatenate(new).astype(old.dtype)
        assert len(z["dur"]) == len(old)
        moved = np.abs(np.cumsum(z["dur"].astype(np.int64)) - np.cumsum(old.astype(np.int64)))
        k = min(n, 400)
        rep = dict(split=split, clips=n, boundaries_moved=round(float((moved > 0).mean()), 3), mean_shift_frames=round(float(moved.mean()), 2),
                   short_phones_before=round(float(((old > 0) & (old <= 2)).mean()), 3), short_phones_after=round(float(((z["dur"] > 0) & (z["dur"] <= 2)).mean()), 3),
                   sharpness_before=round(boundary_sharpness(z["mel"], z["mel_off"], old, z["ph_off"], k), 3),
                   sharpness_after=round(boundary_sharpness(z["mel"], z["mel_off"], z["dur"], z["ph_off"], k), 3))
        print(json.dumps(rep), flush=True)
        np.savez(os.path.join(a.out, split + ".npz"), **z)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
