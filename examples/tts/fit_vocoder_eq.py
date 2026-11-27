"""Fits a fixed correction filter (EQ) for the harmonic vocoder: fed the TRUE spectrum of a real clip, `harmonic_synth` comes out
muffled (too little energy above ~1 kHz). The gain per STFT bin is the energy-weighted mean log-magnitude gap between real clips
and their vocoded copies, smoothed over frequency, refined over a few rounds. `synthesize_voca.py --eq` multiplies the envelope by it.

    python examples/tts/fit_vocoder_eq.py --data voca/data --clips "voca/eval_pf3000/*_M2_real.wav" --out voca/data/vocoder_eq_M2.npy
"""
import argparse
import glob
import json
import os
import sys
import wave

import numpy as np

import oxideon as ox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_voca import SR, HOP  # noqa: E402


def read(path):
    w = wave.open(path)
    return (np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768).astype(np.float32)


def vocode(wav, meta, eq):
    """(real STFT magnitudes, vocoded STFT magnitudes) `[bins, frames]` for the true mel / pitch of `wav`, envelope times `eq`."""
    n_fft, nm = meta["n_fft"], meta["n_mels"]
    spec = ox.audio.Spectrogram(n_fft=n_fft, hop_length=HOP, power=1.0)
    fb = ox.audio.mel_filter_bank(n_mels=nm, n_fft=n_fft, sample_rate=SR).astype(np.float32)
    mag = spec(ox.from_numpy(wav)).to_numpy()
    logmel = np.log(np.maximum(fb @ mag, 1e-5)).T.astype(np.float32)
    f0, conf = (t.to_numpy() for t in ox.audio.pitch_track(ox.from_numpy(wav), SR, HOP, 60.0, 450.0))
    T = min(len(f0), len(logmel))
    env = ox.audio.mel_envelope(logmel[:T], nm, n_fft, SR) * eq[None, :]
    out = ox.audio.harmonic_synth(f0[:T].astype(np.float32), (conf[:T] > 0.7).astype(np.float32), env.astype(np.float32), SR, n_fft, HOP).to_numpy()
    vmag = spec(ox.from_numpy(np.ascontiguousarray(out.astype(np.float32)))).to_numpy()
    n = min(mag.shape[1], vmag.shape[1])
    return mag[:, :n], vmag[:, :n]


def gap(clips, meta, eq):
    """Energy-weighted mean log-magnitude gap (real - vocoded) per bin."""
    num, den = 0.0, 0.0
    for w in clips:
        r, v = vocode(w, meta, eq)
        wgt = r.sum(0, keepdims=True)
        num = num + ((np.log(r + 1e-4) - np.log(v + 1e-4)) * wgt).sum(1)
        den = den + wgt.sum()
    return num / den


def band_share(clips, meta, eq, edges=(0, 1000, 4000, 8000, 12000)):
    f = np.fft.rfftfreq(meta["n_fft"], 1 / SR)
    out = {"real": [], "vocoded": []}
    for w in clips:
        for key, m in zip(out, vocode(w, meta, eq)):
            e = (m ** 2).sum(1)
            out[key].append([e[(f >= a) & (f < b)].sum() / e.sum() for a, b in zip(edges[:-1], edges[1:])])
    return {k: [round(float(x), 4) for x in np.mean(v, 0)] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--clips", required=True, help="glob of real recordings (wav, 24 kHz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    paths = sorted(glob.glob(a.clips))
    clips = [read(p) for p in paths]
    fit, held = clips[0::2], clips[1::2]  # fit on half, report on the other half
    eq = np.ones(meta["n_fft"] // 2 + 1, np.float32)
    print("before (held-out band shares 0-1k / 1-4k / 4-8k / 8-12k Hz):", band_share(held, meta, eq), flush=True)
    k = np.hanning(15) / np.hanning(15).sum()
    for r in range(a.rounds):
        g = np.convolve(np.pad(gap(fit, meta, eq), 7, mode="edge"), k, mode="valid")
        eq = np.clip(eq * np.exp(0.8 * g), 0.25, 8.0).astype(np.float32)
        print(f"round {r + 1}: mean |gap| {np.abs(g).mean():.3f}", flush=True)
    print("after  (held-out):", band_share(held, meta, eq), flush=True)
    np.save(a.out, eq)
    print("wrote", a.out, "gain at 0.5/2/5/10 kHz:", [round(float(eq[int(f / SR * meta["n_fft"])]), 2) for f in (500, 2000, 5000, 10000)])


if __name__ == "__main__":
    main()
