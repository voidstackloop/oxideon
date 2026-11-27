"""Quality ceiling of `SourceFilterHead`: fits its per-frame controls (cepstra + gains) directly to a real recording with the
training losses, no TTS in between. If the fit still sounds robotic, the head's design is the limit, not the model's training.

    python examples/tts/head_ceiling.py --wav real.wav --out fit.wav --steps 400
"""
import argparse
import os
import sys
import wave

import numpy as np

import oxideon as ox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_e2e import spectral  # noqa: E402
from train_vocoder import Spectra  # noqa: E402
from synthesize_voca import write_wav  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    a = ap.parse_args()
    w = wave.open(a.wav)
    sr, hop = w.getframerate(), 256
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    T = len(x) // hop
    x = x[:T * hop]
    f0, c = (t.to_numpy() for t in ox.audio.pitch_track(ox.from_numpy(x), sample_rate=sr, hop=hop, fmin=60.0, fmax=450.0))
    f0, c = f0[:T], c[:T]
    v = np.flatnonzero(c > 0.8)
    f0 = np.clip(np.exp(np.interp(np.arange(T), v, np.log(f0[v]))), 50, 600)[None]
    head = ox.models.SourceFilterHead(dim=1, sample_rate=sr, hop=hop)
    hc = ox.nn.Parameter(ox.from_numpy(np.tile(head.h_b.to_numpy(), (1, T, 1)).astype(np.float32)))
    nc = ox.nn.Parameter(ox.from_numpy(np.tile(head.n_b.to_numpy(), (1, T, 1)).astype(np.float32)))
    sp = Spectra("cpu", sr)
    y = ox.from_numpy(x[None])
    opt = ox.optim.Adam([hc, nc], lr=a.lr)
    noise = np.random.default_rng(0).standard_normal((1, T * hop))
    lens = np.array([T])
    for step in range(a.steps):
        opt.zero_grad()
        stft, mel = spectral(sp, head.synth(hc, nc, f0, lens, noise=noise, phase=np.zeros(1)), y, lens, hop, "cpu")
        (stft * 2.0 + mel * 10.0).backward()
        opt.step()
        if step % 100 == 0 or step == a.steps - 1:
            print(step, "stft %.3f wave_mel %.3f" % (float(stft.item()), float(mel.item())), flush=True)
    with ox.no_grad():
        write_wav(a.out, head.synth(hc, nc, f0, lens, noise=noise, phase=np.zeros(1)).to_numpy()[0], sr)


if __name__ == "__main__":
    main()
