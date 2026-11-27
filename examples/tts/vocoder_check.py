"""Analysis-synthesis test of the vocoders on real clips: true log-mel (+ true pitch) in, waveform out, then the log-mel of the
result is compared with the original. This isolates the vocoder from the acoustic model.

    python examples/tts/vocoder_check.py --root runs/tinyvoca --out runs/voca/vocoder_check
"""
import argparse
import glob
import os
import subprocess
import wave

import numpy as np
import pyarrow.parquet as pq

import oxideon as ox

SR, N_FFT, HOP, N_MELS = 24000, 1024, 256, 80


def decode(b):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"], input=b, capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


def write_wav(path, x):
    x = x / max(1e-6, float(np.abs(x).max())) * 0.9
    with wave.open(path, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--gl-iters", type=int, default=60)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = sorted(glob.glob(os.path.join(a.root, "data", "validation-*.parquet")))[0]
    rows = pq.read_table(path, columns=["id", "audio"]).to_pylist()[: a.clips]
    spec = ox.audio.Spectrogram(n_fft=N_FFT, hop_length=HOP, power=1.0)
    fb = ox.audio.mel_filter_bank(n_mels=N_MELS, n_fft=N_FFT, sample_rate=SR).astype(np.float32)
    logmel = lambda w: np.log(np.maximum(fb @ spec(ox.from_numpy(np.ascontiguousarray(w))).to_numpy(), 1e-5)).T
    clips = []
    for r in rows:
        w = decode(r["audio"]["bytes"])
        mel = logmel(w).astype(np.float32)
        f0, conf = (t.to_numpy() for t in ox.audio.pitch_track(ox.from_numpy(w), SR, HOP, 60.0, 450.0))
        n = min(len(f0), len(mel))
        clips.append((r["id"], w, mel[:n], f0[:n], conf[:n]))
    print("clips:", [(c[0][-2:], round(len(c[1]) / SR, 1), f"voiced {np.mean(c[4] > 0.8):.2f}", f"f0 med {np.median(c[3][c[4] > 0.8]):.0f}") for c in clips])

    def score(make):
        d = []
        for _, w, mel, f0, conf in clips:
            y = make(mel, f0, conf)
            m2 = logmel(y)
            n = min(len(m2), len(mel))
            d.append(np.abs(m2[:n] - mel[:n]).mean())
        return float(np.mean(d))

    def harmonic(gain, breath, thr):
        def make(mel, f0, conf):
            env = ox.audio.mel_envelope(mel, N_MELS, N_FFT, SR)
            v = np.clip((conf - thr) / 0.25, 0, 1)
            return ox.audio.harmonic_synth(f0, v, env, SR, N_FFT, HOP, gain, breath, 0).to_numpy()
        return make

    def griffin(mel, f0, conf):
        mag = np.maximum(np.linalg.pinv(fb) @ np.exp(mel).T, 0).astype(np.float32)
        return ox.audio.griffin_lim(ox.from_numpy(np.ascontiguousarray(mag)), n_fft=N_FFT, hop_length=HOP, n_iter=a.gl_iters, seed=0).to_numpy()

    print(f"Griffin-Lim ({a.gl_iters} it) from true mel:  log-mel L1 = {score(griffin):.3f}")
    best = None
    for gain in (1.0, 1.4, 1.8, 2.4):
        for breath in (0.2, 0.4):
            s = score(harmonic(gain, breath, 0.55))
            print(f"harmonic gain {gain} breath {breath}: log-mel L1 = {s:.3f}")
            if best is None or s < best[0]:
                best = (s, gain, breath)
    print("best", best)
    _, gain, breath = best
    for name, w, mel, f0, conf in clips[:4]:
        write_wav(os.path.join(a.out, f"{name}_original.wav"), w)
        write_wav(os.path.join(a.out, f"{name}_harmonic.wav"), harmonic(gain, breath, 0.55)(mel, f0, conf))
        write_wav(os.path.join(a.out, f"{name}_griffinlim.wav"), griffin(mel, f0, conf))


if __name__ == "__main__":
    main()
