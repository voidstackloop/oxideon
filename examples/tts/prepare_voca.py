"""Turns the tiny-voca dataset (CC BY 4.0, Muhammed Nazeem: https://huggingface.co/datasets/n4ze3m/tiny-voca) into the arrays
`train_tts.py --voca` trains on: log-mel frames, phoneme tokens with stress, forced-aligned durations (in frames), a speaker id,
and per-frame pitch (log F0 and a voicing confidence, from the Rust YIN tracker `ox.audio.pitch_track`).

    python examples/tts/prepare_voca.py --root runs/tinyvoca --out runs/voca/data

`--root` holds `data/*.parquet`, `batch2/data/*.parquet` and `annotations/v4/*.jsonl` (see fetch_tinyvoca.py).
Needs `ffmpeg` on the PATH to decode the FLAC audio inside the Parquet files.
"""
import argparse
import glob
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq

import oxideon as ox

SR, N_FFT, HOP, N_MELS = 24000, 1024, 256, 80


def decode(flac: bytes) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"], input=flac, capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


_W = {}


def _init(ann, audio=False):
    _W["ann"], _W["audio"] = ann, audio
    _W["spec"] = ox.audio.Spectrogram(n_fft=N_FFT, hop_length=HOP, power=1.0)
    _W["fb"] = ox.audio.mel_filter_bank(n_mels=N_MELS, n_fft=N_FFT, sample_rate=SR).astype(np.float32)


def _process_file(path):
    """All annotated clips of one Parquet file -> [(split, clip dict)], in file order."""
    ann, spec_fn, fb = _W["ann"], _W["spec"], _W["fb"]
    voices = {"F2": 0, "M2": 1}
    out = []
    t = pq.read_table(path, columns=["id", "audio"]).to_pylist()
    t = [r for r in t if r["id"] in ann]
    with ThreadPoolExecutor(4) as pool:
        wavs = list(pool.map(lambda r: decode(r["audio"]["bytes"]), t))
    for r, wav in zip(t, wavs):
        m = ann[r["id"]]
        frames = int(m["frames"])
        mag = spec_fn(ox.from_numpy(wav)).to_numpy()  # [freq, frames]
        mel = np.log(np.maximum(fb @ mag, 1e-5)).T.astype(np.float32)
        if abs(mel.shape[0] - frames) > 2:
            continue
        mel = np.concatenate([mel, np.repeat(mel[-1:], max(0, frames - mel.shape[0]), 0)])[:frames]
        f0, conf = (x.to_numpy() for x in ox.audio.pitch_track(ox.from_numpy(wav), sample_rate=SR, hop=HOP, fmin=60.0, fmax=450.0))
        f0 = np.concatenate([f0, np.repeat(f0[-1:], max(0, frames - len(f0)))])[:frames]
        conf = np.concatenate([conf, np.zeros(max(0, frames - len(conf)), np.float32)])[:frames]
        d = np.asarray(m["durations"], np.int16)
        if int(d.sum()) != frames:  # forced alignment rounding: the last phoneme takes the difference
            d[-1] = max(0, d[-1] + frames - int(d.sum()))
        c = dict(id=r["id"], voice=voices[m["speaker"]], mel=mel.astype(np.float16), f0=f0.astype(np.float32),
                 conf=conf.astype(np.float32), tokens=np.asarray(m["tokens"], np.int16), stress=np.asarray(m["stress"], np.uint8), dur=d)
        if _W.get("audio"):  # int16 samples, exactly frames * HOP long (mel frame t is centred on sample t * HOP)
            w16 = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
            c["wav"] = np.concatenate([w16, np.zeros(max(0, frames * HOP - len(w16)), np.int16)])[:frames * HOP]
        out.append((m["split"], c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N annotated clips per split (for tests)")
    ap.add_argument("--audio", action="store_true", help="also write audio_<split>.npz (int16 waveforms) for train_vocoder.py")
    ap.add_argument("--workers", type=int, default=1, help="processes, one Parquet file each (set RAYON_NUM_THREADS so workers x threads ~ cores)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ann = {}
    for split in ("train", "validation", "test"):
        with open(os.path.join(a.root, "annotations", "v4", split + ".jsonl"), encoding="utf8") as f:
            rows = [json.loads(l) for l in f]
        ann.update({r["id"]: r for r in rows[: a.limit]})
    print(len(ann), "annotated clips", flush=True)
    clips = {"train": [], "validation": [], "test": []}
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(a.root, "data", "*.parquet")) + glob.glob(os.path.join(a.root, "batch2", "data", "*.parquet")))
    if a.workers > 1:  # one process per Parquet file (the Rust pitch tracker / STFT hold the GIL, so threads do not scale)
        from multiprocessing import get_context
        pool = get_context("fork").Pool(a.workers, initializer=_init, initargs=(ann, a.audio))
        results = pool.imap(_process_file, files)
    else:
        _init(ann, a.audio)
        results = map(_process_file, files)
    for fi, res in enumerate(results):
        for split, c in res:
            clips[split].append(c)
        print(f"{fi + 1}/{len(files)} files, {sum(map(len, clips.values()))} clips, {time.time() - t0:.0f}s", flush=True)
    tr = np.concatenate([c["mel"].astype(np.float32) for c in clips["train"][::5]])
    mel_mean, mel_std = float(tr.mean()), float(tr.std())
    # log-F0 statistics of confidently voiced frames, per speaker
    f0_stats = {}
    for v in (0, 1):
        x = np.concatenate([np.log(c["f0"][c["conf"] > 0.8]) for c in clips["train"] if c["voice"] == v])
        f0_stats[str(v)] = [float(x.mean()), float(x.std())]

    def save(name, subset):
        off = lambda key: np.cumsum([0] + [len(r[key]) for r in subset]).astype(np.int64)
        np.savez(os.path.join(a.out, name + ".npz"), mel=np.concatenate([r["mel"] for r in subset]), mel_off=off("mel"),
                 f0=np.concatenate([r["f0"] for r in subset]), conf=np.concatenate([r["conf"] for r in subset]),
                 ids=np.concatenate([r["tokens"] for r in subset]), ws=np.concatenate([r["stress"] for r in subset]),
                 dur=np.concatenate([r["dur"] for r in subset]), ph_off=off("tokens"), voice=np.array([r["voice"] for r in subset], dtype=np.int8))
        if a.audio:
            np.savez(os.path.join(a.out, f"audio_{name}.npz"), wav=np.concatenate([r["wav"] for r in subset]), wav_off=off("wav"))
        print(f"{name}: {len(subset)} clips, {sum(len(r['mel']) for r in subset) * HOP / SR / 3600:.2f} hours", flush=True)

    for name, subset in clips.items():
        save(name, subset)
    n_tokens = 1 + max(int(c["tokens"].max()) for s in clips.values() for c in s)
    meta = dict(sample_rate=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, mel_mean=mel_mean, mel_std=mel_std, n_phones=n_tokens,
                silence_id=3, n_flags=int(max(int(c["stress"].max()) for s in clips.values() for c in s)) + 1, f0_stats=f0_stats,
                voices=["F2", "M2"], dataset="tiny-voca (CC BY 4.0, Muhammed Nazeem)")
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("meta", json.dumps({k: v for k, v in meta.items() if k != "voices"}), flush=True)


if __name__ == "__main__":
    main()
