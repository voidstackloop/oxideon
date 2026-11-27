"""Objective check of a TinyTTS trained on tiny-voca: speaks held-out test sentences and compares them with the real recordings.

    python examples/tts/evaluate_voca.py --data runs/voca/data --model runs/voca/model/tinytts.oxd \
        --annotations runs/tinyvoca/annotations/v4 --root runs/tinyvoca --out runs/voca/eval --n 40

Writes, per clip, `<id>_ours.wav` (harmonic source-filter vocoder), `<id>_gl.wav` (Griffin-Lim on the same mel) and `<id>_real.wav`, plus
`eval.json` (text, voice, and pitch / voicing / duration statistics against the real clip). `asr_wer.py` then scores intelligibility.
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
from synthesize_voca import build_lexicon, load, synthesize, write_wav, load_pitchflow, load_vocoder, load_head  # noqa: E402
from prepare_voca import decode, SR, HOP  # noqa: E402


def stats(wav):
    f0, conf = (t.to_numpy() for t in ox.audio.pitch_track(ox.from_numpy(np.ascontiguousarray(wav.astype(np.float32))), SR, HOP, 60.0, 500.0))
    v = conf > 0.7
    return dict(seconds=len(wav) / SR, voiced=float(v.mean()), f0_median=float(np.median(f0[v])) if v.any() else 0.0,
                f0_p5=float(np.percentile(f0[v], 5)) if v.any() else 0.0, f0_p95=float(np.percentile(f0[v], 95)) if v.any() else 0.0), f0, v


def oracle(wav, meta):
    """Vocoders fed the TRUE mel / pitch / voicing of a real clip: what the vocoder alone can do, with a perfect acoustic model."""
    n_fft, nm = meta["n_fft"], meta["n_mels"]
    spec = ox.audio.Spectrogram(n_fft=n_fft, hop_length=HOP, power=1.0)
    fb = ox.audio.mel_filter_bank(n_mels=nm, n_fft=n_fft, sample_rate=SR).astype(np.float32)
    mag = spec(ox.from_numpy(np.ascontiguousarray(wav))).to_numpy()
    logmel = np.log(np.maximum(fb @ mag, 1e-5)).T.astype(np.float32)
    f0, conf = (t.to_numpy() for t in ox.audio.pitch_track(ox.from_numpy(np.ascontiguousarray(wav)), SR, HOP, 60.0, 450.0))
    f0, conf = f0[:len(logmel)], conf[:len(logmel)]
    logmel = logmel[:len(f0)]
    env = ox.audio.mel_envelope(logmel, nm, n_fft, SR)
    ours = ox.audio.harmonic_synth(f0.astype(np.float32), (conf > 0.7).astype(np.float32), env, SR, n_fft, HOP).to_numpy()
    m = np.maximum(np.linalg.pinv(fb) @ np.exp(logmel).T, 0).astype(np.float32)
    gl = ox.audio.griffin_lim(ox.from_numpy(np.ascontiguousarray(m)), n_fft=n_fft, hop_length=HOP, n_iter=60, seed=0).to_numpy()
    return ours, gl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--root", required=True, help="tiny-voca dataset root (parquet files with the real audio)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--pitchflow", default=None, help="a PitchFlow checkpoint: sampled melody instead of the regression pitch head")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--eq", default=None, help="vocoder correction filter (.npy) from fit_vocoder_eq.py")
    ap.add_argument("--specflow", default=None, help="a SpecFlow checkpoint (train_pitchflow.py --target mel)")
    ap.add_argument("--vocoder", default=None, help="a NeuralVocoder checkpoint (train_vocoder.py)")
    ap.add_argument("--sf-head", default=None, help="a SourceFilterHead (train_e2e.py), with that run's TinyTTS as --model")
    ap.add_argument("--jitter", type=float, default=0.0, help="--sf-head: per-cycle relative period deviation of the voice source")
    ap.add_argument("--shimmer", type=float, default=0.0, help="--sf-head: per-cycle relative amplitude deviation of the voice source")
    ap.add_argument("--voiced-vowels", action="store_true", help="keep vowel frames voiced (train_e2e.py --voiced-vowels models)")
    ap.add_argument("--sharpen", type=float, default=1.0, help="--sf-head: formant post-filter strength (1 = off)")
    ap.add_argument("--frontend", default=None, help="frontend.json from learn_frontend.py (needs --cmudict); 'ours' then uses it")
    ap.add_argument("--cmudict", default=None)
    ap.add_argument("--speed", type=float, default=0.91, help="speaking-rate factor for the text-driven variants (<1 = slower); 0.91 makes the median duration equal the real recordings")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, meta = load(a.data, a.model)
    pf = dict(pitchflow=load_pitchflow(a.pitchflow), temperature=a.temperature) if a.pitchflow else {}
    if a.eq:
        pf["eq"] = np.load(a.eq)
    if a.specflow:
        pf["specflow"] = load_pitchflow(a.specflow)
    if a.vocoder:
        pf["vocoder"] = load_vocoder(a.vocoder)
    if a.sf_head:
        pf["voiced_vowels"] = a.voiced_vowels
        pf["head"] = load_head(a.sf_head, jitter=a.jitter, shimmer=a.shimmer, sharpen=a.sharpen)
    lex = build_lexicon(a.annotations, os.path.join(a.data, "lexicon.json"))
    fe = None
    if a.frontend:
        from frontend_cmu import CmuFrontEnd, load_cmudict
        fe = CmuFrontEnd.load(a.frontend, load_cmudict(a.cmudict), lex)
    rows = [json.loads(l) for l in open(os.path.join(a.annotations, "test.jsonl"), encoding="utf8")]
    voices = {"F2": 0, "M2": 1}
    rng = np.random.default_rng(0)
    pick = []
    for sp in ("F2", "M2"):  # an equal number per voice, short enough to be fast
        cand = [r for r in rows if r["speaker"] == sp and 25 <= len(r["text"]) <= 140]
        pick += [cand[i] for i in rng.choice(len(cand), min(a.n // 2, len(cand)), replace=False)]
    want = {r["id"] for r in pick}
    import pyarrow.parquet as pq
    real = {}
    for path in sorted(glob.glob(os.path.join(a.root, "data", "*.parquet")) + glob.glob(os.path.join(a.root, "batch2", "data", "*.parquet"))):
        for r in pq.read_table(path, columns=["id", "audio"]).to_pylist():
            if r["id"] in want:
                real[r["id"]] = decode(r["audio"]["bytes"])
        if len(real) == len(want):
            break
    out = []
    for r in pick:
        if r["id"] not in real:
            continue
        v = voices[r["speaker"]]
        res = synthesize(model, meta, lex, r["text"], voice=v, speed=a.speed, compare=True, frontend=fe, **pf)
        rec = dict(id=r["id"], text=r["text"], voice=v, unknown=res["unknown"])
        lexwav = synthesize(model, meta, lex, r["text"], voice=v, speed=a.speed, **pf)["wav"] if fe else res["wav"]  # the old label-lexicon front end
        o_ours, o_gl = oracle(real[r["id"]], meta)
        gt = synthesize(model, meta, lex, r["text"], voice=v, phonemes=(r["tokens"], r["stress"]), **pf)["wav"]  # the dataset's own phoneme labels
        for tag, wav in (("ours", res["wav"]), ("ours_lex", lexwav), ("ours_gt", gt), ("gl", res["griffinlim"]), ("real", real[r["id"]]), ("oracle_ours", o_ours), ("oracle_gl", o_gl)):
            write_wav(os.path.join(a.out, f"{r['id']}_{tag}.wav"), wav, SR)
            rec[tag] = stats(wav)[0]
        out.append(rec)
        print(f"{len(out)}/{len(pick)} {r['id']}: {rec['ours']['seconds']:.1f}s vs real {rec['real']['seconds']:.1f}s", flush=True)
    json.dump(out, open(os.path.join(a.out, "eval.json"), "w"), indent=1)

    def agg(tag, key, v=None):
        xs = [c[tag][key] for c in out if v is None or c["voice"] == v]
        return float(np.median(xs))

    summary = {}
    for v, name in ((0, "F2"), (1, "M2")):
        summary[name] = {tag: {k: round(agg(tag, k, v), 3) for k in ("seconds", "voiced", "f0_median", "f0_p5", "f0_p95")} for tag in ("ours", "ours_lex", "ours_gt", "gl", "real", "oracle_ours", "oracle_gl")}
    ratio = [c["ours"]["seconds"] / c["real"]["seconds"] for c in out]
    summary["duration_ratio_median"] = round(float(np.median(ratio)), 3)
    summary["duration_ratio_p10_p90"] = [round(float(np.percentile(ratio, 10)), 3), round(float(np.percentile(ratio, 90)), 3)]
    summary["unknown_word_clips"] = sum(1 for c in out if c["unknown"])
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
