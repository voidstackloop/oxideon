"""A/B of the pitch source for an end-to-end (train_e2e.py) model: the same validation sentences (true durations) rendered with the
recording's own pitch, the TinyTTS pitch head, and (optionally) a PitchFlow sample, so a listener can tell whether a robotic
sound comes from the pitch or from the waveform head.

    python examples/tts/pitch_ab.py --data voca/data --model tts_e2e.oxd --head head.oxd --pitchflow pitchflow.oxd --out ab --voice 1
"""
import argparse
import json
import os
import sys

import numpy as np

import oxideon as ox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_tts import Data  # noqa: E402
from train_e2e import forward, predicted_f0, true_f0  # noqa: E402
from synthesize_voca import load, load_head, load_pitchflow, flow_pitch, write_wav  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--pitchflow", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", type=int, default=1)
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    val = Data(os.path.join(a.data, "validation.npz"), meta)
    tts, _ = load(a.data, a.model)
    head = load_head(a.head)
    flow = load_pitchflow(a.pitchflow) if a.pitchflow else None
    mu, sd = meta["f0_stats"][str(a.voice)]
    pick = [i for i in np.flatnonzero(val.voice == a.voice) if 200 <= val.frames[i] <= 330][:a.n]
    for k, i in enumerate(pick):
        t = int(val.frames[i])
        b = val.collate([int(i)], "cpu")
        noise = np.random.default_rng(0).standard_normal((1, t * meta["hop"]))
        with ox.no_grad():
            enc, _ = tts.encode(b["ph"], b["ws"], b["voice"], b["ph_mask"])
            hidden = tts.decode_hidden(enc, b["A"], b["phase"], b["fmask"])
            _, aux = tts.outputs(hidden)
            pitches = {"1_true_pitch": true_f0(val, i, t), "2_head_pitch": predicted_f0(aux, a.voice, meta, t)}
            if flow is not None:
                lf, vo = flow_pitch(flow, hidden, b["fmask"], 0.7, 16, k)
                v = np.flatnonzero(vo > 0.5)
                hz = np.exp(lf * sd + mu)
                pitches["3_pitchflow"] = np.clip(np.interp(np.arange(t), v, hz[v]) if len(v) else hz, 50, 600)
            for name, f0 in pitches.items():
                w = head(hidden, f0[None], [t], noise=noise, phase=np.zeros(1)).to_numpy()[0]
                write_wav(os.path.join(a.out, f"s{k}_{name}.wav"), w, meta["sample_rate"])
                print(k, name, "spread %.1f semitones" % (12 * np.std(np.log2(f0))))


if __name__ == "__main__":
    main()
